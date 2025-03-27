"""
Portfolio Management Module

This module provides tools for managing a portfolio of positions,
tracking performance, calculating aggregate risk metrics, and
applying portfolio-level constraints.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .position import Position, OptionPosition
from .position_inventory import PositionInventory


class Portfolio:
    """
    Manages a collection of positions with portfolio-level metrics and constraints.
    
    This class tracks a portfolio of positions, calculates aggregate metrics like
    exposure, risk, and performance, and enforces portfolio-level constraints.
    """
    
    def __init__(
        self, 
        initial_capital: float,
        max_position_size_pct: float = 0.25,
        max_portfolio_delta: float = 0.20, 
        logger: Optional[logging.Logger] = None,
        margin_calculator = None
    ):
        """
        Initialize the portfolio.
        
        Args:
            initial_capital: Initial capital in dollars
            max_position_size_pct: Maximum single position size as percentage of portfolio
            max_portfolio_delta: Maximum absolute portfolio delta as percentage of portfolio value
            logger: Logger instance
            margin_calculator: Margin calculator instance
        """
        self.logger = logger or logging.getLogger('trading_engine')
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.max_position_size_pct = max_position_size_pct
        self.max_portfolio_delta = max_portfolio_delta
        
        # Store the margin calculator
        self.margin_calculator = margin_calculator
        
        # Initialize position inventory (centralized position management)
        self.inventory = PositionInventory(logger=self.logger)
        
        # For backward compatibility - reference the positions from inventory
        self.positions = self.inventory.positions
        
        # Performance tracking
        self.transactions: List[Dict[str, Any]] = []
        self.daily_returns: List[Dict[str, Any]] = []
        self.equity_history: Dict[datetime, float] = {}
        
        # Track position value separately from cash
        self.position_value = 0
        self.total_value = initial_capital
        
        # Initial equity history entry
        self.equity_history[datetime.now()] = initial_capital
        
        # Track realized PnL for the current day
        self.today_realized_pnl = 0
        
        # Add daily metrics dictionary for tracking
        self.daily_metrics = {}
        
        # Log initialization parameters in a standardized format
        self.logger.info("=" * 40)
        self.logger.info(f"PORTFOLIO INITIALIZATION")
        self.logger.info(f"  Initial capital: ${initial_capital:,.2f}")
        self.logger.info(f"  Max position size: {max_position_size_pct:.2%} of portfolio")
        self.logger.info(f"  Max portfolio delta: {max_portfolio_delta:.2%} of portfolio value")
        
        # Log margin calculator details if available
        if self.margin_calculator:
            # Get a user-friendly name for the margin calculator
            calculator_class_name = type(self.margin_calculator).__name__
            calculator_display_name = calculator_class_name
            if calculator_class_name == "SPANMarginCalculator":
                calculator_display_name = "SPAN"
            elif calculator_class_name == "OptionMarginCalculator":
                calculator_display_name = "Option"
            elif calculator_class_name == "MarginCalculator":
                calculator_display_name = "Basic"
            
            self.logger.info(f"  Margin calculator: {calculator_display_name}")
            
            # Log margin calculator's max leverage if available
            if hasattr(self.margin_calculator, 'max_leverage'):
                self.logger.info(f"  Max leverage: {self.margin_calculator.max_leverage:.2f}x")
        else:
            self.logger.info("  Margin calculator: None (will be set later)")
            
        self.logger.info("=" * 40)
    
    def set_margin_calculator(self, margin_calculator):
        """
        Set the margin calculator for the portfolio.
        
        Args:
            margin_calculator: Margin calculator instance
        """
        self.margin_calculator = margin_calculator
        
        # Get a user-friendly name for the margin calculator
        calculator_class_name = type(margin_calculator).__name__
        calculator_display_name = calculator_class_name
        if calculator_class_name == "SPANMarginCalculator":
            calculator_display_name = "SPAN"
        elif calculator_class_name == "OptionMarginCalculator":
            calculator_display_name = "Option"
        elif calculator_class_name == "MarginCalculator":
            calculator_display_name = "Basic"
            
        self.logger.info(f"Portfolio margin calculator set to: {calculator_display_name}")
    
    # Add dictionary-like access methods for easier reporting
    def keys(self):
        """
        Return the portfolio attribute keys for dictionary-like access.
        Used by reporting system to access portfolio data.
        """
        # Return the main attributes that should be accessible 
        return [
            'initial_capital', 'cash_balance', 'positions', 'transactions',
            'daily_returns', 'equity_history', 'position_value', 'total_value',
            'daily_metrics'
        ]
    
    def items(self):
        """
        Return portfolio attributes as (key, value) pairs for dictionary-like access.
        Used by reporting system for iteration.
        """
        return [(key, getattr(self, key)) for key in self.keys()]
    
    def __getitem__(self, key):
        """
        Make the Portfolio object subscriptable (e.g., portfolio['cash_balance']).
        
        Args:
            key: The attribute name to access
            
        Returns:
            The attribute value
            
        Raises:
            KeyError: If the attribute doesn't exist
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Portfolio has no attribute '{key}'")
    
    def get(self, key, default=None):
        """
        Get a portfolio attribute with a default value if not found.
        Used by reporting system for safe access to attributes.
        
        Args:
            key: Attribute name
            default: Default value if attribute doesn't exist
            
        Returns:
            The attribute value or default
        """
        return getattr(self, key, default)
        
    def get_daily_return(self, date=None):
        """
        Get the daily return for a specific date or the latest daily return if no date provided.
        Used by performance reporting functions.
        
        Args:
            date: The date to get the return for (optional, defaults to latest date)
            
        Returns:
            float: Daily return as a decimal (0.01 = 1%)
        """
        # If no date provided, use the latest date in equity history
        if date is None:
            if hasattr(self, 'daily_returns') and self.daily_returns:
                # Return the most recent daily return
                return self.daily_returns[-1].get('return', 0.0)
            elif self.equity_history:
                # Calculate from equity history
                dates = sorted(self.equity_history.keys())
                if len(dates) > 1:
                    prev_value = self.equity_history[dates[-2]]
                    current_value = self.equity_history[dates[-1]]
                    if prev_value > 0:
                        return (current_value - prev_value) / prev_value
            return 0.0
            
        # Check if we have daily metrics for this date
        if hasattr(self, 'daily_metrics') and date in self.daily_metrics:
            # If we have a previous date's value, calculate return
            prev_dates = [d for d in self.daily_metrics.keys() if d < date]
            if prev_dates:
                prev_date = max(prev_dates)
                prev_value = self.daily_metrics[prev_date].get('portfolio_value', self.initial_capital)
                current_value = self.daily_metrics[date].get('portfolio_value', self.initial_capital)
                
                if prev_value > 0:
                    return (current_value - prev_value) / prev_value
                    
        # Check equity history
        if date in self.equity_history:
            dates = sorted(self.equity_history.keys())
            date_idx = dates.index(date)
            if date_idx > 0:
                prev_value = self.equity_history[dates[date_idx - 1]]
                current_value = self.equity_history[date]
                
                if prev_value > 0:
                    return (current_value - prev_value) / prev_value
        
        # Default to no return
        return 0.0
    
    def get_daily_return_percent(self, date=None):
        """
        Get the daily return for a specific date as a percentage.
        
        Args:
            date: The date to get the return for (optional, defaults to latest date)
            
        Returns:
            float: Daily return as a percentage (1.0 = 1%)
        """
        # Get the return as a decimal and convert to percentage
        return self.get_daily_return(date) * 100.0
    
    def get_returns_series(self):
        """
        Get a series of all daily returns for reporting purposes.
        
        Returns:
            dict: Dictionary with dates as keys and return values (as decimals) as values
        """
        returns_dict = {}
        
        # First try to use daily_returns if available
        if hasattr(self, 'daily_returns') and self.daily_returns:
            for return_entry in self.daily_returns:
                # Ensure dates are converted to strings for consistent handling
                date_key = return_entry['date']
                if not isinstance(date_key, str):
                    date_key = date_key.strftime('%Y-%m-%d')
                returns_dict[date_key] = float(return_entry.get('return', 0.0))
        
        # If no daily returns, calculate from equity history
        elif self.equity_history:
            dates = sorted(self.equity_history.keys())
            for i in range(1, len(dates)):
                prev_value = float(self.equity_history[dates[i-1]])
                current_value = float(self.equity_history[dates[i]])
                
                # Ensure dates are converted to strings for consistent handling
                date_key = dates[i]
                if not isinstance(date_key, str):
                    date_key = date_key.strftime('%Y-%m-%d')
                    
                if prev_value > 0:
                    returns_dict[date_key] = float((current_value - prev_value) / prev_value)
                else:
                    returns_dict[date_key] = 0.0
                    
        # If no equity history either, check daily metrics
        elif hasattr(self, 'daily_metrics') and self.daily_metrics:
            dates = sorted(self.daily_metrics.keys())
            for i in range(1, len(dates)):
                prev_value = float(self.daily_metrics[dates[i-1]].get('portfolio_value', self.initial_capital))
                current_value = float(self.daily_metrics[dates[i]].get('portfolio_value', self.initial_capital))
                
                # Ensure dates are converted to strings for consistent handling
                date_key = dates[i]
                if not isinstance(date_key, str):
                    date_key = date_key.strftime('%Y-%m-%d')
                
                if prev_value > 0:
                    returns_dict[date_key] = float((current_value - prev_value) / prev_value)
                else:
                    returns_dict[date_key] = 0.0
        
        return returns_dict
    
    def get_option_pnl(self):
        """
        Get option P&L data for reporting purposes.
        
        Returns:
            dict: Dictionary with total option P&L and by position
        """
        pnl_data = {
            'total': 0.0,
            'by_position': {}
        }
        
        # Calculate total realized PnL from option positions
        for symbol, position in self.positions.items():
            if hasattr(position, 'realized_pnl'):
                position_pnl = float(position.realized_pnl)
                pnl_data['total'] += position_pnl
                pnl_data['by_position'][symbol] = position_pnl
            
        # Also include historical positions that were closed
        for transaction in self.transactions:
            if 'symbol' in transaction and 'pnl' in transaction:
                symbol = transaction['symbol']
                if symbol not in pnl_data['by_position']:
                    pnl_data['by_position'][symbol] = 0.0
                pnl_data['by_position'][symbol] += float(transaction.get('pnl', 0.0))
        
        return pnl_data
    
    def get_hedge_pnl(self):
        """
        Get hedge P&L data for reporting purposes.
        
        Returns:
            dict: Dictionary with total hedge P&L and by position
        """
        hedge_pnl_data = {
            'total': 0.0,
            'by_position': {}
        }
        
        # Filter transactions for hedge positions
        for transaction in self.transactions:
            if 'symbol' in transaction and 'pnl' in transaction and transaction.get('is_hedge', False):
                symbol = transaction['symbol']
                if symbol not in hedge_pnl_data['by_position']:
                    hedge_pnl_data['by_position'][symbol] = 0.0
                    
                pnl = float(transaction.get('pnl', 0.0))
                hedge_pnl_data['by_position'][symbol] += pnl
                hedge_pnl_data['total'] += pnl
        
        return hedge_pnl_data
    
    def get_trade_history(self):
        """
        Get the trade history in a format suitable for reporting.
        
        Returns:
            list: List of trade dictionaries with standardized fields
        """
        trade_history = []
        
        for transaction in self.transactions:
            # Convert all numeric values to simple floats
            trade = {}
            for key, value in transaction.items():
                if isinstance(value, (int, float)):
                    trade[key] = float(value)
                elif key == 'date' and not isinstance(value, str):
                    # Convert date to string format
                    trade[key] = value.strftime('%Y-%m-%d')
                else:
                    trade[key] = value
            
            trade_history.append(trade)
        
        return trade_history
    
    def get_equity_history_as_list(self):
        """
        Get a standardized equity history in a list format for reporting.
        This avoids issues with inconsistent data structures in reporting.
        
        Returns:
            List of tuples: [(date_str, equity_value), ...]
        """
        # Convert equity history to a list of (date, value) tuples
        # Use string dates for consistent handling in reports
        equity_list = []
        
        # Start with initial value
        equity_list.append(('initial', float(self.initial_capital)))
        
        # Sort dates to ensure chronological order
        for date in sorted(self.equity_history.keys()):
            # Convert date to string for consistent handling
            date_str = date
            if not isinstance(date, str):
                date_str = date.strftime('%Y-%m-%d')
                
            # Ensure value is a simple float, not numpy or other array-like type
            value = float(self.equity_history[date])
            
            # Add to list
            equity_list.append((date_str, value))
        
        return equity_list
    
    def record_daily_metrics(self, date):
        """
        Record daily portfolio metrics for the given date.
        This method is called by the trading engine to track portfolio performance.
        
        Args:
            date: The date to record metrics for
        """
        portfolio_value = self.get_portfolio_value()
        cash = getattr(self, 'cash_balance', self.initial_capital)
        
        # Initialize daily metrics dictionary if it doesn't exist
        if not hasattr(self, 'daily_metrics'):
            self.daily_metrics = {}
            
        # Initialize daily_returns if it doesn't exist
        if not hasattr(self, 'daily_returns'):
            self.daily_returns = []
            
        # Ensure all values are simple types (not numpy arrays or other objects)
        # This prevents issues when saving metrics to reports
        position_value = float(self.position_value) if hasattr(self, 'position_value') else 0.0
        
        # Record metrics
        self.daily_metrics[date] = {
            'date': date,
            'portfolio_value': float(portfolio_value),  # Convert to simple float
            'cash': float(cash),  # Convert to simple float
            'position_value': position_value  # Use the converted value
        }
        
        # Also update equity history
        if not hasattr(self, 'equity_history'):
            self.equity_history = {}
            
        self.equity_history[date] = float(portfolio_value)  # Convert to simple float
        
        # Filter equity history to only include backtest dates (not current time)
        # This ensures we're only using dates from the backtest for calculations
        backtest_dates = [d for d in self.equity_history.keys() if isinstance(d, (datetime, pd.Timestamp)) and d.year < 2025]
        backtest_dates.sort()
        
        if len(backtest_dates) > 1:
            # Find the previous date in the backtest
            current_index = backtest_dates.index(date) if date in backtest_dates else -1
            
            if current_index > 0:  # We have a previous date
                prev_date = backtest_dates[current_index - 1]
                prev_value = self.equity_history[prev_date]
                prev_position_value = self.daily_metrics.get(prev_date, {}).get('position_value', 0)
                
                if prev_value > 0:
                    daily_return = (portfolio_value - prev_value) / prev_value
                    daily_pnl = portfolio_value - prev_value
                    
                    # Calculate P&L components more accurately, similar to trade summary
                    # Option P&L: sum of price differences * contracts * 100 for option positions
                    option_pnl = 0.0
                    equity_pnl = 0.0
                    
                    for symbol, position in self.positions.items():
                        if hasattr(position, 'previous_day_price') and hasattr(position, 'current_price'):
                            if isinstance(position, OptionPosition):
                                # For options, calculate P&L based on price change (matching _log_pre_trade_summary)
                                price_diff = position.previous_day_price - position.current_price if position.is_short else position.current_price - position.previous_day_price
                                # Use previous_day_contracts instead of current contracts for P&L calculation
                                pos_option_pnl = price_diff * position.previous_day_contracts * 100
                                option_pnl += pos_option_pnl
                                if self.logger and self.logger.level <= logging.DEBUG:
                                    self.logger.debug(f"Debug - Option P&L in record_metrics for {symbol}: previous_day_price={position.previous_day_price}, current_price={position.current_price}, previous_day_contracts={position.previous_day_contracts}, P&L=${pos_option_pnl:.2f}")
                            elif hasattr(position, 'contracts'):
                                # For equities, calculate P&L based on price change
                                # Use previous_day_contracts instead of current contracts for P&L calculation
                                pos_equity_pnl = position.previous_day_contracts * (position.current_price - position.previous_day_price)
                                equity_pnl += pos_equity_pnl
                                if self.logger and self.logger.level <= logging.DEBUG:
                                    self.logger.debug(f"Debug - Equity P&L in record_metrics for {symbol}: previous_day_price={position.previous_day_price}, current_price={position.current_price}, previous_day_contracts={position.previous_day_contracts}, P&L=${pos_equity_pnl:.2f}")
                    
                    # Cash/other P&L is what's left after accounting for options and equities
                    cash_pnl = daily_pnl - option_pnl - equity_pnl
                    
                    # Add to daily returns list
                    return_entry = {
                        'date': date,
                        'return': float(daily_return),  # Convert to simple float
                        'portfolio_value': float(portfolio_value),  # Convert to simple float
                        'pnl': float(daily_pnl),  # Add the actual PnL value
                        'option_pnl': float(option_pnl),
                        'equity_pnl': float(equity_pnl),
                        'cash_pnl': float(cash_pnl)
                    }
                    
                    # Log the components for debugging - use debug level to not clutter output
                    self.logger.debug(f"Daily PnL components (record_metrics): Total=${daily_pnl:.2f}, Options=${option_pnl:.2f}, Equity=${equity_pnl:.2f}, Cash/Other=${cash_pnl:.2f}")
                    
                    self.daily_returns.append(return_entry)
                    
                    # Reset today's realized PnL for the next day
                    if hasattr(self, 'today_realized_pnl'):
                        self.today_realized_pnl = 0
    
    def get_portfolio_value(self) -> float:
        """
        Get the current total portfolio value.
        
        Returns:
            float: Portfolio value in dollars
        """
        # Update portfolio value before returning
        self._update_portfolio_value()
        return self.total_value
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio metrics.
        
        Returns:
            dict: Dictionary of portfolio metrics
        """
        # Calculate portfolio value
        portfolio_value = self.get_portfolio_value()
        
        # Calculate Greeks
        greeks = self.get_portfolio_greeks()
        
        # Calculate exposure metrics using margin calculator when available
        total_margin = 0
        margin_by_position = {}
        hedging_benefits = 0
        
        # Log margin calculation approach
        if hasattr(self, 'logger'):
            if self.margin_calculator:
                # Get user-friendly calculator type name
                calc_class_name = type(self.margin_calculator).__name__
                calc_display_name = calc_class_name
                
                # Ensure we correctly identify SPAN calculators - improved logic
                if "SPAN" in calc_class_name:
                    calc_display_name = "SPAN"
                elif hasattr(self.margin_calculator, 'is_delegating_to_span') and self.margin_calculator.is_delegating_to_span:
                    calc_display_name = "SPAN"
                elif hasattr(self.margin_calculator, 'initial_margin_percentage'):
                    calc_display_name = "SPAN"
                # Check if it's a basic calculator that delegates to SPAN
                elif calc_class_name == "MarginCalculator" and self.has_option_positions():
                    # Override to SPAN for option portfolios - the MarginCalculator delegates to SPAN internally
                    calc_display_name = "SPAN"
                elif calc_class_name == "OptionMarginCalculator":
                    calc_display_name = "Option"
                else:
                    calc_display_name = "Basic"
                    
                if self.logger.level <= logging.INFO:
                    self.logger.info(f"[Portfolio] Using {calc_display_name} margin calculation")
                elif self.logger.level <= logging.DEBUG:
                    self.logger.debug(f"[Portfolio] Using {calc_display_name} for margin calculation")
            else:
                if self.logger.level <= logging.DEBUG:
                    self.logger.debug("[Portfolio] No margin calculator set, using position-level calculation")
        
        # If margin calculator is available, use it for portfolio-level margin calculation
        if self.margin_calculator:
            try:
                # Use the portfolio margin calculator
                margin_result = self.margin_calculator.calculate_portfolio_margin(self.positions)
                
                # Extract results
                total_margin = margin_result.get('total_margin', 0)
                margin_by_position = margin_result.get('margin_by_position', {})
                hedging_benefits = margin_result.get('hedging_benefits', 0)
                
                if hasattr(self, 'logger') and self.logger.level <= logging.DEBUG:
                    # Get user-friendly calculator type name
                    calc_class_name = type(self.margin_calculator).__name__
                    calc_display_name = calc_class_name
                    
                    if "SPAN" in calc_class_name or hasattr(self.margin_calculator, 'initial_margin_percentage'):
                        calc_display_name = "SPAN"
                    elif calc_class_name == "OptionMarginCalculator":
                        calc_display_name = "Option"
                    elif calc_class_name == "MarginCalculator":
                        calc_display_name = "Basic"
                        
                    self.logger.debug(f"[Portfolio] Portfolio margin calculation using {calc_display_name}:")
                    self.logger.debug(f"  Total margin: ${total_margin:.2f}")
                    self.logger.debug(f"  Hedging benefits: ${hedging_benefits:.2f}")
                    # Add extra debug info for troubleshooting
                    self.logger.debug(f"[Portfolio] DEBUG - Margin calculator in metrics: {type(self.margin_calculator).__name__}")
                    if hasattr(self.margin_calculator, 'initial_margin_percentage'):
                        self.logger.debug(f"[Portfolio] DEBUG - Has initial_margin_percentage: {self.margin_calculator.initial_margin_percentage}")
                
            except Exception as e:
                # Log the error and fall back to position-level calculation
                if hasattr(self, 'logger'):
                    self.logger.error(f"[Portfolio] Error using portfolio margin calculator: {e}")
                    self.logger.info(f"[Portfolio] Falling back to position-level margin calculation")
                
                # Use position-level calculation as fallback
                for position in self.positions.values():
                    if hasattr(position, 'calculate_margin_requirement'):
                        position_margin = position.calculate_margin_requirement(1.0)  # Use basic margin without leverage
                        total_margin += position_margin
                        if hasattr(position, 'symbol'):
                            margin_by_position[position.symbol] = position_margin
        else:
            # No margin calculator available, use position-level margin calculation
            for position in self.positions.values():
                if hasattr(position, 'calculate_margin_requirement'):
                    position_margin = position.calculate_margin_requirement(1.0)  # Use basic margin without leverage
                    total_margin += position_margin
                    if hasattr(position, 'symbol'):
                        margin_by_position[position.symbol] = position_margin
        
        # Calculate available margin and leverage
        max_margin = portfolio_value  # Maximum margin is 1x NLV by default
        available_margin = max(max_margin - total_margin, 0)
        current_leverage = total_margin / portfolio_value if portfolio_value > 0 else 0
        
        # Create the metrics dictionary
        metrics = {
            'portfolio_value': portfolio_value,
            'cash_balance': self.cash_balance,
            'position_value': self.position_value,
            'position_count': len(self.positions),
            'realized_pnl': sum(p.realized_pnl for p in self.positions.values()),
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values()),
            'total_margin': total_margin,
            'available_margin': available_margin,
            'current_leverage': current_leverage,
            'max_leverage': 1.0,  # Default max leverage is 1x
            'net_liquidation_value': portfolio_value,
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'theta': greeks['theta'],
            'vega': greeks['vega'],
            'delta_pct': greeks['delta_pct'],
            'dollar_delta': greeks['dollar_delta'],
            'dollar_gamma': greeks['dollar_gamma'],
            'dollar_theta': greeks['dollar_theta'],
            'dollar_vega': greeks['dollar_vega'],
            'margin_by_position': margin_by_position,
            'hedging_benefits': hedging_benefits,
            'margin_calculator_type': type(self.margin_calculator).__name__ if self.margin_calculator else 'None'
        }
        
        return metrics
    
    def add_position(
        self,
        symbol: str,
        instrument_data: Dict[str, Any],
        quantity: int,
        price: Optional[float] = None,
        position_type: str = 'option',
        is_short: bool = False,
        execution_data: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        skip_margin_calc: bool = False
    ) -> Optional[Union[Position, OptionPosition]]:
        """
        Add a new position to the portfolio.
        
        Args:
            symbol: Instrument symbol
            instrument_data: Data about the instrument
            quantity: Number of contracts/shares
            price: Execution price
            position_type: Type of position (option/stock)
            is_short: Whether position is short
            execution_data: Additional execution data
            reason: Reason for adding the position
            skip_margin_calc: Whether to skip margin calculation (default: False)
            
        Returns:
            Position: The new position object
        """
        # Skip if quantity is zero
        if quantity == 0:
            self.logger.info(f"Skipping adding position with 0 quantity: {symbol}")
            return None
            
        # Check if we already have this position
        if symbol in self.positions:
            existing_position = self.positions[symbol]
            if existing_position.is_short == is_short:
                # Same direction - add to existing position
                # Calculate the cost/proceeds of the new contracts before adding
                position_value = self._calculate_position_value(position_type, quantity, price, is_short)
                
                # Update cash balance based on this new position addition
                if position_type.lower() == 'option':
                    # For options, the premium costs/generates cash
                    # Short options generate cash, long options cost cash
                    self.cash_balance += position_value if is_short else -position_value
                else:
                    # For stocks, we reduce cash by the position value for long positions
                    # For short positions, we assume margin borrowing requirements are handled separately
                    self.cash_balance -= position_value if not is_short else 0
                
                # Now add the contracts to the existing position
                existing_position.add_contracts(quantity, price)
                self.logger.info(f"Added {quantity} contracts to existing position: {symbol}")
                
                # Log the change in cash balance
                self.logger.info(f"  Position value: ${position_value:,.2f}")
                self.logger.info(f"  New cash balance: ${self.cash_balance:,.2f}")
                
                # Update portfolio value after adding to position
                self._update_portfolio_value()
                
                # Record the transaction
                transaction = {
                    'date': datetime.now(),
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'value': position_value,
                    'type': 'option' if isinstance(existing_position, OptionPosition) else 'stock',
                    'action': 'buy' if not is_short else 'sell',
                    'reason': reason or 'Add to Position'
                }
                self.transactions.append(transaction)
                
                return existing_position
            else:
                # Opposite direction - reduce or close existing position
                if quantity <= existing_position.contracts:
                    # Reduce position
                    existing_position.remove_contracts(quantity, price)
                    if existing_position.contracts == 0:
                        # Position closed - remove from portfolio
                        self.remove_position(symbol)
                        self.logger.info(f"Closed position: {symbol}")
                    else:
                        self.logger.info(f"Reduced position: {symbol} by {quantity} contracts")
                    return existing_position
                else:
                    # Close existing and open new in opposite direction
                    remaining = quantity - existing_position.contracts
                    self.remove_position(symbol)
                    self.logger.info(f"Closed and reversed position: {symbol}")
                    # Continue to create new position with remaining quantity
                    quantity = remaining

        # Create new position with the appropriate type
        try:
            if position_type.lower() == 'option':
                position = OptionPosition(
                    symbol=symbol,
                    option_data=instrument_data,
                    contracts=0,  # Start with 0 contracts and add them with add_contracts
                    entry_price=price,
                    is_short=is_short,
                    logger=self.logger
                )
            else:
                # Stock position
                position = Position(
                    symbol=symbol,
                    contracts=0,  # Start with 0 contracts and add them with add_contracts
                    entry_price=price,
                    is_short=is_short,
                    position_type='stock',
                    instrument_data=instrument_data,
                    logger=self.logger
                )

            # Add contracts to the position
            position.add_contracts(quantity, price)
            
            # Add position to the inventory
            self.inventory.add_position(position)
            
            # Update portfolio value and performance metrics
            self._update_portfolio_value()
            
            # Log position addition
            self.logger.info(f"Added position: {quantity} {'short' if is_short else 'long'} {position_type} {symbol} @ {price}")
            
            # Calculate and apply margin requirement (unless skipped)
            if not skip_margin_calc:
                margin_requirement = self.calculate_margin_requirement()
            
            # Check if we have enough cash for the position
            position_value = self._calculate_position_value(position_type, quantity, price, is_short)
            
            if position_type.lower() == 'option':
                # For options, the premium costs/generates cash
                # Short options generate cash, long options cost cash
                self.cash_balance += position_value if is_short else -position_value
            else:
                # For stocks, we reduce cash by the position value for long positions
                # For short positions, we assume margin borrowing requirements are handled separately
                self.cash_balance -= position_value if not is_short else 0
            
            self.logger.info(f"  Position value: ${position_value:,.2f}")
            self.logger.info(f"  New cash balance: ${self.cash_balance:,.2f}")
            
            # Record the transaction
            transaction = {
                'date': datetime.now(),
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'value': position_value,
                'type': 'option' if isinstance(position, OptionPosition) else 'stock',
                'action': 'buy' if not is_short else 'sell',
                'reason': reason or 'New Position'
            }
            self.transactions.append(transaction)
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
            return None
            
    def _calculate_position_value(self, position_type: str, quantity: int, price: float, is_short: bool) -> float:
        """
        Calculate the value of a position for cash balance updates.
        
        Args:
            position_type: Type of position (option/stock)
            quantity: Number of contracts/shares
            price: Execution price
            is_short: Whether position is short
            
        Returns:
            float: Position value
        """
        if position_type.lower() == 'option':
            return price * quantity * 100  # Options have multiplier of 100
        else:
            return price * quantity  # Stocks have no multiplier

    def remove_position(
        self, 
        symbol: str, 
        quantity: Optional[int] = None, 
        price: Optional[float] = None,
        execution_data: Optional[Dict[str, Any]] = None,
        reason: str = "Close",
        skip_margin_calc: bool = False
    ) -> float:
        """
        Remove a position or reduce its size.
        
        Args:
            symbol: Symbol of the position to remove
            quantity: Number of contracts to remove (None = all)
            price: Execution price (None = use current price)
            execution_data: Additional execution data
            reason: Reason for removing the position
            skip_margin_calc: Whether to skip margin calculation (default: False)
            
        Returns:
            float: Realized P&L from closing the position
        """
        if symbol not in self.positions:
            self.logger.warning(f"Cannot remove position {symbol} - not in portfolio")
            return 0.0
            
        position = self.positions[symbol]
        original_value = position.get_position_value()
        
        # Use current price if none provided
        if price is None:
            price = position.current_price
        
        # Default to closing the entire position
        if quantity is None or quantity >= position.contracts:
            quantity = position.contracts
            is_full_close = True
        else:
            is_full_close = False
        
        # Calculate P&L
        pnl = position.calculate_realized_pnl(quantity, price)
        self.today_realized_pnl += pnl
        
        # Update cash balance based on position type and direction
        if isinstance(position, OptionPosition):
            # For options, closing positions have the opposite cash impact of opening
            # Closing short options costs cash, closing long options generates cash
            self.cash_balance -= position.get_position_value(quantity) if position.is_short else -position.get_position_value(quantity)
        else:
            # For stocks, closing long positions generates cash, closing short positions costs cash
            self.cash_balance += position.get_position_value(quantity) if not position.is_short else -position.get_position_value(quantity)
        
        # Remove contracts from position
        if is_full_close:
            # Remove the entire position from inventory
            self.inventory.remove_position(symbol)
        else:
            # Just reduce the position size
            position.remove_contracts(quantity, price)
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Recalculate margin requirement (unless skipped)
        if not skip_margin_calc:
            self.calculate_margin_requirement()
        
        # Record transaction
        transaction = {
            'date': datetime.now(),
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': quantity * price * 100 if isinstance(position, OptionPosition) else quantity * price,
            'pnl': pnl,
            'type': 'option' if isinstance(position, OptionPosition) else 'stock',
            'action': 'buy' if position.is_short else 'sell',  # Closing action is opposite of position direction
            'reason': reason
        }
        self.transactions.append(transaction)
        
        # Log the operation
        self.logger.info(f"{'Closed' if is_full_close else 'Reduced'} position: {symbol} ({quantity} contracts at {price})")
        self.logger.info(f"  P&L: ${pnl:,.2f}")
        self.logger.info(f"  New cash balance: ${self.cash_balance:,.2f}")
        
        return pnl

    def update_market_data(self, market_data_by_symbol: Dict[str, Any], current_date: Optional[datetime] = None) -> None:
        """
        Update position data with latest market prices.
        
        Args:
            market_data_by_symbol: Latest market data by symbol
            current_date: Current trading date (optional)
        """
        if not market_data_by_symbol:
            self.logger.warning("No market data provided to update positions")
            return
            
        # Process each position
        for symbol, position in list(self.positions.items()):
            # Skip if no market data for this symbol
            if symbol not in market_data_by_symbol:
                continue
                
            market_data = market_data_by_symbol[symbol]
            
            # Update the position with new market data
            if isinstance(position, OptionPosition):
                # Handle option-specific data
                try:
                    # Update price
                    if hasattr(market_data, 'get'):
                        # Dictionary or object with get method
                        mid_price = market_data.get('Last', 0.0)
                        bid = market_data.get('Bid', 0.0)
                        ask = market_data.get('Ask', 0.0)
                        
                        # Calculate mid price if available
                        if bid > 0 and ask > 0:
                            mid_price = (bid + ask) / 2
                        
                        # Update Greeks if available
                        delta = market_data.get('Delta', position.current_delta)
                        gamma = market_data.get('Gamma', position.current_gamma)
                        theta = market_data.get('Theta', position.current_theta)
                        vega = market_data.get('Vega', position.current_vega)
                        
                        # Update underlying price
                        if 'UnderlyingPrice' in market_data:
                            position.underlying_price = market_data.get('UnderlyingPrice')
                    else:
                        # DataFrame row or other object with direct attribute access
                        mid_price = getattr(market_data, 'Last', 0.0)
                        bid = getattr(market_data, 'Bid', 0.0)
                        ask = getattr(market_data, 'Ask', 0.0)
                        
                        # Calculate mid price if available
                        if bid > 0 and ask > 0:
                            mid_price = (bid + ask) / 2
                        
                        # Update Greeks if available
                        delta = getattr(market_data, 'Delta', position.current_delta)
                        gamma = getattr(market_data, 'Gamma', position.current_gamma)
                        theta = getattr(market_data, 'Theta', position.current_theta)
                        vega = getattr(market_data, 'Vega', position.current_vega)
                        
                        # Update underlying price
                        if hasattr(market_data, 'UnderlyingPrice'):
                            position.underlying_price = market_data.UnderlyingPrice
                    
                    # Update position with new price and Greeks
                    position.update_market_data(
                        price=mid_price, 
                        delta=delta,
                        gamma=gamma,
                        theta=theta,
                        vega=vega
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error updating option position {symbol}: {e}")
            else:
                # Handle stock position
                try:
                    # Get latest price
                    if hasattr(market_data, 'get'):
                        # Dictionary or object with get method
                        price = market_data.get('Close', market_data.get('Last', position.current_price))
                    else:
                        # DataFrame row or other object with direct attribute access
                        price = getattr(market_data, 'Close', getattr(market_data, 'Last', position.current_price))
                    
                    # Update position with new price
                    position.update_market_data(price)
                    
                except Exception as e:
                    self.logger.error(f"Error updating stock position {symbol}: {e}")
        
        # Update portfolio value after updating all positions
        self._update_portfolio_value()

    def get_portfolio_greeks(self) -> Dict[str, float]:
        """
        Calculate portfolio-level Greek risk metrics.
        
        Returns:
            dict: Dictionary of portfolio Greeks with proper normalization
            (option delta and equity delta in the same unit of 'option contracts')
        """
        # Use the centralized position inventory to calculate Greeks
        # Get underlying price from any option position we have
        underlying_price = 100.0  # Default
        
        for position in self.inventory.option_positions.values():
            if hasattr(position, 'underlying_price') and position.underlying_price > 0:
                underlying_price = position.underlying_price
                break
        
        return self.inventory.get_portfolio_greeks(underlying_price)
    
    # Use the inventory methods for Greeks calculation
    def get_option_delta(self) -> float:
        """
        Get the total delta from option positions only.
        
        Returns:
            float: Option delta in option contract equivalents (negative for short positions)
        """
        return self.inventory.get_option_delta()

    def get_hedge_delta(self) -> float:
        """
        Get the delta from hedge positions (stocks used for hedging).
        
        Returns:
            float: Hedge delta in option contract equivalents (negative for short positions)
        """
        return self.inventory.get_stock_delta()

    def get_total_delta(self) -> float:
        """
        Get the total portfolio delta.
        
        Returns:
            float: Total portfolio delta
        """
        return self.inventory.get_total_delta()

    def _update_portfolio_value(self):
        """
        Update total portfolio value based on current positions.
        
        Net liquidation value (NLV) should equal:
        Cash Balance - Short Position Liabilities + Long Position Values
        """
        # Track liabilities (short positions) and assets (long positions) separately
        short_option_value = 0
        long_position_value = 0
        
        for pos in self.positions.values():
            if isinstance(pos, OptionPosition):
                # For options, calculate contract value (price * contracts * 100)
                position_value = pos.current_price * pos.contracts * 100
                
                if pos.is_short:
                    # For short options, this is a liability
                    short_option_value += position_value
                else:
                    # For long options, this is an asset
                    long_position_value += position_value
            else:
                # For non-options (like stocks)
                position_value = pos.current_price * pos.contracts
                
                if hasattr(pos, 'is_short') and pos.is_short:
                    short_option_value += position_value  # Short stock would be a liability
                else:
                    long_position_value += position_value  # Long stock would be an asset

        # Store values for reporting and other calculations
        self.short_option_value = short_option_value
        self.long_position_value = long_position_value
        
        # The total "position value" for reporting purposes
        self.position_value = long_position_value
        
        # Calculate net liquidation value (NLV)
        # NLV = Cash + Long Assets - Short Liabilities
        self.total_value = self.cash_balance + long_position_value - short_option_value

        # Record in equity history - use current date/time as key
        current_time = datetime.now()
        self.equity_history[current_time] = self.total_value

    def calculate_margin_requirement(self) -> float:
        """
        Calculate the total margin requirement for the portfolio.
        This method uses the portfolio's margin calculator if available.
        
        Returns:
            float: Total margin requirement
        """
        # If margin calculator is set, use it for consistent margin calculation
        if self.margin_calculator:
            try:
                # Get a user-friendly name for the margin calculator
                calc_class_name = type(self.margin_calculator).__name__
                calc_display_name = calc_class_name
                
                # Improved SPAN calculator detection logic
                if "SPAN" in calc_class_name:
                    calc_display_name = "SPAN"
                elif hasattr(self.margin_calculator, 'is_delegating_to_span') and self.margin_calculator.is_delegating_to_span:
                    calc_display_name = "SPAN"
                elif hasattr(self.margin_calculator, 'initial_margin_percentage'):
                    calc_display_name = "SPAN"
                # Check if we have option positions - in this case we should use SPAN
                elif calc_class_name == "MarginCalculator" and self.has_option_positions():
                    # Override to SPAN for option portfolios - assume the MarginCalculator delegates to SPAN internally
                    calc_display_name = "SPAN"
                    # Add is_delegating_to_span attribute to remember this decision
                    setattr(self.margin_calculator, 'is_delegating_to_span', True)
                elif calc_class_name == "OptionMarginCalculator":
                    calc_display_name = "Option"
                else:
                    calc_display_name = "Basic"
                
                if hasattr(self, 'logger'):
                    if self.logger.level <= logging.INFO:
                        self.logger.info(f"[Portfolio] Using {calc_display_name} margin calculation")
                    elif self.logger.level <= logging.DEBUG:
                        self.logger.debug(f"[Portfolio] Using {calc_display_name} for margin calculation")
                    # Add extra debug info only in debug level
                    if self.logger.level <= logging.DEBUG:
                        self.logger.debug(f"[Portfolio] DEBUG - Margin calculator actual class: {type(self.margin_calculator).__name__}")
                        if hasattr(self.margin_calculator, 'initial_margin_percentage'):
                            self.logger.debug(f"[Portfolio] DEBUG - Has initial_margin_percentage attribute")
                
                # Use the portfolio margin calculator's calculate_portfolio_margin method
                margin_result = self.margin_calculator.calculate_portfolio_margin(self.positions)
                if hasattr(self, 'logger') and self.logger.level <= logging.DEBUG:
                    self.logger.debug(f"[Portfolio] Portfolio margin calculation using {calc_display_name}:")
                    self.logger.debug(f"  Total margin: ${margin_result.get('total_margin', 0):.2f}")
                    self.logger.debug(f"  Hedging benefits: ${margin_result.get('hedging_benefits', 0):.2f}")
                    self.logger.debug(f"[Portfolio] DEBUG - Margin calculator in metrics: {calc_class_name}")
                    
                # Return the total margin from the result
                return margin_result.get('total_margin', 0)
            except Exception as e:
                # Log the error and fall back to position-level calculation
                if hasattr(self, 'logger'):
                    self.logger.error(f"[Portfolio] Error using margin calculator: {e}")
                    self.logger.info("[Portfolio] Falling back to position-level margin calculation")
        else:
            if hasattr(self, 'logger'):
                self.logger.warning(f"[Portfolio] No margin calculator set - using basic position-level calculation")
        
        # If no margin calculator or error, calculate each position's margin separately
        total_margin = 0
        for position in self.positions.values():
            if hasattr(position, 'calculate_margin_requirement'):
                margin = position.calculate_margin_requirement(1.0)  # Use basic margin without leverage
                total_margin += margin
                
        return total_margin
        
    def get_total_liability(self) -> float:
        """
        Calculate the total liability of the portfolio based on short positions.
        
        Returns:
            float: Total liability value in dollars
        """
        # If we've already calculated the short option value, use it
        if hasattr(self, 'short_option_value'):
            return self.short_option_value
            
        # Otherwise calculate it from positions
        total_liability = 0
        for position in self.positions.values():
            if position.is_short:
                # Use current price if available, otherwise use entry price
                price = position.current_price if position.current_price > 0 else position.avg_entry_price
                
                # Calculate position value based on position type
                if isinstance(position, OptionPosition):
                    position_value = price * position.contracts * 100
                else:
                    position_value = price * position.contracts
                    
                total_liability += position_value
        
        return total_liability

    def get_total_position_exposure(self) -> float:
        """
        Calculate the total position exposure as a ratio of portfolio value.
        
        Returns:
            float: Ratio of total position exposure to portfolio value (0-1)
        """
        # Calculate the notional value of all positions
        total_exposure = 0
        for position in self.positions.values():
            # Use current price if available, otherwise use entry price
            price = position.current_price if position.current_price > 0 else position.avg_entry_price
            
            if isinstance(position, OptionPosition):
                position_value = abs(price * position.contracts * 100)
            else:
                position_value = abs(price * position.contracts)
            
            total_exposure += position_value
        
        portfolio_value = self.get_portfolio_value()
        
        # Calculate exposure as percentage of portfolio value
        if portfolio_value > 0:
            return total_exposure / portfolio_value
        else:
            return 0
            
    def get_net_liquidation_value(self) -> float:
        """
        Get the net liquidation value (NLV) of the portfolio.
        
        NLV = Cash + Long Assets - Short Liabilities
        
        Returns:
            float: Net liquidation value in dollars
        """
        # Update portfolio value first (this will correctly calculate NLV)
        self._update_portfolio_value()
        
        # Return the calculated total value
        return self.total_value
        
    def get_available_margin(self) -> float:
        """
        Calculate available margin for new positions.
        
        Returns:
            float: Available margin in dollars
        """
        # NLV - Margin Requirements = Available Margin
        nlv = self.get_net_liquidation_value()
        margin_req = self.calculate_margin_requirement()
        
        # Make sure NLV is not less than zero
        if nlv <= 0:
            return 0
            
        return nlv - margin_req
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance and risk metrics for the portfolio.
        
        This is an alias of get_portfolio_metrics for backward compatibility.
        
        Returns:
            dict: Dictionary of portfolio performance metrics
        """
        # This is a wrapper around get_portfolio_metrics to maintain backward compatibility
        metrics = self.get_portfolio_metrics()
        
        # Add additional performance metrics that might be expected
        if metrics:
            # Calculate return metrics if we have equity history
            if hasattr(self, 'equity_history') and len(self.equity_history) > 1:
                dates = sorted(self.equity_history.keys())
                initial_value = self.equity_history[dates[0]]
                current_value = self.equity_history[dates[-1]]
                
                if initial_value > 0:
                    total_return = (current_value - initial_value) / initial_value
                    metrics['total_return'] = float(total_return)
                    
            # Add performance specific metrics that might not be in portfolio_metrics
            metrics['realized_pnl'] = sum(p.realized_pnl for p in self.positions.values())
            metrics['unrealized_pnl'] = sum(p.unrealized_pnl for p in self.positions.values())
            
            # Add risk metrics
            metrics['margin_utilization'] = metrics.get('total_margin', 0) / metrics.get('portfolio_value', 1) if metrics.get('portfolio_value', 0) > 0 else 0
            
            # Add any metrics specifically needed by the position sizing logic
            metrics['net_liquidation_value'] = self.get_net_liquidation_value()
            metrics['available_margin'] = self.get_available_margin()
            
        return metrics

    def has_option_positions(self) -> bool:
        """
        Check if the portfolio contains any option positions.
        
        Returns:
            bool: True if there are any option positions, False otherwise
        """
        for position in self.positions.values():
            if hasattr(position, 'option_type') or getattr(position, 'is_option', False):
                return True
        return False