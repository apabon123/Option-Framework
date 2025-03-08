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
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the portfolio.
        
        Args:
            initial_capital: Initial capital in dollars
            max_position_size_pct: Maximum single position size as percentage of portfolio
            max_portfolio_delta: Maximum absolute portfolio delta as percentage of portfolio value
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger('trading_engine')
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.max_position_size_pct = max_position_size_pct
        self.max_portfolio_delta = max_portfolio_delta
        
        # Track positions by symbol
        self.positions: Dict[str, Position] = {}
        
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
        
        # Log initialization
        self.logger.info(f"Portfolio initialized with ${initial_capital:,.2f} capital")
        self.logger.info(f"  Max position size: {max_position_size_pct:.1%} of portfolio")
        self.logger.info(f"  Max portfolio delta: {max_portfolio_delta:.1%} of portfolio value")
    
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
                    
                    # Calculate unrealized PnL change more accurately
                    unrealized_pnl_change = position_value - prev_position_value
                    realized_pnl = daily_pnl - unrealized_pnl_change
                    
                    # Add to daily returns list
                    return_entry = {
                        'date': date,
                        'return': float(daily_return),  # Convert to simple float
                        'portfolio_value': float(portfolio_value),  # Convert to simple float
                        'pnl': float(daily_pnl),  # Add the actual PnL value
                        'unrealized_pnl_change': float(unrealized_pnl_change),  # More accurate change
                        'realized_pnl': float(realized_pnl)  # Corrected realized PnL
                    }
                    
                    # Log the components for debugging
                    self.logger.debug(f"Daily PnL components: Total=${daily_pnl:.2f}, Unrealized=${unrealized_pnl_change:.2f}, Realized=${realized_pnl:.2f}")
                    
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
        
        # Calculate exposure metrics (margin based)
        total_margin = 0
        for position in self.positions.values():
            if hasattr(position, 'calculate_margin_requirement'):
                margin = position.calculate_margin_requirement(1.0)  # Use basic margin without leverage
                total_margin += margin
        
        # Calculate available margin and leverage
        max_margin = portfolio_value  # Maximum margin is 1x NLV by default
        available_margin = max(max_margin - total_margin, 0)
        current_leverage = total_margin / portfolio_value if portfolio_value > 0 else 0
        
        return {
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
            'dollar_vega': greeks['dollar_vega']
        }
    
    def add_position(
        self, 
        symbol: str, 
        instrument_data: Dict[str, Any],
        quantity: int = 1, 
        price: Optional[float] = None,
        position_type: str = 'option',
        is_short: bool = True,
        execution_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Position]:
        """
        Add a new position or add to an existing position.
        
        Args:
            symbol: Position symbol
            instrument_data: Dictionary of instrument data
            quantity: Number of contracts
            price: Execution price (None = use current price)
            position_type: Type of position ('option', 'stock', etc.)
            is_short: Whether this is a short position
            execution_data: Additional execution data
            
        Returns:
            Position: Added or updated position object, or None if failed
        """
        if quantity <= 0:
            self.logger.error(f"Cannot add position for {symbol}: Invalid quantity {quantity}")
            return None
            
        if price is None:
            # Get price from market data
            if hasattr(instrument_data, 'get'):
                price = instrument_data.get('MidPrice', 0)
            else:
                price = instrument_data['MidPrice'] if 'MidPrice' in instrument_data else 0
                
        if price <= 0:
            self.logger.error(f"Cannot add position for {symbol}: Invalid price ${price}")
            return None
            
        # Calculate position value
        position_value = price * quantity * 100 if position_type == 'option' else price * quantity
        
        # Check if position exceeds maximum allowed size
        portfolio_value = self.get_portfolio_value()
        if position_value > portfolio_value * self.max_position_size_pct:
            self.logger.warning(f"Position value ${position_value:,.2f} exceeds maximum allowed size (${portfolio_value * self.max_position_size_pct:,.2f})")
            
            # Reduce quantity to fit within maximum allowed size
            max_allowed_value = portfolio_value * self.max_position_size_pct
            if position_type == 'option':
                max_quantity = int(max_allowed_value / (price * 100))
            else:
                max_quantity = int(max_allowed_value / price)
                
            if max_quantity <= 0:
                self.logger.error("Cannot add position: Maximum allowed quantity is zero")
                return None
                
            self.logger.warning(f"Reduced quantity from {quantity} to {max_quantity}")
            quantity = max_quantity
            position_value = price * quantity * 100 if position_type == 'option' else price * quantity
        
        # Get existing position or create new one
        if symbol in self.positions:
            position = self.positions[symbol]
            position.add_contracts(quantity, price, execution_data)
        else:
            # Create appropriate position type
            if position_type.lower() == 'option':
                position = OptionPosition(symbol, instrument_data, 0, is_short, self.logger)
                position.add_contracts(quantity, price, execution_data)
            else:
                # Default to standard position for non-options
                position = Position(symbol, instrument_data, 0, is_short, self.logger)
                position.add_contracts(quantity, price, execution_data)
                
            # Add to portfolio
            self.positions[symbol] = position
        
        # Update cash balance
        cost = position_value
        if is_short:
            self.cash_balance += cost  # Short positions add cash (receive premium)
        else:
            self.cash_balance -= cost  # Long positions reduce cash (pay premium)
            
        # Record transaction
        transaction_date = datetime.now()
        if execution_data and 'date' in execution_data:
            transaction_date = execution_data['date']
            
        transaction = {
            'date': transaction_date,
            'symbol': symbol,
            'action': 'SELL' if is_short else 'BUY',
            'quantity': quantity,
            'price': price,
            'value': position_value,
            'type': position_type,
            'cash_balance': self.cash_balance
        }
        self.transactions.append(transaction)
        
        self.logger.info(f"Added position: {quantity} {'short' if is_short else 'long'} {position_type} {symbol} @ ${price:.2f}")
        self.logger.info(f"  Position value: ${position_value:,.2f}")
        self.logger.info(f"  New cash balance: ${self.cash_balance:,.2f}")
        
        # Update portfolio value immediately after adding the position
        self._update_portfolio_value()
        
        # Check if new portfolio delta exceeds limits
        self._check_portfolio_delta_constraint()
        
        return position
    
    def remove_position(
        self, 
        symbol: str, 
        quantity: Optional[int] = None, 
        price: Optional[float] = None,
        execution_data: Optional[Dict[str, Any]] = None,
        reason: str = "Close"
    ) -> float:
        """
        Remove all or part of a position.
        
        Args:
            symbol: Position symbol
            quantity: Quantity to remove (None = all)
            price: Execution price (None = use current price)
            execution_data: Additional execution data
            reason: Reason for closing
            
        Returns:
            float: Realized P&L from the closure
        """
        if symbol not in self.positions:
            self.logger.warning(f"Cannot remove position {symbol}: Position not found")
            return 0
            
        position = self.positions[symbol]
        
        # If quantity not specified, close entire position
        if quantity is None:
            quantity = position.contracts
            
        if quantity <= 0 or quantity > position.contracts:
            self.logger.warning(f"Invalid quantity to remove: {quantity} (current: {position.contracts})")
            return 0
            
        # If price not specified, use current price
        if price is None:
            price = position.current_price
            
        if price <= 0:
            self.logger.warning(f"Invalid price for removal: ${price}")
            return 0
            
        # Calculate the value of the removed contracts based on position type
        position_value = price * quantity
        if isinstance(position, OptionPosition):
            position_value *= 100  # Option contracts are x100
            
        # Update realized PnL
        pnl = position.remove_contracts(quantity, price, execution_data, reason)
        
        # Track today's realized PnL
        if hasattr(self, 'today_realized_pnl'):
            self.today_realized_pnl += pnl
        
        # Update cash balance - inverse of add position
        if position.is_short:
            self.cash_balance -= position_value  # Short closing reduces cash (pay to close)
        else:
            self.cash_balance += position_value  # Long closing adds cash (receive proceeds)
            
        # Record transaction
        transaction_date = datetime.now()
        if execution_data and 'date' in execution_data:
            transaction_date = execution_data['date']
            
        transaction = {
            'date': transaction_date,
            'symbol': symbol,
            'action': 'BUY' if position.is_short else 'SELL',  # Closing action is opposite of position type
            'quantity': quantity,
            'price': price,
            'value': position_value,
            'type': 'option' if isinstance(position, OptionPosition) else 'stock',
            'cash_balance': self.cash_balance,
            'pnl': pnl,
            'reason': reason
        }
        self.transactions.append(transaction)
        
        self.logger.info(f"Removed position {symbol}: {quantity} contracts at ${price:.2f}")
        self.logger.info(f"  Realized P&L: ${pnl:.2f}")
        self.logger.info(f"  New cash balance: ${self.cash_balance:,.2f}")
        
        # If all contracts removed, delete position
        if position.contracts == 0:
            self.logger.info(f"Position {symbol} closed entirely")
            del self.positions[symbol]
            
        # Update portfolio value immediately after removing the position
        self._update_portfolio_value()
            
        return pnl
    
    def update_market_data(self, market_data_by_symbol: Dict[str, Any], current_date: Optional[datetime] = None) -> None:
        """
        Update all positions with latest market data.
        
        Args:
            market_data_by_symbol: Dictionary of market data by symbol
            current_date: Current date for this update (optional). If None, no POST-TRADE summary will be logged.
        """
        # Skip if there are no positions
        if not self.positions:
            return
            
        # Store portfolio value before update for return calculation
        prev_value = self.get_portfolio_value()
        
        # Reset today's realized PnL
        self.today_realized_pnl = 0
        
        # Keep track of positions updated
        updated_positions = []
        positions_to_remove = []
        
        # Extract underlying prices for consistency
        underlying_prices = {}
        for symbol, market_data in market_data_by_symbol.items():
            if hasattr(market_data, 'get'):
                underlying_symbol = market_data.get('UnderlyingSymbol', 'SPY')
                underlying_price = market_data.get('UnderlyingPrice')
                if underlying_price and underlying_symbol not in underlying_prices:
                    underlying_prices[underlying_symbol] = underlying_price
        
        # Update each position with latest market data
        for symbol, position in list(self.positions.items()):
            if symbol in market_data_by_symbol:
                # Get market data for this position
                market_data = market_data_by_symbol[symbol]
                
                # Store previous market data
                position.prev_price = position.current_price
                
                # Extract market data
                if hasattr(market_data, 'get'):
                    # Dict-like object
                    # First check for MidPrice, then try to calculate from Bid/Ask
                    price = market_data.get('MidPrice')
                    if price is None or price == 0:
                        bid = market_data.get('Bid', 0)
                        ask = market_data.get('Ask', 0)
                        if bid > 0 and ask > 0:
                            price = (bid + ask) / 2
                        elif bid > 0:
                            price = bid
                        elif ask > 0:
                            price = ask
                        else:
                            # If still no price, use entry price as last resort
                            price = position.avg_entry_price
                    
                    delta = market_data.get('Delta', position.current_delta)
                    gamma = market_data.get('Gamma', position.current_gamma)
                    theta = market_data.get('Theta', position.current_theta)
                    vega = market_data.get('Vega', position.current_vega)
                    days_to_expiry = market_data.get('DaysToExpiry', position.days_to_expiry if hasattr(position, 'days_to_expiry') else None)
                    
                    # Get underlying symbol and use consistent price
                    underlying_symbol = market_data.get('UnderlyingSymbol', 'SPY')
                    underlying_price = underlying_prices.get(underlying_symbol, market_data.get('UnderlyingPrice'))
                else:
                    # Pandas Series or DataFrame row
                    # First check for MidPrice, then try to calculate from Bid/Ask
                    if 'MidPrice' in market_data and market_data['MidPrice'] > 0:
                        price = market_data['MidPrice']
                    elif 'Bid' in market_data and 'Ask' in market_data and market_data['Bid'] > 0 and market_data['Ask'] > 0:
                        price = (market_data['Bid'] + market_data['Ask']) / 2
                    elif 'Bid' in market_data and market_data['Bid'] > 0:
                        price = market_data['Bid']
                    elif 'Ask' in market_data and market_data['Ask'] > 0:
                        price = market_data['Ask']
                    elif 'Last' in market_data and market_data['Last'] > 0:
                        price = market_data['Last']
                    else:
                        # If still no price, use entry price as last resort
                        price = position.avg_entry_price
                    
                    delta = market_data['Delta'] if 'Delta' in market_data else position.current_delta
                    gamma = market_data['Gamma'] if 'Gamma' in market_data else position.current_gamma
                    theta = market_data['Theta'] if 'Theta' in market_data else position.current_theta
                    vega = market_data['Vega'] if 'Vega' in market_data else position.current_vega
                    days_to_expiry = market_data['DaysToExpiry'] if 'DaysToExpiry' in market_data else (position.days_to_expiry if hasattr(position, 'days_to_expiry') else None)
                    
                    # Get underlying symbol and use consistent price
                    underlying_symbol = market_data['UnderlyingSymbol'] if 'UnderlyingSymbol' in market_data else 'SPY'
                    underlying_price = underlying_prices.get(underlying_symbol, market_data['UnderlyingPrice'] if 'UnderlyingPrice' in market_data else None)
                
                # Update position with new market data
                position.current_price = price
                position.current_delta = delta
                position.current_gamma = gamma
                position.current_theta = theta
                position.current_vega = vega
                
                # Update underlying price if available
                if underlying_price is not None:
                    position.underlying_price = underlying_price
                
                # Update days to expiry for option positions
                if isinstance(position, OptionPosition) and days_to_expiry is not None:
                    position.days_to_expiry = days_to_expiry
                    
                    # Check for expired options
                    if days_to_expiry <= 0:
                        positions_to_remove.append((symbol, "Expired"))
                
                # Recalculate unrealized P&L
                position.update_unrealized_pnl()
                
                # Mark position as updated
                updated_positions.append(symbol)
            else:
                self.logger.debug(f"No market data available for {symbol}")
                
        # Log update summary only if current_date is provided
        if updated_positions and current_date is not None:
            self.logger.debug(f"Updated market data for {len(updated_positions)} positions")
            
            # Calculate new portfolio value
            current_value = self.get_portfolio_value()
            daily_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
            
            # Log daily return if date is provided
            self.logger.info("===========================================")
            self.logger.info(f"POST-TRADE Summary [{current_date.strftime('%Y-%m-%d')}]:")
            self.logger.info(f"Daily P&L: ${current_value - prev_value:.0f} ({daily_return:.2%})")
            self.logger.info(f"  Option PnL: ${current_value - prev_value - self.today_realized_pnl:.0f}")
            self.logger.info(f"  Realized PnL: ${self.today_realized_pnl:.0f}")
            self.logger.info(f"Open Trades: {len(self.positions)}")
            
            # Calculate total exposure as percentage of NLV
            total_exposure = 0
            for pos in self.positions.values():
                if isinstance(pos, OptionPosition):
                    pos_value = abs(pos.current_price * pos.contracts * 100)
                else:
                    pos_value = abs(pos.current_price * pos.contracts)
                total_exposure += pos_value
            
            exposure_pct = total_exposure / current_value if current_value > 0 else 0
            self.logger.info(f"Total Position Exposure: {exposure_pct:.1%} of NLV")
            self.logger.info(f"Net Liq: ${current_value:.0f}")
            
            # Get portfolio metrics for additional information
            metrics = self.get_portfolio_metrics()
            self.logger.info(f"  Cash Balance: ${metrics['cash_balance']:.0f}")
            self.logger.info(f"  Total Liability: ${metrics['position_value']:.0f}")
            self.logger.info(f"Total Margin Requirement: ${metrics['total_margin']:.0f}")
            self.logger.info(f"Available Margin: ${metrics['available_margin']:.0f}")
            self.logger.info(f"Margin-Based Leverage: {metrics['current_leverage']:.2f}")
            
            # Portfolio Greeks section
            self.logger.info("\nPortfolio Greek Risk:")
            self.logger.info(f"  Option Delta: {metrics['delta']:.3f} (${metrics['dollar_delta']:.2f})")
            self.logger.info(f"  Gamma: {metrics['gamma']:.6f} (${metrics['dollar_gamma']:.2f} per 1% move)")
            self.logger.info(f"  Theta: ${metrics['dollar_theta']:.2f} per day")
            self.logger.info(f"  Vega: ${metrics['dollar_vega']:.2f} per 1% IV")
            
            # Performance metrics if we have enough data
            if len(self.daily_returns) >= 5:
                perf_metrics = self.get_performance_metrics()
                self.logger.info("\nRolling Metrics:")
                self.logger.info(f"  Sharpe: {perf_metrics.get('sharpe_ratio', 0):.2f}, Volatility: {perf_metrics.get('volatility', 0):.2%}")
            
            # Open positions table
            self.logger.info("\nOpen Trades Table:")
            self.logger.info("-" * 120)
            self.logger.info(f"{'Symbol':<20}{'Contracts':>10}{'Entry':>8}{'Current':>10}{'Value':>10}{'NLV%':>8}{'Underlying':>10}{'Delta':>10}{'Gamma':>10}{'Theta':>10}{'Vega':>10}{'Margin':>10}{'DTE':>5}")
            self.logger.info("-" * 120)
            
            for symbol, pos in self.positions.items():
                if isinstance(pos, OptionPosition):
                    pos_value = pos.current_price * pos.contracts * 100
                else:
                    pos_value = pos.current_price * pos.contracts
                
                pos_pct = pos_value / current_value if current_value > 0 else 0
                
                # Get margin requirement if available
                margin = pos.get_margin_requirement() if hasattr(pos, 'get_margin_requirement') else 0
                
                # Get underlying price
                underlying_price = pos.underlying_price if hasattr(pos, 'underlying_price') else 0
                
                # Get DTE (days to expiry)
                dte = pos.days_to_expiry if hasattr(pos, 'days_to_expiry') else 0
                
                self.logger.info(f"{symbol:<20}{pos.contracts:>10d}${pos.avg_entry_price:>6.2f}${pos.current_price:>8.2f}${pos_value:>9.0f}{pos_pct:>7.1%}${underlying_price:>8.2f}{pos.current_delta:>10.3f}{pos.current_gamma:>10.6f}${pos.current_theta:>9.2f}${pos.current_vega:>9.2f}${margin:>9.0f}{dte:>5d}")
            
            self.logger.info("-" * 120)
            self.logger.info(f"TOTAL{' ':>30}${self.position_value:>9.0f}{exposure_pct:>7.1%}{' ':>20}${self.get_margin_requirement():>9.0f}")
            self.logger.info("-" * 120)
            self.logger.info("===========================================")
            
            # Record daily return with components
            self.daily_returns.append({
                'date': current_date,
                'return': daily_return,
                'value': current_value,
                'pnl': current_value - prev_value,
                'unrealized_pnl_change': current_value - prev_value - self.today_realized_pnl,
                'realized_pnl': self.today_realized_pnl
            })
        elif updated_positions:
            # Just log that we updated without the full summary
            self.logger.debug(f"Updated market data for {len(updated_positions)} positions without logging POST-TRADE summary")
                
        # Handle expired options
        for symbol, reason in positions_to_remove:
            if symbol in self.positions:
                self.remove_position(symbol, reason=reason)
                
    def get_portfolio_greeks(self) -> Dict[str, float]:
        """
        Calculate portfolio-level Greek risk metrics.
        
        Returns:
            dict: Dictionary of portfolio Greeks
        """
        portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'dollar_delta': 0.0,
            'dollar_gamma': 0.0,
            'dollar_theta': 0.0,
            'dollar_vega': 0.0
        }
        
        # Calculate portfolio value for delta percentage
        portfolio_value = self.get_portfolio_value()
        
        # Sum up position Greeks
        for position in self.positions.values():
            # Get position Greeks - handle both option and non-option positions
            if hasattr(position, 'get_greeks'):
                position_greeks = position.get_greeks()
                # Sum up all Greek values
                for greek, value in position_greeks.items():
                    if greek in portfolio_greeks:
                        portfolio_greeks[greek] += value
            else:
                # For non-option positions with simple delta
                sign = -1 if position.is_short else 1
                delta = sign * position.current_delta * position.contracts
                portfolio_greeks['delta'] += delta
                
                # Calculate dollar delta
                if hasattr(position, 'underlying_price') and position.underlying_price > 0:
                    portfolio_greeks['dollar_delta'] += delta * position.underlying_price
        
        # Calculate delta as percentage of portfolio value
        if portfolio_value > 0:
            portfolio_greeks['delta_pct'] = portfolio_greeks['dollar_delta'] / portfolio_value
        else:
            portfolio_greeks['delta_pct'] = 0
            
        return portfolio_greeks
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            dict: Dictionary of performance metrics
        """
        # Get portfolio value history
        if not self.equity_history:
            return {
                'return': 0,
                'sharpe_ratio': 0,
                'volatility': 0,
                'max_drawdown': 0,
                'cagr': 0
            }
        
        # Convert equity history to series
        dates = sorted(self.equity_history.keys())
        equity_values = [self.equity_history[date] for date in dates]
        
        if len(equity_values) < 2:
            return {
                'return': 0,
                'sharpe_ratio': 0,
                'volatility': 0,
                'max_drawdown': 0,
                'cagr': 0
            }
        
        # Calculate time-weighted returns
        returns = []
        for i in range(1, len(equity_values)):
            ret = (equity_values[i] / equity_values[i-1]) - 1
            returns.append(ret)
        
        # Calculate total return
        total_return = (equity_values[-1] / equity_values[0]) - 1
        
        # Calculate annualized return (CAGR)
        first_date = dates[0]
        last_date = dates[-1]
        
        # Handle case where dates are the same
        if first_date == last_date:
            years = 0.00273  # 1 day as fraction of year
        else:
            years = (last_date - first_date).days / 365.25
            
        # Minimum period to avoid division by zero
        years = max(years, 0.00273)  # Minimum of 1 day
        
        # Calculate annualization factor based on return frequency
        # This is an approximation assuming daily returns
        annualization_factor = 252
        
        # Calculate CAGR
        cagr = (equity_values[-1] / equity_values[0]) ** (1 / years) - 1
        
        # Calculate volatility
        returns_series = pd.Series(returns)
        volatility = returns_series.std() * np.sqrt(annualization_factor)
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        avg_return = returns_series.mean() * annualization_factor
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + returns_series).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'cagr': cagr
        }

    def _check_portfolio_delta_constraint(self) -> bool:
        """
        Check if portfolio delta is within constraints.

        Returns:
            bool: True if within constraints, False otherwise
        """
        greeks = self.get_portfolio_greeks()
        portfolio_value = self.get_portfolio_value()

        # Skip the check if portfolio value is zero or negative
        if portfolio_value <= 0:
            return True

        # Calculate delta percentage
        delta_pct = abs(greeks['delta_pct'])

        if delta_pct > self.max_portfolio_delta:
            # Log at DEBUG level instead of WARNING to reduce console output
            # Only log once per day to reduce spam
            if not hasattr(self, '_last_delta_warning_date') or self._last_delta_warning_date != datetime.now().date():
                self.logger.debug(
                    f"[DELTA WARNING] Portfolio delta ({delta_pct:.2%}) exceeds maximum allowed ({self.max_portfolio_delta:.2%})")
                self.logger.debug(
                    f"  Portfolio value: ${portfolio_value:,.2f}, Total delta: {greeks['delta']:.3f}")

                # Store positions info for debugging but don't log to console
                positions_info = []
                for symbol, pos in self.positions.items():
                    pos_delta = pos.current_delta * pos.contracts
                    pos_delta_pct = pos_delta / portfolio_value if portfolio_value > 0 else 0
                    positions_info.append(f"{symbol}: {pos_delta:.3f} ({pos_delta_pct:.2%})")

                self.logger.debug(f"  Position deltas: {', '.join(positions_info)}")
                self.logger.debug(f"  Consider adding hedges to reduce delta exposure")

                # Track the date to prevent multiple warnings on the same day
                self._last_delta_warning_date = datetime.now().date()

            return False

        return True

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

    def update_positions(self, current_date, daily_data):
        """
        Update all positions with the latest market data.

        Args:
            current_date: Current simulation date
            daily_data: DataFrame containing market data for the current date
        """
        self.logger.debug(f"Updating positions for {current_date}")

        # Extract all unique underlying prices to ensure consistency
        underlying_prices = {}
        for _, row in daily_data.iterrows():
            if 'UnderlyingSymbol' in row and 'UnderlyingPrice' in row:
                underlying_symbol = row['UnderlyingSymbol']
                if underlying_symbol not in underlying_prices:
                    underlying_prices[underlying_symbol] = row['UnderlyingPrice']

        # Create a dictionary of option data by symbol for quick lookup
        market_data = {}
        for _, row in daily_data.iterrows():
            if 'OptionSymbol' in row:
                # Make a copy of the row to avoid modifying the original
                data = row.copy()
                
                # Override the underlying price with the consistent value for this symbol
                if 'UnderlyingSymbol' in row and row['UnderlyingSymbol'] in underlying_prices:
                    data['UnderlyingPrice'] = underlying_prices[row['UnderlyingSymbol']]
                    
                market_data[row['OptionSymbol']] = data

        # Update each position
        for symbol, position in list(self.positions.items()):
            # Skip if no market data available for this position
            if symbol not in market_data:
                self.logger.warning(f"No market data available for {symbol}, skipping update")
                continue

            # Get current market data
            current_data = market_data[symbol]

            # Store previous price before updating
            position.prev_price = position.current_price
            
            # Update position values
            position.current_price = current_data.get('MidPrice', 0)

            # Update days to expiry for option positions
            if isinstance(position, OptionPosition) and 'DaysToExpiry' in current_data:
                position.days_to_expiry = current_data['DaysToExpiry']

            # Calculate unrealized P&L
            if position.is_short:
                # For short positions, profit when price decreases
                position.unrealized_pnl = (position.avg_entry_price - position.current_price) * position.contracts * 100
            else:
                # For long positions, profit when price increases
                position.unrealized_pnl = (position.current_price - position.avg_entry_price) * position.contracts * 100

        # Update portfolio value
        self._update_portfolio_value()

    def get_position_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of all current positions for reporting.
        
        Returns:
            dict: Position snapshot
        """
        snapshot = {
            'date': datetime.now(),
            'positions': {}
        }
        
        for symbol, position in self.positions.items():
            snapshot['positions'][symbol] = position.get_position_summary()
            
        return snapshot
    
    def get_position_allocation(self) -> pd.DataFrame:
        """
        Get position allocation as percentage of portfolio.
        
        Returns:
            DataFrame: Position allocation data
        """
        if not self.positions:
            return pd.DataFrame()
            
        portfolio_value = self.get_portfolio_value()
        position_data = []
        
        for symbol, position in self.positions.items():
            # Calculate position value
            if isinstance(position, OptionPosition):
                position_value = abs(position.current_price * position.contracts * 100)
            else:
                position_value = abs(position.current_price * position.contracts)
            
            # Calculate allocation percentage - handle zero or negative portfolio value
            allocation = position_value / portfolio_value if portfolio_value > 0 else 0
            
            # Compile position data
            position_data.append({
                'symbol': symbol,
                'type': 'option' if isinstance(position, OptionPosition) else 'stock',
                'direction': 'short' if position.is_short else 'long',
                'contracts': position.contracts,
                'avg_price': position.avg_entry_price,
                'current_price': position.current_price,
                'value': position_value,
                'allocation': allocation,
                'unrealized_pnl': position.unrealized_pnl,
                'delta': position.current_delta * position.contracts
            })
            
        # Convert to DataFrame
        df = pd.DataFrame(position_data)
        
        # Sort by allocation (descending)
        if not df.empty and 'allocation' in df.columns:
            df = df.sort_values('allocation', ascending=False)
            
        return df
    
    def get_transaction_history(self) -> pd.DataFrame:
        """
        Get transaction history as a DataFrame.
        
        Returns:
            DataFrame: Transaction history
        """
        if not self.transactions:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(self.transactions)
        
        # Sort by date
        if not df.empty and 'date' in df.columns:
            df = df.sort_values('date')
            
        return df

    def get_total_liability(self) -> float:
        """
        Calculate the total liability of the portfolio based on short positions.
        
        Returns:
            float: Total liability value in dollars
        """
        # If we've already calculated the short option value, use it
        if hasattr(self, 'short_option_value'):
            return self.short_option_value
            
        # Otherwise calculate it (legacy method)
        total_liability = 0
        for position in self.positions.values():
            if position.is_short:
                # Use entry price if current price is 0
                price = position.current_price if position.current_price > 0 else position.avg_entry_price
                
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
            # Use entry price if current price is 0
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
    
    def get_open_positions(self) -> list:
        """
        Get all open positions.
        
        Returns:
            list: List of Position objects
        """
        return list(self.positions.values())

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

    def get_cash_balance(self) -> float:
        """
        Get the current cash balance.
        
        Returns:
            float: Cash balance in dollars
        """
        return self.cash_balance

    def get_hedge_value(self) -> float:
        """
        Get the current value of hedge positions.
        
        Returns:
            float: Hedge value in dollars, defaults to 0 if not implemented
        """
        # This is a placeholder - actual hedge values would be implemented
        # in a derived class that implements hedging
        return 0

    def get_total_margin_requirement(self) -> float:
        """
        Get the total margin requirement for all positions.
        
        Returns:
            float: Total margin requirement in dollars
        """
        total_margin = 0
        for position in self.positions.values():
            if hasattr(position, 'calculate_margin_requirement'):
                margin = position.calculate_margin_requirement(1.0)  # Use basic margin without leverage
                total_margin += margin
        return total_margin

    def get_available_margin(self) -> float:
        """
        Calculate available margin for new positions.
        
        Returns:
            float: Available margin in dollars
        """
        # NLV - Margin Requirements = Available Margin
        nlv = self.get_net_liquidation_value()
        margin_req = self.get_total_margin_requirement()
        
        # Return cash balance minus margin requirements
        # Make sure NLV is not less than zero
        if nlv <= 0:
            return 0
            
        return nlv - margin_req

    def get_margin_based_leverage(self) -> float:
        """
        Calculate margin-based leverage.
        
        Returns:
            float: Leverage ratio
        """
        portfolio_value = self.get_portfolio_value()
        total_margin = self.get_total_margin_requirement()
        return total_margin / portfolio_value if portfolio_value > 0 else 0

    def get_option_delta(self) -> float:
        """
        Get the total delta from option positions only.
        
        Returns:
            float: Option delta
        """
        greeks = self.get_portfolio_greeks()
        return greeks.get('delta', 0)

    def get_hedge_delta(self) -> float:
        """
        Get the delta from hedge positions.
        
        Returns:
            float: Hedge delta, defaults to 0 if not implemented
        """
        # This is a placeholder - actual hedge delta would be implemented
        # in a derived class that implements hedging
        return 0

    def get_total_delta(self) -> float:
        """
        Get the total portfolio delta.
        
        Returns:
            float: Total portfolio delta
        """
        return self.get_option_delta() + self.get_hedge_delta()

    def get_gamma(self) -> float:
        """
        Get the portfolio gamma.
        
        Returns:
            float: Portfolio gamma
        """
        greeks = self.get_portfolio_greeks()
        return greeks.get('gamma', 0)

    def get_theta(self) -> float:
        """
        Get the portfolio theta.
        
        Returns:
            float: Portfolio theta
        """
        greeks = self.get_portfolio_greeks()
        return greeks.get('dollar_theta', 0)

    def get_vega(self) -> float:
        """
        Get the portfolio vega.
        
        Returns:
            float: Portfolio vega
        """
        greeks = self.get_portfolio_greeks()
        return greeks.get('dollar_vega', 0)

    def get_daily_return(self) -> float:
        """
        Get the daily return in dollars.
        
        Returns:
            float: Daily return in dollars
        """
        if not self.daily_returns:
            return 0
        return self.daily_returns[-1].get('pnl', 0)

    def get_daily_return_percent(self) -> float:
        """
        Get the daily return as a percentage.
        
        Returns:
            float: Daily return as a percentage (0-1)
        """
        if not self.daily_returns:
            return 0
        return self.daily_returns[-1].get('return', 0)

    def get_option_pnl(self) -> float:
        """
        Get the option P&L component.
        
        Returns:
            float: Option P&L in dollars
        """
        if not self.daily_returns:
            return 0
        return self.daily_returns[-1].get('unrealized_pnl_change', 0)

    def get_hedge_pnl(self) -> float:
        """
        Get the hedge P&L component.
        
        Returns:
            float: Hedge P&L in dollars, defaults to 0 if not implemented
        """
        # This is a placeholder - actual hedge PnL would be implemented
        # in a derived class that implements hedging
        return 0

    def get_rolling_metrics(self) -> dict:
        """
        Get rolling performance metrics.
        
        Returns:
            dict: Dictionary of rolling metrics
        """
        # If we don't have enough history, return empty metrics
        if not hasattr(self, 'daily_returns') or len(self.daily_returns) < 5:
            self.logger.debug(f"Not enough history for rolling metrics: {len(self.daily_returns) if hasattr(self, 'daily_returns') else 0} observations")
            if hasattr(self, 'daily_returns'):
                self.logger.debug(f"Daily returns: {self.daily_returns}")
            return {
                'expanding_sharpe': 0,
                'expanding_volatility': 0,
                'short_sharpe': 0,
                'short_volatility': 0,
                'medium_sharpe': 0,
                'medium_volatility': 0,
                'long_sharpe': 0,
                'long_volatility': 0
            }
        
        # Otherwise, calculate metrics from performance history
        metrics = {}
        
        # Create a DataFrame from daily returns
        df = pd.DataFrame(self.daily_returns)
        self.logger.debug(f"Daily returns DataFrame: {df}")
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Calculate expanding window metrics (all available history)
        returns_series = df['return']
        
        # Force calculation even with limited data
        self.logger.debug(f"Calculating rolling metrics with {len(returns_series)} observations")
        self.logger.debug(f"Returns series: {returns_series}")
        
        # Calculate expanding window metrics (all available history)
        metrics['expanding_volatility'] = returns_series.std() * np.sqrt(252)  # Annualized
        metrics['expanding_sharpe'] = (returns_series.mean() * 252) / (returns_series.std() * np.sqrt(252)) if returns_series.std() > 0 else 0
        
        # Calculate short window metrics (21 trading days)
        window_size = min(21, len(returns_series))
        short_window = returns_series.iloc[-window_size:]
        metrics['short_volatility'] = short_window.std() * np.sqrt(252)
        metrics['short_sharpe'] = (short_window.mean() * 252) / (short_window.std() * np.sqrt(252)) if short_window.std() > 0 else 0
        
        # Calculate medium window metrics (63 trading days)
        window_size = min(63, len(returns_series))
        medium_window = returns_series.iloc[-window_size:]
        metrics['medium_volatility'] = medium_window.std() * np.sqrt(252)
        metrics['medium_sharpe'] = (medium_window.mean() * 252) / (medium_window.std() * np.sqrt(252)) if medium_window.std() > 0 else 0
        
        # Calculate long window metrics (252 trading days)
        window_size = min(252, len(returns_series))
        long_window = returns_series.iloc[-window_size:]
        metrics['long_volatility'] = long_window.std() * np.sqrt(252)
        metrics['long_sharpe'] = (long_window.mean() * 252) / (long_window.std() * np.sqrt(252)) if long_window.std() > 0 else 0
        
        # Log that we calculated the metrics
        self.logger.debug(f"Calculated rolling metrics: {metrics}")
        
        return metrics