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
        self.logger = logger or logging.getLogger('trading')
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.max_position_size_pct = max_position_size_pct
        self.max_portfolio_delta = max_portfolio_delta
        
        # Track positions by symbol
        self.positions: Dict[str, Position] = {}
        
        # Performance tracking
        self.equity_history: Dict[datetime, float] = {}
        self.equity_history[datetime.now()] = initial_capital
        
        # Daily performance metrics
        self.daily_returns: List[Dict[str, Any]] = []
        
        # Transaction log
        self.transactions: List[Dict[str, Any]] = []
        
        self.logger.info(f"Portfolio initialized with ${initial_capital:,.2f} capital")
        self.logger.info(f"  Max position size: {max_position_size_pct:.1%} of portfolio")
        self.logger.info(f"  Max portfolio delta: {max_portfolio_delta:.1%} of portfolio value")
    
    def add_position(
        self, 
        symbol: str, 
        instrument_data: Dict[str, Any], 
        quantity: int, 
        price: float, 
        position_type: str = 'option',
        is_short: bool = False,
        execution_data: Optional[Dict[str, Any]] = None
    ) -> Position:
        """
        Add a new position or add to an existing position in the portfolio.
        
        Args:
            symbol: Instrument symbol
            instrument_data: Instrument data dictionary
            quantity: Quantity to add
            price: Price per unit
            position_type: Type of position ('option', 'stock', etc.)
            is_short: Whether this is a short position
            execution_data: Additional execution data
            
        Returns:
            Position: The position object
        """
        # Check position size constraint
        position_value = price * quantity * 100 if position_type == 'option' else price * quantity
        portfolio_value = self.get_portfolio_value()
        
        if position_value / portfolio_value > self.max_position_size_pct:
            self.logger.warning(
                f"Position size (${position_value:,.2f}) exceeds maximum allowed ({self.max_position_size_pct:.1%} of ${portfolio_value:,.2f})")
            
            # Reduce quantity to max allowed
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
            
        # Close the position
        pnl = position.remove_contracts(quantity, price, execution_data, reason)
        
        # Update cash balance
        position_value = price * quantity * 100
        if position.is_short:
            self.cash_balance -= position_value  # Short positions reduce cash when closed
        else:
            self.cash_balance += position_value  # Long positions add cash when closed
            
        # Record transaction
        transaction_date = datetime.now()
        if execution_data and 'date' in execution_data:
            transaction_date = execution_data['date']
            
        transaction = {
            'date': transaction_date,
            'symbol': symbol,
            'action': 'BUY' if position.is_short else 'SELL',  # Opposite of original action
            'quantity': quantity,
            'price': price,
            'value': position_value,
            'pnl': pnl,
            'reason': reason,
            'cash_balance': self.cash_balance
        }
        self.transactions.append(transaction)
        
        # Remove position if fully closed
        if position.contracts == 0:
            del self.positions[symbol]
            
        self.logger.info(f"Removed {quantity} {'short' if position.is_short else 'long'} {symbol} @ ${price:.2f}")
        self.logger.info(f"  P&L: ${pnl:,.2f}")
        self.logger.info(f"  New cash balance: ${self.cash_balance:,.2f}")
        
        return pnl
    
    def update_market_data(
        self, 
        market_data_by_symbol: Dict[str, Dict[str, Any]],
        current_date: Optional[datetime] = None
    ) -> None:
        """
        Update all positions with latest market data.
        
        Args:
            market_data_by_symbol: Dictionary of market data by symbol
            current_date: Current date for tracking
        """
        if not current_date:
            current_date = datetime.now()
            
        # Store previous portfolio value for return calculation
        previous_value = self.get_portfolio_value()
        
        # Track PnL components
        unrealized_pnl_change = 0
        realized_pnl = 0
        
        # Update each position
        for symbol, position in list(self.positions.items()):
            if symbol in market_data_by_symbol:
                # Get previous unrealized PnL
                prev_unrealized = position.unrealized_pnl
                
                # Update position with market data
                position.update_market_data(market_data_by_symbol[symbol])
                
                # Calculate change in unrealized PnL
                unrealized_pnl_change += (position.unrealized_pnl - prev_unrealized)
            else:
                self.logger.warning(f"No market data for position {symbol}")
                
        # Calculate new portfolio value
        new_value = self.get_portfolio_value()
        
        # Calculate daily return
        if previous_value > 0:
            daily_return = (new_value - previous_value) / previous_value
        else:
            daily_return = 0
            
        # Record equity history
        self.equity_history[current_date] = new_value
        
        # Record daily return
        self.daily_returns.append({
            'date': current_date,
            'return': daily_return,
            'portfolio_value': new_value,
            'unrealized_pnl_change': unrealized_pnl_change,
            'realized_pnl': realized_pnl
        })
        
        self.logger.info(f"Updated market data for {len(self.positions)} positions")
        self.logger.info(f"  Current portfolio value: ${new_value:,.2f}")
        self.logger.info(f"  Daily return: {daily_return:.2%} (${new_value - previous_value:,.2f})")
    
    def get_portfolio_value(self) -> float:
        """
        Get current total portfolio value (NLV).
        
        Returns:
            float: Portfolio value in dollars
        """
        # Add up position values and cash
        total_value = self.cash_balance
        
        for symbol, position in self.positions.items():
            # For short positions, unrealized PnL is a liability
            # For long positions, unrealized PnL is an asset
            position_value = position.current_price * position.contracts * 100
            if position.is_short:
                # For short positions, subtract the position value (liability)
                # but add the unrealized P&L (negative P&L increases liability)
                total_value -= position_value
            else:
                # For long positions, add the position value (asset)
                total_value += position_value
                
        return total_value
    
    def get_portfolio_greeks(self) -> Dict[str, float]:
        """
        Calculate aggregate portfolio-level Greeks with separate hedge tracking.
        
        Returns:
            dict: Portfolio Greeks with option and hedge component separation
        """
        greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'dollar_delta': 0.0,
            'dollar_gamma': 0.0,
            'dollar_theta': 0.0,
            'dollar_vega': 0.0,
            'hedge_delta': 0.0,
            'dollar_hedge_delta': 0.0,
            'total_delta': 0.0,
            'dollar_total_delta': 0.0
        }
        
        # Sum the option greeks separately from hedge positions
        for symbol, position in self.positions.items():
            position_greeks = position.get_greeks()
            
            # Check if this is a hedge position (non-option position)
            is_hedge = not isinstance(position, OptionPosition)
            
            if is_hedge:
                # For hedge positions, we only care about delta
                if position.is_short:
                    greeks['hedge_delta'] -= position.contracts  # Short hedge is negative delta
                else:
                    greeks['hedge_delta'] += position.contracts  # Long hedge is positive delta
                
                # Calculate dollar delta for hedge
                greeks['dollar_hedge_delta'] = greeks['hedge_delta'] * position.underlying_price * 100
            else:
                # For option positions, add to all option greeks
                for greek, value in position_greeks.items():
                    if greek in greeks:
                        greeks[greek] += value
        
        # Calculate total delta (options + hedge)
        greeks['total_delta'] = greeks['delta'] + greeks['hedge_delta'] 
        greeks['dollar_total_delta'] = greeks['dollar_delta'] + greeks['dollar_hedge_delta']
                    
        # Calculate delta percentage (relative to portfolio value)
        portfolio_value = self.get_portfolio_value()
        if portfolio_value > 0:
            greeks['delta_pct'] = greeks['dollar_delta'] / portfolio_value
            greeks['total_delta_pct'] = greeks['dollar_total_delta'] / portfolio_value
        else:
            greeks['delta_pct'] = 0
            greeks['total_delta_pct'] = 0
            
        return greeks
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio metrics.
        
        Returns:
            dict: Portfolio metrics
        """
        portfolio_value = self.get_portfolio_value()
        greeks = self.get_portfolio_greeks()
        
        # Count positions by type
        option_positions = sum(1 for p in self.positions.values() if isinstance(p, OptionPosition))
        other_positions = len(self.positions) - option_positions
        
        # Calculate exposure metrics
        total_exposure = 0
        for position in self.positions.values():
            if isinstance(position, OptionPosition):
                # For options, use notional exposure (underlying_price * contracts * 100)
                exposure = position.underlying_price * position.contracts * 100
            else:
                # For other instruments, use position value
                exposure = position.current_price * position.contracts
                
            total_exposure += exposure
            
        exposure_ratio = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        return {
            'portfolio_value': portfolio_value,
            'cash_balance': self.cash_balance,
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'theta': greeks['theta'],
            'vega': greeks['vega'],
            'dollar_delta': greeks['dollar_delta'],
            'delta_pct': greeks['delta_pct'],
            'position_count': len(self.positions),
            'option_positions': option_positions,
            'other_positions': other_positions,
            'exposure': total_exposure,
            'exposure_ratio': exposure_ratio
        }
    
    def get_performance_metrics(
        self, 
        annualization_factor: int = 252
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            annualization_factor: Annualization factor (252 for daily returns)
            
        Returns:
            dict: Performance metrics
        """
        # Need at least 2 equity values to calculate returns
        if len(self.equity_history) < 2:
            return {
                'return': 0,
                'sharpe_ratio': 0,
                'volatility': 0,
                'max_drawdown': 0,
                'cagr': 0
            }
            
        # Extract equity values and sort by date
        dates = sorted(self.equity_history.keys())
        equity_values = [self.equity_history[date] for date in dates]
        
        # Calculate returns
        returns = []
        for i in range(1, len(equity_values)):
            if equity_values[i-1] > 0:
                returns.append((equity_values[i] / equity_values[i-1]) - 1)
                
        if not returns:
            return {
                'return': 0,
                'sharpe_ratio': 0,
                'volatility': 0,
                'max_drawdown': 0,
                'cagr': 0
            }
            
        # Calculate total return
        total_return = (equity_values[-1] / equity_values[0]) - 1
        
        # Calculate CAGR
        days = (dates[-1] - dates[0]).days
        years = days / 365 if days > 0 else 1
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
        
        # Calculate delta percentage
        delta_pct = abs(greeks['delta_pct'])
        
        if delta_pct > self.max_portfolio_delta:
            # Create a more informative warning message with position details
            positions_info = []
            for symbol, pos in self.positions.items():
                pos_delta = pos.current_delta * pos.contracts
                pos_delta_pct = pos_delta / portfolio_value if portfolio_value > 0 else 0
                positions_info.append(f"{symbol}: {pos_delta:.3f} ({pos_delta_pct:.2%})")
                
            self.logger.warning(
                f"[DELTA WARNING] Portfolio delta ({delta_pct:.2%}) exceeds maximum allowed ({self.max_portfolio_delta:.2%})")
            self.logger.warning(
                f"  Portfolio value: ${portfolio_value:,.2f}, Total delta: {greeks['delta']:.3f}")
            self.logger.warning(
                f"  Position deltas: {', '.join(positions_info)}")
            self.logger.warning(
                f"  Consider adding hedges to reduce delta exposure")
            return False
            
        return True

    def _update_portfolio_value(self):
        """
        Update total portfolio value based on current positions.
        """
        # Calculate position value
        position_value = sum(pos.current_price * pos.contracts for pos in self.positions.values())

        # Update total value
        self.position_value = position_value
        self.total_value = self.cash_balance + position_value

        # Record in equity history
        self.equity_history[datetime.now()] = self.total_value

    def update_positions(self, current_date, daily_data):
        """
        Update all positions with the latest market data.

        Args:
            current_date: Current simulation date
            daily_data: DataFrame containing market data for the current date
        """
        self.logger.debug(f"Updating positions for {current_date}")

        # Create a dictionary of option data by symbol for quick lookup
        market_data = {}
        for _, row in daily_data.iterrows():
            if 'OptionSymbol' in row:
                market_data[row['OptionSymbol']] = row

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
            position_value = abs(position.current_price * position.contracts * 100)
            
            # Calculate allocation percentage
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
            
        # Convert to DataFrame and sort by allocation
        df = pd.DataFrame(position_data)
        if not df.empty:
            df = df.sort_values('allocation', ascending=False)
            
        return df