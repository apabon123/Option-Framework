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
        
        # Log initialization
        self.logger.info(f"Portfolio initialized with ${initial_capital:,.2f} capital")
        self.logger.info(f"  Max position size: {max_position_size_pct:.1%} of portfolio")
        self.logger.info(f"  Max portfolio delta: {max_portfolio_delta:.1%} of portfolio value")
    
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
        pnl = position.remove_contracts(quantity, price, execution_data)
        
        # Track today's realized PnL
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
            
        return pnl
    
    def update_market_data(self, market_data_by_symbol: Dict[str, Any], current_date: Optional[datetime] = None) -> None:
        """
        Update all positions with latest market data.
        
        Args:
            market_data_by_symbol: Dictionary of market data by symbol
            current_date: Current date for this update (optional)
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
        positions_to_remove = []  # For expired options
        
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
                    price = market_data.get('MidPrice', position.current_price)
                    delta = market_data.get('Delta', position.current_delta)
                    gamma = market_data.get('Gamma', position.current_gamma)
                    theta = market_data.get('Theta', position.current_theta)
                    vega = market_data.get('Vega', position.current_vega)
                    days_to_expiry = market_data.get('DaysToExpiry', position.days_to_expiry if hasattr(position, 'days_to_expiry') else None)
                    underlying_price = market_data.get('UnderlyingPrice')
                else:
                    # Pandas Series or DataFrame row
                    price = market_data['MidPrice'] if 'MidPrice' in market_data else position.current_price
                    delta = market_data['Delta'] if 'Delta' in market_data else position.current_delta
                    gamma = market_data['Gamma'] if 'Gamma' in market_data else position.current_gamma
                    theta = market_data['Theta'] if 'Theta' in market_data else position.current_theta
                    vega = market_data['Vega'] if 'Vega' in market_data else position.current_vega
                    days_to_expiry = market_data['DaysToExpiry'] if 'DaysToExpiry' in market_data else (position.days_to_expiry if hasattr(position, 'days_to_expiry') else None)
                    underlying_price = market_data['UnderlyingPrice'] if 'UnderlyingPrice' in market_data else None
                
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
                
        # Log update summary
        if updated_positions:
            self.logger.debug(f"Updated market data for {len(updated_positions)} positions")
            
            # Calculate new portfolio value
            current_value = self.get_portfolio_value()
            daily_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
            
            # Log daily return if date is provided
            if current_date:
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
                greeks = self.get_portfolio_greeks()
                
                self.logger.info(f"  Cash Balance: ${self.cash_balance:.0f}")
                self.logger.info(f"  Total Liability: ${-self.position_value:.0f}")
                self.logger.info(f"Total Margin Requirement: ${metrics['total_margin']:.0f}")
                self.logger.info(f"Available Margin: ${metrics['available_margin']:.0f}")
                self.logger.info(f"Margin-Based Leverage: {metrics['current_leverage']:.2f}")
                
                # Portfolio Greek risk section
                self.logger.info("\nPortfolio Greek Risk:")
                self.logger.info(f"  Option Delta: {greeks['delta']:.3f} (${greeks['dollar_delta']:.2f})")
                self.logger.info(f"  Gamma: {greeks['gamma']:.6f} (${greeks['dollar_gamma']:.2f} per 1% move)")
                self.logger.info(f"  Theta: ${greeks['dollar_theta']:.2f} per day")
                self.logger.info(f"  Vega: ${greeks['dollar_vega']:.2f} per 1% IV")
                
                # Get performance metrics if we have enough history
                if len(self.daily_returns) >= 5:
                    perf = self.get_performance_metrics()
                    self.logger.info("\nRolling Metrics:")
                    self.logger.info(f"  Sharpe: {perf['sharpe_ratio']:.2f}, Volatility: {perf['volatility']:.2%}")
                
                # Print position table
                self.logger.info("\nOpen Trades Table:")
                self.logger.info("-" * 120)
                self.logger.info(f"{'Symbol':<20}{'Contracts':>10}{'Entry':>8}{'Current':>10}{'Value':>10}{'NLV%':>8}{'Delta':>10}")
                self.logger.info("-" * 120)
                
                for symbol, pos in self.positions.items():
                    if isinstance(pos, OptionPosition):
                        pos_value = pos.current_price * pos.contracts * 100
                    else:
                        pos_value = pos.current_price * pos.contracts
                    
                    pos_pct = pos_value / current_value if current_value > 0 else 0
                    
                    self.logger.info(f"{symbol:<20}{pos.contracts:>10d}${pos.avg_entry_price:>6.2f}${pos.current_price:>8.2f}${pos_value:>9.0f}{pos_pct:>7.1%}{pos.current_delta:>10.3f}")
                
                self.logger.info("-" * 120)
                self.logger.info(f"TOTAL{' ':>30}${self.position_value:>9.0f}{exposure_pct:>7.1%}")
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
        """
        # Calculate position value
        position_value = 0
        for pos in self.positions.values():
            if isinstance(pos, OptionPosition):
                position_value += pos.current_price * pos.contracts * 100
            else:
                position_value += pos.current_price * pos.contracts

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