"""
Position Management Module

This module provides classes and utilities for managing trading positions,
tracking PnL, and calculating risk metrics.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np

class Position:
    """
    Represents a position in a single financial instrument with PnL tracking.

    This class manages a position's lifecycle including adding and removing
    contracts, tracking market data, and calculating performance metrics.
    It can be used for various instrument types, not just options.
    """

    def __init__(
        self, 
        symbol: str, 
        contracts: int, 
        entry_price: float, 
        is_short: bool = False,
        position_type: str = 'unknown',
        instrument_data: Optional[Dict[str, Any]] = None,
        date: Optional[datetime] = None,
        logger: Optional[logging.Logger] = None,
        current_price: Optional[float] = None
    ):
        """
        Initialize a new position.
        
        Args:
            symbol: Position symbol/ticker
            contracts: Number of contracts/shares
            entry_price: Average entry price per contract/share
            is_short: Whether this is a short position
            position_type: Type of position (e.g., 'option', 'stock')
            instrument_data: Additional data about the instrument
            date: Date of position creation
            logger: Logger instance
            current_price: Current price of the instrument (defaults to entry_price if not provided)
        """
        self.symbol = symbol
        self.contracts = contracts
        self.avg_entry_price = entry_price
        self.is_short = is_short
        self.position_type = position_type
        self.instrument_data = instrument_data or {}
        self.creation_date = date or datetime.now()
        self.logger = logger
        
        # Market data tracking
        self.current_price = current_price if current_price is not None else entry_price
        self.prev_price = entry_price  # Initialize prev_price to entry_price
        self.previous_day_price = entry_price  # New field to track previous day's closing price
        self.previous_day_contracts = contracts  # New field to track previous day's contract count
        self.underlying_price = 0.0
        self.daily_data = []
        
        # Performance tracking
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.max_profit = 0.0
        self.max_drawdown = 0.0
        self.days_held = 0
        
        # Transaction tracking
        self.transactions = []
        
        # Record the entry transaction
        transaction = {
            'date': self.creation_date,
            'type': 'Entry',
            'contracts': contracts,
            'price': entry_price,
            'value': self._calculate_position_value(contracts, entry_price, is_short)
        }
        self.transactions.append(transaction)
        
        # Initialize with basic instrument data
        self._process_instrument_data(instrument_data or {})

    def _process_instrument_data(self, instrument_data: Union[Dict[str, Any], pd.Series]) -> None:
        """
        Process instrument data and set relevant attributes.
        
        This method extracts and sets key attributes from the provided instrument data.
        Called during initialization and when updating market data.
        
        Args:
            instrument_data: A dictionary or pandas Series containing instrument data
        """
        try:
            # Log the instrument data for debugging
            if self.logger:
                self.logger.debug(f"Processing instrument data for {self.symbol}: {instrument_data}")
                
            # Handle dictionary-like objects
            if hasattr(instrument_data, 'get') and not hasattr(instrument_data, 'iloc'):
                # Extract option-specific data if available
                self.strike = instrument_data.get('Strike')
                if 'Expiration' in instrument_data:
                    self.expiration = instrument_data.get('Expiration')
                
                # Safely set type attribute
                type_value = instrument_data.get('Type')
                # Make sure type is a string value
                if isinstance(type_value, str):
                    self.type = type_value
                else:
                    # Default to 'stock' for non-option instruments if type is not a string
                    self.type = 'stock'
                    if self.logger:
                        self.logger.debug(f"Type value is not a string: {type(type_value)}, value: {type_value}. Setting to 'stock'")
                
                self.underlying_price = instrument_data.get('UnderlyingPrice', 0)
                
                # Set initial price if not already set
                if self.current_price == 0:
                    # Try to get mid price first, then fall back to last or bid/ask
                    if 'MidPrice' in instrument_data:
                        self.current_price = instrument_data.get('MidPrice')
                    elif 'Last' in instrument_data and instrument_data.get('Last') > 0:
                        self.current_price = instrument_data.get('Last')
                    elif 'Bid' in instrument_data and 'Ask' in instrument_data:
                        bid = instrument_data.get('Bid', 0)
                        ask = instrument_data.get('Ask', 0)
                        if bid > 0 and ask > 0:
                            self.current_price = (bid + ask) / 2
                
                # Set initial Greeks if available
                self.current_delta = instrument_data.get('Delta', 0)
                self.current_gamma = instrument_data.get('Gamma', 0)
                self.current_theta = instrument_data.get('Theta', 0)
                self.current_vega = instrument_data.get('Vega', 0)
                
            # Handle pandas Series objects
            elif hasattr(instrument_data, 'iloc'):
                # Extract option-specific data if available
                self.strike = instrument_data['Strike'] if 'Strike' in instrument_data.index else None
                if 'Expiration' in instrument_data.index:
                    self.expiration = instrument_data['Expiration']
                
                # Safely set type attribute
                if 'Type' in instrument_data.index:
                    type_value = instrument_data['Type']
                    # Make sure type is a string value
                    if isinstance(type_value, str):
                        self.type = type_value
                    else:
                        # Default to 'stock' for non-option instruments if type is not a string
                        self.type = 'stock'
                else:
                    self.type = 'stock'
                
                self.underlying_price = instrument_data['UnderlyingPrice'] if 'UnderlyingPrice' in instrument_data.index else 0
                
                # Set initial price if not already set
                if self.current_price == 0:
                    # Try to get mid price first, then fall back to last or bid/ask
                    if 'MidPrice' in instrument_data.index:
                        self.current_price = instrument_data['MidPrice']
                    elif 'Last' in instrument_data.index and instrument_data['Last'] > 0:
                        self.current_price = instrument_data['Last']
                    elif 'Bid' in instrument_data.index and 'Ask' in instrument_data.index:
                        bid = instrument_data['Bid'] if 'Bid' in instrument_data.index else 0
                        ask = instrument_data['Ask'] if 'Ask' in instrument_data.index else 0
                        if bid > 0 and ask > 0:
                            self.current_price = (bid + ask) / 2
                
                # Set initial Greeks if available
                self.current_delta = instrument_data['Delta'] if 'Delta' in instrument_data.index else 0
                self.current_gamma = instrument_data['Gamma'] if 'Gamma' in instrument_data.index else 0
                self.current_theta = instrument_data['Theta'] if 'Theta' in instrument_data.index else 0
                self.current_vega = instrument_data['Vega'] if 'Vega' in instrument_data.index else 0
        
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error processing instrument data: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

    def is_option_position(self) -> bool:
        """
        Determine if this is an option position, with proper type checking.
        
        Returns:
            bool: True if this is an option position, False otherwise
        """
        if hasattr(self, 'option_type') and self.option_type:
            return True
        
        if hasattr(self, 'type'):
            if isinstance(self.type, str) and self.type.lower() == 'option':
                return True
        
        return False

    def get_position_value(self, quantity: Optional[int] = None) -> float:
        """
        Calculate the total value of the position.
        
        Args:
            quantity: Optional number of contracts to calculate value for (default is all contracts)
            
        Returns:
            float: Total value of the position in dollars
        """
        if quantity is None:
            quantity = self.contracts
            
        if quantity <= 0:
            return 0.0
            
        # Check if we're dealing with an option
        if self.is_option_position():
            # For options, the value is price * quantity * 100 (option multiplier)
            return self.current_price * quantity * 100
        else:
            # For stocks, the value is just price * quantity
            return self.current_price * quantity

    def _calculate_position_value(self, quantity: int, price: float, is_short: bool) -> float:
        """
        Calculate the value of a position.
        
        Args:
            quantity: Number of contracts or shares
            price: Price per contract or share
            is_short: Whether the position is short
            
        Returns:
            float: Position value in dollars
        """
        if quantity <= 0:
            return 0.0
            
        # Check if we're dealing with an option
        if self.is_option_position():
            # For options, the value is price * quantity * 100 (option multiplier)
            return price * quantity * 100
        else:
            # For stocks, the value is just price * quantity
            return price * quantity

    def calculate_realized_pnl(self, quantity: int, price: float) -> float:
        """
        Calculate realized PnL when closing a portion of the position.
        
        Args:
            quantity: Number of contracts to close
            price: Current price per contract
            
        Returns:
            float: Realized PnL for this closure
        """
        if quantity <= 0 or quantity > self.contracts:
            return 0.0
            
        # Determine multiplier based on position type
        multiplier = 100 if self.is_option_position() else 1
        
        # Calculate PnL
        # For short positions: entry_price - exit_price is the P&L
        # For long positions: exit_price - entry_price is the P&L
        if self.is_short:
            pnl = (self.avg_entry_price - price) * quantity * multiplier
        else:
            pnl = (price - self.avg_entry_price) * quantity * multiplier
            
        return pnl

    def add_contracts(
        self, 
        quantity: int, 
        price: float, 
        execution_data: Optional[Union[Dict[str, Any], pd.Series]] = None
    ) -> float:
        """
        Add contracts to the position with proper cost basis tracking.

        Args:
            quantity: Number of contracts to add
            price: Price per contract
            execution_data: Additional data about the execution

        Returns:
            float: Updated average entry price
        """
        if quantity <= 0:
            return self.avg_entry_price
        
        # Record transaction - handle both dict and Series
        transaction_date = None
        if execution_data is not None:
            try:
                # Check if execution_data is a Pandas Series
                if hasattr(execution_data, 'get') and not hasattr(execution_data, 'iloc'):
                    # Dictionary style access
                    transaction_date = execution_data.get('DataDate')
                elif hasattr(execution_data, 'iloc'):
                    # Series style access
                    transaction_date = execution_data['DataDate'] if 'DataDate' in execution_data.index else None
                else:
                    transaction_date = None
            except (TypeError, ValueError) as e:
                # Log the error but continue with None date
                if self.logger:
                    self.logger.warning(f"Error extracting transaction date: {e}")
                transaction_date = None
        
        # Determine if this is an option position for value calculation
        is_option = self.is_option_position()
        
        transaction = {
            'date': transaction_date,
            'action': 'SELL' if self.is_short else 'BUY',
            'contracts': quantity,
            'price': price,
            'value': price * quantity * 100 if is_option else price * quantity
        }
        self.transactions.append(transaction)
        
        # Calculate weighted average entry price
        old_value = self.avg_entry_price * self.contracts
        new_value = price * quantity
        total_contracts = self.contracts + quantity
        
        if total_contracts > 0:
            self.avg_entry_price = (old_value + new_value) / total_contracts
        
        # Update contract count
        old_contracts = self.contracts
        self.contracts = total_contracts
        
        # Set current price to execution price
        self.current_price = price
        
        # Log the transaction
        if self.logger:
            self.logger.info(f"[Position] Added {quantity} contracts of {self.symbol} at ${price:.2f}")
            self.logger.info(f"  Previous: {old_contracts} at ${self.avg_entry_price:.2f}")
            self.logger.info(f"  New position: {self.contracts} contracts at avg price ${self.avg_entry_price:.2f}")
        
        # Update total trade value (for accounting)
        self.total_value = self.avg_entry_price * self.contracts * 100 if is_option else self.avg_entry_price * self.contracts
        
        return self.avg_entry_price

    def remove_contracts(
        self, 
        quantity: int, 
        price: float, 
        execution_data: Optional[Union[Dict[str, Any], pd.Series]] = None, 
        reason: str = "Close"
    ) -> float:
        """
        Remove contracts from position and calculate realized PnL.

        Args:
            quantity: Number of contracts to remove
            price: Current price per contract
            execution_data: Additional data about the execution
            reason: Reason for closing (e.g., "Profit Target", "Stop Loss")

        Returns:
            float: Realized PnL for this closure
        """
        if quantity <= 0:
            if self.logger:
                self.logger.warning(f"Invalid quantity {quantity} for remove_contracts on {self.symbol}")
            return 0
            
        if self.contracts <= 0:
            if self.logger:
                self.logger.warning(f"Cannot remove contracts from {self.symbol} - position already has 0 contracts")
            return 0
            
        if quantity > self.contracts:
            if self.logger:
                self.logger.warning(f"Trying to remove {quantity} contracts from {self.symbol} but only {self.contracts} available - adjusting to {self.contracts}")
            quantity = self.contracts
        
        # Store current average entry price
        entry_price = self.avg_entry_price
        
        # Determine if this is an option position for value calculation
        is_option = self.is_option_position()
        
        # Calculate realized PnL
        # For short positions: entry_price - exit_price is the P&L
        # For long positions: exit_price - entry_price is the P&L
        multiplier = 100 if is_option else 1
        if self.is_short:
            pnl = (entry_price - price) * quantity * multiplier
        else:
            pnl = (price - entry_price) * quantity * multiplier
        
        # Record transaction - handle different types properly
        transaction_date = None
        
        # Safely extract transaction date from execution_data
        if execution_data is not None:
            try:
                # Check if execution_data is a dictionary-like object
                if hasattr(execution_data, 'get') and not hasattr(execution_data, 'iloc'):
                    transaction_date = execution_data.get('DataDate')
                # Check if execution_data is a pandas Series
                elif hasattr(execution_data, 'iloc'):
                    transaction_date = execution_data['DataDate'] if 'DataDate' in execution_data.index else None
                else:
                    transaction_date = None
            except (TypeError, ValueError) as e:
                # Log the error but continue with None date
                if self.logger:
                    self.logger.warning(f"Error extracting transaction date: {e}")
                transaction_date = None
        
        transaction = {
            'date': transaction_date,
            'action': 'BUY' if self.is_short else 'SELL',
            'contracts': quantity,
            'price': price,
            'pnl': pnl,
            'reason': reason
        }
        self.transactions.append(transaction)
        
        # Update realized PnL total
        self.realized_pnl += pnl
        
        # Update contract count
        old_contracts = self.contracts
        self.contracts -= quantity
        
        # Reset average price if position closed completely
        if self.contracts == 0:
            self.avg_entry_price = 0
            self.total_value = 0
        else:
            # No need to adjust avg_entry_price when removing contracts
            self.total_value = self.avg_entry_price * self.contracts * 100
        
        # Log the transaction
        if self.logger:
            self.logger.info(f"[Position] Removed {quantity} contracts of {self.symbol} at ${price:.2f}")
            self.logger.info(f"  Entry price: ${entry_price:.2f}, Exit price: ${price:.2f}")
            # Calculate percentage P&L, handle division by zero
            pct_pnl = (pnl / (entry_price * quantity * 100) * 100) if (entry_price > 0) else 0
            self.logger.info(f"  P&L: ${pnl:.2f} ({'+' if pnl >= 0 else ''}{pct_pnl:.2f}%)")
            self.logger.info(f"  Remaining: {self.contracts} contracts")
        
        return pnl

    def update_market_data(
        self, 
        market_data: Union[Dict[str, Any], pd.Series]
    ) -> float:
        """
        Update position with latest market data and calculate unrealized PnL.
        
        Properly handles the signs for Greeks:
        - Delta: Negative for short calls, positive for short puts (based on option type)
        - Gamma: Always positive for both long and short positions
        - Theta: Always negative for both long and short positions
        - Vega: Always positive for both long and short positions
        
        Args:
            market_data: Current market data for this instrument

        Returns:
            float: Updated unrealized PnL
        """
        # Store the previous price for P&L calculations
        self.prev_price = self.current_price
        
        # Store the data
        try:
            # If market_data has a copy method (DataFrame or Series), use it
            if hasattr(market_data, 'copy'):
                self.daily_data.append(market_data.copy())
            else:
                # Otherwise just append the data as is
                self.daily_data.append(market_data)
        except Exception as e:
            # If there's an error, just append the data as is and log the error
            self.daily_data.append(market_data)
            if self.logger:
                self.logger.warning(f"Error copying market data: {e}")
        
        # Update key metrics - handle different types of market_data
        try:
            # Check if market_data is a dictionary-like object
            if hasattr(market_data, 'get') and not hasattr(market_data, 'iloc'):
                # Dictionary style access
                self.current_price = market_data.get('MidPrice', 0)
                delta = market_data.get('Delta', 0)
                gamma = market_data.get('Gamma', 0)
                theta = market_data.get('Theta', 0)
                vega = market_data.get('Vega', 0)
                self.underlying_price = market_data.get('UnderlyingPrice', 0)
                
                # Calculate days to expiry if possible
                if 'DataDate' in market_data and self.expiration:
                    self.days_to_expiry = (self.expiration - market_data.get('DataDate')).days
            # Check if market_data is a pandas Series
            elif hasattr(market_data, 'iloc'):
                # Series style access
                self.current_price = market_data['MidPrice'] if 'MidPrice' in market_data.index else 0
                delta = market_data['Delta'] if 'Delta' in market_data.index else 0
                gamma = market_data['Gamma'] if 'Gamma' in market_data.index else 0
                theta = market_data['Theta'] if 'Theta' in market_data.index else 0
                vega = market_data['Vega'] if 'Vega' in market_data.index else 0
                self.underlying_price = market_data['UnderlyingPrice'] if 'UnderlyingPrice' in market_data.index else 0
                
                # Calculate days to expiry if possible
                if 'DataDate' in market_data.index and self.expiration is not None:
                    self.days_to_expiry = (self.expiration - market_data['DataDate']).days
                    
            # Determine if this is a call or put
            is_call = False
            is_put = False
            if hasattr(self, 'option_type'):
                is_call = self.option_type.upper() in ['C', 'CALL']
                is_put = self.option_type.upper() in ['P', 'PUT']
            elif hasattr(self, 'type') and isinstance(self.type, str):
                is_call = self.type.upper() in ['C', 'CALL']
                is_put = self.type.upper() in ['P', 'PUT']
                
            # For short positions, adjust delta based on option type
            if self.is_short:
                if is_call:
                    # For short calls, delta should be negative
                    self.current_delta = -abs(delta)
                elif is_put:  # Short puts should have positive delta
                    self.current_delta = abs(delta)
                else:
                    # If we can't determine the option type, default to -delta for shorts
                    self.current_delta = -delta
                    
                # For short positions, adjust other Greeks as well
                # Gamma is negative for short positions
                self.current_gamma = -abs(gamma)
                # Theta is positive for short positions (profit from time decay)
                self.current_theta = abs(theta)
                # Vega is negative for short positions (profit from volatility decreases)
                self.current_vega = -abs(vega)
            else:
                # For long positions, keep the original sign (negative for puts, positive for calls)
                self.current_delta = delta
                
                # For long positions, signs are standard
                # Gamma is positive for long positions
                self.current_gamma = abs(gamma)
                # Theta is negative for long positions (loss from time decay)
                self.current_theta = -abs(theta)
                # Vega is positive for long positions (profit from volatility increases)
                self.current_vega = abs(vega)
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error updating market data: {e}")
        
        # Determine if this is an option position for value calculation
        is_option = self.is_option_position()
        
        # Calculate unrealized PnL (positive values mean profit)
        if self.contracts > 0:
            if self.is_short:
                # For short options: entry_price - current_price is the P&L per unit
                # If current_price goes down, that's profit for short options
                pnl_per_contract = self.avg_entry_price - self.current_price
            else:
                # For long options: current_price - entry_price is the P&L per unit
                pnl_per_contract = self.current_price - self.avg_entry_price
            
            # Calculate total PnL based on contract size and count
            multiplier = 100 if is_option else 1
            self.unrealized_pnl = pnl_per_contract * self.contracts * multiplier
        else:
            self.unrealized_pnl = 0
        
        # Update max drawdown tracking (drawdown is negative PnL)
        if self.unrealized_pnl < -self.max_drawdown:
            self.max_drawdown = -self.unrealized_pnl
        
        return self.unrealized_pnl

    def update_unrealized_pnl(self):
        """
        Update the unrealized PnL based on current price and position type.
        
        This is a helper method called by the Portfolio class when updating
        position market data.
        """
        if self.contracts <= 0:
            self.unrealized_pnl = 0
            return
            
        if self.is_short:
            # For short positions: entry_price - current_price is the P&L per unit
            # If current_price goes down, that's profit for short options
            pnl_per_contract = self.avg_entry_price - self.current_price
        else:
            # For long positions: current_price - entry_price is the P&L per unit
            pnl_per_contract = self.current_price - self.avg_entry_price
            
        # Determine if this is an option position for value calculation
        is_option = self.is_option_position()
        
        # Calculate total PnL based on contract size and count
        multiplier = 100 if is_option else 1
        self.unrealized_pnl = pnl_per_contract * self.contracts * multiplier
        
        # Update max drawdown tracking (drawdown is negative PnL)
        if self.unrealized_pnl < -self.max_drawdown:
            self.max_drawdown = -self.unrealized_pnl

    def calculate_margin_requirement(self, max_leverage: float) -> float:
        """
        Calculate margin requirement for this position using span margining approach.
        
        Span margining is a portfolio-based approach to calculating margin that accounts for
        risk across the entire portfolio, typically resulting in lower margin requirements
        than traditional Reg T margining.

        Args:
            max_leverage: Maximum leverage allowed

        Returns:
            float: Margin requirement in dollars
        """
        if self.contracts <= 0:
            return 0
        
        # Check if this is a stock position (no option-specific attributes)
        is_stock_position = self.type == 'stock' or (not hasattr(self, 'implied_volatility') and not hasattr(self, 'option_chain_id'))
        
        if is_stock_position:
            # For stock positions, use span margining approach
            # Long positions: typically 15-20% for index-based ETFs
            # Short positions: typically 20-30% for index-based ETFs
            # Adjust as needed based on the underlying's volatility
            
            # Base values
            long_margin_rate = 0.20  # 20% margin for long positions
            short_margin_rate = 0.30  # 30% margin for short positions
            
            if self.is_short:
                initial_margin = self.avg_entry_price * self.contracts * short_margin_rate
            else:
                initial_margin = self.avg_entry_price * self.contracts * long_margin_rate
        else:
            # For options, use a risk-based approach based on option properties
            # Higher IV options require more margin
            iv_factor = getattr(self, 'implied_volatility', 0.30) / 0.30  # Normalize to 30% IV as baseline
            initial_margin = self.avg_entry_price * self.contracts * 100 * max_leverage * iv_factor
        
        # Adjust margin for unrealized PnL (both gains and losses)
        adjusted_margin = initial_margin - self.unrealized_pnl
        
        return max(adjusted_margin, 0)  # Ensure no negative margin

    def get_greeks(self, for_display: bool = False) -> Dict[str, float]:
        """
        Get position Greeks with correct signs for either calculation or display.
        
        The mathematical signs (for_display=False) are:
        - Delta: Positive for long calls, negative for long puts, POSITIVE for short puts, NEGATIVE for short calls
        - Gamma: Positive for long positions, negative for short positions
        - Theta: Negative for long positions, positive for short positions
        - Vega: Positive for long positions, negative for short positions
        
        The trading convention signs for display (for_display=True) are:
        - Delta: Negative for short calls, positive for short puts
        - Gamma: Negative for short positions, positive for long positions  
        - Theta: Positive for short positions, negative for long positions
        - Vega: Negative for short positions, positive for long positions
        
        Args:
            for_display: If True, return Greeks with signs adjusted for display purposes
                        If False, return Greeks with signs as used in calculations
        
        Returns:
            dict: Dictionary of position Greeks with appropriate signs
        """
        # Start with the raw Greeks (multiplied by contracts)
        greeks = {
            'delta': self.current_delta * self.contracts,
            'gamma': self.current_gamma * self.contracts,
            'theta': self.current_theta * self.contracts,
            'vega': self.current_vega * self.contracts,
            'dollar_delta': self.current_delta * self.contracts * self.underlying_price * 100,
            'dollar_gamma': self.current_gamma * self.contracts * (self.underlying_price ** 2) * 0.01,
            'dollar_theta': self.current_theta * self.contracts,
            'dollar_vega': self.current_vega * self.contracts
        }
        
        # Check if option is a put (needed for both calculation and display)
        is_put = False
        if hasattr(self, 'option_type') and self.option_type:
            is_put = self.option_type.upper() in ['P', 'PUT']
        elif hasattr(self, 'type') and isinstance(self.type, str):
            is_put = self.type.upper() in ['P', 'PUT']
        
        # For short puts, delta should ALWAYS be positive (regardless of display mode)
        # This is critical for correct hedging calculations
        if self.is_short and is_put and greeks['delta'] < 0:
            # Make short put delta positive for both calculation and display
            greeks['delta'] = abs(greeks['delta'])
            # Also adjust dollar delta
            greeks['dollar_delta'] = abs(greeks['dollar_delta'])
        
        # If requesting display format, apply additional trading convention signs
        if for_display:
            # For short positions, adjust other Greeks for display conventions
            if self.is_short:
                # Gamma should be negative for short positions
                greeks['gamma'] = -abs(greeks['gamma'])
                greeks['dollar_gamma'] = -abs(greeks['dollar_gamma'])
                
                # Theta should be positive for short positions (time decay is beneficial)
                greeks['theta'] = abs(greeks['theta'])
                greeks['dollar_theta'] = abs(greeks['dollar_theta'])
                
                # Vega should be negative for short positions (volatility decrease is beneficial)
                greeks['vega'] = -abs(greeks['vega'])
                greeks['dollar_vega'] = -abs(greeks['dollar_vega'])
        
        return greeks

    def get_position_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the position for reporting

        Returns:
            dict: Position summary dictionary
        """
        return {
            'symbol': self.symbol,
            'contracts': self.contracts,
            'entry_price': self.avg_entry_price,
            'current_price': self.current_price,
            'underlying_price': self.underlying_price,
            'value': self.current_price * self.contracts * 100,
            'delta': self.current_delta,
            'gamma': self.current_gamma,
            'theta': self.current_theta,
            'vega': self.current_vega,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'dte': self.days_to_expiry,
            'is_short': self.is_short
        }

    def mark_end_of_day(self) -> None:
        """
        Mark the end of the trading day by recording the current price as the previous day price.
        This ensures accurate daily P&L calculations.
        """
        self.previous_day_price = self.current_price
        self.previous_day_contracts = self.contracts  # Store the current contract count for next day's P&L calculation
        
        # Log the operation if logger is available
        if self.logger:
            self.logger.debug(f"[Position] Marked end of day for {self.symbol}: price=${self.current_price:.2f}, contracts={self.contracts}")


class OptionPosition(Position):
    """
    Specialized Position class for options with additional option-specific functionality.
    """
    
    @classmethod
    def parse_option_symbol(cls, symbol: str, logger=None) -> Dict[str, Any]:
        """
        Class method to parse an option symbol and extract key information.
        
        Args:
            symbol: Option symbol (e.g., SPY240419P00461000)
            logger: Optional logger for debugging
            
        Returns:
            Dict with strike, expiry, and option_type
        """
        result = {
            'strike': None,
            'expiry': None,
            'option_type': None,
            'underlying': None
        }
        
        if logger:
            logger.debug(f"Parsing option symbol: {symbol}")
        else:
            # Create a temporary logger for debugging
            logger = logging.getLogger('temp_logger')
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
            
        if not symbol or len(symbol) < 10:
            logger.warning(f"Symbol too short to parse: {symbol}")
            return result
            
        try:
            # First, find the underlying symbol (first 3 letters)
            result['underlying'] = symbol[:3]
            logger.debug(f"Extracted underlying: {result['underlying']}")
            
            # The option type should be at position 9 (after SPYYYMMDD)
            type_idx = 9
            if type_idx < len(symbol) and symbol[type_idx] in ['C', 'P']:
                result['option_type'] = symbol[type_idx]
                logger.debug(f"Found option type {result['option_type']} at position {type_idx}")
            else:
                logger.warning(f"Could not find option type (C/P) in symbol: {symbol}")
                return result
            
            # Extract date components (format is SPY + YYMMDD + C/P + Strike)
            try:
                date_part = symbol[3:9]  # Fixed position for date
                year = int('20' + date_part[:2])
                month = int(date_part[2:4])
                day = int(date_part[4:6])
                result['expiry'] = datetime(year, month, day)
                logger.debug(f"Extracted expiry: {result['expiry'].strftime('%Y-%m-%d')}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing expiry: {e}")
                    
            # Extract strike price
            try:
                # Strike is everything after the option type
                strike_str = symbol[type_idx+1:]
                logger.debug(f"Strike string: {strike_str}")
                if strike_str:
                    # Convert from standard format (e.g., 00461000 -> 461.0)
                    strike_value = float(strike_str)
                    result['strike'] = strike_value / 1000.0
                    logger.debug(f"Extracted strike: {result['strike']}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing strike: {e}")
                logger.debug(f"Strike string: {strike_str}")
                    
            return result
            
        except Exception as e:
            logger.warning(f"Error parsing option symbol {symbol}: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return result
    
    def __init__(
        self, 
        symbol: str, 
        option_data: Optional[Dict] = None,
        contracts: int = 0, 
        entry_price: float = 0.0,
        current_price: float = 0.0,
        is_short: bool = False,
        strategy_id: str = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize an option position.
        
        Args:
            symbol: Option symbol (e.g., SPY230616C00410000)
            option_data: Dictionary with option data
            contracts: Number of contracts
            entry_price: Entry price per contract
            current_price: Current price per contract
            is_short: Whether this is a short position
            strategy_id: Identifier for the strategy that created this position
            logger: Logger instance
        """
        super().__init__(
            symbol=symbol,
            contracts=contracts,
            entry_price=entry_price,
            is_short=is_short,
            position_type='option',
            logger=logger
        )
        
        # Set this as an option type position
        self.position_type = 'option'
        
        # Option specific properties
        self.option_type = None
        self.underlying = None  # Initialize underlying to None
        self.strike = None  # Initialize strike to None
        
        # Initialize option_data if None
        if option_data is None:
            option_data = {}
            
        if logger:
            logger.debug(f"Initializing OptionPosition for {symbol} with data: {option_data}")
            logger.debug("Raw option_data keys: " + ", ".join(str(k) for k in option_data.keys()))
            if 'Strike' in option_data:
                logger.debug(f"Strike in option_data: {option_data['Strike']} (type: {type(option_data['Strike'])})")
        
        # Parse the option symbol first to get basic information
        parsed_data = self.parse_option_symbol(symbol, logger)
        if logger:
            logger.debug(f"Parsed symbol data: {parsed_data}")
        
        # Set option type from parsed data if available
        if parsed_data['option_type']:
            self.option_type = parsed_data['option_type']
            if logger:
                logger.debug(f"Set option_type from parsed data: {self.option_type}")
        
        # Try to extract option_type from data if not from symbol
        if not self.option_type and 'Type' in option_data:
            option_type = str(option_data['Type']).lower()
            if 'call' in option_type or 'c' in option_type:
                self.option_type = 'C'
            elif 'put' in option_type or 'p' in option_type:
                self.option_type = 'P'
            if logger:
                logger.debug(f"Set option_type from option_data: {self.option_type}")
            
        # Log warning if option_type could not be determined
        if not self.option_type and logger:
            logger.warning(f"Could not determine option type from {symbol}")
        
        # Set underlying from parsed data
        if parsed_data['underlying']:
            self.underlying = parsed_data['underlying']
            if logger:
                logger.debug(f"Set underlying from symbol parsing: {self.underlying}")
        
        # Set underlying from option_data if not set from parsing
        if (not self.underlying or self.underlying == "") and 'UnderlyingSymbol' in option_data:
            self.underlying = option_data['UnderlyingSymbol']
            if logger:
                logger.debug(f"Set underlying from option_data: {self.underlying}")
        
        # If still no underlying, try to extract it from symbol (first few chars before date)
        if (not self.underlying or self.underlying == "") and symbol and len(symbol) >= 3:
            # Assuming the symbol starts with the underlying ticker, e.g., SPY240328C00518000
            # Try to extract the underlying symbol - this is a fallback method
            # For SPY options, the underlying would be "SPY"
            for i in range(min(6, len(symbol))):
                if symbol[i].isdigit():
                    self.underlying = symbol[:i]
                    if logger:
                        logger.debug(f"Set underlying as fallback from symbol prefix: {self.underlying}")
                    break
        
        # Set strike price - try multiple sources
        if parsed_data['strike'] is not None:
            self.strike = float(parsed_data['strike'])  # Ensure strike is float
            if logger:
                logger.debug(f"Set strike from symbol parsing: {self.strike}")
        elif 'Strike' in option_data:
            try:
                self.strike = float(option_data['Strike'])  # Ensure strike is float
                if logger:
                    logger.debug(f"Set strike from option_data: {self.strike}")
            except (ValueError, TypeError) as e:
                if logger:
                    logger.error(f"Error converting strike price from option_data: {e}")
                    logger.debug(f"Strike value in option_data: {option_data['Strike']} (type: {type(option_data['Strike'])})")
        
        # Set expiration date - try multiple sources
        if parsed_data['expiry'] is not None:
            self.expiration = parsed_data['expiry']
            if logger:
                logger.debug(f"Set expiry from symbol parsing: {self.expiration}")
        elif 'Expiration' in option_data:
            self.expiration = option_data['Expiration']
            if logger:
                logger.debug(f"Set expiry from option_data: {self.expiration}")
        
        # Store the full option data
        self.option_data = option_data
        
        # Extract and set option greeks from option_data
        if hasattr(option_data, 'get'):
            # Dictionary or object with get method
            self.current_delta = option_data.get('Delta', 0)
            self.current_gamma = option_data.get('Gamma', 0)
            self.current_theta = option_data.get('Theta', 0)
            self.current_vega = option_data.get('Vega', 0)
        else:
            # DataFrame row or other object with direct attribute access
            self.current_delta = getattr(option_data, 'Delta', 0)
            self.current_gamma = getattr(option_data, 'Gamma', 0)
            self.current_theta = getattr(option_data, 'Theta', 0)
            self.current_vega = getattr(option_data, 'Vega', 0)
                
        # Adjust delta sign for short positions
        if is_short and self.current_delta != 0:
            self.current_delta = -abs(self.current_delta)
        
        # Option-specific reference data (for specialized calculations)
        self.option_chain_id = None  # An identifier to link to the option chain
        self.implied_volatility = option_data.get('IV', 0) if hasattr(option_data, 'get') else (
            option_data['IV'] if 'IV' in option_data.index else 0
        )
        
        # For CPD (Constant Portfolio Delta) reference
        self.cpd_reference_symbol = None
        self.cpd_reference_price = None
        self.cpd_reference_delta = None
        self.cpd_reference_expiry = None
        self.cpd_effective_contracts = 0
        
        # Final check to make sure we have Strike, Expiration, and Underlying attributes
        if logger:
            logger.debug(f"OptionPosition initialized: Underlying={self.underlying}, "
                        f"Strike={getattr(self, 'strike', 'None')}, "
                        f"Expiry={getattr(self, 'expiration', 'None')}, "
                        f"Type={self.option_type}")
            if not hasattr(self, 'strike') or self.strike is None:
                logger.warning("Strike price is missing after initialization")
                
    def get_strike(self) -> Optional[float]:
        """
        Get the strike price of the option.
        
        Returns:
            float or None: Strike price if available, None otherwise
        """
        if hasattr(self, 'strike') and self.strike is not None:
            return float(self.strike)
        return None

    def _calculate_position_value(self, quantity: int, price: float, is_short: bool) -> float:
        """
        Calculate the value of an option position.
        
        Args:
            quantity: Number of contracts
            price: Price per contract
            is_short: Whether the position is short
            
        Returns:
            float: Position value
        """
        # For options, multiply by 100 (option multiplier)
        return price * quantity * 100

    def update_market_data(
        self,
        price: Optional[float] = None,
        delta: Optional[float] = None,
        gamma: Optional[float] = None,
        theta: Optional[float] = None,
        vega: Optional[float] = None,
        implied_volatility: Optional[float] = None,
        market_data: Optional[Union[Dict[str, Any], pd.Series]] = None
    ) -> float:
        """
        Update position with latest market data and calculate unrealized PnL.
        This overloaded version allows updating with individual parameters.

        Args:
            price: Current price
            delta: Current delta
            gamma: Current gamma
            theta: Current theta
            vega: Current vega
            implied_volatility: Current implied volatility
            market_data: Full market data dictionary/Series (optional)

        Returns:
            float: Updated unrealized PnL
        """
        # If market_data is provided, use the parent method
        if market_data is not None:
            return super().update_market_data(market_data)
        
        # Store the previous price for P&L calculations
        self.prev_price = self.current_price
        
        # Otherwise, update with individual parameters
        if price is not None:
            self.current_price = price
        
        if delta is not None:
            self.current_delta = delta
        
        if gamma is not None:
            self.current_gamma = gamma
        
        if theta is not None:
            self.current_theta = theta
        
        if vega is not None:
            self.current_vega = vega
        
        if implied_volatility is not None:
            self.implied_volatility = implied_volatility
        
        # Calculate unrealized PnL
        self.update_unrealized_pnl()
        
        return self.unrealized_pnl
    
    def is_itm(self) -> bool:
        """
        Check if the option is in-the-money based on current market data.

        Returns:
            bool: True if in-the-money, False otherwise
        """
        if not self.underlying_price or self.strike is None:
            return False
            
        if self.type:
            option_type = self.type.lower() if isinstance(self.type, str) else str(self.type).lower()
            if option_type == 'call':
                return self.underlying_price > self.strike
            elif option_type == 'put':
                return self.underlying_price < self.strike
                
        return False
    
    def calculate_moneyness(self) -> float:
        """
        Calculate option moneyness (distance from ATM).

        Returns:
            float: Moneyness ratio (>1 for ITM calls, <1 for ITM puts)
        """
        if not self.underlying_price or self.strike is None:
            return 0
            
        return self.underlying_price / self.strike
    
    def set_cpd_reference(
        self, 
        reference_symbol: str, 
        reference_price: float, 
        reference_delta: float, 
        reference_expiry: Optional[datetime] = None
    ) -> float:
        """
        Set the CPD reference information for this position

        Args:
            reference_symbol: Reference option symbol
            reference_price: Reference option price
            reference_delta: Reference option delta
            reference_expiry: Reference option expiration

        Returns:
            float: Calculated effective contracts
        """
        self.cpd_reference_symbol = reference_symbol
        self.cpd_reference_price = reference_price
        self.cpd_reference_delta = reference_delta
        self.cpd_reference_expiry = reference_expiry
        
        # Calculate effective contracts based on price ratio - only done once
        if reference_price > 0:
            price_ratio = self.avg_entry_price / reference_price
            self.cpd_effective_contracts = self.contracts * price_ratio
            
            self.logger.info(f"[CPD Reference] Position {self.symbol}: Set reference price ${reference_price:.2f}")
            self.logger.info(f"  Price ratio: {price_ratio:.2f}")
            self.logger.info(f"  Effective contracts: {self.cpd_effective_contracts:.2f}")
            
            return self.cpd_effective_contracts
        
        return 0
    
    def update_cpd_effective_contracts_for_closure(self, closed_contracts: int) -> float:
        """
        Update effective contracts when some contracts are closed

        Args:
            closed_contracts: Number of contracts closed

        Returns:
            float: Change in effective contracts
        """
        if self.contracts <= 0 or closed_contracts <= 0:
            return 0
        
        # Calculate the proportion of contracts being closed
        proportion_closed = closed_contracts / (self.contracts + closed_contracts)
        
        # Calculate the effective contracts being reduced
        reduced_effective = self.cpd_effective_contracts * proportion_closed
        
        # Update the effective contracts
        self.cpd_effective_contracts -= reduced_effective
        
        self.logger.info(f"[CPD Reference] Position {self.symbol}: Reduced effective contracts")
        self.logger.info(f"  Closed {closed_contracts} contracts ({proportion_closed:.2%})")
        self.logger.info(f"  Reduced effective contracts by {reduced_effective:.2f}")
        self.logger.info(f"  Remaining effective contracts: {self.cpd_effective_contracts:.2f}")
        
        return reduced_effective