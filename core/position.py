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
        instrument_data: Dict[str, Any], 
        initial_contracts: int = 0, 
        is_short: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize a position with instrument data.

        Args:
            symbol: Unique identifier for the instrument
            instrument_data: Initial data for the instrument
            initial_contracts: Starting number of contracts
            is_short: Whether this is a short position
            logger: Logger instance
        """
        self.symbol = symbol
        self.logger = logger or logging.getLogger('trading_engine')
        
        # Handle pandas Series objects correctly
        if hasattr(instrument_data, 'get') and not hasattr(instrument_data, 'iloc'):
            # This is a dictionary
            self.strike = instrument_data.get('Strike')
            self.expiration = instrument_data.get('Expiration')
            self.type = instrument_data.get('Type')
        else:
            # This is a pandas Series
            self.strike = instrument_data['Strike'] if 'Strike' in instrument_data else None
            self.expiration = instrument_data['Expiration'] if 'Expiration' in instrument_data else None
            self.type = instrument_data['Type'] if 'Type' in instrument_data else None

        self.contracts = initial_contracts
        self.is_short = is_short
        
        # Log the position type for debugging
        instrument_type = "unknown"
        if self.type is not None:
            instrument_type = self.type.lower() if isinstance(self.type, str) else str(self.type).lower()
        
        self.logger.debug(f"[Position] Created {instrument_type} position, is_short={self.is_short}")
        
        # For average price calculation
        self.total_value = 0
        self.avg_entry_price = 0
        
        # For P&L tracking
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.max_drawdown = 0
        
        # For market data
        self.current_price = 0
        self.prev_price = None  # Store previous price for daily P&L calculation
        self.current_delta = 0
        self.current_gamma = 0
        self.current_theta = 0
        self.current_vega = 0
        self.underlying_price = 0
        self.days_to_expiry = 0
        
        # Transaction history
        self.transactions = []
        self.daily_data = []
        
        # Update with initial data if provided
        if instrument_data is not None:
            self.update_market_data(instrument_data)

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
        
        transaction = {
            'date': transaction_date,
            'action': 'SELL' if self.is_short else 'BUY',
            'contracts': quantity,
            'price': price,
            'value': price * quantity * 100
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
        self.total_value = self.avg_entry_price * self.contracts * 100
        
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
        if quantity <= 0 or quantity > self.contracts:
            return 0
        
        # Store current average entry price
        entry_price = self.avg_entry_price
        
        # Calculate realized PnL
        # For short positions: entry_price - exit_price is the P&L
        # For long positions: exit_price - entry_price is the P&L
        if self.is_short:
            pnl = (entry_price - price) * quantity * 100
        else:
            pnl = (price - entry_price) * quantity * 100
        
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

        Args:
            market_data: Current market data for this instrument

        Returns:
            float: Updated unrealized PnL
        """
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
                self.current_delta = market_data.get('Delta', 0)
                self.current_gamma = market_data.get('Gamma', 0)
                self.current_theta = market_data.get('Theta', 0)
                self.current_vega = market_data.get('Vega', 0)
                self.underlying_price = market_data.get('UnderlyingPrice', 0)
                
                # Calculate days to expiry if possible
                if 'DataDate' in market_data and self.expiration:
                    self.days_to_expiry = (self.expiration - market_data.get('DataDate')).days
            # Check if market_data is a pandas Series
            elif hasattr(market_data, 'iloc'):
                # Series style access
                self.current_price = market_data['MidPrice'] if 'MidPrice' in market_data.index else 0
                self.current_delta = market_data['Delta'] if 'Delta' in market_data.index else 0
                self.current_gamma = market_data['Gamma'] if 'Gamma' in market_data.index else 0
                self.current_theta = market_data['Theta'] if 'Theta' in market_data.index else 0
                self.current_vega = market_data['Vega'] if 'Vega' in market_data.index else 0
                self.underlying_price = market_data['UnderlyingPrice'] if 'UnderlyingPrice' in market_data.index else 0
                
                # Calculate days to expiry if possible
                if 'DataDate' in market_data.index and self.expiration is not None:
                    self.days_to_expiry = (self.expiration - market_data['DataDate']).days
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error updating market data: {e}")
        
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
            self.unrealized_pnl = pnl_per_contract * self.contracts * 100
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
            
        # Calculate total PnL based on contract size and count
        # Check if this is an OptionPosition by checking for option-specific attributes
        is_option = hasattr(self, 'option_symbol') or hasattr(self, 'strike') or hasattr(self, 'expiration')
        
        if is_option or self.type is not None:
            # This is an OptionPosition - multiply by 100
            self.unrealized_pnl = pnl_per_contract * self.contracts * 100
        else:
            # This is a regular Position (e.g., stock)
            self.unrealized_pnl = pnl_per_contract * self.contracts
        
        # Update max drawdown tracking (drawdown is negative PnL)
        if self.unrealized_pnl < -self.max_drawdown:
            self.max_drawdown = -self.unrealized_pnl

    def calculate_margin_requirement(self, max_leverage: float) -> float:
        """
        Calculate margin requirement for this position, accounting for unrealized PnL.
        
        Initial margin is based on entry price, and unrealized PnL (both gains and losses) is factored in.

        Args:
            max_leverage: Maximum leverage allowed

        Returns:
            float: Margin requirement in dollars
        """
        if self.contracts <= 0:
            return 0
        
        # Initial margin calculation based on entry price
        initial_margin = self.avg_entry_price * self.contracts * 100 * max_leverage
        
        # Adjust margin for unrealized PnL (both gains and losses)
        adjusted_margin = initial_margin - self.unrealized_pnl
        
        return adjusted_margin

    def get_greeks(self) -> Dict[str, float]:
        """
        Get position Greeks

        Returns:
            dict: Dictionary of position Greeks
        """
        sign = -1 if self.is_short else 1
        return {
            'delta': sign * self.current_delta * self.contracts,
            'gamma': sign * self.current_gamma * self.contracts,
            'theta': sign * self.current_theta * self.contracts,
            'vega': sign * self.current_vega * self.contracts,
            'dollar_delta': sign * self.current_delta * self.contracts * self.underlying_price * 100,
            'dollar_gamma': sign * self.current_gamma * self.contracts * (self.underlying_price ** 2) * 0.01,
            'dollar_theta': sign * self.current_theta * self.contracts,
            'dollar_vega': sign * self.current_vega * self.contracts
        }

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


class OptionPosition(Position):
    """
    Specialized Position class for options with additional option-specific functionality.
    """
    
    def __init__(
        self, 
        option_symbol: str, 
        option_data: Dict[str, Any], 
        initial_contracts: int = 0, 
        is_short: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize an option position with option-specific data.

        Args:
            option_symbol: Unique identifier for the option
            option_data: Initial data for the option
            initial_contracts: Starting number of contracts
            is_short: Whether this is a short position
            logger: Logger instance
        """
        super().__init__(option_symbol, option_data, initial_contracts, is_short, logger)
        
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