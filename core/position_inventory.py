"""
Position Inventory Module

This module provides a centralized repository for portfolio positions,
serving as a single source of truth for position data, valuation,
and risk metrics calculation.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .position import Position, OptionPosition


class PositionInventory:
    """
    Centralized repository for all positions in the portfolio.
    
    This class serves as a single source of truth for position data,
    handling position management, valuation, and metrics calculation
    in a standardized way.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the position inventory.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger('trading_engine')
        self.positions: Dict[str, Position] = {}
        self.option_positions: Dict[str, Position] = {}
        self.stock_positions: Dict[str, Position] = {}
        
    def add_position(self, position: Position) -> None:
        """
        Add a position to the inventory.
        
        Args:
            position: The position to add
        """
        symbol = position.symbol
        self.positions[symbol] = position
        
        # Also store in the appropriate specialized dictionary
        if isinstance(position, OptionPosition):
            self.option_positions[symbol] = position
        else:
            self.stock_positions[symbol] = position
            
        self.logger.debug(f"Added position to inventory: {symbol} ({type(position).__name__})")
        
    def remove_position(self, symbol: str) -> Optional[Position]:
        """
        Remove a position from the inventory.
        
        Args:
            symbol: The symbol of the position to remove
            
        Returns:
            Position or None: The removed position or None if not found
        """
        position = self.positions.pop(symbol, None)
        
        if position:
            # Also remove from the specialized dictionary
            if isinstance(position, OptionPosition):
                self.option_positions.pop(symbol, None)
            else:
                self.stock_positions.pop(symbol, None)
                
            self.logger.debug(f"Removed position from inventory: {symbol}")
            
        return position
        
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get a position by symbol.
        
        Args:
            symbol: The symbol of the position
            
        Returns:
            Position or None: The position or None if not found
        """
        return self.positions.get(symbol)
        
    def get_all_positions(self) -> Dict[str, Position]:
        """
        Get all positions.
        
        Returns:
            dict: Dictionary of all positions
        """
        return self.positions
        
    def get_option_positions(self) -> Dict[str, Position]:
        """
        Get option positions only.
        
        Returns:
            dict: Dictionary of option positions
        """
        return self.option_positions
        
    def get_stock_positions(self) -> Dict[str, Position]:
        """
        Get stock positions only.
        
        Returns:
            dict: Dictionary of stock positions
        """
        return self.stock_positions
        
    def has_position(self, symbol: str) -> bool:
        """
        Check if a position exists.
        
        Args:
            symbol: The symbol to check
            
        Returns:
            bool: True if the position exists
        """
        return symbol in self.positions
        
    def get_total_value(self) -> float:
        """
        Get the total value of all positions.
        
        Returns:
            float: Total position value
        """
        return sum(position.get_position_value() for position in self.positions.values())
        
    def get_option_delta(self) -> float:
        """
        Get the total delta from option positions only.
        
        Returns:
            float: Option delta in option contract equivalents (negative for short calls, positive for short puts)
        """
        option_delta = 0.0
        
        # Calculate only from option positions
        for position in self.option_positions.values():
            # Get position Greeks with proper calculation signs, not display signs
            position_greeks = position.get_greeks(for_display=False)
            option_delta += position_greeks['delta']
        
        return option_delta
        
    def get_stock_delta(self) -> float:
        """
        Get the delta from stock positions.
        
        Returns:
            float: Stock delta in option contract equivalents (negative for short positions)
        """
        stock_delta = 0.0
        
        # Calculate only from stock positions
        for position in self.stock_positions.values():
            # For stocks, delta is shares with sign for direction
            raw_delta = -position.contracts if position.is_short else position.contracts
            # Normalize to option contract equivalents (100 shares = 1 contract)
            normalized_delta = raw_delta / 100.0
            stock_delta += normalized_delta
        
        return stock_delta
        
    def get_total_delta(self) -> float:
        """
        Get the total delta from all positions.
        
        Returns:
            float: Total delta in option contract equivalents
        """
        return self.get_option_delta() + self.get_stock_delta()
    
    def get_dollar_delta(self, underlying_price: float = 100.0) -> Dict[str, float]:
        """
        Calculate dollar delta values for all positions.
        
        Args:
            underlying_price: Price of the underlying for calculation

        Returns:
            dict: Dictionary with option, stock, and total dollar delta
        """
        # Calculate option dollar delta
        option_dollar_delta = 0.0
        for position in self.option_positions.values():
            sign = -1 if position.is_short else 1
            if hasattr(position, 'underlying_price') and position.underlying_price > 0:
                price = position.underlying_price
            else:
                price = underlying_price
            
            position_delta = position.current_delta * position.contracts * sign
            position_dollar_delta = position_delta * 100 * price
            option_dollar_delta += position_dollar_delta
        
        # Calculate stock dollar delta
        stock_dollar_delta = 0.0
        for position in self.stock_positions.values():
            raw_delta = -position.contracts if position.is_short else position.contracts
            if hasattr(position, 'current_price') and position.current_price > 0:
                price = position.current_price
            else:
                price = underlying_price
            
            position_dollar_delta = raw_delta * price
            stock_dollar_delta += position_dollar_delta
        
        return {
            'option_dollar_delta': option_dollar_delta,
            'stock_dollar_delta': stock_dollar_delta,
            'total_dollar_delta': option_dollar_delta + stock_dollar_delta
        }
        
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio metrics for all positions.
        
        Returns:
            dict: Dictionary of portfolio metrics
        """
        metrics = {
            'total_value': self.get_total_value(),
            'option_delta': self.get_option_delta(),
            'stock_delta': self.get_stock_delta(),
            'total_delta': self.get_total_delta(),
            'option_count': len(self.option_positions),
            'stock_count': len(self.stock_positions),
            'total_position_count': len(self.positions)
        }
        
        return metrics
    
    def get_portfolio_greeks(self, underlying_price: float = 100.0) -> Dict[str, float]:
        """
        Calculate portfolio-level Greek risk metrics.
        
        Args:
            underlying_price: Price of the underlying for calculating dollar Greeks
            
        Returns:
            dict: Dictionary of portfolio Greeks with proper normalization
        """
        # Initialize the Greeks dictionary
        greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'dollar_delta': 0.0,
            'dollar_gamma': 0.0,
            'dollar_theta': 0.0,
            'dollar_vega': 0.0
        }
        
        # Calculate option Greeks
        for position in self.option_positions.values():
            # Get position Greeks with calculation signs, not display signs
            position_greeks = position.get_greeks(for_display=False)
            
            # Add up the Greeks
            greeks['delta'] += position_greeks['delta']
            greeks['gamma'] += position_greeks['gamma']
            greeks['theta'] += position_greeks['theta']
            greeks['vega'] += position_greeks['vega']
            greeks['dollar_delta'] += position_greeks['dollar_delta']
            greeks['dollar_gamma'] += position_greeks['dollar_gamma']
            greeks['dollar_theta'] += position_greeks['dollar_theta']
            greeks['dollar_vega'] += position_greeks['dollar_vega']
        
        # Add stock delta
        stock_delta = self.get_stock_delta()
        stock_dollar_delta = 0.0
        
        for position in self.stock_positions.values():
            raw_delta = -position.contracts if position.is_short else position.contracts
            price = position.current_price if hasattr(position, 'current_price') and position.current_price > 0 else underlying_price
            stock_dollar_delta += raw_delta * price
        
        greeks['delta'] += stock_delta
        greeks['dollar_delta'] += stock_dollar_delta
        
        # Calculate delta percentage if we have the portfolio value
        portfolio_value = self.get_total_value()
        if portfolio_value > 0:
            greeks['delta_pct'] = greeks['dollar_delta'] / portfolio_value
        else:
            greeks['delta_pct'] = 0.0
        
        return greeks
    
    def __len__(self) -> int:
        """
        Get the number of positions.
        
        Returns:
            int: Number of positions
        """
        return len(self.positions)
    
    def __iter__(self):
        """
        Iterate over positions.
        
        Returns:
            iterator: Iterator over positions
        """
        return iter(self.positions.values())
    
    def items(self):
        """
        Get position items (symbol, position).
        
        Returns:
            items: Dictionary items
        """
        return self.positions.items() 