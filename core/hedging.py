"""
Delta Hedging Module

This module provides tools for dynamically managing portfolio delta exposure
through hedging operations. It supports various hedging strategies including
ratio-based and constant-delta approaches with configurable tolerances.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .position import Position, OptionPosition
from .portfolio import Portfolio

class HedgingManager:
    """
    Manages delta hedging of portfolio positions to maintain target delta exposure.
    
    This class implements various delta hedging strategies to keep portfolio 
    delta exposure within the desired range, calculating hedge requirements,
    generating hedge signals, and tracking hedging performance.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        portfolio: Portfolio,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the hedging manager.
        
        Args:
            config: Hedging configuration parameters
            portfolio: Portfolio instance to manage hedging for
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger('trading')
        self.portfolio = portfolio
        self.config = config
        
        # Extract hedging parameters
        self.enable_hedging = config.get('enable_hedging', False)
        self.hedge_mode = config.get('hedge_mode', 'ratio').lower()
        
        # Get delta target
        self.delta_target = config.get('delta_target', 0.0)
        
        # For ratio mode: target as ratio of portfolio value
        self.target_delta_ratio = config.get('target_delta_ratio', config.get('delta_target', 0.0))  
        
        # For constant mode: absolute target delta
        self.target_portfolio_delta = config.get('constant_portfolio_delta', 0.0)
        
        # Tolerance - only hedge when delta exposure is outside this band
        self.delta_tolerance = config.get('hedge_tolerance', config.get('delta_tolerance', 0.05))
        
        # Underlying symbol for hedging
        self.hedge_symbol = config.get('hedge_symbol', 'SPY')
        
        # Flag to determine if we should hedge with the underlying from option data
        self.hedge_with_underlying = config.get('hedge_with_underlying', False)
        
        # Track hedging history
        self.hedge_history = []
        
        # Log initialization
        if self.enable_hedging:
            self.logger.info(f"[HedgingManager] Initialized with {self.hedge_mode} mode")
            
            if self.hedge_mode == 'ratio':
                self.logger.info(f"  Target delta ratio: {self.target_delta_ratio:.2f}")
            else:  # constant mode
                self.logger.info(f"  Target portfolio delta: {self.target_portfolio_delta:.2f}")
                
            self.logger.info(f"  Delta tolerance: {self.delta_tolerance:.2f}")
            self.logger.info(f"  Hedge symbol: {self.hedge_symbol}")
            self.logger.info(f"  Hedge with underlying price: {self.hedge_with_underlying}")
        else:
            self.logger.info("[HedgingManager] Hedging disabled")
    
    def calculate_hedge_requirements(
        self, 
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate hedging requirements based on current portfolio exposure.
        
        Args:
            market_data: Dictionary of market data including underlying price
            
        Returns:
            dict: Hedge requirements including target contracts and direction
        """
        if not self.enable_hedging:
            return {
                'required': False,
                'contracts': 0,
                'direction': None,
                'current_delta': 0,
                'target_delta': 0,
                'delta_deviation': 0
            }
        
        # Get portfolio metrics
        portfolio_metrics = self.portfolio.get_portfolio_metrics()
        portfolio_value = portfolio_metrics.get('portfolio_value', 0)
        
        # Get current portfolio delta
        option_greeks = self.portfolio.get_portfolio_greeks()
        current_delta = option_greeks.get('delta', 0)
        dollar_delta = option_greeks.get('dollar_delta', 0)
        
        # Add hedge delta to get total portfolio delta
        hedge_position = self.get_hedge_position()
        hedge_delta = 0
        
        if hedge_position:
            hedge_delta = hedge_position.contracts
            # Adjust for hedge position direction
            if hedge_position.is_short:
                hedge_delta *= -1
        
        total_delta = current_delta + hedge_delta
        
        # Calculate target delta based on mode
        if self.hedge_mode == 'ratio':
            # Ratio mode: target delta as percentage of portfolio value
            if portfolio_value > 0:
                target_delta_dollars = portfolio_value * self.target_delta_ratio
                
                # Normalize by underlying price to get target delta in contracts
                underlying_price = self._get_underlying_price(market_data)
                if underlying_price > 0:
                    target_delta = target_delta_dollars / underlying_price
                else:
                    target_delta = 0
            else:
                target_delta = 0
        else:
            # Constant mode: fixed target delta
            target_delta = self.target_portfolio_delta
        
        # Calculate deviation from target
        delta_deviation = total_delta - target_delta
        
        # Determine if hedging is required based on tolerance
        needs_hedging = abs(delta_deviation) > self.delta_tolerance
        
        # Calculate required hedging contracts
        if needs_hedging:
            # Round to nearest contract
            hedge_contracts = int(round(delta_deviation))
            
            # Determine direction
            if delta_deviation > 0:
                # Need to hedge long delta: sell shares
                direction = 'sell'
            else:
                # Need to hedge short delta: buy shares
                direction = 'buy'
                # Make contracts positive for buying
                hedge_contracts = abs(hedge_contracts)
        else:
            hedge_contracts = 0
            direction = None
        
        # Special handling for existing hedge position
        if hedge_position and hedge_contracts > 0:
            current_contracts = hedge_position.contracts
            current_direction = 'sell' if hedge_position.is_short else 'buy'
            
            # If we need to reverse direction
            if current_direction != direction:
                # First close existing position
                hedge_contracts = current_contracts + hedge_contracts
            else:
                # Just add to existing position in same direction
                hedge_contracts = abs(hedge_contracts - current_contracts)
        
        # Return detailed requirements
        return {
            'required': needs_hedging,
            'contracts': hedge_contracts,
            'direction': direction,
            'current_delta': total_delta,
            'target_delta': target_delta,
            'delta_deviation': delta_deviation,
            'delta_dollars': dollar_delta,
            'portfolio_value': portfolio_value,
            'delta_ratio': dollar_delta / portfolio_value if portfolio_value > 0 else 0,
            'tolerance': self.delta_tolerance
        }
    
    def get_hedge_position(self) -> Optional[Position]:
        """
        Get the current hedge position if it exists.
        
        Returns:
            Position or None: The current hedge position or None if no hedge exists
        """
        return self.portfolio.positions.get(self.hedge_symbol)
    
    def generate_hedge_signals(
        self, 
        market_data: Dict[str, Any],
        current_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals to implement required hedging.
        
        Args:
            market_data: Dictionary of market data
            current_date: Current simulation date
            
        Returns:
            list: List of trading signals for hedging
        """
        if not self.enable_hedging:
            return []
        
        # Calculate hedge requirements
        requirements = self.calculate_hedge_requirements(market_data)
        
        # Log hedge analysis
        self.logger.info(f"[Hedging Analysis] Date: {current_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"  Current delta: {requirements['current_delta']:.2f} (${requirements['delta_dollars']:.2f})")
        self.logger.info(f"  Target delta: {requirements['target_delta']:.2f}")
        
        # Calculate and log the acceptable delta range
        lower_bound = requirements['target_delta'] - requirements['tolerance']
        upper_bound = requirements['target_delta'] + requirements['tolerance']
        self.logger.info(f"  Acceptable delta range: {lower_bound:.2f} to {upper_bound:.2f}")
        
        self.logger.info(f"  Deviation: {requirements['delta_deviation']:.2f}")
        self.logger.info(f"  Portfolio Value: ${requirements['portfolio_value']:.2f}")
        self.logger.info(f"  Delta/NLV Ratio: {requirements['delta_ratio']:.2%}")
        self.logger.info(f"  Tolerance band: ±{requirements['tolerance']:.2f}")
        
        # Add underlying hedge status
        underlying_price = self._get_underlying_price(market_data)
        if underlying_price > 0:
            self.logger.info(f"  Underlying price available: ${underlying_price:.2f}")
        else:
            self.logger.info(f"  Underlying price not available - hedging may not execute")
        
        # Generate hedging signals if required
        signals = []
        
        if requirements['required'] and requirements['contracts'] > 0:
            # Get data for the hedge symbol
            underlying_price = self._get_underlying_price(market_data)
            
            if underlying_price > 0:
                # Create hedge signal
                hedge_signal = {
                    'action': 'SELL' if requirements['direction'] == 'sell' else 'BUY',
                    'symbol': self.hedge_symbol,
                    'quantity': requirements['contracts'],
                    'price': underlying_price,
                    'position_type': 'stock',
                    'is_short': requirements['direction'] == 'sell',
                    'data': {
                        'Underlying': self.hedge_symbol,
                        'UnderlyingPrice': underlying_price,
                        'DataDate': current_date
                    },
                    'reason': 'Delta Hedge'
                }
                
                signals.append(hedge_signal)
                
                # Log hedging action
                self.logger.info(f"  Hedging Action: {hedge_signal['action']} {hedge_signal['quantity']} {self.hedge_symbol} @ ${underlying_price:.2f}")
                
                # Record hedge in history
                self.hedge_history.append({
                    'date': current_date,
                    'action': hedge_signal['action'],
                    'symbol': self.hedge_symbol,
                    'contracts': requirements['contracts'],
                    'price': underlying_price,
                    'value': requirements['contracts'] * underlying_price,
                    'current_delta': requirements['current_delta'],
                    'target_delta': requirements['target_delta'],
                    'delta_deviation': requirements['delta_deviation'],
                    'portfolio_value': requirements['portfolio_value'],
                    'delta_ratio': requirements['delta_ratio']
                })
            else:
                self.logger.warning(f"[Hedging] Underlying price not available for {self.hedge_symbol}, cannot hedge")
        else:
            if not requirements['required']:
                self.logger.info(f"  No hedging required - within tolerance band")
            else:
                self.logger.info(f"  Hedge size too small ({requirements['contracts']} contracts) - skipping")
        
        return signals
    
    def _get_underlying_price(self, market_data: Dict[str, Any]) -> float:
        """
        Get the price of the underlying security from market data.
        
        Args:
            market_data: Dictionary of market data
            
        Returns:
            float: Price of the underlying security
        """
        # Check for hedge with underlying configuration
        hedge_with_underlying = self.config.get('hedge_with_underlying', False)
        
        if hedge_with_underlying:
            # Prioritize using the UnderlyingPrice column for the hedge symbol
            # Check if any market data has 'UnderlyingPrice' we can use
            if isinstance(market_data, dict):
                for symbol, data in market_data.items():
                    if hasattr(data, 'get'):
                        # Direct access to UnderlyingPrice
                        underlying_price = data.get('UnderlyingPrice')
                        if underlying_price is not None and underlying_price > 0:
                            self.logger.info(f"Using underlying price from market data: ${underlying_price:.2f}")
                            return underlying_price
                    elif hasattr(data, 'UnderlyingPrice'):
                        # Attribute access
                        underlying_price = data.UnderlyingPrice
                        if underlying_price is not None and underlying_price > 0:
                            self.logger.info(f"Using underlying price from market data: ${underlying_price:.2f}")
                            return underlying_price
            
            # If market_data is a DataFrame, try to extract from it
            if isinstance(market_data, pd.DataFrame):
                if 'UnderlyingPrice' in market_data.columns:
                    if not market_data.empty and market_data['UnderlyingPrice'].iloc[0] > 0:
                        underlying_price = market_data['UnderlyingPrice'].iloc[0]
                        self.logger.info(f"Using underlying price from DataFrame: ${underlying_price:.2f}")
                        return underlying_price
        
        # Standard approach for specific hedge symbol lookup
        # Try to get from market data directly
        if self.hedge_symbol in market_data:
            if hasattr(market_data[self.hedge_symbol], 'get'):
                price = market_data[self.hedge_symbol].get('Close', 0)
                if price > 0:
                    return price
            else:
                if 'Close' in market_data[self.hedge_symbol]:
                    price = market_data[self.hedge_symbol]['Close']
                    if price > 0:
                        return price
        
        # If market_data is a DataFrame, try to extract from it
        if isinstance(market_data, pd.DataFrame):
            if 'UnderlyingSymbol' in market_data.columns and 'UnderlyingPrice' in market_data.columns:
                matching_rows = market_data[market_data['UnderlyingSymbol'] == self.hedge_symbol]
                if not matching_rows.empty:
                    return matching_rows['UnderlyingPrice'].iloc[0]
        
        # Check if any market data has specific hedge symbol and price
        if isinstance(market_data, dict):
            for symbol, data in market_data.items():
                underlying_symbol = None
                underlying_price = None
                
                if hasattr(data, 'get'):
                    underlying_symbol = data.get('UnderlyingSymbol')
                    underlying_price = data.get('UnderlyingPrice')
                elif hasattr(data, 'UnderlyingSymbol') and hasattr(data, 'UnderlyingPrice'):
                    underlying_symbol = data.UnderlyingSymbol
                    underlying_price = data.UnderlyingPrice
                
                if underlying_symbol == self.hedge_symbol and underlying_price is not None:
                    return underlying_price
        
        # If all else fails, default to 0
        self.logger.warning(f"Could not find price for {self.hedge_symbol} in market data")
        return 0