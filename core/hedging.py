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
        portfolio,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the hedging manager.
        
        Args:
            portfolio: Portfolio instance
            config: Hedging configuration
            logger: Logger instance
        """
        self.portfolio = portfolio
        self.logger = logger or logging.getLogger('trading_engine')
        
        # Store the config for reference
        self.config = config
        
        # Initialize hedging parameters
        self.enable_hedging = config.get('enabled', False)
        self.hedge_mode = config.get('mode', 'constant').lower()
        self.hedge_with_underlying = config.get('hedge_with_underlying', True)
        
        # For ratio mode: target as ratio of portfolio value
        self.target_delta_ratio = config.get('target_delta_ratio', config.get('delta_target', 0.0))
        
        # For constant mode: absolute target delta
        self.target_portfolio_delta = config.get('target_delta', 0.0)
        
        # Common parameters
        self.delta_tolerance = config.get('delta_tolerance', 0.05)
        self.hedge_symbol = config.get('hedge_symbol', 'SPY')
        
        # Maximum hedge ratio to prevent excessive hedging
        self.max_hedge_ratio = config.get('max_hedge_ratio', 2.0)
        
        # Initialize tracking variables
        self.hedge_history = []
        self.current_hedge_delta = 0
        self.current_dollar_delta = 0
        
        # Initialize cached prices dictionary
        self.cached_prices = {}
        
        # Log initialization in a standardized format
        if self.logger:
            self.logger.info("=" * 40)
            self.logger.info("HEDGING MANAGER INITIALIZATION")
            
            if self.enable_hedging:
                self.logger.info(f"  Hedging: Enabled")
                self.logger.info(f"  Hedge mode: {self.hedge_mode}")
                
                if self.hedge_mode == 'ratio':
                    self.logger.info(f"  Target delta ratio: {self.target_delta_ratio:.2f}")
                else:  # constant mode
                    self.logger.info(f"  Target portfolio delta: {self.target_portfolio_delta:.2f}")
                    
                self.logger.info(f"  Delta tolerance: {self.delta_tolerance:.2f}")
                self.logger.info(f"  Hedge symbol: {self.hedge_symbol}")
                self.logger.info(f"  Hedge with underlying: {self.hedge_with_underlying}")
                self.logger.info(f"  Max hedge ratio: {self.max_hedge_ratio:.2f}x")
                
                # Log additional hedging parameters if present
                rebalance_threshold = config.get('rebalance_threshold', None)
                if rebalance_threshold is not None:
                    self.logger.info(f"  Rebalance threshold: {rebalance_threshold:.2f}")
                    
                min_hedge_size = config.get('min_hedge_size', None)
                if min_hedge_size is not None:
                    self.logger.info(f"  Min hedge size: {min_hedge_size}")
                    
                hedge_interval = config.get('hedge_interval', None)
                if hedge_interval is not None:
                    self.logger.info(f"  Hedge interval: {hedge_interval} days")
            else:
                self.logger.info(f"  Hedging: Disabled")
                
            self.logger.info("=" * 40)
    
    def set_cached_prices(self, prices: Dict[str, float]) -> None:
        """
        Set the cached prices dictionary for consistent price access across all hedging operations.
        
        Args:
            prices: Dictionary of symbol -> price mappings
        """
        self.cached_prices = prices
        if self.logger:
            if self.hedge_symbol in prices:
                self.logger.info(f"Set cached price for primary hedge symbol {self.hedge_symbol}: ${prices[self.hedge_symbol]:.2f}")
            self.logger.info(f"Cached prices set for {len(prices)} symbols")
    
    def calculate_hedge_requirements(self, market_data: pd.DataFrame, current_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate hedging requirements based on current portfolio delta.
        
        Args:
            market_data: Market data for the current day
            current_date: Current trading date
            
        Returns:
            dict: Hedging requirement details
        """
        # If hedging is disabled, return no requirements
        if not self.enable_hedging:
            return {'required': False}
        
        # Get portfolio delta (option positions)
        option_delta = self.portfolio.get_option_delta()  # Already normalized to option contracts
        option_dollar_delta = 0
        
        # Get hedge delta (stock positions) - already normalized to option contracts
        hedge_delta = self.portfolio.get_hedge_delta()
        hedge_dollar_delta = 0
        
        # Get underlying price from market data
        underlying_price = self._get_underlying_price(market_data)
        
        # Calculate dollar deltas if we have a valid underlying price
        if underlying_price > 0:
            option_dollar_delta = option_delta * underlying_price * 100  # Convert to dollars
            
            # Get the hedge position and dollar delta
            hedge_position = self.get_hedge_position()
            if hedge_position:
                # Dollar delta uses actual shares (not normalized)
                hedge_dollar_delta = hedge_position.current_delta * hedge_position.current_price
            else:
                hedge_dollar_delta = hedge_delta * underlying_price * 100  # Convert to dollars
        
        # Total delta (in option contract equivalents)
        total_delta = option_delta + hedge_delta
        
        # Total dollar delta
        total_dollar_delta = option_dollar_delta + hedge_dollar_delta
        
        # Get portfolio value
        portfolio_value = self.portfolio.get_portfolio_value()
        
        # Set target_delta and tolerance based on hedge mode
        if self.hedge_mode == 'ratio':
            # Get target delta ratio based on portfolio value
            target_dollar_delta = portfolio_value * self.target_delta_ratio
            
            # Convert to option contract equivalents for delta
            if underlying_price > 0:
                target_delta = target_dollar_delta / (underlying_price * 100)
            else:
                target_delta = 0
                
            # Calculate tolerance in dollar terms
            tolerance_dollars = portfolio_value * self.delta_tolerance
            if underlying_price > 0:
                tolerance_delta = tolerance_dollars / (underlying_price * 100)
            else:
                tolerance_delta = 0
        else:
            # Constant mode: fixed target delta
            target_delta = self.target_portfolio_delta
            tolerance_delta = self.delta_tolerance
        
        # Calculate deviation from target
        delta_deviation = total_delta - target_delta
        
        # Determine if hedging is required based on tolerance
        needs_hedging = abs(delta_deviation) > tolerance_delta
        
        # Define adjusted target and deviation variables
        adjusted_target = None
        adjusted_deviation = None
        
        # Calculate required hedging contracts
        if needs_hedging:
            # Adjust target to hedge to boundary of range instead of exact target
            if self.hedge_mode == 'ratio':
                if delta_deviation > 0:
                    # We're over the target by more than tolerance, adjust to upper bound
                    adjusted_target = target_delta + tolerance_delta
                else:
                    # We're under the target by more than tolerance, adjust to lower bound
                    adjusted_target = target_delta - tolerance_delta
                
                # Recalculate deviation using the adjusted target
                adjusted_deviation = total_delta - adjusted_target
                self.logger.info(f"[Hedge Calculation] Original target: {target_delta:.4f}, Adjusted target (boundary): {adjusted_target:.4f}")
                self.logger.info(f"[Hedge Calculation] Original deviation: {delta_deviation:.4f}, Adjusted deviation: {adjusted_deviation:.4f}")
                
                # Log dollar values for clarity
                if underlying_price > 0:
                    target_dollar = target_delta * underlying_price * 100
                    adjusted_target_dollar = adjusted_target * underlying_price * 100
                    adjusted_deviation_dollar = adjusted_deviation * underlying_price * 100
                    self.logger.info(f"[Hedge Calculation] Original target: ${target_dollar:.2f}, Adjusted target (boundary): ${adjusted_target_dollar:.2f}")
                    self.logger.info(f"[Hedge Calculation] Original deviation: ${delta_deviation * underlying_price * 100:.2f}, Adjusted deviation: ${adjusted_deviation_dollar:.2f}")
                
                # Use the adjusted deviation for hedging calculations
                delta_deviation = adjusted_deviation
            else:  # constant mode
                if delta_deviation > 0:
                    # We're over the target by more than tolerance, adjust to upper bound
                    adjusted_target = target_delta + tolerance_delta
                else:
                    # We're under the target by more than tolerance, adjust to lower bound
                    adjusted_target = target_delta - tolerance_delta
                
                # Recalculate deviation using the adjusted target
                adjusted_deviation = total_delta - adjusted_target
                self.logger.info(f"[Hedge Calculation] Original target: {target_delta:.4f}, Adjusted target (boundary): {adjusted_target:.4f}")
                self.logger.info(f"[Hedge Calculation] Original deviation: {delta_deviation:.4f}, Adjusted deviation: {adjusted_deviation:.4f}")
                
                # Use the adjusted deviation for hedging calculations
                delta_deviation = adjusted_deviation
            
            # Convert delta to shares (100 shares = 1.0 delta)
            # delta_deviation is in option contract equivalents, multiply by 100 to get shares
            hedge_contracts = int(round(abs(delta_deviation) * 100))
            
            # Determine direction based on deviation sign
            if delta_deviation > 0:
                # Need to reduce delta - sell shares (is_short=True)
                direction = 'sell'
            else:
                # Need to increase delta - buy shares (is_short=False)
                direction = 'buy'
                
            # Enforce maximum hedge ratio to prevent excessive hedging
            max_hedge_contracts = int(portfolio_value * self.max_hedge_ratio / underlying_price)
            if hedge_contracts > max_hedge_contracts:
                self.logger.warning(f"[Hedge Calculation] Limiting hedge from {hedge_contracts} to {max_hedge_contracts} contracts (max hedge ratio: {self.max_hedge_ratio})")
                hedge_contracts = max_hedge_contracts
            
            # Additional safeguard - limit hedge to reasonable multiple of option delta
            option_delta_contracts = int(abs(option_delta) * 100)  # Convert option delta to contracts
            max_hedge_to_option_ratio = 1.5  # Maximum hedge position as multiple of option position
            
            if option_delta_contracts > 0:
                max_hedge_by_option = int(option_delta_contracts * max_hedge_to_option_ratio)
                if hedge_contracts > max_hedge_by_option:
                    self.logger.warning(f"[Hedge Calculation] Further limiting hedge from {hedge_contracts} to {max_hedge_by_option} contracts (max hedge-to-option ratio: {max_hedge_to_option_ratio}x)")
                    hedge_contracts = max_hedge_by_option
            
            # Final hard constraint - ensure hedge value is no more than 50% of portfolio value
            if underlying_price > 0 and portfolio_value > 0:
                max_hedge_value = portfolio_value * 0.5
                max_hedge_contracts_by_value = int(max_hedge_value / underlying_price)
                
                if hedge_contracts > max_hedge_contracts_by_value:
                    self.logger.warning(f"[Hedge Calculation] Hard limit applied: reducing hedge from {hedge_contracts} to {max_hedge_contracts_by_value} contracts (max 50% of portfolio value)")
                    hedge_contracts = max_hedge_contracts_by_value
                    
            # Set minimum threshold - if we end up with a very small hedge, don't bother
            min_hedge_contracts = 5
            if hedge_contracts < min_hedge_contracts:
                self.logger.info(f"[Hedge Calculation] Hedge size {hedge_contracts} below minimum threshold of {min_hedge_contracts}, skipping hedge")
                hedge_contracts = 0
                needs_hedging = False
                direction = None
        else:
            hedge_contracts = 0
            direction = None
        
        # Special handling for existing hedge position
        if hedge_position and hedge_contracts > 0:
            current_contracts = hedge_position.contracts
            current_direction = 'sell' if hedge_position.is_short else 'buy'
            
            # Log current position details
            self.logger.info(f"[Hedge Calculation] Current position: {current_contracts} contracts ({current_direction})")
            self.logger.info(f"[Hedge Calculation] Initial hedge plan: {hedge_contracts} contracts ({direction})")
            
            # If we need to reverse direction
            if current_direction != direction:
                # Calculate required contracts based on delta deviation
                # If we have 562 long and need to be 341 short, that's a 903 total adjustment
                # But we should check that this doesn't exceed our delta deviation requirement
                total_adjustment = current_contracts + hedge_contracts
                
                # Convert back to delta to verify we're not overshooting
                total_delta_adjustment = total_adjustment / 100  # Convert shares back to delta
                self.logger.info(f"[Hedge Calculation] Direction reversal: {current_direction} to {direction}")
                self.logger.info(f"[Hedge Calculation] Total adjustment: {total_adjustment} contracts ({total_delta_adjustment} delta)")
                self.logger.info(f"[Hedge Calculation] Adjusted deviation: {adjusted_deviation}, Abs: {abs(adjusted_deviation)}")
                
                # If total adjustment would overshoot our target, limit it
                if total_delta_adjustment > abs(adjusted_deviation):
                    # Limit to the actual delta deviation
                    original_contracts = hedge_contracts
                    hedge_contracts = int(round(abs(adjusted_deviation) * 100))
                    self.logger.warning(f"[Hedge Calculation] Limiting hedge adjustment from {original_contracts} to {hedge_contracts} contracts to avoid overshooting target")
                else:
                    # Use the calculated value which includes closing the existing position
                    hedge_contracts = total_adjustment
                    self.logger.info(f"[Hedge Calculation] Using total adjustment: {hedge_contracts} contracts")
            else:
                # Just add to existing position in same direction
                original_contracts = hedge_contracts
                hedge_contracts = abs(hedge_contracts - current_contracts)
                self.logger.info(f"[Hedge Calculation] Same direction adjustment: from {original_contracts} to {hedge_contracts} contracts")
                if hedge_contracts == 0:
                    needs_hedging = False
                    direction = None
                    self.logger.info("[Hedge Calculation] No adjustment needed - setting needs_hedging to False")
        
        self.logger.info(f"[Hedge Calculation] Final hedge plan: {hedge_contracts} contracts ({direction})")
        
        # Return detailed requirements
        return {
            'required': needs_hedging,
            'contracts': hedge_contracts,
            'direction': direction,
            'current_delta': total_delta,
            'target_delta': target_delta,
            'delta_deviation': delta_deviation,
            'dollar_delta': total_dollar_delta,
            'target_dollar_delta': target_dollar_delta if self.hedge_mode == 'ratio' else target_delta * 100 * underlying_price,
            'portfolio_value': portfolio_value,
            'delta_ratio': total_dollar_delta / portfolio_value if portfolio_value > 0 else 0,
            'tolerance': tolerance_delta,
            'tolerance_dollars': tolerance_dollars if self.hedge_mode == 'ratio' else tolerance_delta * 100 * underlying_price,
            'adjusted_target': adjusted_target if needs_hedging else None,
            'adjusted_deviation': adjusted_deviation if needs_hedging else None,
            'adjusted_target_dollar': adjusted_target * underlying_price * 100 if needs_hedging and underlying_price > 0 else None,
            'adjusted_deviation_dollar': adjusted_deviation * underlying_price * 100 if needs_hedging and underlying_price > 0 else None
        }
    
    def get_hedge_position(self) -> Optional[Position]:
        """
        Get the current hedge position if it exists.
        Also updates the current_hedge_delta and current_dollar_delta attributes.
        
        Returns:
            Position or None: The current hedge position or None if no hedge exists
        """
        hedge_position = self.portfolio.positions.get(self.hedge_symbol)
        
        # Update hedge delta values if position exists
        if hedge_position:
            # For stock positions, delta is the number of shares (positive for long, negative for short)
            total_shares = hedge_position.contracts  # This is the raw number of shares
            sign = -1 if hedge_position.is_short else 1
            
            # Set the position's current_delta to be the raw number of shares with direction
            # This is what will be displayed in position tables
            hedge_position.current_delta = total_shares * sign
            
            # Store the normalized delta (option contract equivalent) for internal calculations
            # 1 option contract = 100 shares, so divide by 100
            normalized_delta = (total_shares / 100.0) * sign
            
            # Store as attributes for external access
            self.current_hedge_delta = normalized_delta
            
            # Calculate dollar delta using the current price
            if hasattr(hedge_position, 'current_price') and hedge_position.current_price > 0:
                # Dollar delta is based on the actual number of shares (not normalized)
                self.current_dollar_delta = hedge_position.current_delta * hedge_position.current_price
            else:
                # Fallback to average entry price if current price not available
                self.current_dollar_delta = hedge_position.current_delta * hedge_position.avg_entry_price
            
            self.logger.debug(f"Hedge position: {self.hedge_symbol}, Shares: {total_shares}, Direction: {'Short' if hedge_position.is_short else 'Long'}")
            self.logger.debug(f"Hedge delta: {normalized_delta:.3f} (option contracts equivalent), Dollar delta: ${self.current_dollar_delta:,.2f}")
        else:
            # No hedge position exists
            self.current_hedge_delta = 0
            self.current_dollar_delta = 0
            self.logger.debug("No hedge position exists")
        
        return hedge_position
    
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
        
        # Calculate the original delta deviation before boundary adjustments
        original_delta_deviation = requirements['current_delta'] - requirements['target_delta']
        original_dollar_deviation = requirements['dollar_delta'] - requirements['target_dollar_delta']
        
        # Log details for reporting
        self.logger.info(f"[Hedging Analysis] Date: {current_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"  Current delta: {requirements['current_delta']:.2f} (${requirements['dollar_delta']:.2f})")
        self.logger.info(f"  Target delta: {requirements['target_delta']:.2f} (${requirements['target_dollar_delta']:.2f})")
        self.logger.info(f"  Deviation: {original_delta_deviation:.2f} (${original_dollar_deviation:.2f})")
        self.logger.info(f"  Tolerance:  {requirements['tolerance']:.2f} ( ${requirements['tolerance_dollars']:.2f})")
        
        if requirements['required']:
            # Add adjusted target information if hedging is needed
            if 'adjusted_target' in requirements:
                self.logger.info(f"  Adjusted target (boundary): {requirements['adjusted_target']:.2f} (${requirements['adjusted_target_dollar']:.2f})")
                self.logger.info(f"  Adjusted deviation: {requirements['adjusted_deviation']:.2f} (${requirements['adjusted_deviation_dollar']:.2f})")
        
        # Generate hedge signals
        signals = []
        
        if requirements['required'] and requirements['contracts'] > 0:
            # Get the underlying price
            underlying_price = self._get_underlying_price(market_data)
            
            if underlying_price > 0:
                self.logger.info(f"[Hedging] Required {requirements['direction']} {requirements['contracts']} shares of {self.hedge_symbol} @ ${underlying_price:.2f}")
                
                # Create hedge signal
                hedge_signal = {
                    'symbol': self.hedge_symbol,
                    'action': requirements['direction'],
                    'quantity': requirements['contracts'],
                    'price': underlying_price,
                    'position_type': 'stock',
                    'reason': 'Delta hedge',
                    'instrument_data': {
                        'Symbol': self.hedge_symbol,
                        'UnderlyingSymbol': self.hedge_symbol,
                        'Type': 'stock',
                        'Price': underlying_price,
                        'Delta': 1.0 if requirements['direction'] == 'buy' else -1.0,
                        'HedgeRatio': requirements['delta_ratio']
                    }
                }
                
                signals.append(hedge_signal)
                
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
        
    def apply_hedging(self, current_date: datetime, market_data: pd.DataFrame) -> None:
        """
        Apply hedging adjustments to the portfolio based on calculated hedge requirements.
        
        This method:
        1. Generates hedge signals
        2. Executes the signals to modify portfolio hedges
        3. Logs the hedging activity
        
        Args:
            current_date: Current trading date
            market_data: Market data for hedging calculations
        """
        if not self.enable_hedging:
            self.logger.info("[Hedging] Hedging is disabled")
            return
            
        # Generate hedge signals
        hedge_signals = self.generate_hedge_signals(market_data, current_date)
        
        # If we have signals, apply them to the portfolio
        if hedge_signals:
            self.logger.info(f"[Hedging] Applying {len(hedge_signals)} hedge adjustments")
            
            for signal in hedge_signals:
                symbol = signal['symbol']
                action = signal['action']
                quantity = signal['quantity']
                price = signal['price']
                reason = signal['reason']
                
                # Create instrument data dictionary if not provided
                instrument_data = signal.get('instrument_data', {
                    'Symbol': symbol,
                    'Type': 'stock',
                    'Price': price,
                    'Delta': 1.0 if action == 'buy' else -1.0
                })
                
                # For buy/sell actions, convert to add/remove position operations
                if action.lower() in ['buy', 'sell']:
                    # ACTION CORRESPONDS DIRECTLY TO is_short:
                    # 'sell' action means creating a short position (is_short=True, negative delta)
                    # 'buy' action means creating a long position (is_short=False, positive delta)
                    is_short = action.lower() == 'sell'
                    
                    # Log the action and explain direction
                    self.logger.info(f"[Hedging] Action: {action.upper()}, Setting is_short={is_short}")
                    self.logger.info(f"[Hedging] Direction explanation: {'Short position will have negative delta' if is_short else 'Long position will have positive delta'}")
                    
                    # Check if we already have a position in this symbol
                    existing_position = self.portfolio.positions.get(symbol)
                    
                    if existing_position:
                        # If position exists and is in opposite direction, we need to adjust it
                        if existing_position.is_short != is_short:
                            self.logger.info(f"[Hedging] Direction reversal: {symbol} from {'short' if existing_position.is_short else 'long'} to {'short' if is_short else 'long'}")
                            
                            # Calculate the net adjustment needed
                            if existing_position.contracts <= quantity:
                                # Close the existing position and open a new one in the opposite direction
                                # with the remaining quantity
                                remaining_quantity = quantity - existing_position.contracts
                                
                                self.logger.info(f"[Hedging] Closing existing position: {symbol} ({existing_position.contracts} contracts)")
                                self.portfolio.remove_position(
                                    symbol=symbol,
                                    quantity=quantity,
                                    price=price,
                                    reason=f"Hedge reversal: {reason}"
                                )
                                
                                if remaining_quantity > 0:
                                    # Now add the new position with remaining quantity
                                    self.portfolio.add_position(
                                        symbol=symbol,
                                        quantity=remaining_quantity,
                                        price=price,
                                        position_type='stock',
                                        is_short=is_short,
                                        instrument_data=instrument_data,
                                        reason=reason
                                    )
                                    
                                    self.logger.info(f"[Hedging] Added new position: {remaining_quantity} {'short' if is_short else 'long'} {symbol} @ ${price:.2f}")
                            else:
                                # Reduce the existing position
                                self.logger.info(f"[Hedging] Reducing existing position: {symbol} from {existing_position.contracts} to {existing_position.contracts - quantity}")
                                self.portfolio.remove_position(
                                    symbol=symbol,
                                    quantity=quantity,
                                    price=price,
                                    reason=f"Hedge adjustment: {reason}"
                                )
                        else:
                            # Same direction, just add to position
                            self.logger.info(f"[Hedging] Adding to existing position: {symbol} (+{quantity} contracts)")
                            self.portfolio.add_position(
                                symbol=symbol,
                                quantity=quantity,
                                price=price,
                                position_type='stock',
                                is_short=is_short,
                                instrument_data=instrument_data,
                                reason=reason
                            )
                    else:
                        # No existing position, create new one
                        self.portfolio.add_position(
                            symbol=symbol,
                            quantity=quantity,
                            price=price,
                            position_type='stock',
                            is_short=is_short,
                            instrument_data=instrument_data,
                            reason=reason
                        )
                        
                        self.logger.info(f"[Hedging] Added new position: {quantity} {'short' if is_short else 'long'} {symbol} @ ${price:.2f}")
                        self.logger.info(f"[Hedging] Position delta: {-quantity if is_short else quantity} (before normalizing to option contract equivalents)")
                        
                elif action.lower() == 'close':
                    # Close an existing position
                    if symbol in self.portfolio.positions:
                        self.logger.info(f"[Hedging] Closing position: {symbol}")
                        self.portfolio.remove_position(
                            symbol=symbol,
                            price=price,
                            reason=reason
                        )
                    else:
                        self.logger.warning(f"[Hedging] Cannot close position {symbol} - not found")
                
                else:
                    self.logger.warning(f"[Hedging] Unknown action: {action}")
        else:
            self.logger.info("[Hedging] No hedge adjustments required")
            
        # Update hedge position data
        self.get_hedge_position()
    
    def _get_underlying_price(self, market_data: Dict[str, Any]) -> float:
        """
        Get the underlying price from market data.
        
        Args:
            market_data: Market data
            
        Returns:
            float: Underlying price or 0 if not found
        """
        underlying_price = 0
        
        # FIRST PRIORITY: Use cached prices if available
        if hasattr(self, 'cached_prices') and self.cached_prices and self.hedge_symbol in self.cached_prices:
            underlying_price = self.cached_prices[self.hedge_symbol]
            if underlying_price > 0:
                self.logger.info(f"Using underlying price from cached prices: ${underlying_price:.2f}")
                return underlying_price
        
        # SECOND PRIORITY: Check for hedge with underlying configuration
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
                    if not market_data.empty:
                        underlying_price = market_data['UnderlyingPrice'].iloc[0]
                        # Make sure we're comparing a numeric value, not a dictionary
                        if isinstance(underlying_price, (int, float)) and underlying_price > 0:
                            self.logger.info(f"Using underlying price from DataFrame: ${underlying_price:.2f}")
                            return underlying_price
                        # Handle case where underlying_price is a dictionary
                        elif isinstance(underlying_price, dict) and 'close' in underlying_price:
                            value = underlying_price.get('close')
                            if isinstance(value, (int, float)) and value > 0:
                                self.logger.info(f"Using underlying price from dictionary: ${value:.2f}")
                                return value
                        # Log warning for non-numeric values
                        elif underlying_price is not None:
                            self.logger.warning(f"Underlying price is not numeric: {type(underlying_price)}, value: {underlying_price}")
        
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
                    underlying_price = matching_rows['UnderlyingPrice'].iloc[0]
                    if isinstance(underlying_price, (int, float)) and underlying_price > 0:
                        self.logger.info(f"Using underlying price from matching rows: ${underlying_price:.2f}")
                        return underlying_price
                    elif isinstance(underlying_price, dict) and 'close' in underlying_price:
                        value = underlying_price.get('close')
                        if isinstance(value, (int, float)) and value > 0:
                            self.logger.info(f"Using underlying price from dictionary in matching rows: ${value:.2f}")
                            return value
        
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
        
        # NEW SECTION: Additional sources to check when market_data doesn't contain the price
        # If the portfolio has access to a data manager, try to use it
        if hasattr(self, 'portfolio') and self.portfolio:
            if hasattr(self.portfolio, 'data_manager') and self.portfolio.data_manager:
                underlying_price = self.portfolio.data_manager.get_latest_price(self.hedge_symbol) or 0
                if underlying_price > 0:
                    self.logger.info(f"Using underlying price from data manager: ${underlying_price:.2f}")
                    return underlying_price
            
            # Try to get price from portfolio's market data
            if hasattr(self.portfolio, 'market_data'):
                market_data = self.portfolio.market_data
                if isinstance(market_data, pd.DataFrame) and not market_data.empty:
                    # Try to extract price from DataFrame
                    if self.hedge_symbol in market_data.index:
                        price_col = next((col for col in ['Close', 'close', 'Price', 'price'] 
                                          if col in market_data.columns), None)
                        if price_col:
                            underlying_price = market_data.loc[self.hedge_symbol, price_col]
                            if underlying_price > 0:
                                self.logger.info(f"Using underlying price from portfolio market data frame: ${underlying_price:.2f}")
                                return underlying_price
        
        # If we're in a TradingEngine, try to use its price lookup method
        if hasattr(self, 'trading_engine'):
            try:
                if hasattr(self.trading_engine, '_get_underlying_price'):
                    underlying_price = self.trading_engine._get_underlying_price(self.hedge_symbol) or 0
                    if underlying_price > 0:
                        self.logger.info(f"Using underlying price from trading engine: ${underlying_price:.2f}")
                        return underlying_price
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error getting underlying price from trading engine: {str(e)}")
        
        # As a fallback, use a realistic default price based on major indices
        # Default prices for common indices/ETFs
        default_prices = {
            'SPY': 475.0,
            'QQQ': 430.0,
            'IWM': 200.0,
            'DIA': 380.0
        }
        
        # Use specific price if available, otherwise use a general default
        default_price = default_prices.get(self.hedge_symbol, 450.0)
        
        self.logger.warning(f"Could not find price for {self.hedge_symbol} in market data")
        self.logger.warning(f"No underlying price available for {self.hedge_symbol}, using default price of ${default_price:.2f}")
        
        return default_price

    def verify_hedge_calculation(self, total_delta: float, target_delta: float, tolerance_delta: float, underlying_price: float = 100.0) -> Dict[str, Any]:
        """
        Verify that the hedge calculation is working as expected.
        This is mainly used for testing and debugging the hedging logic.
        
        Args:
            total_delta: Total delta of the portfolio
            target_delta: Target delta
            tolerance_delta: Delta tolerance
            underlying_price: Underlying price
            
        Returns:
            dict: Dictionary with verification results
        """
        # Calculate original delta deviation
        delta_deviation = total_delta - target_delta
        
        # Check if hedging is needed
        needs_hedging = abs(delta_deviation) > tolerance_delta
        
        # Apply the new logic to adjust to boundary
        if needs_hedging:
            if delta_deviation > 0:
                adjusted_target = target_delta + tolerance_delta
            else:
                adjusted_target = target_delta - tolerance_delta
            
            adjusted_deviation = total_delta - adjusted_target
        else:
            adjusted_target = target_delta
            adjusted_deviation = delta_deviation
        
        # Convert to dollar values for clarity
        target_dollar = target_delta * underlying_price * 100
        tolerance_dollar = tolerance_delta * underlying_price * 100
        total_dollar = total_delta * underlying_price * 100
        adjusted_target_dollar = adjusted_target * underlying_price * 100
        adjusted_deviation_dollar = adjusted_deviation * underlying_price * 100
        
        # Calculate required hedge contracts
        hedge_contracts = int(round(abs(adjusted_deviation) * 100)) if needs_hedging else 0
        
        # Direction - EXISTING LOGIC IS CORRECT (matches line ~220)
        # When adjusted_deviation > 0, we need to sell shares (is_short=True, negative delta)
        # When adjusted_deviation < 0, we need to buy shares (is_short=False, positive delta)
        direction = 'sell' if adjusted_deviation > 0 else 'buy' if adjusted_deviation < 0 else None
        
        return {
            'total_delta': total_delta,
            'target_delta': target_delta,
            'tolerance_delta': tolerance_delta,
            'needs_hedging': needs_hedging,
            'delta_deviation': delta_deviation,
            'adjusted_target': adjusted_target,
            'adjusted_deviation': adjusted_deviation,
            'hedge_contracts': hedge_contracts,
            'direction': direction,
            'total_dollar': total_dollar,
            'target_dollar': target_dollar,
            'tolerance_dollar': tolerance_dollar,
            'adjusted_target_dollar': adjusted_target_dollar,
            'adjusted_deviation_dollar': adjusted_deviation_dollar
        }
    
    @staticmethod
    def example_hedge_to_boundary():
        """
        Run a specific example of hedging to boundary.
        This is the example from the user's request.
        
        Example case:
        - Current delta: -1.17 ($-55413.49)
        - Target delta: 0.21 ($10000.00)
        - Tolerance: 0.63 ($30000.00)
        
        Original behavior: Hedge to target delta of 0.21
        New behavior: Hedge to boundary (target - tolerance) = -0.42
        """
        # Setup example values
        target = 0.21
        total = -1.17
        tolerance = 0.63
        underlying_price = 472.65
        
        # Print initial values
        print('=== Hedging to Boundary Test Case ===')
        print(f'Target: {target}, Current: {total}, Tolerance: {tolerance}')
        
        # Original approach (hedge to target)
        original_deviation = total - target
        original_shares = int(round(abs(original_deviation) * 100))
        print(f'\n--- Original Approach (Hedge to Target) ---')
        print(f'Deviation from target: {original_deviation}')
        print(f'Required shares: {original_shares}')
        
        # New approach (hedge to boundary)
        adjusted_target = target - tolerance
        print(f'\n--- New Approach (Hedge to Boundary) ---')
        print(f'Target boundary: {adjusted_target}')
        adjusted_deviation = total - adjusted_target
        print(f'Deviation from boundary: {adjusted_deviation}')
        adjusted_shares = int(round(abs(adjusted_deviation) * 100))
        print(f'Required shares: {adjusted_shares}')
        
        # Dollar values
        target_dollar = target * underlying_price * 100
        total_dollar = total * underlying_price * 100
        tolerance_dollar = tolerance * underlying_price * 100
        adjusted_target_dollar = adjusted_target * underlying_price * 100
        adjusted_deviation_dollar = adjusted_deviation * underlying_price * 100
        original_deviation_dollar = original_deviation * underlying_price * 100
        
        print('\n=== Dollar Values ===')
        print(f'Target: ${target_dollar:.2f}')
        print(f'Current: ${total_dollar:.2f}')
        print(f'Tolerance: ${tolerance_dollar:.2f}')
        print(f'Original delta deviation: ${original_deviation_dollar:.2f}')
        print(f'Target boundary: ${adjusted_target_dollar:.2f}')
        print(f'Boundary deviation: ${adjusted_deviation_dollar:.2f}')
        
        return {
            'original_shares': original_shares,
            'adjusted_shares': adjusted_shares,
            'difference': original_shares - adjusted_shares
        }
        
    def calculate_hedge_position(self, portfolio, target_delta):
        """
        Calculate a hedge position for a portfolio to reach a target delta
        
        Args:
            portfolio: The portfolio to hedge
            target_delta: The target delta to reach
            
        Returns:
            Position: The hedge position
        """
        
        # Calculate the current portfolio delta
        current_delta = self.calculate_portfolio_delta(portfolio)
        self.logger.debug(f"Portfolio Delta: {current_delta}, Target Delta: {target_delta}")
        
        # Calculate the delta gap
        delta_gap = target_delta - current_delta
        
        # Skip hedging if delta gap is too small
        if abs(delta_gap) < self.delta_threshold:
            self.logger.debug(f"Delta gap ({delta_gap}) is below threshold ({self.delta_threshold}), skipping hedge")
            return None
        
        # Calculate the number of shares needed to hedge
        shares_needed = delta_gap / 1  # Each share of the underlying has a delta of 1
        
        # Round to nearest whole number
        shares_needed = round(shares_needed)
        
        # Skip hedging if number of shares is too small
        if abs(shares_needed) < self.min_shares:
            self.logger.debug(f"Shares needed ({shares_needed}) is below minimum ({self.min_shares}), skipping hedge")
            return None
        
        # Create a position for the hedge
        is_short = shares_needed < 0
        contracts = abs(shares_needed)
        
        # Get the latest price for the underlying
        underlying_price = self.data_manager.get_latest_price(self.underlying_symbol)
        
        if not underlying_price:
            self.logger.warning(f"Could not get price for {self.underlying_symbol}, skipping hedge")
            return None
        
        # Create the hedge position
        hedge_position = Position(
            symbol=self.underlying_symbol,
            contracts=contracts,
            entry_price=underlying_price,
            is_short=is_short,
            strategy_id="Hedge"
        )
        
        # Set the current price
        hedge_position.current_price = underlying_price
        
        # IMPORTANT: Set the delta directly on the hedge position
        # A long position will have a delta of 1 per share, a short position will have -1 per share
        hedge_position.current_delta = -contracts if is_short else contracts
        
        # Calculate the dollar value of the delta
        dollar_delta = hedge_position.current_delta * underlying_price
        
        self.logger.debug(f"Created hedge position: {hedge_position.symbol}, Contracts: {hedge_position.contracts}, Is Short: {hedge_position.is_short}")
        self.logger.debug(f"Hedge Delta: {hedge_position.current_delta}, Dollar Delta: ${dollar_delta:.2f}")
        
        return hedge_position

    def create_theoretical_hedge_position(self, option_position) -> Optional[Position]:
        """
        Create a theoretical hedge position for a given option position.
        This is used for margin calculations before actual execution.
        
        Args:
            option_position: The option position to hedge
            
        Returns:
            Position: A theoretical hedge position (not added to portfolio)
        """
        if not option_position:
            return None
            
        # Get delta from the option position
        option_delta = option_position.current_delta if hasattr(option_position, 'current_delta') else 0
        
        # Skip if delta is zero
        if option_delta == 0:
            return None
            
        # Get number of contracts
        contracts = option_position.contracts if hasattr(option_position, 'contracts') else 1
        
        # Determine if this is a put or call
        is_put = False
        if hasattr(option_position, 'option_type'):
            is_put = option_position.option_type.upper() in ['P', 'PUT']
        elif hasattr(option_position, 'symbol') and 'P' in option_position.symbol:
            is_put = True
            
        # Determine if position is short
        is_option_short = False
        if hasattr(option_position, 'is_short'):
            is_option_short = option_position.is_short
            
        # Calculate total position delta - this represents the dollar risk per $1 move in the underlying
        # For options, we need to convert from "per share" delta to "per contract" delta
        # For consistency with how the margin calculator handles deltas, we'll express this as dollar delta
        position_delta = option_delta * contracts * 100  # 100 shares per contract
        
        # IMPROVED HEDGE DIRECTION LOGIC:
        # For a short put: Use a short hedge position (negative delta)
        # For a long put: Use a long hedge position (positive delta)
        # For a short call: Use a long hedge position (positive delta)
        # For a long call: Use a short hedge position (negative delta)
        
        # For proper delta hedging, a short put should be hedged with a short stock position
        # even though the hedge delta calculation might suggest otherwise
        
        # The adjusted hedge direction based on option type and position
        if is_put:
            # For puts: short put = short hedge, long put = long hedge
            is_short = is_option_short
        else:
            # For calls: short call = long hedge, long call = short hedge
            is_short = not is_option_short
            
        # Calculate hedge delta (opposite of position delta)
        hedge_delta = -position_delta
        
        # Convert to hedge shares
        hedge_shares = int(round(abs(hedge_delta / 100)))  # Each share has delta of 1.0 or -1.0
        
        # Skip if hedge size is too small
        if hedge_shares < 1:
            return None
            
        # Get underlying symbol and price
        if hasattr(option_position, 'underlying') and option_position.underlying:
            underlying_symbol = option_position.underlying
        elif hasattr(option_position, 'instrument_data') and option_position.instrument_data:
            underlying_symbol = option_position.instrument_data.get('UnderlyingSymbol', self.hedge_symbol)
        else:
            underlying_symbol = self.hedge_symbol
            
        # Get underlying price - FIRST CHECK CACHED PRICES FOR CONSISTENCY
        underlying_price = 0
        
        # First priority: Check cached prices for consistency across all operations
        if hasattr(self, 'cached_prices') and self.cached_prices and underlying_symbol in self.cached_prices:
            underlying_price = self.cached_prices[underlying_symbol]
            if underlying_price > 0:
                self.logger.info(f"Using underlying price from cached prices: ${underlying_price:.2f}")
        
        # Second priority: Check option position attributes if we didn't find a cached price
        if underlying_price <= 0:
            if hasattr(option_position, 'underlying_price') and option_position.underlying_price > 0:
                underlying_price = option_position.underlying_price
            elif hasattr(option_position, 'instrument_data') and option_position.instrument_data:
                underlying_price = option_position.instrument_data.get('UnderlyingPrice', 0)
        
        # Third priority: Try other data sources if still no price
        if underlying_price <= 0:
            if hasattr(self, 'portfolio') and self.portfolio:
                # If the portfolio has access to a data manager, try to use it
                if hasattr(self.portfolio, 'data_manager') and self.portfolio.data_manager:
                    underlying_price = self.portfolio.data_manager.get_latest_price(underlying_symbol) or 0
                    if underlying_price > 0:
                        self.logger.info(f"Using underlying price from data manager: ${underlying_price:.2f}")
                    elif underlying_price <= 0 and self.logger:
                        self.logger.warning(f"Failed to get underlying price for {underlying_symbol} from portfolio's data manager")
                
                # Try to get price from portfolio's market data
                if underlying_price <= 0 and hasattr(self.portfolio, 'market_data'):
                    market_data = self.portfolio.market_data
                    if isinstance(market_data, pd.DataFrame) and not market_data.empty:
                        # Try to extract price from DataFrame
                        if underlying_symbol in market_data.index:
                            price_col = next((col for col in ['Close', 'close', 'Price', 'price'] 
                                              if col in market_data.columns), None)
                            if price_col:
                                underlying_price = market_data.loc[underlying_symbol, price_col]
                                if underlying_price > 0:
                                    self.logger.info(f"Using underlying price from market data frame: ${underlying_price:.2f}")
            
            # If we still don't have a price and we're in a TradingEngine, try to use its price lookup method
            if underlying_price <= 0 and hasattr(self, 'trading_engine'):
                try:
                    if hasattr(self.trading_engine, '_get_underlying_price'):
                        underlying_price = self.trading_engine._get_underlying_price(underlying_symbol) or 0
                        if underlying_price > 0:
                            self.logger.info(f"Using underlying price from trading engine: ${underlying_price:.2f}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error getting underlying price from trading engine: {str(e)}")
        
        # If we still don't have a price, check if there's a _get_underlying_price method
        if underlying_price <= 0 and hasattr(self, '_get_underlying_price'):
            try:
                # Try to get price from our own method with an empty dict
                underlying_price = self._get_underlying_price({}) or 0
                if underlying_price > 0:
                    self.logger.info(f"Using underlying price from _get_underlying_price: ${underlying_price:.2f}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error getting underlying price: {str(e)}")
                underlying_price = 0
        
        # As a fallback, use a realistic default price based on major indices
        if underlying_price <= 0:
            # Default prices for common indices/ETFs
            default_prices = {
                'SPY': 475.0,
                'QQQ': 430.0,
                'IWM': 200.0,
                'DIA': 380.0
            }
            
            # Use specific price if available, otherwise use a general default
            default_price = default_prices.get(underlying_symbol, 450.0)
            
            if self.logger:
                self.logger.warning(f"No underlying price available for {underlying_symbol}, using default price of ${default_price:.2f}")
            
            underlying_price = default_price
    
        # Log the hedge position details
        if self.logger:
            self.logger.info(f"[HedgingManager] Creating theoretical hedge for {option_position.symbol}")
            self.logger.info(f"  Option delta: {option_delta:.4f} × {contracts} contracts = {position_delta/100:.4f}")
            self.logger.info(f"  Hedge delta needed: {abs(hedge_delta/100):.4f}")
            self.logger.info(f"  Hedge shares: {hedge_shares} of {underlying_symbol} at ${underlying_price:.2f}")
            self.logger.info(f"  Direction: {'Short' if is_short else 'Long'}")
            
        # Create the hedge position
        from .position import Position
        
        hedge_position = Position(
            symbol=underlying_symbol,
            contracts=hedge_shares,
            entry_price=underlying_price,
            current_price=underlying_price,
            is_short=is_short,
            position_type='stock',
            logger=self.logger
        )
        
        # CRITICAL FIX: Make sure the underlying_price is set correctly for margin calculation
        hedge_position.underlying_price = underlying_price
        
        # Set the delta explicitly in dollar terms for consistency with margin calculator
        # For stock positions, the dollar delta is simply shares * price * direction
        direction = -1 if is_short else 1
        stock_delta_dollars = hedge_shares * direction * underlying_price
        
        # This is the critical part - for consistency with how the actual positions are handled in margin calculation
        # We need to express the delta in dollar terms to match how the margin calculator calculates net_delta
        hedge_position.current_delta = hedge_shares * direction  # Raw delta (shares count with direction)
        hedge_position.delta_dollars = stock_delta_dollars  # Dollar delta for margin calculations
        
        # Also set the option delta in dollar terms on the option position to ensure consistent calculations
        option_position.delta_dollars = position_delta * underlying_price / 100 if not hasattr(option_position, 'delta_dollars') else option_position.delta_dollars
        
        # Ensure all required attributes for margin calculation are present
        # Stock positions don't use these for margin, but set to zero to avoid attribute errors
        hedge_position.current_gamma = 0
        hedge_position.current_vega = 0
        hedge_position.current_theta = 0
        
        return hedge_position
    
    def create_hedged_position_pair(self, option_position) -> Dict[str, Position]:
        """
        Create a pair of positions (option + hedge) for margin calculation.
        
        Args:
            option_position: The option position to hedge
            
        Returns:
            dict: Dictionary with both positions keyed by symbol
        """
        # Create a theoretical hedge position
        hedge_position = self.create_theoretical_hedge_position(option_position)
        
        # Create the pair dictionary
        positions_dict = {option_position.symbol: option_position}
        
        # Add hedge position if it exists
        if hedge_position:
            positions_dict[hedge_position.symbol] = hedge_position
            
        return positions_dict
    
    def calculate_margin_for_hedged_option(self, option_position, margin_calculator) -> Dict[str, Any]:
        """
        Calculate margin for an option position with its theoretical hedge.
        
        Args:
            option_position: The option position to evaluate
            margin_calculator: The margin calculator to use
            
        Returns:
            dict: Margin calculation results including hedging benefits
        """
        # Make sure option_position has all required attributes for margin calculation
        if not hasattr(option_position, 'underlying_price') or option_position.underlying_price <= 0:
            if hasattr(self, 'cached_prices') and self.cached_prices:
                underlying_symbol = option_position.underlying if hasattr(option_position, 'underlying') else self.hedge_symbol
                if underlying_symbol in self.cached_prices:
                    option_position.underlying_price = self.cached_prices[underlying_symbol]
                    if self.logger:
                        self.logger.info(f"Setting option position underlying price from cache: ${option_position.underlying_price:.2f}")
        
        # Create the hedged position pair
        positions_dict = self.create_hedged_position_pair(option_position)
        
        # Calculate margin using portfolio approach
        margin_result = margin_calculator.calculate_portfolio_margin(positions_dict)
        
        return margin_result
        
    # Add a method to calculate theoretical margin for a new option before adding it
    def calculate_theoretical_margin(self, option_data: Dict[str, Any], quantity: int) -> Dict[str, float]:
        """
        Calculate theoretical margin for a new option position with hedging.
        
        Args:
            option_data: Option data dictionary 
            quantity: Number of contracts
            
        Returns:
            dict: Dictionary with margin amounts and related metrics
        """
        # Create a temporary option position
        from .position import OptionPosition
        
        # Extract key option data
        symbol = option_data.get('symbol', option_data.get('OptionSymbol', 'Unknown'))
        price = option_data.get('price', option_data.get('MidPrice', option_data.get('Ask', 0)))
        underlying_price = option_data.get('UnderlyingPrice', 0)
        delta = option_data.get('Delta', 0)
        option_type = option_data.get('Type', 'P' if 'P' in symbol else 'C')
        strike = option_data.get('Strike', 0)
        expiry = option_data.get('Expiration', None)
        
        # Prepare option data for position creation
        position_data = {
            'Symbol': symbol,
            'UnderlyingSymbol': option_data.get('UnderlyingSymbol', 'SPY'),
            'UnderlyingPrice': underlying_price,
            'Type': option_type,
            'Strike': strike,
            'Expiration': expiry,
        }
        
        # Create the temporary option position
        temp_position = OptionPosition(
            symbol=symbol,
            option_data=position_data,
            contracts=quantity,
            entry_price=price,
            current_price=price,
            is_short=True,  # Assuming short positions for selling options
            logger=self.logger
        )
        
        # Set greeks on the position
        temp_position.current_delta = delta
        temp_position.current_gamma = option_data.get('Gamma', 0)
        temp_position.current_theta = option_data.get('Theta', 0)
        temp_position.current_vega = option_data.get('Vega', 0)
        
        # Ensure underlying price is set
        if temp_position.underlying_price == 0 and underlying_price > 0:
            temp_position.underlying_price = underlying_price
            
        # Get the portfolio's margin calculator
        margin_calculator = None
        if self.portfolio and hasattr(self.portfolio, 'margin_calculator'):
            margin_calculator = self.portfolio.margin_calculator
        else:
            # Create a temporary margin calculator
            from .margin import SPANMarginCalculator
            margin_calculator = SPANMarginCalculator(
                max_leverage=12.0,
                hedge_credit_rate=0.8,
                logger=self.logger
            )
            
        # Calculate margin with hedging
        margin_result = self.calculate_margin_for_hedged_option(temp_position, margin_calculator)
        
        # Return the calculation results
        return margin_result