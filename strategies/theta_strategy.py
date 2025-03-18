"""
Theta Decay Strategy Module

This module implements a theta decay focused options trading strategy
that sells out-of-the-money options to capture premium decay over time.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# Handle imports properly whether run as script or imported as module
import os
import sys
if __name__ == "__main__":
    # Add the parent directory to the path so we can run this file directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.trading_engine import Strategy
    from core.position import Position
else:
    # When imported as a module, use absolute imports instead of relative imports
    from core.trading_engine import Strategy
    from core.position import Position


class ThetaDecayStrategy(Strategy):
    """
    A strategy focused on capturing theta decay from option premium.
    
    This strategy:
    1. Sells out-of-the-money options with specific delta targets
    2. Implements delta hedging to manage directional risk
    3. Uses profit taking and stop loss rules for risk management
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the theta decay strategy.
        
        Args:
            config: Strategy configuration parameters
            logger: Logger instance for recording events
        """
        super().__init__("ThetaDecayStrategy", config, logger)
        
        # Strategy parameters (with defaults)
        self.days_to_expiry_min = config.get('days_to_expiry_min', 30)
        self.days_to_expiry_max = config.get('days_to_expiry_max', 45)
        self.delta_target = config.get('delta_target', -0.05)
        self.delta_tolerance = config.get('delta_tolerance', 0.02)
        self.profit_target = config.get('profit_target', 0.65)
        self.stop_loss_threshold = config.get('stop_loss_threshold', 2.0)
        self.close_days_to_expiry = config.get('close_days_to_expiry', 14)
        self.is_short = config.get('is_short', True)
        
        # Strategy state variables
        self.last_trade_date = None
        
        if self.logger:
            self.logger.info(f"[ThetaDecayStrategy] Initialized with delta target: {self.delta_target}")
    
    def generate_signals(self, current_date: datetime, daily_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate trading signals for the current date.
        
        Args:
            current_date: Current backtest date
            daily_data: DataFrame containing daily market data
        
        Returns:
            List of trading signal dictionaries
        """
        signals = []
        
        # Apply trading logic only on certain days or conditions
        if self.last_trade_date and (current_date - self.last_trade_date).days < 5:
            if self.logger:
                self.logger.debug("[ThetaDecayStrategy] Skipping signal generation - cooling period")
            return signals
        
        # Filter options meeting our criteria
        filtered_options = self._filter_candidate_options(daily_data)
        
        if filtered_options.empty:
            if self.logger:
                self.logger.debug("[ThetaDecayStrategy] No suitable options found")
            return signals
        
        # Find the best candidates based on our selection criteria
        best_options = self._select_best_candidates(filtered_options)
        
        # Generate signals for the selected options
        for _, option in best_options.iterrows():
            signal = {
                'symbol': option['OptionSymbol'] if 'OptionSymbol' in option else option['UnderlyingSymbol'],
                'action': 'sell' if self.is_short else 'buy',
                'option_type': option['Type'],
                'strike': option['Strike'],
                'expiry': option['Expiration'],
                'quantity': 1,  # Base quantity, will be adjusted by position sizing logic
                'price': option['Ask'] if self.is_short else option['Bid'],  # Sell at ask, buy at bid
                'reason': f"Theta decay setup: delta {option['Delta']:.3f}, DTE {option['DaysToExpiry']}",
                'type': 'option',  # Specify that this is an option position
                'instrument_data': option.to_dict()  # Include all option data
            }
            signals.append(signal)
            
            # Update last trade date
            self.last_trade_date = current_date
            
            if self.logger:
                self.logger.info(f"[ThetaDecayStrategy] Generated {signal['action']} signal for {signal['symbol']}")
        
        return signals
    
    def check_exit_conditions(self, position, market_data):
        """
        Check if exit conditions are met for the given position.
        
        Args:
            position: Position to evaluate
            market_data: Dictionary of current market data
            
        Returns:
            Tuple of (should_exit, reason)
        """
        # Get position details
        symbol = position.symbol if hasattr(position, 'symbol') else 'Unknown'
        
        # Get the entry and current prices for P&L calculation
        entry_price = position.avg_entry_price if hasattr(position, 'avg_entry_price') else 0
        current_price = position.current_price if hasattr(position, 'current_price') else 0
        
        # Skip if we don't have valid prices
        if entry_price <= 0 or current_price <= 0:
            self.logger.warning(f"Invalid prices for {symbol}: entry_price={entry_price}, current_price={current_price}")
            return False, "Invalid prices"
        
        # Use the is_short flag directly from the position
        is_short = position.is_short if hasattr(position, 'is_short') else False
        contracts = position.contracts if hasattr(position, 'contracts') else 0
        
        # Calculate raw P&L in dollars
        if hasattr(position, 'contracts'):
            qty = abs(contracts)
            if is_short:
                raw_pnl = (entry_price - current_price) * qty * 100  # Positive when price drops
            else:
                raw_pnl = (current_price - entry_price) * qty * 100  # Positive when price rises
        else:
            raw_pnl = 0
            
        # Calculate profit percentage CORRECTLY for short vs long positions
        if is_short:
            # SHORT position logic: Make money when price goes DOWN
            # Profit % is positive when price decreases, negative when price increases
            profit_pct = (entry_price - current_price) / entry_price
        else:
            # LONG position logic: Make money when price goes UP
            # Profit % is positive when price increases, negative when price decreases
            profit_pct = (current_price - entry_price) / entry_price
            
        # Debug full details for the specific position
        if hasattr(position, 'symbol') and position.symbol == 'SPY240328C00520000':
            self.logger.info(f"==================== DETAILED POSITION DEBUG ====================")
            self.logger.info(f"Position: {symbol}, contracts: {contracts}, is_short: {is_short}")
            self.logger.info(f"Entry: ${entry_price}, Current: ${current_price}")
            self.logger.info(f"Raw P&L: ${raw_pnl}")
            self.logger.info(f"Profit %: {profit_pct:.4f} ({profit_pct:.2%})")
            self.logger.info(f"Profit target: {self.profit_target:.4f} ({self.profit_target:.2%})")
            self.logger.info(f"Stop loss: {-self.stop_loss_threshold:.4f} ({-self.stop_loss_threshold:.2%})")
            
            if is_short and current_price > entry_price:
                self.logger.warning(f"SHORT POSITION LOSING MONEY - price increased from ${entry_price} to ${current_price}")
            elif is_short and current_price < entry_price:
                self.logger.info(f"SHORT POSITION MAKING MONEY - price decreased from ${entry_price} to ${current_price}")
            
            self.logger.info(f"================================================================")
        
        # Get days to expiry if it exists
        days_to_expiry = position.days_to_expiry if hasattr(position, 'days_to_expiry') else None
        
        # --------------------------------------------------------------------
        # Condition 1: Check approach to expiry - close regardless of PNL
        # --------------------------------------------------------------------
        if days_to_expiry is not None and days_to_expiry <= self.close_days_to_expiry:
            self.logger.info(f"[ThetaStrategy] Closing {symbol} due to approaching expiry: {days_to_expiry} days remaining (threshold: {self.close_days_to_expiry})")
            return True, f"Approaching expiry: {days_to_expiry} days remaining (threshold: {self.close_days_to_expiry})"
        
        # --------------------------------------------------------------------
        # Condition 2: Check for profit target reached
        # --------------------------------------------------------------------
        # For a profit target to be hit:
        # - For SHORT positions: price decreased, profit_pct is positive, profit_pct >= profit_target
        # - For LONG positions: price increased, profit_pct is positive, profit_pct >= profit_target
        if profit_pct >= self.profit_target:
            profit_message = f"Profit target reached for {symbol}: {profit_pct:.2%} vs target {self.profit_target:.2%}, Entry: ${entry_price}, Current: ${current_price}, P&L: ${raw_pnl}"
            self.logger.info(f"[ThetaStrategy] {profit_message}")
            return True, profit_message
        
        # --------------------------------------------------------------------
        # Condition 3: Check for stop loss triggered
        # --------------------------------------------------------------------
        # For a stop loss to be hit:
        # - For SHORT positions: price increased, profit_pct is negative, profit_pct <= -stop_loss_threshold
        # - For LONG positions: price decreased, profit_pct is negative, profit_pct <= -stop_loss_threshold
        if profit_pct <= -self.stop_loss_threshold:
            # This is a significant LOSS situation
            stop_loss_message = f"Stop loss triggered for {symbol}: {profit_pct:.2%} vs threshold {-self.stop_loss_threshold:.2%}, Entry: ${entry_price}, Current: ${current_price}, P&L: ${raw_pnl}"
            self.logger.info(f"[ThetaStrategy] {stop_loss_message}")
            return True, stop_loss_message
            
        # Keep holding the position if no exit conditions met
        return False, "Holding position"
    
    def update_metrics(self, portfolio_metrics: Dict[str, Any]) -> None:
        """
        Update strategy-specific metrics within the portfolio metrics.
        
        Args:
            portfolio_metrics: Dictionary of portfolio metrics to update
        """
        portfolio_metrics['strategy_name'] = 'ThetaDecayStrategy'
        portfolio_metrics['delta_target'] = self.delta_target
        portfolio_metrics['avg_days_to_expiry'] = self.days_to_expiry_min + (self.days_to_expiry_max - self.days_to_expiry_min) / 2
    
    def _filter_candidate_options(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter options that meet our strategy criteria.
        
        Args:
            daily_data: DataFrame containing daily market data
        
        Returns:
            Filtered DataFrame of candidate options
        """
        if 'DaysToExpiry' not in daily_data.columns:
            if self.logger:
                self.logger.warning("[ThetaDecayStrategy] Missing 'days_to_expiry' column in data")
            return pd.DataFrame()
        
        filtered = daily_data.copy()
        
        # Basic filtering conditions
        filtered = filtered[
            (filtered['DaysToExpiry'] >= self.days_to_expiry_min) &
            (filtered['DaysToExpiry'] <= self.days_to_expiry_max) &
            (filtered['Volume'] > 0)  # Ensure there is some liquidity
        ]
        
        # Delta-based filtering
        if self.is_short:
            # For short strategies, we want options with delta close to our target
            if 'Delta' in filtered.columns:
                # For puts (negative delta), we want to be close to our target
                put_options = filtered[filtered['Type'] == 'put']
                put_options = put_options[
                    (put_options['Delta'] >= self.delta_target - self.delta_tolerance) &
                    (put_options['Delta'] <= self.delta_target + self.delta_tolerance)
                ]
                
                # For calls (positive delta), we convert our target to positive
                positive_delta_target = -self.delta_target
                call_options = filtered[filtered['Type'] == 'call']
                call_options = call_options[
                    (call_options['Delta'] >= positive_delta_target - self.delta_tolerance) &
                    (call_options['Delta'] <= positive_delta_target + self.delta_tolerance)
                ]
                
                # Combine the filtered puts and calls
                filtered = pd.concat([put_options, call_options])
        
        return filtered
    
    def _select_best_candidates(self, filtered_options: pd.DataFrame) -> pd.DataFrame:
        """
        Select the best option candidates based on our criteria.
        
        Args:
            filtered_options: DataFrame of pre-filtered options
        
        Returns:
            DataFrame of the best options to trade
        """
        if filtered_options.empty:
            return filtered_options
        
        # Sort options by our priority criteria
        if self.is_short:
            # For short strategies, maximize theta/vega ratio
            if 'Theta' in filtered_options.columns and 'Vega' in filtered_options.columns:
                # Avoid division by zero
                filtered_options['Vega'] = filtered_options['Vega'].apply(lambda x: max(0.0001, abs(x)))
                filtered_options['theta_vega_ratio'] = filtered_options['Theta'] / filtered_options['Vega']
                
                # Sort by the best theta/vega ratio (highest absolute ratio)
                filtered_options = filtered_options.sort_values(
                    by='theta_vega_ratio',
                    ascending=False
                )
        else:
            # For long strategies, we might have different priorities
            # For example, maximizing gamma/theta ratio for long options
            pass
        
        # Take top N candidates (here just top 1 for simplicity)
        return filtered_options.head(1) 