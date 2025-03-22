"""
Put Selling Strategy Module

This module implements a put selling strategy that focuses on selling out-of-the-money 
put options to capture premium decay over time.
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


class PutSellStrat(Strategy):
    """
    A strategy focused on selling out-of-the-money put options to capture premium decay.
    
    This strategy:
    1. Sells out-of-the-money put options with specific delta targets
    2. Implements profit taking and stop loss rules for risk management
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the put selling strategy.
        
        Args:
            config: Strategy configuration parameters
            logger: Logger instance for recording events
        """
        super().__init__("PutSellStrat", config, logger)
        
        # Strategy parameters (with defaults)
        self.days_to_expiry_min = config.get('days_to_expiry_min', 30)
        self.days_to_expiry_max = config.get('days_to_expiry_max', 45)
        self.delta_target = config.get('delta_target', -0.20)
        self.delta_tolerance = config.get('delta_tolerance', 0.02)
        self.profit_target = config.get('profit_target', 0.65)
        self.stop_loss_threshold = config.get('stop_loss_threshold', 2.0)
        self.close_days_to_expiry = config.get('close_days_to_expiry', 14)
        self.is_short = True  # Always short for put selling strategy
        
        # Strategy state variables
        self.last_trade_date = None
        
        if self.logger:
            self.logger.info(f"[PutSellStrat] Initialized with delta target: {self.delta_target}")
    
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
        
        # Filter options meeting our criteria
        filtered_options = self._filter_candidate_options(daily_data)
        
        if filtered_options.empty:
            if self.logger:
                self.logger.debug("[PutSellStrat] No suitable put options found")
            return signals
        
        # Find the best candidates based on our selection criteria
        best_options = self._select_best_candidates(filtered_options)
        
        # Generate signals for the selected options
        for _, option in best_options.iterrows():
            signal = {
                'symbol': option['OptionSymbol'] if 'OptionSymbol' in option else option['Symbol'],
                'action': 'sell',  # Always sell for this strategy
                'option_type': 'put',  # Always put options
                'strike': option['Strike'],
                'expiry': option['Expiration'],
                'quantity': 1,  # Base quantity, will be adjusted by position sizing logic
                'price': option['Ask'],  # Sell at ask price
                'reason': f"Put selling setup: delta {option['Delta']:.3f}, DTE {option['DaysToExpiry']}",
                'type': 'option',  # Specify that this is an option position
                'instrument_data': option.to_dict()  # Include all option data
            }
            signals.append(signal)
            
            # Log the signal
            if self.logger:
                self.logger.info(f"[PutSellStrat] Generated sell signal for {signal['symbol']}")
                self.logger.info(f"  Strike: {signal['strike']}, DTE: {option['DaysToExpiry']}")
                self.logger.info(f"  Delta: {option['Delta']:.3f}, Price: ${option['Ask']:.2f}")
        
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
        
        # For short put positions, profit is when price decreases
        profit_pct = (entry_price - current_price) / entry_price
            
        # Get days to expiry if it exists
        days_to_expiry = position.days_to_expiry if hasattr(position, 'days_to_expiry') else None
        
        # Condition 1: Check approach to expiry - close regardless of PNL
        if days_to_expiry is not None and days_to_expiry <= self.close_days_to_expiry:
            self.logger.info(f"[PutSellStrat] Closing {symbol} due to approaching expiry: {days_to_expiry} days remaining (threshold: {self.close_days_to_expiry})")
            return True, f"Approaching expiry: {days_to_expiry} days remaining (threshold: {self.close_days_to_expiry})"
        
        # Condition 2: Check for profit target reached
        if profit_pct >= self.profit_target:
            profit_message = f"Profit target reached for {symbol}: {profit_pct:.2%} vs target {self.profit_target:.2%}, Entry: ${entry_price}, Current: ${current_price}"
            self.logger.info(f"[PutSellStrat] {profit_message}")
            return True, profit_message
        
        # Condition 3: Check for stop loss triggered
        if profit_pct <= -self.stop_loss_threshold:
            stop_loss_message = f"Stop loss triggered for {symbol}: {profit_pct:.2%} vs threshold {-self.stop_loss_threshold:.2%}, Entry: ${entry_price}, Current: ${current_price}"
            self.logger.info(f"[PutSellStrat] {stop_loss_message}")
            return True, stop_loss_message
            
        # Keep holding the position if no exit conditions met
        return False, "Holding position"
    
    def update_metrics(self, portfolio_metrics: Dict[str, Any]) -> None:
        """
        Update strategy-specific metrics within the portfolio metrics.
        
        Args:
            portfolio_metrics: Dictionary of portfolio metrics to update
        """
        portfolio_metrics['strategy_name'] = 'PutSellStrat'
        portfolio_metrics['delta_target'] = self.delta_target
        portfolio_metrics['avg_days_to_expiry'] = self.days_to_expiry_min + (self.days_to_expiry_max - self.days_to_expiry_min) / 2
    
    def _filter_candidate_options(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter put options that meet our strategy criteria.
        
        Args:
            daily_data: DataFrame containing daily market data
        
        Returns:
            Filtered DataFrame of candidate put options
        """
        if 'DaysToExpiry' not in daily_data.columns:
            if self.logger:
                self.logger.warning("[PutSellStrat] Missing 'days_to_expiry' column in data")
            return pd.DataFrame()
        
        filtered = daily_data.copy()
        
        # Only keep put options
        filtered = filtered[filtered['Type'] == 'put']
        
        # Apply DTE filter
        filtered = filtered[
            (filtered['DaysToExpiry'] >= self.days_to_expiry_min) &
            (filtered['DaysToExpiry'] <= self.days_to_expiry_max) &
            (filtered['Volume'] > 0)  # Ensure there is some liquidity
        ]
        
        # Apply delta filter
        if 'Delta' in filtered.columns:
            # For puts (negative delta), we want to be close to our target
            filtered = filtered[
                (filtered['Delta'] >= self.delta_target - self.delta_tolerance) &
                (filtered['Delta'] <= self.delta_target + self.delta_tolerance)
            ]
        
        return filtered
    
    def _select_best_candidates(self, filtered_options: pd.DataFrame) -> pd.DataFrame:
        """
        Select the best put option candidates based on our criteria.
        
        Args:
            filtered_options: DataFrame of pre-filtered options
        
        Returns:
            DataFrame of the best options to trade
        """
        if filtered_options.empty:
            return filtered_options
        
        # Sort options by our priority criteria
        if 'Theta' in filtered_options.columns and 'Vega' in filtered_options.columns:
            # Avoid division by zero
            filtered_options['Vega'] = filtered_options['Vega'].apply(lambda x: max(0.0001, abs(x)))
            filtered_options['theta_vega_ratio'] = filtered_options['Theta'] / filtered_options['Vega']
            
            # Sort by the best theta/vega ratio (highest absolute ratio)
            filtered_options = filtered_options.sort_values(
                by='theta_vega_ratio',
                ascending=False
            )
        
        # Take top 1 candidate
        return filtered_options.head(1) 