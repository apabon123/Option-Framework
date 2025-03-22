"""
Call and Put Selling Strategy Implementation

This strategy sells both call and put options based on delta targets
to create a short strangle position.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from core.trading_engine import Strategy


class CallPutStrat(Strategy):
    """
    A strategy that sells both call and put options to create a short strangle position
    and manages positions with predetermined exit criteria.
    """

    def __init__(self, config, logger=None):
        """Initialize the strategy with configuration."""
        super().__init__("CallPutStrat", config, logger)

        # Extract strategy parameters
        self.days_to_expiry_min = config.get('days_to_expiry_min', 30)
        self.days_to_expiry_max = config.get('days_to_expiry_max', 45)
        self.put_delta_target = config.get('delta_target', -0.20)  # For puts
        self.call_delta_target = -self.put_delta_target  # For calls (positive value)
        self.delta_tolerance = config.get('delta_tolerance', 0.05)
        self.profit_target = config.get('profit_target', 0.50)
        self.stop_loss_threshold = config.get('stop_loss_threshold', 2.0)
        self.close_days_to_expiry = config.get('close_days_to_expiry', 14)
        self.is_short = True  # Always short for this strategy

        self.logger.info(f"[CallPutStrat] Initialized")
        self.logger.info(f"  DTE Range: {self.days_to_expiry_min} to {self.days_to_expiry_max} days")
        self.logger.info(f"  Put Delta Target: {self.put_delta_target:.2f}±{self.delta_tolerance:.2f}")
        self.logger.info(f"  Call Delta Target: {self.call_delta_target:.2f}±{self.delta_tolerance:.2f}")
        self.logger.info(f"  Profit Target: {self.profit_target:.0%}")
        self.logger.info(f"  Stop Loss: {self.stop_loss_threshold:.1f}x premium")
        self.logger.info(f"  Close DTE: {self.close_days_to_expiry} days")

    def generate_signals(self, current_date, daily_data):
        """
        Generate trading signals for the current date for both calls and puts.

        Args:
            current_date: Current simulation date
            daily_data: Data for the current date

        Returns:
            list: List of signal dictionaries
        """
        signals = []

        # Generate put signals
        put_signals = self._generate_option_signals(current_date, daily_data, 'put', self.put_delta_target)
        signals.extend(put_signals)
        
        # Generate call signals
        call_signals = self._generate_option_signals(current_date, daily_data, 'call', self.call_delta_target)
        signals.extend(call_signals)
        
        return signals

    def _generate_option_signals(self, current_date, daily_data, option_type, delta_target):
        """
        Generate signals for a specific option type (call or put).
        
        Args:
            current_date: Current date for the backtest
            daily_data: Market data for the current date
            option_type: 'call' or 'put'
            delta_target: Target delta value for this option type
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        # Filter for appropriate options
        candidates = daily_data[
            (daily_data['Type'].str.lower() == option_type) &
            (daily_data['DaysToExpiry'] >= self.days_to_expiry_min) &
            (daily_data['DaysToExpiry'] <= self.days_to_expiry_max) &
            (daily_data['Volume'] > 0)  # Ensure there is some liquidity
        ]

        # Exit if no candidates
        if candidates.empty:
            self.logger.debug(f"[CallPutStrat] No suitable {option_type} candidates found for {current_date}")
            return signals

        # Calculate delta distance and find best match
        candidates = candidates.copy()  # Create a copy to avoid the warning
        candidates.loc[:, 'delta_distance'] = abs(candidates['Delta'] - delta_target)
        candidates = candidates.sort_values('delta_distance')

        # Select top candidate
        best_candidate = candidates.iloc[0]

        # Check if delta is within tolerance
        if best_candidate['delta_distance'] > self.delta_tolerance:
            self.logger.debug(f"[CallPutStrat] Best {option_type} candidate delta too far from target: {best_candidate['Delta']:.2f} vs {delta_target:.2f}")
            return signals

        # Create signal
        symbol = best_candidate['OptionSymbol'] if 'OptionSymbol' in best_candidate else best_candidate['Symbol']
        
        # Default to 1 contract for simplicity
        quantity = 1

        self.logger.info(f"[CallPutStrat] Generated SELL signal for {symbol} ({option_type})")
        self.logger.info(f"  Strike: {best_candidate['Strike']}, DTE: {best_candidate['DaysToExpiry']}")
        self.logger.info(f"  Delta: {best_candidate['Delta']:.3f}, Price: ${best_candidate['Ask']:.2f}")

        signals.append({
            'action': 'sell',
            'symbol': symbol,
            'option_type': option_type,
            'strike': best_candidate['Strike'],
            'expiry': best_candidate['Expiration'] if 'Expiration' in best_candidate else None,
            'quantity': quantity,
            'price': best_candidate['Ask'],
            'type': 'option',
            'reason': f"{option_type.capitalize()} selling: delta {best_candidate['Delta']:.3f}, DTE {best_candidate['DaysToExpiry']}",
            'instrument_data': best_candidate.to_dict()
        })

        return signals

    def check_exit_conditions(self, position, market_data):
        """
        Check if a position should be exited based on predefined criteria.

        Args:
            position: Position to check
            market_data: Current market data

        Returns:
            tuple: (exit_flag, reason) - Whether to exit and why
        """
        # Safety check - no contracts means no exit conditions
        if not hasattr(position, 'contracts') or position.contracts <= 0:
            return False, None

        # Make sure we have days_to_expiry attribute
        if not hasattr(position, 'days_to_expiry'):
            return False, None
            
        # Get symbol for more detailed reasons
        symbol = position.symbol if hasattr(position, 'symbol') else "Unknown"
        option_type = 'call' if 'C' in symbol else 'put' if 'P' in symbol else 'unknown'

        # Time-based exit - close by DTE
        if position.days_to_expiry <= self.close_days_to_expiry:
            detailed_reason = f"Close {option_type} due to DTE for {symbol}: {position.days_to_expiry} days remaining vs threshold {self.close_days_to_expiry} days"
            self.logger.info(f"[CallPutStrat] {detailed_reason}")
            return True, detailed_reason

        # Safety check for entry and current prices
        if not hasattr(position, 'avg_entry_price') or not hasattr(position, 'current_price'):
            return False, None

        # Calculate profit percentage (for short positions)
        if position.avg_entry_price <= 0:
            profit_pct = 0
        else:
            # For short options: entry_price > current_price is profit
            profit_pct = (position.avg_entry_price - position.current_price) / position.avg_entry_price

        # Profit target exit
        if profit_pct >= self.profit_target:
            detailed_reason = f"Profit target reached for {symbol} ({option_type}): {profit_pct:.2%} vs target {self.profit_target:.2%}, Entry: ${position.avg_entry_price:.2f}, Current: ${position.current_price:.2f}"
            self.logger.info(f"[CallPutStrat] {detailed_reason}")
            return True, detailed_reason

        # Stop loss exit
        if profit_pct <= -self.stop_loss_threshold:
            detailed_reason = f"Stop loss triggered for {symbol} ({option_type}): {profit_pct:.2%} vs threshold -{self.stop_loss_threshold:.2%}, Entry: ${position.avg_entry_price:.2f}, Current: ${position.current_price:.2f}"
            self.logger.info(f"[CallPutStrat] {detailed_reason}")
            return True, detailed_reason

        # No exit condition met
        return False, None
        
    def update_metrics(self, portfolio_metrics: Dict[str, Any]) -> None:
        """
        Update strategy-specific metrics within the portfolio metrics.
        
        Args:
            portfolio_metrics: Dictionary of portfolio metrics to update
        """
        portfolio_metrics['strategy_name'] = 'CallPutStrat'
        portfolio_metrics['put_delta_target'] = self.put_delta_target
        portfolio_metrics['call_delta_target'] = self.call_delta_target
        portfolio_metrics['avg_days_to_expiry'] = self.days_to_expiry_min + (self.days_to_expiry_max - self.days_to_expiry_min) / 2 