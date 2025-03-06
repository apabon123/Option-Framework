"""
Example Strategy Implementation

This script demonstrates how to use the core modules to implement
and backtest a simple options trading strategy.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from core.trading_engine import Strategy


class SimpleOptionStrategy(Strategy):
    """
    A simple option strategy that sells put options below support levels
    and manages positions with predetermined exit criteria.
    """

    def __init__(self, config, logger=None):
        """Initialize the strategy with configuration."""
        super().__init__(config, logger)

        # Extract strategy parameters
        self.is_short = config.get('is_short', True)
        self.days_to_expiry_min = config.get('days_to_expiry_min', 30)
        self.days_to_expiry_max = config.get('days_to_expiry_max', 45)
        self.delta_target = config.get('delta_target', -0.20)
        self.delta_tolerance = config.get('delta_tolerance', 0.05)
        self.profit_target = config.get('profit_target', 0.50)
        self.stop_loss = config.get('stop_loss', 2.0)
        self.close_dte = config.get('close_dte', 7)

        self.logger.info(f"[Strategy] Initialized {self.name}")
        self.logger.info(f"  Position Type: {'Short' if self.is_short else 'Long'}")
        self.logger.info(f"  DTE Range: {self.days_to_expiry_min} to {self.days_to_expiry_max} days")
        self.logger.info(f"  Delta Target: {self.delta_target:.2f}±{self.delta_tolerance:.2f}")
        self.logger.info(f"  Profit Target: {self.profit_target:.0%}")
        self.logger.info(f"  Stop Loss: {self.stop_loss:.0f}x premium")
        self.logger.info(f"  Close DTE: {self.close_dte} days")

    def generate_signals(self, current_date, daily_data):
        """
        Generate trading signals for the current date based on delta target.

        Args:
            current_date: Current simulation date
            daily_data: Data for the current date

        Returns:
            list: List of signal dictionaries
        """
        signals = []

        # Determine option type based on delta target sign
        option_type = 'call' if self.delta_target > 0 else 'put'

        # Filter for appropriate options
        candidates = daily_data[
            (daily_data['Type'].str.lower() == option_type) &
            (daily_data['DaysToExpiry'] >= self.days_to_expiry_min) &
            (daily_data['DaysToExpiry'] <= self.days_to_expiry_max) &
            (daily_data['MidPrice'] > 0.10)
        ]

        # Exit if no candidates
        if candidates.empty:
            self.logger.debug(f"[Strategy] No suitable candidates found for {current_date}")
            return signals

        # Calculate delta distance and find best match
        candidates = candidates.copy()  # Create a copy to avoid the warning
        candidates.loc[:, 'delta_distance'] = abs(candidates['Delta'] - self.delta_target)
        candidates = candidates.sort_values('delta_distance')

        # Select top candidate
        best_candidate = candidates.iloc[0]

        # Check if delta is within tolerance
        if best_candidate['delta_distance'] > self.delta_tolerance:
            self.logger.debug(f"[Strategy] Best candidate delta too far from target: {best_candidate['Delta']:.2f} vs {self.delta_target:.2f}")
            return signals

        # Create signal
        action = 'SELL' if self.is_short else 'BUY'
        symbol = best_candidate['OptionSymbol']

        # Default to 1 contract for simplicity
        quantity = 1

        self.logger.info(f"[Strategy] Generated {action} signal for {symbol}")
        self.logger.info(f"  Strike: {best_candidate['Strike']}, DTE: {best_candidate['DaysToExpiry']}")
        self.logger.info(f"  Delta: {best_candidate['Delta']:.3f}, Price: ${best_candidate['MidPrice']:.2f}")

        signals.append({
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'type': 'option',
            'data': best_candidate
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

        # Time-based exit - close by DTE
        if position.days_to_expiry <= self.close_dte:
            self.logger.debug(f"[Strategy] Exit signal: DTE {position.days_to_expiry} ≤ {self.close_dte}")
            return True, f"Close by DTE {self.close_dte}"

        # Safety check for entry and current prices
        if not hasattr(position, 'avg_entry_price') or not hasattr(position, 'current_price'):
            return False, None

        # Calculate profit percentage
        if position.avg_entry_price <= 0:
            # Handle division by zero - no profit if no entry price
            profit_pct = 0
        elif self.is_short:
            # For short options: entry_price > current_price is profit
            profit_pct = (position.avg_entry_price - position.current_price) / position.avg_entry_price
        else:
            # For long options: current_price > entry_price is profit
            profit_pct = (position.current_price - position.avg_entry_price) / position.avg_entry_price

        # Safety checks for profit target and stop loss
        try:
            # Profit target exit
            if profit_pct >= self.profit_target:
                self.logger.debug(f"[Strategy] Exit signal: Profit {profit_pct:.2%} ≥ Target {self.profit_target:.2%}")
                return True, 'Profit Target'

            # Stop loss exit - for short options, this is when loss exceeds stop_loss times premium
            if profit_pct <= -self.stop_loss:
                self.logger.debug(f"[Strategy] Exit signal: Loss {profit_pct:.2%} ≤ Stop Loss -{self.stop_loss:.2%}")
                return True, 'Stop Loss'
        except Exception as e:
            # Log the error but don't exit the position due to calculation errors
            if hasattr(self, 'logger'):
                self.logger.error(f"Error checking profit/loss exit conditions: {e}")

        # No exit condition met
        return False, None


# For testing purposes - can be called directly to test the strategy
if __name__ == "__main__":
    print("This script should not be run directly.")
    print("To backtest the SimpleOptionStrategy, use main.py with the appropriate config:")
    print("python main.py -c config/config.yaml")