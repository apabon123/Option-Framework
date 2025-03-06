"""
Example Strategy Implementation

This script demonstrates how to use the core modules to implement
and backtest a simple options trading strategy.
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml  # Explicitly import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

# Import from the core package properly
from core.trading_engine import Strategy, TradingEngine, load_configuration, main
from core.data_manager import DataManager
from core.position import Position, OptionPosition
from core.reporting import ReportingSystem


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


if __name__ == "__main__":
    # Create a custom configuration for our strategy
    config = {
        # File paths - adjust these to your actual file paths
        'paths': {
            'input_file': r"C:\Users\alexp\OneDrive\Gdrive\Trading\Data Outputs\SPY_Combined.csv",  # Change to your actual data file
            'output_dir': r"C:\Users\alexp\OneDrive\Gdrive\Trading\Data Outputs\scenario_results",
        },
        
        # Date range for backtesting
        'dates': {
            'start_date': "2024-01-01",
            'end_date': "2024-12-31"
        },
        
        # Portfolio settings
        'portfolio': {
            'initial_capital': 100000,
            'max_position_size_pct': 0.5,  # Max % per position
            'max_portfolio_delta': 0.25,  # Max % portfolio delta
        },
        
        # Risk management settings
        'risk': {
            'max_leverage': .14285,  # 7x leverage
        },
        
        # Strategy parameters
        'strategy': {
            'name': 'SimpleOptionStrategy',
            'instrument_type': 'option',
            'is_short': True,
            'days_to_expiry_min': 45,
            'days_to_expiry_max': 75,
            'delta_target': -0.20,
            'delta_tolerance': 0.05,
            'profit_target': 0.65,  # 65% of premium
            'stop_loss': 2.5,  # 2.5x premium
            'close_dte': 14,
            # Delta hedging parameters
            'enable_hedging': True,
            'hedge_mode': 'ratio',
            'target_delta_ratio': 0.0,  # Target delta as % of portfolio value (0.0 means market neutral)
            'hedge_tolerance': 0.5,  # Hedge when outside this tolerance
            'hedge_symbol': 'SPY'
        },
        
        # Data settings
        'data': {
            'max_spread': 0.30,  # 30% max spread
        },
        
        # Backtest settings
        'backtest': {
            'verbose': True,
        }
    }
    
    # Run the backtest with our custom strategy
    # We need to create the strategy instance first
    strategy = SimpleOptionStrategy(config['strategy'])
    
    # Add a minimal message about the strategy configuration
    print(f"Strategy: {config['strategy']['name']} | Hedge: {config['strategy']['hedge_symbol']} | Target Delta: {config['strategy']['delta_target']}")
    
    # Then create the engine with our strategy
    engine = TradingEngine(config, strategy)
    
    # Initialize dates in case they weren't set in init
    from datetime import datetime
    import pandas as pd
    
    # Explicitly set start_date and end_date on the engine
    if hasattr(engine, 'config') and 'dates' in engine.config:
        dates_config = engine.config.get('dates', {})
        if 'start_date' in dates_config:
            start_date = dates_config['start_date']
            if isinstance(start_date, str):
                engine.start_date = pd.to_datetime(start_date)
            else:
                engine.start_date = start_date
                
        if 'end_date' in dates_config:
            end_date = dates_config['end_date']
            if isinstance(end_date, str):
                engine.end_date = pd.to_datetime(end_date)
            else:
                engine.end_date = end_date
    
    # Load data and run the backtest
    if engine.load_data():
        results = engine.run_backtest()
        
        # Print summary
        print("\n=== Backtest Results ===")
        print(f"Initial Capital: ${results.get('initial_capital', 0):,.2f}")
        print(f"Final Value: ${results.get('final_value', 0):,.2f}")
        print(f"Total Return: {results.get('total_return', 0):.2%}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        
        # Print report path
        if 'report_path' in results:
            print(f"Detailed report saved to: {results['report_path']}")
    else:
        print("Failed to load data for backtesting")