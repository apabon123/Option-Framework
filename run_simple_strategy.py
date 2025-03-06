#!/usr/bin/env python
"""
Convenience script for running the SimpleOptionStrategy

This script provides a simple way to run the SimpleOptionStrategy
with predefined configuration.
"""

import sys
import os
import yaml
import logging
from datetime import datetime

# Ensure the parent directory is in the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trading_engine import TradingEngine, LoggingManager
from strategies.example_strategy import SimpleOptionStrategy


def run_simple_strategy():
    """Run the SimpleOptionStrategy backtest."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Setup logging
    logging_manager = LoggingManager()
    logger = logging_manager.setup_logging(config, verbose_console=False)

    # Print initial info
    strategy_config = config.get('strategy', {})
    print(f"Strategy: {strategy_config.get('name', 'SimpleOptionStrategy')} | "
          f"Hedge: {strategy_config.get('hedge_symbol', 'SPY')} | "
          f"Target Delta: {strategy_config.get('delta_target', -0.2)}")

    # Create strategy instance
    strategy = SimpleOptionStrategy(strategy_config, logger)

    # Create trading engine
    engine = TradingEngine(config, strategy, logger)

    # Run backtest
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

        return 0
    else:
        print("Failed to load data for backtesting")
        return 1


if __name__ == "__main__":
    sys.exit(run_simple_strategy())