#!/usr/bin/env python
"""
Main entry point for the trading system.

This module provides a command-line interface for running backtests
with different strategies and configuration options.
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime
import importlib

from core.trading_engine import TradingEngine, Strategy, LoggingManager


def load_yaml_config(config_path):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_strategy_class(strategy_name):
    """
    Dynamically import and return the strategy class by name.

    Args:
        strategy_name: Name of the strategy class to import

    Returns:
        Strategy class
    """
    try:
        # First try to import from strategies package
        module_name = f"strategies.{strategy_name.lower()}"
        module = importlib.import_module(module_name)
        return getattr(module, strategy_name)
    except (ImportError, AttributeError):
        try:
            # Fall back to direct import if it's a fully qualified name
            module_name, class_name = strategy_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ValueError, ImportError, AttributeError):
            raise ImportError(f"Could not import strategy class: {strategy_name}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run trading strategy backtest")

    parser.add_argument(
        "-c", "--config",
        default="config/config.yaml",
        help="Path to configuration YAML file (default: config/config.yaml)"
    )

    parser.add_argument(
        "-s", "--strategy",
        help="Override strategy name from config"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode with extra logging"
    )

    parser.add_argument(
        "--start-date",
        help="Override start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date",
        help="Override end date (YYYY-MM-DD)"
    )

    return parser.parse_args()


def main():
    """Main function to run the trading system."""
    args = parse_args()

    try:
        # Load configuration
        config = load_yaml_config(args.config)

        # Override configuration with command-line arguments
        if args.strategy:
            if 'strategy' not in config:
                config['strategy'] = {}
            config['strategy']['name'] = args.strategy

        if args.start_date:
            if 'dates' not in config:
                config['dates'] = {}
            config['dates']['start_date'] = args.start_date

        if args.end_date:
            if 'dates' not in config:
                config['dates'] = {}
            config['dates']['end_date'] = args.end_date

        # Configure logging
        logging_manager = LoggingManager()
        logger = logging_manager.setup_logging(
            config,
            verbose_console=args.verbose,
            debug_mode=args.debug,
            clean_format=not args.debug
        )

        # Get strategy class and create instance
        strategy_name = config.get('strategy', {}).get('name')
        if not strategy_name:
            logger.error("No strategy name specified in configuration")
            return 1

        logger.info(f"[INIT] Using strategy: {strategy_name}")

        try:
            strategy_class = get_strategy_class(strategy_name)
            strategy = strategy_class(config.get('strategy', {}), logger)
        except ImportError as e:
            logger.error(f"Error loading strategy: {e}")
            return 1

        # Create trading engine
        engine = TradingEngine(config, strategy, logger)

        # Load data
        logger.info("[INIT] Loading market data...")
        if not engine.load_data():
            logger.error("Failed to load data")
            return 1

        # Run backtest
        results = engine.run_backtest()

        # Print summary
        logger.info("\n=== Backtest Results ===")
        logger.info(f"Initial Capital: ${results.get('initial_value', 0):,.2f}")
        logger.info(f"Final Value: ${results.get('final_value', 0):,.2f}")
        logger.info(f"Total Return: {results.get('total_return', 0):.2%}")
        logger.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")

        # Print report path
        if 'report_path' in results:
            logger.info(f"Detailed report saved to: {results['report_path']}")

        return 0

    except Exception as e:
        logging.error(f"Error running backtest: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())