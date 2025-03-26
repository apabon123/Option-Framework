#!/usr/bin/env python
"""
Script to run the Intraday Momentum Strategy with enhanced logging.

This script provides a convenient way to run the Intraday Momentum Strategy
with proper logging configuration to see detailed output.
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime

# Add parent directory to path so imports work correctly from the runners folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trading_engine import TradingEngine
from strategies.intraday_momentum import IntradayMomentumStrategy
from utils.simple_logging import SimpleLoggingManager


def main():
    """Run the Intraday Momentum Strategy with enhanced logging."""
    print("\n=== Starting Intraday Momentum Strategy with Enhanced Logging ===")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Intraday Momentum Strategy with detailed logging")
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/strategy/intraday_momentum_config.yaml"),
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--start-date",
        help="Override start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        help="Override end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode logging"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    print(f"Loading configuration from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(f"Loaded configuration successfully")

    # Override configuration with command-line arguments
    if args.start_date:
        if 'dates' not in config:
            config['dates'] = {}
        config['dates']['start_date'] = args.start_date
        print(f"Start date override: {args.start_date}")

    if args.end_date:
        if 'dates' not in config:
            config['dates'] = {}
        config['dates']['end_date'] = args.end_date
        print(f"End date override: {args.end_date}")

    # Configure strategy
    config['strategy']['name'] = "IntradayMomentumStrategy"
    print(f"Strategy set to: IntradayMomentumStrategy")

    # Setup logging configuration if not specified
    if 'logging' not in config:
        config['logging'] = {}
    
    # Only set logging level if not specified in the config
    if 'level' not in config['logging']:
        config['logging']['level'] = 'INFO'
    
    # Ensure file logging is enabled
    config['logging']['log_to_file'] = True

    # Make sure component_levels exists
    if 'component_levels' not in config['logging']:
        config['logging']['component_levels'] = {}

    # Only set component levels if not already specified
    if 'margin' not in config['logging']['component_levels']:
        config['logging']['component_levels']['margin'] = 'INFO'
    
    if 'portfolio' not in config['logging']['component_levels']:
        config['logging']['component_levels']['portfolio'] = 'INFO'
        
    if 'trading' not in config['logging']['component_levels']:
        config['logging']['component_levels']['trading'] = 'INFO'

    # Configure logging
    print("Setting up logging...")
    print(f"Logging level from config: {config['logging']['level']}")
    print(f"Component levels: {config['logging']['component_levels']}")
    
    logging_manager = SimpleLoggingManager()
    logger = logging_manager.setup_logging(
        config_dict=config,
        verbose_console=True,  # Enable verbose console output
        debug_mode=args.debug, # Only enable debug mode if --debug flag is provided
        clean_format=False     # Include timestamp and level in logs
    )
    
    log_file_path = logging_manager.get_log_file_path()
    if log_file_path:
        print(f"Logging to file: {log_file_path}")
    
    # Print date range for clarity
    start_date = config.get('dates', {}).get('start_date', 'Not specified')
    end_date = config.get('dates', {}).get('end_date', 'Not specified')
    print(f"Backtest date range: {start_date} to {end_date}")
    
    # Initialize the trading engine and run the backtest
    print("Initializing trading engine...")
    try:
        # Create the IntradayMomentumStrategy instance
        strategy = IntradayMomentumStrategy(config.get('strategy', {}), logger)
        
        # Create the TradingEngine with the strategy instance
        engine = TradingEngine(config, strategy, logger)
        
        # Run the backtest
        print("\nStarting backtest...")
        engine.run_backtest()
        
        # Print summary
        print("\n=== Backtest Complete ===")
        print(f"Check the log file for detailed output: {log_file_path}")
        
    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"Error during backtest: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 