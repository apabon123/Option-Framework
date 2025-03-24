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
from simple_logging import SimpleLoggingManager


def main():
    """Run the Intraday Momentum Strategy with detailed logging."""
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
        "--input-file",
        help="Override the input data file path"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging"
    )
    parser.add_argument(
        "--margin-log-level",
        choices=["minimal", "standard", "verbose", "debug"],
        default="standard",
        help="Set the verbosity level for margin calculations"
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

    # Override with input file if provided
    if args.input_file:
        if os.path.exists(args.input_file):
            print(f"Using input file: {args.input_file}")
            if 'intraday' not in config:
                config['intraday'] = {}
            config['intraday']['file_path'] = args.input_file
        else:
            print(f"WARNING: Specified input file not found: {args.input_file}")

    # Check if configured input file exists
    if 'intraday' in config and 'file_path' in config['intraday']:
        input_file = config['intraday']['file_path']
        print(f"Intraday data file from config: {input_file}")
        
        if not os.path.exists(input_file):
            print(f"WARNING: Intraday data file does not exist: {input_file}")
            print("Please provide a valid intraday data file")
            return 1
    else:
        print("WARNING: No intraday file_path specified in config['intraday']")
        print("Intraday momentum strategy requires intraday data")
        return 1

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
    if 'strategy' not in config:
        config['strategy'] = {}
    config['strategy']['name'] = "IntradayMomentumStrategy"
    print(f"Strategy set to: IntradayMomentumStrategy")

    # Enable verbose logging
    if 'logging' not in config:
        config['logging'] = {}
    config['logging']['level'] = 'DEBUG' if args.debug else 'INFO'
    config['logging']['file'] = True
    
    # Enhanced component logging
    if 'components' not in config['logging']:
        config['logging']['components'] = {}
    
    # Set margin logging level
    if 'margin' not in config['logging']['components']:
        config['logging']['components']['margin'] = {}
    config['logging']['components']['margin']['level'] = args.margin_log_level
    
    # Configure logging
    print("Setting up enhanced logging...")
    logging_manager = SimpleLoggingManager()
    logger = logging_manager.setup_logging(
        config_dict=config,
        verbose_console=True,
        debug_mode=args.debug,
        clean_format=False
    )
    
    log_file_path = logging_manager.get_log_file_path()
    if log_file_path:
        print(f"Logging to file: {log_file_path}")
    
    # Print date range for clarity
    start_date = config.get('dates', {}).get('start_date', 'Not specified')
    end_date = config.get('dates', {}).get('end_date', 'Not specified')
    print(f"Backtest date range: {start_date} to {end_date}")
    
    # Initialize the trading engine with the configuration
    print("Initializing trading engine...")
    try:
        engine = TradingEngine(config, logger)
        
        # Run the backtest
        print("Starting backtest...")
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