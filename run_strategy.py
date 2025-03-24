#!/usr/bin/env python
"""
Strategy Runner Script

This script provides a unified way to run any trading strategy with
enhanced logging and proper data handling. It includes automatic
detection and configuration for test data.
"""

import os
import sys
import argparse
import yaml
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

from core.trading_engine import TradingEngine
from simple_logging import SimpleLoggingManager


def main():
    """Run a trading strategy with enhanced logging and proper data handling."""
    print("\n=== Strategy Runner with Enhanced Logging ===")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run trading strategies with enhanced logging")
    parser.add_argument(
        "-c", "--config",
        default="config/config.yaml",
        help="Path to configuration YAML file (default: config/config.yaml)"
    )
    parser.add_argument(
        "-s", "--strategy",
        default=None,
        help="Strategy to run (e.g., PutSellStrat, ThetaDecayStrategy)"
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Override start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Override end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Override the input data file path"
    )
    parser.add_argument(
        "--use-test-data",
        action="store_true",
        help="Use test data from tests/data/test_data.csv"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
        help="Set the logging level"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable extra debug information"
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

    # HANDLE INPUT FILE
    test_data_path = "tests/data/test_data.csv"
    
    # Always use test data if specified
    if args.use_test_data:
        if os.path.exists(test_data_path):
            print(f"Using test data from: {test_data_path}")
            if 'paths' not in config:
                config['paths'] = {}
            config['paths']['input_file'] = test_data_path
        else:
            print(f"WARNING: Test data file not found: {test_data_path}")
    
    # Override with explicit input file if provided
    if args.input_file:
        if os.path.exists(args.input_file):
            print(f"Using input file: {args.input_file}")
            if 'paths' not in config:
                config['paths'] = {}
            config['paths']['input_file'] = args.input_file
        else:
            print(f"WARNING: Specified input file not found: {args.input_file}")
    
    # Check if configured input file exists
    if 'paths' in config and 'input_file' in config['paths']:
        input_file = config['paths']['input_file']
        print(f"Input file from config: {input_file}")
        
        if not os.path.exists(input_file):
            print(f"WARNING: Input file does not exist: {input_file}")
            
            # Fall back to test data if available
            if os.path.exists(test_data_path):
                print(f"Falling back to test data: {test_data_path}")
                config['paths']['input_file'] = test_data_path
            else:
                print("ERROR: No valid input data file available")
                return 1
    else:
        print("WARNING: No input_file specified in config['paths']")
        
        # Fall back to test data if available
        if os.path.exists(test_data_path):
            print(f"Using test data as default: {test_data_path}")
            if 'paths' not in config:
                config['paths'] = {}
            config['paths']['input_file'] = test_data_path
        else:
            print("ERROR: No valid input data file available")
            return 1
    
    # Analyze the data file for date range
    input_file = config['paths']['input_file']
    try:
        if args.debug:
            print(f"\nAnalyzing data file: {input_file}")
        
        df = pd.read_csv(input_file)
        if args.debug:
            print(f"Data shape: {df.shape}")
            print(f"Columns: {', '.join(df.columns)}")
        
        # Check and process date column
        if 'date' in df.columns:
            # Convert to datetime if needed
            if df['date'].dtype != 'datetime64[ns]':
                df['date'] = pd.to_datetime(df['date'])
            
            # Get unique dates
            unique_dates = df['date'].dt.date.unique()
            
            if args.debug:
                print(f"Data contains dates: {min(unique_dates)} to {max(unique_dates)}")
            
            # Override dates if not explicitly specified
            if not args.start_date:
                if 'dates' not in config or 'start_date' not in config['dates']:
                    start_date = min(unique_dates)
                    if 'dates' not in config:
                        config['dates'] = {}
                    config['dates']['start_date'] = str(start_date)
                    print(f"Setting start date to match data: {start_date}")
            
            if not args.end_date:
                if 'dates' not in config or 'end_date' not in config['dates']:
                    end_date = max(unique_dates)
                    if 'dates' not in config:
                        config['dates'] = {}
                    config['dates']['end_date'] = str(end_date)
                    print(f"Setting end date to match data: {end_date}")
    except Exception as e:
        print(f"Warning: Error analyzing data file: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    
    # HANDLE DATE RANGE
    # Override date range with command-line arguments
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
    
    # HANDLE STRATEGY
    # Override strategy from command line
    if args.strategy:
        if 'strategy' not in config:
            config['strategy'] = {}
        config['strategy']['name'] = args.strategy
        print(f"Strategy override: {args.strategy}")
    
    # Display strategy
    strategy_name = config.get('strategy', {}).get('name', 'DefaultStrategy')
    print(f"Running strategy: {strategy_name}")

    # CONFIGURE LOGGING
    # Enable verbose logging
    if 'logging' not in config:
        config['logging'] = {}
    config['logging']['level'] = args.log_level
    config['logging']['file'] = True
    
    # Enhanced component logging
    if 'components' not in config['logging']:
        config['logging']['components'] = {}
    
    # Set margin and portfolio logging to verbose
    for component in ['margin', 'portfolio']:
        if component not in config['logging']['components']:
            config['logging']['components'][component] = {}
        config['logging']['components'][component]['level'] = 'verbose'

    # Set up logging
    print("Setting up enhanced logging...")
    logging_manager = SimpleLoggingManager()
    logger = logging_manager.setup_logging(
        config_dict=config,
        verbose_console=True,
        debug_mode=True,
        clean_format=False
    )
    
    log_file_path = logging_manager.get_log_file_path()
    if log_file_path:
        print(f"Logging to file: {log_file_path}")
    
    # Print date range for clarity
    start_date = config.get('dates', {}).get('start_date', 'Not specified')
    end_date = config.get('dates', {}).get('end_date', 'Not specified')
    print(f"Backtest date range: {start_date} to {end_date}")
    
    # RUN BACKTEST
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
        if args.debug:
            print("\nStack trace:")
            traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 