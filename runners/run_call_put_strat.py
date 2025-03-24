#!/usr/bin/env python
"""
Script to run the Call-Put Strategy with enhanced logging.

This script provides a convenient way to run the Call-Put Strategy
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
from strategies.call_put_strat import CallPutStrat
from simple_logging import SimpleLoggingManager


def main():
    """Run the Call-Put Strategy with detailed logging."""
    print("\n=== Starting Call-Put Strategy with Enhanced Logging ===")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Call-Put Strategy with detailed logging")
    parser.add_argument(
        "-c", "--config",
        default="../config/strategy/call_put_config.yaml",
        help="Path to configuration YAML file (default: ../config/strategy/call_put_config.yaml)"
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
        "--use-test-data",
        action="store_true",
        help="Use test data from ../tests/data/test_data.csv"
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

    # Check for test data flag
    if args.use_test_data:
        test_data_path = "../tests/data/test_data.csv"
        if os.path.exists(test_data_path):
            print(f"Using test data from: {test_data_path}")
            if 'paths' not in config:
                config['paths'] = {}
            config['paths']['input_file'] = test_data_path
        else:
            print(f"WARNING: Test data file not found: {test_data_path}")

    # Print important configuration values for debugging
    if 'paths' in config and 'input_file' in config['paths']:
        print(f"Input file from config: {config['paths']['input_file']}")
        
        # Check if input file exists
        input_file = config['paths']['input_file']
        if not os.path.exists(input_file):
            print(f"WARNING: Input file does not exist: {input_file}")
            print("This will cause 'No trading dates available' error")
            
            # Try to use the test data file as a fallback
            test_data_path = "../tests/data/test_data.csv"
            if os.path.exists(test_data_path):
                config['paths']['input_file'] = test_data_path
                print(f"Using test data as fallback: {test_data_path}")
            elif args.input_file:
                # Override input file path if provided
                config['paths']['input_file'] = args.input_file
                print(f"Overriding input file path to: {args.input_file}")
                
                # Check if override file exists
                if not os.path.exists(args.input_file):
                    print(f"WARNING: Override input file does not exist: {args.input_file}")
            else:
                print("ERROR: No valid input data file available")
                print("Please provide a valid input file with the --input-file option")
                return 1
    else:
        print("WARNING: No input_file specified in config['paths']")
        print("Setting up a default input file path")
        
        # Create paths section if it doesn't exist
        if 'paths' not in config:
            config['paths'] = {}
            
        # Try to use the test data as a default
        test_data_path = "../tests/data/test_data.csv"
        if os.path.exists(test_data_path):
            config['paths']['input_file'] = test_data_path
            print(f"Using test data as default: {test_data_path}")
        else:
            print("ERROR: No valid input data file available")
            print("Please provide a valid input file with the --input-file option")
            return 1

    # Override configuration with command-line arguments
    if args.start_date:
        if 'dates' not in config:
            config['dates'] = {}
        config['dates']['start_date'] = args.start_date
        print(f"Start date override: {args.start_date}")
    else:
        # Ensure we have a default start date that matches the test data
        if 'dates' not in config:
            config['dates'] = {}
        if 'start_date' not in config['dates']:
            config['dates']['start_date'] = "2024-01-02"  # First date in test data
            print(f"Using default start date: {config['dates']['start_date']}")

    if args.end_date:
        if 'dates' not in config:
            config['dates'] = {}
        config['dates']['end_date'] = args.end_date
        print(f"End date override: {args.end_date}")
    else:
        # Ensure we have a default end date that matches the test data
        if 'dates' not in config:
            config['dates'] = {}
        if 'end_date' not in config['dates']:
            config['dates']['end_date'] = "2024-01-05"  # Last date in test data could be later
            print(f"Using default end date: {config['dates']['end_date']}")

    # Configure strategy
    config['strategy']['name'] = "CallPutStrat"
    print(f"Strategy set to: CallPutStrat")

    # Enable verbose logging
    if 'logging' not in config:
        config['logging'] = {}
    config['logging']['level'] = 'DEBUG'
    config['logging']['file'] = True
    
    # Enhanced component logging
    if 'components' not in config['logging']:
        config['logging']['components'] = {}
    
    # Set margin logging to verbose
    if 'margin' not in config['logging']['components']:
        config['logging']['components']['margin'] = {}
    config['logging']['components']['margin']['level'] = 'verbose'
    
    # Set portfolio logging to verbose
    if 'portfolio' not in config['logging']['components']:
        config['logging']['components']['portfolio'] = {}
    config['logging']['components']['portfolio']['level'] = 'verbose'

    # Configure logging
    print("Setting up enhanced logging...")
    logging_manager = SimpleLoggingManager()
    logger = logging_manager.setup_logging(
        config_dict=config,
        verbose_console=True,  # Enable verbose console output
        debug_mode=True,       # Enable debug mode
        clean_format=False     # Include timestamp and level in logs
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