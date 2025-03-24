#!/usr/bin/env python
"""
Enhanced debug script to diagnose issues with loading date ranges for the PutSellStrat.
This script adds detailed debugging output to help diagnose the "No trading dates available" error.
"""

import os
import sys
import argparse
import yaml
import logging
import pandas as pd
from datetime import datetime

from core.trading_engine import TradingEngine
from strategies.put_sell_strat import PutSellStrat
from simple_logging import SimpleLoggingManager


def main():
    """Run a detailed debugging setup for the trading engine."""
    print("\n=== Starting Enhanced Debug Script ===")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Debug the trading engine setup")
    parser.add_argument(
        "-c", "--config",
        default="config/config.yaml",
        help="Path to configuration YAML file (default: config/config.yaml)"
    )
    parser.add_argument(
        "--use-test-data",
        action="store_true",
        help="Use test data from tests/data/test_data.csv"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    print(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(f"Loaded configuration successfully")

    # Always use test data for debugging
    test_data_path = "tests/data/test_data.csv"
    print(f"Using test data from: {test_data_path}")
    
    if 'paths' not in config:
        config['paths'] = {}
    config['paths']['input_file'] = test_data_path
    
    # Set specific date range matching data
    if 'dates' not in config:
        config['dates'] = {}
    config['dates']['start_date'] = "2024-01-02"
    config['dates']['end_date'] = "2024-01-03"
    
    # Set strategy
    config['strategy']['name'] = "PutSellStrat"
    
    # Set up logging
    logging_manager = SimpleLoggingManager()
    logger = logging_manager.setup_logging(
        config_dict=config,
        verbose_console=True,
        debug_mode=True,
        clean_format=False
    )
    
    # MANUALLY READ AND DIAGNOSE DATA FILE
    print("\n=== MANUALLY CHECKING TEST DATA FILE ===")
    
    # Check if test data file exists
    if not os.path.exists(test_data_path):
        print(f"ERROR: Test data file not found: {test_data_path}")
        return 1
    
    # Read the test data
    try:
        df = pd.read_csv(test_data_path)
        print(f"Successfully read data file: {test_data_path}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {', '.join(df.columns)}")
        
        # Check date column format
        if 'date' in df.columns:
            print("\n=== Date Column Analysis ===")
            print(f"Date column type: {df['date'].dtype}")
            print(f"First 5 dates: {df['date'].head(5).tolist()}")
            
            # Convert to datetime if it's not already
            if df['date'].dtype != 'datetime64[ns]':
                print("Converting date column to datetime format...")
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    print("Conversion successful")
                    print(f"Date column type after conversion: {df['date'].dtype}")
                    print(f"First 5 dates after conversion: {df['date'].head(5).tolist()}")
                except Exception as e:
                    print(f"Error converting dates: {e}")
            
            # Extract unique dates for comparison with config
            unique_dates = df['date'].dt.date.unique()
            print(f"\nUnique dates in dataset: {unique_dates}")
            
            # Compare with config date range
            start_date = pd.to_datetime(config['dates']['start_date']).date()
            end_date = pd.to_datetime(config['dates']['end_date']).date()
            print(f"Config date range: {start_date} to {end_date}")
            
            # Check if there are dates in the range
            filtered_dates = [d for d in unique_dates if start_date <= d <= end_date]
            print(f"Dates in range: {filtered_dates}")
            
            if not filtered_dates:
                print("WARNING: No dates in the specified range - this will cause 'No trading dates available'")
                
                # Suggest fixing by updating date range
                print("\nSuggested fix:")
                if len(unique_dates) > 0:
                    suggested_start = min(unique_dates)
                    suggested_end = max(unique_dates)
                    print(f"Update date range to: {suggested_start} to {suggested_end}")
                    
                    # Update config with suggested dates
                    config['dates']['start_date'] = str(suggested_start)
                    config['dates']['end_date'] = str(suggested_end)
                    print(f"Updated config date range to: {config['dates']['start_date']} to {config['dates']['end_date']}")
        else:
            print("WARNING: No 'date' column found in data")
    except Exception as e:
        print(f"Error reading or analyzing data file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n=== Trying to run the engine with corrected dates ===")
    print(f"Date range: {config['dates']['start_date']} to {config['dates']['end_date']}")
    
    # Initialize the trading engine with the modified configuration
    try:
        print("Initializing trading engine...")
        engine = TradingEngine(config, logger)
        
        # Run the backtest
        print("Starting backtest...")
        engine.run_backtest()
        
        print("\n=== Backtest Complete ===")
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