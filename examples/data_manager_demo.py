#!/usr/bin/env python
"""
Data Managers Demo

This script demonstrates how to use the various data managers in the Option-Framework
to load and process different types of financial data.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Ensure we can import from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_managers import (
    OptionDataManager, 
    IntradayDataManager, 
    DailyDataManager
)
from data_managers.utils import (
    detect_data_frequency,
    convert_to_appropriate_format
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataManagerDemo")


def demo_data_conversion():
    """Demonstrate data conversion utilities."""
    logger.info("\n=== Data Conversion Demo ===")
    
    input_files = [
        # Add your input files here
        # For example: "data/raw/spy_daily.csv", "data/raw/spy_minute.csv"
    ]
    
    if not input_files:
        logger.warning("No input files provided for conversion demo")
        logger.info("To use this demo, add your data file paths to the input_files list")
        return
    
    for input_file in input_files:
        if not os.path.exists(input_file):
            logger.warning(f"File not found: {input_file}")
            continue
            
        # Detect data frequency
        frequency = detect_data_frequency(input_file)
        logger.info(f"Detected frequency for {input_file}: {frequency}")
        
        # Convert to standard format
        output_file = convert_to_appropriate_format(
            input_file,
            output_dir="data/processed",
            verbose=True
        )
        
        logger.info(f"Converted {input_file} to {output_file}")


def demo_intraday_data_manager():
    """Demonstrate intraday data manager functionality."""
    logger.info("\n=== Intraday Data Manager Demo ===")
    
    # Create an intraday data manager with NYC timezone
    config = {"timezone": "America/New_York"}
    intraday_dm = IntradayDataManager(config)
    
    # Specify the intraday data file
    intraday_file = "path/to/your/intraday_data.csv"
    
    if not os.path.exists(intraday_file):
        logger.warning(f"File not found: {intraday_file}")
        logger.info("To use this demo, provide a valid intraday data file path")
        return
    
    # Load and prepare data
    minute_data = intraday_dm.prepare_data_for_analysis(
        intraday_file,
        days_to_analyze=20,
        lookback_buffer=5
    )
    
    if minute_data.empty:
        logger.warning("No intraday data loaded")
        return
    
    # Show data info
    logger.info(f"Loaded {len(minute_data)} intraday data points")
    
    # Filter to market hours
    market_hours_data = intraday_dm.filter_market_hours(minute_data)
    logger.info(f"After filtering to market hours: {len(market_hours_data)} data points")
    
    # Plot a single day of data
    recent_date = minute_data.index.date[-1]
    single_day = minute_data[minute_data.index.date == recent_date]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(single_day.index, single_day['Close'], label='Price')
    if 'vwap' in single_day.columns:
        ax.plot(single_day.index, single_day['vwap'], label='VWAP', linestyle='--')
    
    ax.set_title(f"Intraday Data for {recent_date}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    
    # Save plot
    plt.savefig("intraday_demo.png")
    logger.info("Saved intraday plot to intraday_demo.png")


def demo_daily_data_manager():
    """Demonstrate daily data manager functionality."""
    logger.info("\n=== Daily Data Manager Demo ===")
    
    # Create a daily data manager
    daily_dm = DailyDataManager()
    
    # Specify the daily data file
    daily_file = "path/to/your/daily_data.csv"
    
    if not os.path.exists(daily_file):
        logger.warning(f"File not found: {daily_file}")
        logger.info("To use this demo, provide a valid daily data file path")
        return
    
    # Load and prepare data
    daily_data = daily_dm.prepare_data_for_analysis(
        daily_file,
        lookback_days=252  # ~1 year of trading days
    )
    
    if daily_data.empty:
        logger.warning("No daily data loaded")
        return
    
    # Show data info
    logger.info(f"Loaded {len(daily_data)} days of data")
    
    # Plot price and moving averages
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_data.index, daily_data['Close'], label='Close Price')
    
    for window in [20, 50, 200]:
        ma_col = f'ma{window}'
        if ma_col in daily_data.columns:
            ax.plot(daily_data.index, daily_data[ma_col], label=f'{window}-day MA')
    
    ax.set_title("Daily Price with Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    
    # Save plot
    plt.savefig("daily_demo.png")
    logger.info("Saved daily plot to daily_demo.png")


def demo_option_data_manager():
    """Demonstrate option data manager functionality."""
    logger.info("\n=== Option Data Manager Demo ===")
    
    # Create an option data manager
    option_dm = OptionDataManager()
    
    # Specify the option data file
    option_file = "path/to/your/option_data.csv"
    
    if not os.path.exists(option_file):
        logger.warning(f"File not found: {option_file}")
        logger.info("To use this demo, provide a valid option data file path")
        return
    
    # Load and prepare data
    option_data = option_dm.prepare_data_for_analysis(
        option_file,
        option_type='BOTH',
        min_dte=5,
        max_dte=60,
        min_moneyness=0.8,
        max_moneyness=1.2
    )
    
    if option_data.empty:
        logger.warning("No option data loaded")
        return
    
    # Show data info
    logger.info(f"Loaded {len(option_data)} option contracts")
    
    # Get the options chain for the nearest expiration
    chain = option_dm.get_options_chain(option_data)
    logger.info(f"Options chain for nearest expiration: {len(chain)} contracts")
    
    # Plot implied volatility by strike for calls and puts
    if 'option_type' in chain.columns and 'strike' in chain.columns and 'iv' in chain.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        calls = chain[chain['option_type'] == 'C']
        puts = chain[chain['option_type'] == 'P']
        
        ax.scatter(calls['strike'], calls['iv'], label='Calls', color='green', marker='o')
        ax.scatter(puts['strike'], puts['iv'], label='Puts', color='red', marker='x')
        
        ax.set_title("Implied Volatility Smile")
        ax.set_xlabel("Strike Price")
        ax.set_ylabel("Implied Volatility")
        ax.legend()
        ax.grid(True)
        
        # Save plot
        plt.savefig("option_iv_demo.png")
        logger.info("Saved option IV plot to option_iv_demo.png")
        
        # Try to plot volatility surface if we have enough data
        if len(option_data['dte'].unique()) > 3:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            option_dm.plot_volatility_surface(
                option_data[option_data['option_type'] == 'C'],
                z_column='iv',
                ax=ax,
                title="Call Option Implied Volatility Surface"
            )
            
            # Save plot
            plt.savefig("vol_surface_demo.png")
            logger.info("Saved volatility surface plot to vol_surface_demo.png")


if __name__ == "__main__":
    # Run all demos
    demo_data_conversion()
    demo_intraday_data_manager()
    demo_daily_data_manager()
    demo_option_data_manager()
    
    logger.info("\nAll demos completed. Check the generated plots if data was available.")
    
    # Display plots if running in interactive mode
    plt.show() 