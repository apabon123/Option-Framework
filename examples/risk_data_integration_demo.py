#!/usr/bin/env python
"""
Risk Management and Data Manager Integration Demo

This script demonstrates how to integrate the risk management module with
different data managers for comprehensive backtesting with risk controls.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yaml

# Ensure we can import from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import data managers
from data_managers import IntradayDataManager, DailyDataManager

# Import risk management classes
from core.risk.factory import RiskManagerFactory
from core.risk.metrics import RiskMetrics
from core.risk.parameters import RiskParameters, VolatilityRiskParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RiskDataIntegrationDemo")


def load_config(config_path="config/risk_config.yaml"):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def demo_daily_data_with_risk():
    """
    Demonstrate daily data with risk management
    """
    logger.info("\n=== Daily Data with Risk Management Demo ===")
    
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
    
    # Create risk metrics calculator
    risk_metrics = RiskMetrics(
        short_window=20,
        medium_window=60,
        long_window=120,
        risk_free_rate=0.03
    )
    
    # Calculate risk metrics from daily returns
    if 'daily_return' in daily_data.columns:
        returns_series = daily_data['daily_return'].dropna()
        
        # Update risk metrics with returns data
        risk_metrics.update_metrics(returns_series)
        
        # Log key risk metrics
        logger.info(f"Volatility (annualized): {risk_metrics.get_volatility():.2%}")
        logger.info(f"Sharpe Ratio: {risk_metrics.get_sharpe_ratio():.2f}")
        logger.info(f"Maximum Drawdown: {risk_metrics.get_max_drawdown():.2%}")
        logger.info(f"Value at Risk (95%): {risk_metrics.get_var(0.95):.2%}")
        
        # Create volatility-targeting risk manager
        vol_params = VolatilityRiskParameters(
            target_volatility=0.15,
            max_leverage=2.0,
            volatility_lookback=60
        )
        
        # Create risk manager
        risk_manager = RiskManagerFactory.create(
            {
                "risk_manager_type": "volatility",
                "risk_parameters": vol_params.to_dict()
            }, 
            risk_metrics
        )
        
        # Calculate position sizes over time based on volatility
        position_sizes = []
        dates = []
        
        # Simulate position sizing through time
        for i in range(20, len(daily_data)):
            window = daily_data.iloc[i-20:i]
            window_returns = window['daily_return'].dropna()
            window_volatility = window_returns.std() * np.sqrt(252)  # Annualize
            
            # Calculate position size (capital = 100,000)
            position_size = risk_manager.calculate_position_size(
                {
                    "volatility": window_volatility,
                    "price": window.iloc[-1]['Close'],
                    "expected_return": 0.0001  # Daily expected return (example)
                },
                100000  # Capital
            )
            
            position_sizes.append(position_size / 100000)  # As percentage of capital
            dates.append(window.index[-1])
        
        # Plot position sizes over time
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price
        ax1.plot(daily_data.index, daily_data['Close'], label='Price')
        ax1.set_title("Price and Position Size")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True)
        
        # Plot position size
        ax2.plot(dates, position_sizes, label='Position Size (% of Capital)', color='orange')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Position Size")
        ax2.legend()
        ax2.grid(True)
        
        # Save plot
        plt.tight_layout()
        plt.savefig("daily_risk_demo.png")
        logger.info("Saved daily risk plot to daily_risk_demo.png")


def demo_intraday_data_with_risk():
    """
    Demonstrate intraday data with risk management
    """
    logger.info("\n=== Intraday Data with Risk Management Demo ===")
    
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
    intraday_data = intraday_dm.prepare_data_for_analysis(
        intraday_file,
        days_to_analyze=20,
        lookback_buffer=5
    )
    
    if intraday_data.empty:
        logger.warning("No intraday data loaded")
        return
    
    # Resample to hourly data for risk analysis
    hourly_data = intraday_data.copy()
    hourly_data = hourly_data.resample('1H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # Calculate hourly returns
    hourly_data['hourly_return'] = hourly_data['Close'].pct_change()
    
    # Create Adaptive Risk Manager with heat-based position sizing
    config = {
        "risk_manager_type": "adaptive",
        "risk_parameters": {
            "base_size": 1.0,
            "max_size": 2.0,
            "min_size": 0.2,
            "heat_rate": 0.25,
            "cool_rate": 0.10
        }
    }
    
    # Create risk metrics calculator
    risk_metrics = RiskMetrics(
        short_window=24,  # 1 day (with hourly data)
        medium_window=72,  # 3 days (with hourly data)
        long_window=240,  # 10 days (with hourly data)
        risk_free_rate=0.03/24  # Hourly risk-free rate
    )
    
    # Calculate risk metrics from hourly returns
    returns_series = hourly_data['hourly_return'].dropna()
    risk_metrics.update_metrics(returns_series)
    
    # Create risk manager
    risk_manager = RiskManagerFactory.create(config, risk_metrics)
    
    # Simulate an intraday trading strategy with adaptive position sizing
    logger.info("Simulating an intraday trading strategy with adaptive position sizing")
    
    # Get a single day's data for visualization
    recent_date = intraday_data.index.date[-1]
    single_day = intraday_data[intraday_data.index.date == recent_date]
    
    # Simulate position sizing based on market hours
    market_hours = single_day['market_hours'] if 'market_hours' in single_day.columns else np.ones(len(single_day))
    
    # Simulate PnL to affect the heat
    cumulative_pnl = 0
    position_sizes = []
    
    # Start with neutral heat
    heat = 0
    
    # Simulate trades throughout the day
    for i in range(len(single_day)):
        # Get current data point
        current = single_day.iloc[i]
        
        # Skip if outside market hours
        if market_hours[i] != 1:
            position_sizes.append(0)
            continue
        
        # Simulate a trade result (win or loss based on random seed)
        np.random.seed(i)  # For reproducibility
        trade_result = (np.random.random() - 0.48) * 0.01  # Slight edge
        
        # Update cumulative PnL
        cumulative_pnl += trade_result
        
        # Update heat based on trade result
        if trade_result > 0:
            heat -= config["risk_parameters"]["cool_rate"]
        else:
            heat += config["risk_parameters"]["heat_rate"] * abs(trade_result) * 100
        
        # Clamp heat between 0 and 1
        heat = max(0, min(1, heat))
        
        # Calculate position size
        position_info = {
            "heat": heat,
            "time_of_day": current.name.time(),
            "price": current['Close']
        }
        
        position_size = risk_manager.calculate_position_size(position_info, 100000)
        position_sizes.append(position_size / 100000)  # As percentage of capital
    
    # Plot a single day of data with position sizes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot price
    ax1.plot(single_day.index, single_day['Close'], label='Price')
    ax1.set_title(f"Intraday Price for {recent_date}")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)
    
    # Plot position size
    ax2.plot(single_day.index, position_sizes, label='Position Size (% of Capital)', color='orange')
    ax2.set_ylabel("Position Size")
    ax2.legend()
    ax2.grid(True)
    
    # Plot heat
    heat_values = [1 - ps/position_sizes[0] if position_sizes[0] > 0 else 0 for ps in position_sizes]
    ax3.plot(single_day.index, heat_values, label='Heat', color='red')
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Heat")
    ax3.legend()
    ax3.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig("intraday_risk_demo.png")
    logger.info("Saved intraday risk plot to intraday_risk_demo.png")


if __name__ == "__main__":
    # Run demos
    demo_daily_data_with_risk()
    demo_intraday_data_with_risk()
    
    logger.info("\nAll demos completed. Check the generated plots if data was available.")
    
    # Display plots if running in interactive mode
    plt.show() 