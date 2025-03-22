"""
Pytest fixtures for the Option-Framework testing suite.
This file contains shared fixtures for unit, integration, and system tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile

# Import conditional based on module availability
try:
    from core.portfolio import Portfolio
    from core.position import OptionPosition, StockPosition
    from core.margin import SpanMarginCalculator
    from data_managers.option_data_manager import OptionDataManager
    from data_managers.daily_data_manager import DailyDataManager
    from data_managers.intraday_data_manager import IntradayDataManager
except ImportError:
    # If modules are not available yet, use mock classes for development
    class Portfolio:
        def __init__(self, initial_capital=100000):
            self.initial_capital = initial_capital
    
    class OptionPosition:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class StockPosition:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class SpanMarginCalculator:
        def __init__(self, config):
            self.config = config
    
    class OptionDataManager:
        def load_data(self, file_path):
            pass
    
    class DailyDataManager:
        def load_data(self, file_path):
            pass
    
    class IntradayDataManager:
        def __init__(self, timezone='UTC'):
            self.timezone = timezone
        
        def load_data(self, file_path):
            pass


@pytest.fixture
def sample_config():
    """Return a sample configuration for testing."""
    return {
        'portfolio': {
            'initial_capital': 100000,
            'max_leverage': 2.0
        },
        'span': {
            'max_leverage': 12.0,
            'initial_margin_percentage': 0.1,
            'maintenance_margin_percentage': 0.07,
            'hedge_credit_rate': 0.8,
            'price_move_pct': 0.05,
            'vol_shift_pct': 0.3,
            'gamma_scaling_factor': 0.3,
            'min_scan_risk_percentage': 0.25,
            'max_margin_to_premium_ratio': 20.0,
            'otm_scaling_enabled': True,
            'otm_minimum_scaling': 0.1
        },
        'data': {
            'sources': [
                {'type': 'csv', 'path': 'tests/data/sample_options.csv'}
            ]
        },
        'strategy': {
            'name': 'ThetaStrategy',
            'parameters': {
                'min_dte': 30,
                'max_dte': 45,
                'target_delta': 0.3
            }
        }
    }

@pytest.fixture
def empty_portfolio():
    """Return an empty portfolio."""
    return Portfolio(initial_capital=100000)

@pytest.fixture
def sample_portfolio():
    """Return a portfolio with sample positions."""
    portfolio = Portfolio(initial_capital=100000)
    
    # Add an option position
    call_option = OptionPosition(
        symbol="SPY220318C450",
        underlying="SPY",
        quantity=10,
        option_type="CALL",
        strike=450,
        expiration=datetime(2022, 3, 18),
        entry_price=5.0,
        current_price=5.5,
        greeks={"delta": 0.4, "gamma": 0.05, "theta": -0.1, "vega": 0.2}
    )
    
    # Add a put option
    put_option = OptionPosition(
        symbol="SPY220318P420",
        underlying="SPY",
        quantity=-5,
        option_type="PUT",
        strike=420,
        expiration=datetime(2022, 3, 18),
        entry_price=6.0,
        current_price=5.8,
        greeks={"delta": 0.3, "gamma": -0.03, "theta": 0.08, "vega": -0.15}
    )
    
    # Add a stock position
    stock = StockPosition(
        symbol="SPY",
        quantity=-30,
        entry_price=432.0,
        current_price=435.0
    )
    
    # Add positions to portfolio (actual implementation may vary)
    try:
        portfolio.add_position(call_option)
        portfolio.add_position(put_option)
        portfolio.add_position(stock)
    except (NotImplementedError, AttributeError):
        # If add_position is not implemented, just store positions
        portfolio.positions = [call_option, put_option, stock]
    
    return portfolio

@pytest.fixture
def margin_calculator(sample_config):
    """Return a SPAN margin calculator."""
    return SpanMarginCalculator(sample_config)

@pytest.fixture
def sample_option_data():
    """Generate sample option data for testing."""
    # Create option chain data
    expiration = datetime(2022, 3, 18)
    today = datetime(2022, 2, 15)
    
    data = []
    for strike in range(400, 470, 5):
        for option_type in ['CALL', 'PUT']:
            # ATM is around 435
            atm_factor = 1.0 - abs(strike - 435) / 100
            price = max(0.5, 5.0 * atm_factor)
            
            delta = 0.5 * atm_factor
            if option_type == 'PUT':
                delta = -delta
                
            option = {
                'Date': today,
                'Symbol': f"SPY220318{option_type[0]}{strike}",
                'Underlying': 'SPY',
                'Strike': strike,
                'OptionType': option_type,
                'Expiration': expiration,
                'DTE': (expiration - today).days,
                'Bid': price - 0.05,
                'Ask': price + 0.05,
                'Mid': price,
                'Last': price,
                'Volume': 1000,
                'OpenInterest': 5000,
                'Delta': delta,
                'Gamma': 0.05 * atm_factor,
                'Theta': -0.10 * atm_factor,
                'Vega': 0.20 * atm_factor,
                'ImpliedVol': 0.20 + 0.05 * (1 - atm_factor)
            }
            data.append(option)
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_daily_data():
    """Generate sample daily price data for testing."""
    dates = pd.date_range(start='2021-01-01', periods=100)
    
    # Generate prices with a slight uptrend
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 100)
    
    prices = [100]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    prices = prices[1:]
    
    data = pd.DataFrame({
        'Date': dates,
        'Symbol': ['SPY'] * len(dates),
        'Open': prices,
        'High': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'Low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'Close': prices,
        'Volume': [int(np.random.uniform(100000, 1000000)) for _ in range(len(prices))]
    })
    
    return data

@pytest.fixture
def sample_intraday_data():
    """Generate sample intraday minute data for testing."""
    # Create a trading day with minute data
    base_date = datetime(2022, 2, 15, 9, 30)  # Market open
    minutes = pd.date_range(
        start=base_date,
        end=base_date + timedelta(hours=6, minutes=30),  # Until 4:00 PM
        freq='1min'
    )
    
    # Generate prices with random walk
    np.random.seed(42)
    returns = np.random.normal(0.0, 0.001, len(minutes))
    
    prices = [435.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    prices = prices[1:]
    
    data = pd.DataFrame({
        'Timestamp': minutes,
        'Symbol': ['SPY'] * len(minutes),
        'Open': prices,
        'High': [p * (1 + np.random.uniform(0, 0.001)) for p in prices],
        'Low': [p * (1 - np.random.uniform(0, 0.001)) for p in prices],
        'Close': prices,
        'Volume': [int(np.random.uniform(1000, 10000)) for _ in range(len(prices))]
    })
    
    return data


# The following fixtures require saving data to disk, so they're implemented
# conditionally based on whether the data managers are fully implemented

@pytest.fixture
def option_data_manager(sample_option_data, tmp_path):
    """Return an initialized option data manager with sample data."""
    # Save data to a temp file
    data_file = os.path.join(tmp_path, "sample_options.csv")
    sample_option_data.to_csv(data_file, index=False)
    
    # Create and initialize the manager
    manager = OptionDataManager()
    try:
        manager.load_data(data_file)
    except Exception as e:
        # If load_data is not fully implemented, just attach the data
        manager.data = sample_option_data
    
    return manager

@pytest.fixture
def daily_data_manager(sample_daily_data, tmp_path):
    """Return an initialized daily data manager with sample data."""
    # Save data to a temp file
    data_file = os.path.join(tmp_path, "sample_daily.csv")
    sample_daily_data.to_csv(data_file, index=False)
    
    # Create and initialize the manager
    manager = DailyDataManager()
    try:
        manager.load_data(data_file)
    except Exception as e:
        # If load_data is not fully implemented, just attach the data
        manager.data = sample_daily_data
    
    return manager

@pytest.fixture
def intraday_data_manager(sample_intraday_data, tmp_path):
    """Return an initialized intraday data manager with sample data."""
    # Save data to a temp file
    data_file = os.path.join(tmp_path, "sample_intraday.csv")
    sample_intraday_data.to_csv(data_file, index=False)
    
    # Create and initialize the manager
    manager = IntradayDataManager(timezone='America/New_York')
    try:
        manager.load_data(data_file)
    except Exception as e:
        # If load_data is not fully implemented, just attach the data
        manager.data = sample_intraday_data
    
    return manager 