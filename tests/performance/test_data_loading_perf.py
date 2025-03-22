"""
Performance tests for data loading and processing.
These tests measure the efficiency of data loading, filtering, and processing operations.
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import tempfile

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the required modules
try:
    from data_managers.option_data_manager import OptionDataManager
    from data_managers.daily_data_manager import DailyDataManager
    from data_managers.intraday_data_manager import IntradayDataManager
except ImportError:
    pytest.skip("Data manager modules not available for testing", allow_module_level=True)


@pytest.fixture
def large_option_data_file():
    """Generate a large option chain dataset for performance testing."""
    # Number of options to generate
    num_options = 10000
    
    # Create a DataFrame with option data
    data = []
    
    # Current date for calculations
    current_date = datetime.now()
    
    # List of stock symbols for underlyings
    stock_symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM"]
    
    # Generate a range of expiry dates
    expiry_dates = [current_date + timedelta(days=days) for days in [7, 14, 30, 60, 90, 180, 365]]
    
    # Generate random options data
    for _ in range(num_options):
        # Choose a random underlying
        underlying = np.random.choice(stock_symbols)
        
        # Choose a random expiry date
        expiry_date = np.random.choice(expiry_dates)
        expiry_str = expiry_date.strftime('%Y-%m-%d')
        
        # Choose random option parameters
        option_type = np.random.choice(["CALL", "PUT"])
        underlying_price = np.random.uniform(50.0, 500.0)
        strike = underlying_price * np.random.uniform(0.7, 1.3)  # Strike around the underlying price
        
        # Calculate approximate Black-Scholes values for the options
        moneyness = underlying_price / strike
        itm = (option_type == "CALL" and underlying_price > strike) or (option_type == "PUT" and underlying_price < strike)
        
        # Calculate delta based on moneyness and option type
        if option_type == "CALL":
            delta = 0.5 + 0.5 * (moneyness - 1) * 10
            delta = max(0.01, min(0.99, delta))
        else:  # PUT
            delta = -0.5 - 0.5 * (moneyness - 1) * 10
            delta = min(-0.01, max(-0.99, delta))
        
        # Calculate approx price
        if itm:
            if option_type == "CALL":
                price = max(0.1, underlying_price - strike)
            else:  # PUT
                price = max(0.1, strike - underlying_price)
        else:
            price = max(0.1, 5 * (1 - abs(moneyness - 1) * 5))
        
        # Calculate other Greeks
        if option_type == "CALL":
            gamma = max(0.001, 0.05 * (1 - abs(delta - 0.5) * 1.5))
        else:  # PUT
            gamma = max(0.001, 0.05 * (1 - abs(delta + 0.5) * 1.5))
        
        vega = max(0.01, 0.5 * (1 - abs(delta) * 1.5))
        theta = -max(0.01, 0.1 * price)
        
        # Generate a unique option symbol
        expiry_code = expiry_date.strftime('%y%m%d')
        strike_code = f"{int(strike):08d}"
        option_symbol = f"{underlying}{expiry_code}{option_type[0]}{strike_code}"
        
        # Add the option data
        data.append({
            'symbol': option_symbol,
            'underlying': underlying,
            'expiration': expiry_str,
            'strike': strike,
            'option_type': option_type,
            'bid': price * 0.95,
            'ask': price * 1.05,
            'last': price,
            'volume': int(1000 * (1 - abs(delta if option_type == "CALL" else delta + 1) * 0.5)),
            'open_interest': int(5000 * (1 - abs(delta if option_type == "CALL" else delta + 1) * 0.5)),
            'implied_volatility': 0.2 + 0.1 * abs(delta if option_type == "CALL" else delta + 1),
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'underlying_price': underlying_price
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as temp_file:
        df.to_csv(temp_file.name, index=False)
        return temp_file.name


@pytest.fixture
def large_daily_price_data_file():
    """Generate a large daily price dataset for performance testing."""
    # Generate price data for multiple stocks over several years
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM"]
    
    # Generate dates for 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Create a list to hold all data
    all_data = []
    
    # Generate price data for each symbol
    for symbol in symbols:
        # Start with a random price between 50 and 500
        start_price = np.random.uniform(50.0, 500.0)
        
        # Generate a price path with random walk and drift
        price = start_price
        prices = []
        
        for _ in range(len(dates)):
            # Random daily return with slight positive drift
            daily_return = np.random.normal(0.0005, 0.015)
            price *= (1 + daily_return)
            prices.append(price)
        
        # Create a Series of daily prices
        price_series = pd.Series(prices)
        
        # Calculate daily high, low, and volume
        high = price_series * (1 + np.random.uniform(0.0, 0.03, size=len(price_series)))
        low = price_series * (1 - np.random.uniform(0.0, 0.03, size=len(price_series)))
        volume = np.random.randint(100000, 10000000, size=len(price_series))
        
        # Create data for this symbol
        for i, date in enumerate(dates):
            all_data.append({
                'symbol': symbol,
                'date': date.strftime('%Y-%m-%d'),
                'open': prices[i] * (1 - np.random.uniform(0.0, 0.01)),
                'high': high[i],
                'low': low[i],
                'close': prices[i],
                'volume': volume[i],
                'adjusted_close': prices[i]  # For simplicity, assume no adjustments
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as temp_file:
        df.to_csv(temp_file.name, index=False)
        return temp_file.name


@pytest.fixture
def large_intraday_price_data_file():
    """Generate a large intraday price dataset for performance testing."""
    # Generate intraday price data for a single trading day
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    
    # Generate timestamps for one trading day (6.5 hours) at minute intervals
    end_time = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
    start_time = end_time.replace(hour=9, minute=30)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Create a list to hold all data
    all_data = []
    
    # Generate intraday price data for each symbol
    for symbol in symbols:
        # Start with a random price between 50 and 500
        start_price = np.random.uniform(50.0, 500.0)
        
        # Generate a price path with random walk
        price = start_price
        prices = []
        
        for _ in range(len(timestamps)):
            # Random minute return (more volatile than daily)
            minute_return = np.random.normal(0.0, 0.001)
            price *= (1 + minute_return)
            prices.append(price)
        
        # Create a Series of minute prices
        price_series = pd.Series(prices)
        
        # Calculate minute high, low, and volume
        high = price_series * (1 + np.random.uniform(0.0, 0.002, size=len(price_series)))
        low = price_series * (1 - np.random.uniform(0.0, 0.002, size=len(price_series)))
        volume = np.random.randint(1000, 100000, size=len(price_series))
        
        # Create data for this symbol
        for i, timestamp in enumerate(timestamps):
            all_data.append({
                'symbol': symbol,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'price': prices[i],
                'high': high[i],
                'low': low[i],
                'volume': volume[i]
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as temp_file:
        df.to_csv(temp_file.name, index=False)
        return temp_file.name


class TestOptionDataManagerPerformance:
    """Performance tests for option data loading and processing."""
    
    def test_option_data_loading_performance(self, large_option_data_file):
        """Test performance of loading large option datasets."""
        # Create option data manager
        option_manager = OptionDataManager()
        
        # Measure time to load the data
        start_time = time.time()
        option_data = option_manager.load_option_data(large_option_data_file)
        end_time = time.time()
        
        # Calculate loading time
        loading_time = end_time - start_time
        
        # Log the performance
        print(f"Option data loading time: {loading_time:.6f} seconds for {len(option_data)} options")
        
        # Verify data was loaded correctly
        assert not option_data.empty
        assert 'symbol' in option_data.columns
        assert 'delta' in option_data.columns
        
        # Loading should be reasonably fast (adjust threshold based on dataset size)
        assert loading_time < 5.0  # Should load in under 5 seconds
    
    def test_option_data_filtering_performance(self, large_option_data_file):
        """Test performance of filtering large option datasets."""
        # Create option data manager
        option_manager = OptionDataManager()
        
        # Load the option data
        option_data = option_manager.load_option_data(large_option_data_file)
        
        # Test filtering by different criteria
        filters = [
            {"option_type": "CALL"},
            {"option_type": "PUT"},
            {"min_delta": 0.4, "max_delta": 0.6},
            {"min_delta": -0.6, "max_delta": -0.4},
            {"min_volume": 5000},
            {"underlying": "AAPL"}
        ]
        
        for filter_criteria in filters:
            # Measure time to filter the data
            start_time = time.time()
            filtered_data = option_manager.filter_options(option_data, **filter_criteria)
            end_time = time.time()
            
            # Calculate filtering time
            filtering_time = end_time - start_time
            
            # Log the performance
            print(f"Option data filtering time: {filtering_time:.6f} seconds for filter {filter_criteria}")
            
            # Filtering should be reasonably fast
            assert filtering_time < 1.0  # Should filter in under 1 second
            
            # Verify the filter worked correctly
            if "option_type" in filter_criteria:
                assert all(filtered_data['option_type'] == filter_criteria["option_type"])
            if "min_delta" in filter_criteria:
                assert all(filtered_data['delta'] >= filter_criteria["min_delta"])
            if "max_delta" in filter_criteria:
                assert all(filtered_data['delta'] <= filter_criteria["max_delta"])
            if "min_volume" in filter_criteria:
                assert all(filtered_data['volume'] >= filter_criteria["min_volume"])
            if "underlying" in filter_criteria:
                assert all(filtered_data['underlying'] == filter_criteria["underlying"])
    
    def test_option_data_calculation_performance(self, large_option_data_file):
        """Test performance of calculating derived values from option data."""
        # Create option data manager
        option_manager = OptionDataManager()
        
        # Load the option data
        option_data = option_manager.load_option_data(large_option_data_file)
        
        # Measure time to calculate implied volatility surface
        start_time = time.time()
        vol_surface = option_manager.calculate_volatility_surface(option_data)
        end_time = time.time()
        
        # Calculate computation time
        computation_time = end_time - start_time
        
        # Log the performance
        print(f"Volatility surface calculation time: {computation_time:.6f} seconds")
        
        # Calculation should be reasonably fast
        assert computation_time < 5.0  # Should compute in under 5 seconds
        
        # Verify the calculation produced results
        assert vol_surface is not None
        assert not vol_surface.empty if isinstance(vol_surface, pd.DataFrame) else True
    
    def test_option_chain_processing_performance(self, large_option_data_file):
        """Test performance of processing complete option chains."""
        # Create option data manager
        option_manager = OptionDataManager()
        
        # Load the option data
        option_data = option_manager.load_option_data(large_option_data_file)
        
        # Select a symbol to process its full chain
        underlyings = option_data['underlying'].unique()
        test_symbol = underlyings[0]
        
        # Measure time to process the full option chain
        start_time = time.time()
        chain = option_manager.get_option_chain(option_data, underlying=test_symbol)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log the performance
        print(f"Option chain processing time for {test_symbol}: {processing_time:.6f} seconds")
        
        # Processing should be reasonably fast
        assert processing_time < 2.0  # Should process in under 2 seconds
        
        # Verify the chain contains only options for the specified underlying
        assert all(chain['underlying'] == test_symbol)


class TestDailyDataManagerPerformance:
    """Performance tests for daily price data loading and processing."""
    
    def test_daily_data_loading_performance(self, large_daily_price_data_file):
        """Test performance of loading large daily price datasets."""
        # Create daily data manager
        daily_manager = DailyDataManager()
        
        # Measure time to load the data
        start_time = time.time()
        daily_data = daily_manager.load_daily_data(large_daily_price_data_file)
        end_time = time.time()
        
        # Calculate loading time
        loading_time = end_time - start_time
        
        # Log the performance
        print(f"Daily data loading time: {loading_time:.6f} seconds for {len(daily_data)} records")
        
        # Verify data was loaded correctly
        assert not daily_data.empty
        assert 'symbol' in daily_data.columns
        assert 'close' in daily_data.columns
        
        # Loading should be reasonably fast
        assert loading_time < 5.0  # Should load in under 5 seconds
    
    def test_daily_data_calculation_performance(self, large_daily_price_data_file):
        """Test performance of calculating technical indicators from daily data."""
        # Create daily data manager
        daily_manager = DailyDataManager()
        
        # Load the daily data
        daily_data = daily_manager.load_daily_data(large_daily_price_data_file)
        
        # Select a symbol to calculate indicators for
        symbols = daily_data['symbol'].unique()
        test_symbol = symbols[0]
        symbol_data = daily_data[daily_data['symbol'] == test_symbol]
        
        # Measure time to calculate multiple indicators
        start_time = time.time()
        
        # Calculate moving averages
        daily_manager.calculate_sma(symbol_data, 'close', 20)
        daily_manager.calculate_sma(symbol_data, 'close', 50)
        daily_manager.calculate_sma(symbol_data, 'close', 200)
        
        # Calculate RSI
        daily_manager.calculate_rsi(symbol_data, 'close', 14)
        
        # Calculate MACD
        daily_manager.calculate_macd(symbol_data, 'close', 12, 26, 9)
        
        end_time = time.time()
        
        # Calculate computation time
        computation_time = end_time - start_time
        
        # Log the performance
        print(f"Technical indicator calculation time for {test_symbol}: {computation_time:.6f} seconds")
        
        # Calculation should be reasonably fast
        assert computation_time < 2.0  # Should compute in under 2 seconds
    
    def test_daily_data_filtering_performance(self, large_daily_price_data_file):
        """Test performance of filtering and aggregating daily data."""
        # Create daily data manager
        daily_manager = DailyDataManager()
        
        # Load the daily data
        daily_data = daily_manager.load_daily_data(large_daily_price_data_file)
        
        # Test different filtering and aggregation operations
        operations = [
            {"description": "Filter by date range", 
             "func": lambda df: df[(df['date'] >= '2020-01-01') & (df['date'] <= '2020-12-31')]},
            {"description": "Calculate daily returns", 
             "func": lambda df: daily_manager.calculate_returns(df, 'close')},
            {"description": "Resample to weekly", 
             "func": lambda df: daily_manager.resample_to_period(df, 'W')},
            {"description": "Calculate correlation matrix", 
             "func": lambda df: daily_manager.calculate_correlation_matrix(df, 'close')}
        ]
        
        for op in operations:
            # Measure time to perform the operation
            start_time = time.time()
            result = op["func"](daily_data)
            end_time = time.time()
            
            # Calculate operation time
            operation_time = end_time - start_time
            
            # Log the performance
            print(f"{op['description']} time: {operation_time:.6f} seconds")
            
            # Operations should be reasonably fast
            assert operation_time < 5.0  # Should complete in under 5 seconds
            
            # Verify the operation produced results
            assert result is not None


class TestIntradayDataManagerPerformance:
    """Performance tests for intraday price data loading and processing."""
    
    def test_intraday_data_loading_performance(self, large_intraday_price_data_file):
        """Test performance of loading large intraday price datasets."""
        # Create intraday data manager
        intraday_manager = IntradayDataManager()
        
        # Measure time to load the data
        start_time = time.time()
        intraday_data = intraday_manager.load_intraday_data(large_intraday_price_data_file)
        end_time = time.time()
        
        # Calculate loading time
        loading_time = end_time - start_time
        
        # Log the performance
        print(f"Intraday data loading time: {loading_time:.6f} seconds for {len(intraday_data)} records")
        
        # Verify data was loaded correctly
        assert not intraday_data.empty
        assert 'symbol' in intraday_data.columns
        assert 'price' in intraday_data.columns
        
        # Loading should be reasonably fast
        assert loading_time < 5.0  # Should load in under 5 seconds
    
    def test_intraday_data_resampling_performance(self, large_intraday_price_data_file):
        """Test performance of resampling intraday data to different timeframes."""
        # Create intraday data manager
        intraday_manager = IntradayDataManager()
        
        # Load the intraday data
        intraday_data = intraday_manager.load_intraday_data(large_intraday_price_data_file)
        
        # Select a symbol to resample
        symbols = intraday_data['symbol'].unique()
        test_symbol = symbols[0]
        symbol_data = intraday_data[intraday_data['symbol'] == test_symbol]
        
        # Test resampling to different timeframes
        timeframes = ['5min', '15min', '30min', '1H']
        
        for timeframe in timeframes:
            # Measure time to resample the data
            start_time = time.time()
            resampled_data = intraday_manager.resample_data(symbol_data, timeframe)
            end_time = time.time()
            
            # Calculate resampling time
            resampling_time = end_time - start_time
            
            # Log the performance
            print(f"Intraday data resampling to {timeframe} time: {resampling_time:.6f} seconds")
            
            # Resampling should be reasonably fast
            assert resampling_time < 1.0  # Should resample in under 1 second
            
            # Verify the resampling worked correctly
            assert not resampled_data.empty
            assert len(resampled_data) < len(symbol_data)  # Should have fewer rows after resampling
    
    def test_intraday_data_calculation_performance(self, large_intraday_price_data_file):
        """Test performance of calculating intraday indicators and metrics."""
        # Create intraday data manager
        intraday_manager = IntradayDataManager()
        
        # Load the intraday data
        intraday_data = intraday_manager.load_intraday_data(large_intraday_price_data_file)
        
        # Select a symbol to calculate indicators for
        symbols = intraday_data['symbol'].unique()
        test_symbol = symbols[0]
        symbol_data = intraday_data[intraday_data['symbol'] == test_symbol]
        
        # Measure time to calculate VWAP
        start_time = time.time()
        vwap_data = intraday_manager.calculate_vwap(symbol_data)
        end_time = time.time()
        
        # Calculate computation time
        vwap_time = end_time - start_time
        
        # Log the performance
        print(f"VWAP calculation time: {vwap_time:.6f} seconds")
        
        # Calculation should be reasonably fast
        assert vwap_time < 1.0  # Should compute in under 1 second
        
        # Verify the calculation produced results
        assert 'vwap' in vwap_data.columns
        
        # Measure time to calculate intraday volatility
        start_time = time.time()
        volatility = intraday_manager.calculate_intraday_volatility(symbol_data)
        end_time = time.time()
        
        # Calculate computation time
        volatility_time = end_time - start_time
        
        # Log the performance
        print(f"Intraday volatility calculation time: {volatility_time:.6f} seconds")
        
        # Calculation should be reasonably fast
        assert volatility_time < 1.0  # Should compute in under 1 second
        
        # Verify the calculation produced a result
        assert volatility > 0


if __name__ == "__main__":
    # This allows running the tests directly from this file
    pytest.main(["-v", __file__]) 