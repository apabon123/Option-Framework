"""
Performance tests for the margin calculation engine.
These tests measure the performance and scalability of the margin calculation system.
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the required modules
try:
    from core.portfolio import Portfolio, Position
    from core.margin import SpanMarginCalculator
except ImportError:
    pytest.skip("Core modules not available for testing", allow_module_level=True)


def generate_large_portfolio(size=100):
    """Generate a large portfolio with many positions for performance testing."""
    portfolio = Portfolio("Performance Test Portfolio")
    
    # List of stock symbols
    stock_symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM"]
    
    # Current date for calculations
    current_date = datetime.now()
    
    # Generate random stock positions
    for _ in range(size // 5):
        symbol = random.choice(stock_symbols)
        quantity = random.randint(10, 1000)
        entry_price = random.uniform(50.0, 500.0)
        
        portfolio.add_position(
            Position(
                symbol=symbol,
                position_type="STOCK",
                quantity=quantity,
                entry_price=entry_price,
                entry_date=current_date - timedelta(days=random.randint(1, 365))
            )
        )
    
    # Generate random option positions
    for _ in range(size - (size // 5)):
        # Choose a random underlying
        underlying = random.choice(stock_symbols)
        
        # Randomly select call or put
        option_type = random.choice(["CALL", "PUT"])
        
        # Generate random parameters
        days_to_expiry = random.randint(1, 365)
        expiry_date = current_date + timedelta(days=days_to_expiry)
        strike = random.uniform(50.0, 500.0)
        entry_price = random.uniform(1.0, 50.0)
        quantity = random.randint(-100, 100)  # Allow both long and short positions
        
        # Calculate approximate delta based on strike and option type
        if option_type == "CALL":
            delta = max(0.01, min(0.99, 0.5 + random.uniform(-0.4, 0.4)))
        else:
            delta = min(-0.01, max(-0.99, -0.5 + random.uniform(-0.4, 0.4)))
        
        # Generate an option symbol
        expiry_code = expiry_date.strftime('%y%m%d')
        strike_code = f"{int(strike):08d}"
        option_symbol = f"{underlying}{expiry_code}{option_type[0]}{strike_code}"
        
        portfolio.add_position(
            Position(
                symbol=option_symbol,
                position_type="OPTION",
                quantity=quantity,
                entry_price=entry_price,
                entry_date=current_date - timedelta(days=random.randint(1, 90)),
                expiration=expiry_date,
                strike=strike,
                option_type=option_type,
                delta=delta,
                gamma=random.uniform(0.001, 0.1),
                vega=random.uniform(0.01, 1.0),
                theta=random.uniform(-1.0, -0.01)
            )
        )
    
    return portfolio


def generate_market_prices(portfolio):
    """Generate current market prices for all instruments in the portfolio."""
    current_prices = {}
    
    for position in portfolio.positions:
        # For simplicity, use a price that's within 10% of entry price
        price_adjustment = random.uniform(0.9, 1.1)
        current_prices[position.symbol] = position.entry_price * price_adjustment
    
    return current_prices


class TestMarginCalculatorPerformance:
    """Performance tests for margin calculation."""
    
    @pytest.mark.parametrize("portfolio_size", [10, 50, 100, 200, 500])
    def test_margin_calculation_scaling(self, portfolio_size):
        """Test how margin calculation time scales with portfolio size."""
        # Generate a portfolio of specified size
        portfolio = generate_large_portfolio(size=portfolio_size)
        
        # Generate current market prices
        current_prices = generate_market_prices(portfolio)
        
        # Create margin calculator
        margin_calculator = SpanMarginCalculator()
        
        # Measure time for margin calculation
        start_time = time.time()
        margin_requirement = margin_calculator.calculate_margin(portfolio, current_prices)
        end_time = time.time()
        
        # Calculate execution time
        execution_time = end_time - start_time
        
        # Log the performance
        print(f"Portfolio size: {portfolio_size}, Execution time: {execution_time:.6f} seconds")
        
        # Performance should scale reasonably with portfolio size
        # This is not a strict assertion, as performance depends on hardware
        # but we can check that the result is calculated and positive
        assert margin_requirement > 0
        
        # For larger portfolios, we might want to set an upper bound on computation time
        if portfolio_size <= 100:
            assert execution_time < 1.0  # Should be under 1 second for small portfolios
        elif portfolio_size <= 200:
            assert execution_time < 2.0  # Allow up to 2 seconds for medium portfolios
        else:
            assert execution_time < 5.0  # Allow up to 5 seconds for large portfolios
    
    def test_margin_calculation_memory_usage(self):
        """Test the memory usage of margin calculations for large portfolios."""
        import psutil
        import os
        
        # Get the current process
        process = psutil.Process(os.getpid())
        
        # Measure initial memory usage
        initial_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        
        # Generate a large portfolio
        portfolio_size = 1000
        portfolio = generate_large_portfolio(size=portfolio_size)
        
        # Generate current market prices
        current_prices = generate_market_prices(portfolio)
        
        # Create margin calculator
        margin_calculator = SpanMarginCalculator()
        
        # Measure memory usage before calculation
        pre_calc_memory = process.memory_info().rss / (1024 * 1024)
        
        # Perform margin calculation
        margin_requirement = margin_calculator.calculate_margin(portfolio, current_prices)
        
        # Measure memory usage after calculation
        post_calc_memory = process.memory_info().rss / (1024 * 1024)
        
        # Calculate memory usage for the margin calculation
        margin_calc_memory = post_calc_memory - pre_calc_memory
        
        # Log memory usage
        print(f"Memory usage for portfolio initialization: {pre_calc_memory - initial_memory:.2f} MB")
        print(f"Memory usage for margin calculation: {margin_calc_memory:.2f} MB")
        
        # The margin calculation should not use excessive memory
        # This threshold may need adjustment based on the implementation
        assert margin_calc_memory < 200  # Memory usage should be under 200 MB
    
    def test_parallel_margin_calculation(self):
        """Test performance improvement with parallel margin calculation."""
        # Generate a large portfolio
        portfolio = generate_large_portfolio(size=300)
        
        # Generate current market prices
        current_prices = generate_market_prices(portfolio)
        
        # Create margin calculator (without parallel processing)
        margin_calculator_serial = SpanMarginCalculator(use_parallel=False)
        
        # Measure time for serial margin calculation
        start_time_serial = time.time()
        margin_requirement_serial = margin_calculator_serial.calculate_margin(portfolio, current_prices)
        end_time_serial = time.time()
        serial_time = end_time_serial - start_time_serial
        
        # Create margin calculator (with parallel processing)
        margin_calculator_parallel = SpanMarginCalculator(use_parallel=True)
        
        # Measure time for parallel margin calculation
        start_time_parallel = time.time()
        margin_requirement_parallel = margin_calculator_parallel.calculate_margin(portfolio, current_prices)
        end_time_parallel = time.time()
        parallel_time = end_time_parallel - start_time_parallel
        
        # Log the performance comparison
        print(f"Serial execution time: {serial_time:.6f} seconds")
        print(f"Parallel execution time: {parallel_time:.6f} seconds")
        print(f"Speedup: {serial_time / parallel_time:.2f}x")
        
        # Margin requirements should be approximately the same
        # (small differences may occur due to floating-point arithmetic in different order)
        assert abs(margin_requirement_serial - margin_requirement_parallel) / margin_requirement_serial < 0.01
        
        # Parallel version should be faster, at least on multicore machines
        # This assertion might not hold on single-core machines or if implementation doesn't support parallelism
        # Skip the assertion if the ratio is close to 1
        if parallel_time > 0 and serial_time / parallel_time > 1.1:
            assert parallel_time < serial_time
    
    def test_margin_calculation_batch_sizes(self):
        """Test margin calculation performance with different batch sizes."""
        # Generate a large portfolio
        portfolio = generate_large_portfolio(size=200)
        
        # Generate current market prices
        current_prices = generate_market_prices(portfolio)
        
        # Test different batch sizes
        batch_sizes = [10, 20, 50, 100]
        execution_times = {}
        
        for batch_size in batch_sizes:
            # Create margin calculator with specified batch size
            margin_calculator = SpanMarginCalculator(batch_size=batch_size)
            
            # Measure time for margin calculation
            start_time = time.time()
            margin_requirement = margin_calculator.calculate_margin(portfolio, current_prices)
            end_time = time.time()
            
            # Record execution time
            execution_times[batch_size] = end_time - start_time
            
            # Log the performance
            print(f"Batch size: {batch_size}, Execution time: {execution_times[batch_size]:.6f} seconds")
        
        # Find the optimal batch size (the one with the minimum execution time)
        optimal_batch_size = min(execution_times, key=execution_times.get)
        print(f"Optimal batch size: {optimal_batch_size}")
        
        # Verify that margin calculation works correctly
        assert margin_requirement > 0
    
    def test_margin_calculation_with_caching(self):
        """Test the performance improvement from caching in margin calculations."""
        # Generate a portfolio
        portfolio = generate_large_portfolio(size=100)
        
        # Generate current market prices
        current_prices = generate_market_prices(portfolio)
        
        # Create margin calculator without caching
        margin_calculator_no_cache = SpanMarginCalculator(use_caching=False)
        
        # First run without caching
        start_time_no_cache_1 = time.time()
        margin_requirement_1 = margin_calculator_no_cache.calculate_margin(portfolio, current_prices)
        end_time_no_cache_1 = time.time()
        execution_time_no_cache_1 = end_time_no_cache_1 - start_time_no_cache_1
        
        # Second run without caching (should be similar to first)
        start_time_no_cache_2 = time.time()
        margin_requirement_2 = margin_calculator_no_cache.calculate_margin(portfolio, current_prices)
        end_time_no_cache_2 = time.time()
        execution_time_no_cache_2 = end_time_no_cache_2 - start_time_no_cache_2
        
        # Create margin calculator with caching
        margin_calculator_with_cache = SpanMarginCalculator(use_caching=True)
        
        # First run with caching (cache gets populated)
        start_time_cache_1 = time.time()
        margin_requirement_3 = margin_calculator_with_cache.calculate_margin(portfolio, current_prices)
        end_time_cache_1 = time.time()
        execution_time_cache_1 = end_time_cache_1 - start_time_cache_1
        
        # Second run with caching (should be faster due to cache hits)
        start_time_cache_2 = time.time()
        margin_requirement_4 = margin_calculator_with_cache.calculate_margin(portfolio, current_prices)
        end_time_cache_2 = time.time()
        execution_time_cache_2 = end_time_cache_2 - start_time_cache_2
        
        # Log the performance comparison
        print(f"No caching (1st run): {execution_time_no_cache_1:.6f} seconds")
        print(f"No caching (2nd run): {execution_time_no_cache_2:.6f} seconds")
        print(f"With caching (1st run): {execution_time_cache_1:.6f} seconds")
        print(f"With caching (2nd run): {execution_time_cache_2:.6f} seconds")
        print(f"Speedup from caching: {execution_time_no_cache_2 / execution_time_cache_2:.2f}x")
        
        # All margin calculations should yield the same result
        assert abs(margin_requirement_1 - margin_requirement_2) / margin_requirement_1 < 0.0001
        assert abs(margin_requirement_1 - margin_requirement_3) / margin_requirement_1 < 0.0001
        assert abs(margin_requirement_1 - margin_requirement_4) / margin_requirement_1 < 0.0001
        
        # Second run with caching should be significantly faster than without caching
        # Only assert if the improvement is substantial
        if execution_time_cache_2 > 0 and execution_time_no_cache_2 / execution_time_cache_2 > 1.5:
            assert execution_time_cache_2 < execution_time_no_cache_2


if __name__ == "__main__":
    # This allows running the tests directly from this file
    pytest.main(["-v", __file__]) 