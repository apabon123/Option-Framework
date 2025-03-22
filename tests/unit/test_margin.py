"""
Unit tests for the SpanMarginCalculator.
These tests validate the margin calculations for different position types and scenarios.
"""

import pytest
import os
import sys
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the required modules
try:
    from core.margin import SpanMarginCalculator
    from core.position import OptionPosition, StockPosition
except ImportError:
    pytest.skip("Core modules not available for testing", allow_module_level=True)


class TestSpanMarginCalculator:
    """Tests for the SPAN margin calculator."""
    
    def test_initialization(self, sample_config):
        """Test that the calculator initializes correctly with config."""
        margin_calc = SpanMarginCalculator(sample_config)
        assert margin_calc is not None
        assert hasattr(margin_calc, 'config')
    
    def test_single_option_margin(self, margin_calculator):
        """Test margin calculation for a single option position."""
        # Create a test option position
        option = OptionPosition(
            symbol="SPY220318C450",
            underlying="SPY",
            quantity=-1,  # Short position
            option_type="CALL",
            strike=450,
            expiration=datetime(2022, 3, 18),
            entry_price=5.0,
            current_price=5.5,
            greeks={"delta": -0.4, "gamma": -0.05, "theta": 0.1, "vega": -0.2}
        )
        
        # Calculate margin
        margin = margin_calculator.calculate_position_margin(option)
        
        # Basic validation - margin should be positive
        assert margin > 0
        
        # Margin should be greater than the option value
        assert margin > (option.current_price * 100)
    
    def test_stock_position_margin(self, margin_calculator):
        """Test margin calculation for a stock position."""
        # Create a test stock position
        stock = StockPosition(
            symbol="SPY",
            quantity=-10,  # Short position
            entry_price=435.0,
            current_price=435.0
        )
        
        # Calculate margin
        margin = margin_calculator.calculate_position_margin(stock)
        
        # Basic validation - margin should be positive for short position
        assert margin > 0
        
        # Margin should be related to position value
        assert margin >= (stock.current_price * abs(stock.quantity) / margin_calculator.config['span']['max_leverage'])
    
    def test_portfolio_margin_with_hedging(self, margin_calculator):
        """Test margin calculation for a portfolio with hedging."""
        # Create option and stock positions that hedge each other
        option = OptionPosition(
            symbol="SPY220318C450",
            underlying="SPY",
            quantity=-1,  # Short call
            option_type="CALL",
            strike=450,
            expiration=datetime(2022, 3, 18),
            entry_price=5.0,
            current_price=5.5,
            greeks={"delta": -0.4, "gamma": -0.05, "theta": 0.1, "vega": -0.2}
        )
        
        stock = StockPosition(
            symbol="SPY",
            quantity=40,  # Long stock to approximately hedge the delta
            entry_price=435.0,
            current_price=435.0
        )
        
        # Calculate individual margins
        option_margin = margin_calculator.calculate_position_margin(option)
        stock_margin = margin_calculator.calculate_position_margin(stock)
        
        # Calculate portfolio margin
        portfolio_margin = margin_calculator.calculate_portfolio_margin([option, stock])
        
        # Portfolio margin should be less than sum of individual margins due to hedging
        assert portfolio_margin < (option_margin + stock_margin)
        
        # But still greater than zero
        assert portfolio_margin > 0
    
    # Additional test cases can be added here
    # def test_deep_itm_options(self, margin_calculator):
    #     """Test margin calculation for deep in-the-money options."""
    #     pass
    
    # def test_margin_calculation_extreme_volatility(self, margin_calculator):
    #     """Test margin calculation with extreme volatility inputs."""
    #     pass


if __name__ == "__main__":
    # This allows running the tests directly from this file
    pytest.main(["-v", __file__]) 