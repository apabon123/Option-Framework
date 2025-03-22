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
    
    def test_deep_itm_options(self, margin_calculator):
        """Test margin calculation for deep in-the-money options."""
        # Create a deep ITM call option
        deep_itm_call = OptionPosition(
            symbol="SPY220318C350",
            underlying="SPY",
            quantity=-1,  # Short position
            option_type="CALL",
            strike=350,
            expiration=datetime(2022, 3, 18),
            entry_price=85.0,  # Deep ITM - very expensive
            current_price=86.5,
            greeks={"delta": -0.95, "gamma": -0.01, "theta": 0.03, "vega": -0.05}
        )
        
        # Calculate margin
        margin = margin_calculator.calculate_position_margin(deep_itm_call)
        
        # For deep ITM options, margin should be close to intrinsic value
        underlying_price = 435.0  # Assuming current SPY price is 435
        intrinsic_value = (underlying_price - deep_itm_call.strike) * 100  # (Underlying price - strike) * 100
        assert margin >= intrinsic_value * 0.9  # Allow for some flexibility in calculation
        
        # But not excessive (not more than max_margin_to_premium_ratio times premium)
        max_margin = deep_itm_call.current_price * 100 * margin_calculator.config['span']['max_margin_to_premium_ratio']  # Premium * 100 * max ratio
        assert margin <= max_margin
    
    def test_margin_calculation_extreme_volatility(self, margin_calculator, sample_config):
        """Test margin calculation with extreme volatility inputs."""
        # Create a margin calculator with extreme volatility settings
        extreme_config = sample_config.copy()
        extreme_config['span']['price_move_pct'] = 0.15  # 15% price move - extreme scenario
        extreme_config['span']['vol_shift_pct'] = 0.5    # 50% volatility shift - extreme scenario
        
        extreme_margin_calc = SpanMarginCalculator(extreme_config)
        
        # Create a high-gamma option position
        high_gamma_option = OptionPosition(
            symbol="SPY220318C435",
            underlying="SPY",
            quantity=-1,
            option_type="CALL",
            strike=435,
            expiration=datetime(2022, 3, 18),
            entry_price=12.0,
            current_price=12.5,
            greeks={"delta": -0.5, "gamma": -0.10, "theta": 0.15, "vega": -0.3}
        )
        
        # Calculate margin under normal and extreme conditions
        normal_margin = margin_calculator.calculate_position_margin(high_gamma_option)
        extreme_margin = extreme_margin_calc.calculate_position_margin(high_gamma_option)
        
        # Extreme volatility should result in higher margin
        assert extreme_margin > normal_margin
        
        # The increase should be significant due to gamma exposure
        assert extreme_margin > normal_margin * 1.2  # At least 20% higher
        
    def test_zero_delta_positions(self, margin_calculator):
        """Test margin for delta-neutral positions."""
        # Create a delta-neutral position (iron condor)
        short_call = OptionPosition(
            symbol="SPY220318C450",
            underlying="SPY",
            quantity=-1,
            option_type="CALL",
            strike=450,
            expiration=datetime(2022, 3, 18),
            entry_price=5.0,
            current_price=5.5,
            greeks={"delta": -0.3, "gamma": -0.05, "theta": 0.1, "vega": -0.2}
        )
        
        long_call = OptionPosition(
            symbol="SPY220318C460",
            underlying="SPY",
            quantity=1,
            option_type="CALL",
            strike=460,
            expiration=datetime(2022, 3, 18),
            entry_price=3.0,
            current_price=3.3,
            greeks={"delta": 0.2, "gamma": 0.03, "theta": -0.08, "vega": 0.15}
        )
        
        short_put = OptionPosition(
            symbol="SPY220318P400",
            underlying="SPY",
            quantity=-1,
            option_type="PUT",
            strike=400,
            expiration=datetime(2022, 3, 18),
            entry_price=4.0,
            current_price=4.2,
            greeks={"delta": 0.2, "gamma": 0.03, "theta": 0.05, "vega": 0.12}
        )
        
        long_put = OptionPosition(
            symbol="SPY220318P390",
            underlying="SPY",
            quantity=1,
            option_type="PUT",
            strike=390,
            expiration=datetime(2022, 3, 18),
            entry_price=2.5,
            current_price=2.7,
            greeks={"delta": -0.1, "gamma": 0.02, "theta": -0.03, "vega": 0.1}
        )
        
        # Calculate margin for the iron condor
        positions = [short_call, long_call, short_put, long_put]
        margin = margin_calculator.calculate_portfolio_margin(positions)
        
        # Verify margin is less than sum of individual margins
        individual_margins = sum(margin_calculator.calculate_position_margin(p) for p in positions)
        assert margin < individual_margins
        
        # Iron condor has defined risk (difference between short and long strikes)
        call_spread_risk = (long_call.strike - short_call.strike) * 100
        put_spread_risk = (short_put.strike - long_put.strike) * 100
        max_risk = max(call_spread_risk, put_spread_risk)
        
        # Margin should be at least the maximum risk
        assert margin >= max_risk * 0.9  # Allow for some flexibility


if __name__ == "__main__":
    # This allows running the tests directly from this file
    pytest.main(["-v", __file__]) 