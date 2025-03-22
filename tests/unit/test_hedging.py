"""
Unit tests for the hedging system.
These tests validate the delta hedging strategy and implementation.
"""

import pytest
import os
import sys
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the required modules
try:
    from core.hedging import DeltaHedger
    from core.portfolio import Portfolio
    from core.position import OptionPosition, StockPosition
except ImportError:
    pytest.skip("Core modules not available for testing", allow_module_level=True)


class TestDeltaHedger:
    """Tests for the delta hedging system."""
    
    def test_initialization(self):
        """Test initializing the delta hedger with various configs."""
        # Test with ratio mode
        ratio_hedger = DeltaHedger({
            "mode": "ratio",
            "target_delta_ratio": 0.1,
            "delta_tolerance": 0.05,
            "hedge_symbol": "SPY",
            "hedge_with_underlying": True
        })
        
        assert ratio_hedger.mode == "ratio"
        assert ratio_hedger.target_delta_ratio == 0.1
        assert ratio_hedger.delta_tolerance == 0.05
        assert ratio_hedger.hedge_symbol == "SPY"
        assert ratio_hedger.hedge_with_underlying is True
        
        # Test with constant mode
        constant_hedger = DeltaHedger({
            "mode": "constant",
            "target_delta": 5000,
            "delta_tolerance": 1000,
            "hedge_symbol": "SPY",
            "hedge_with_underlying": True
        })
        
        assert constant_hedger.mode == "constant"
        assert constant_hedger.target_delta == 5000
        assert constant_hedger.delta_tolerance == 1000
        assert constant_hedger.hedge_symbol == "SPY"
        assert constant_hedger.hedge_with_underlying is True
    
    def test_calculate_hedge_adjustment_ratio_mode(self):
        """Test calculating hedge adjustment amount in ratio mode."""
        # Create a hedger with ratio mode
        hedger = DeltaHedger({
            "mode": "ratio",
            "target_delta_ratio": 0.0,  # Target neutral
            "delta_tolerance": 0.1,     # 10% tolerance
            "hedge_symbol": "SPY",
            "hedge_with_underlying": True
        })
        
        # Create portfolio with some delta exposure
        portfolio = Portfolio(initial_capital=100000)
        
        # Mock methods to simulate portfolio state
        portfolio.get_portfolio_delta = lambda: 5000  # +5000 delta
        portfolio.get_net_liquidation_value = lambda: 100000  # $100,000 NLV
        
        # Calculate hedge adjustment
        hedge_qty = hedger._calculate_hedge_adjustment(portfolio, 435.0)
        
        # Should be negative to offset positive delta
        assert hedge_qty < 0
        
        # Should offset all the delta (5000 delta = ~11.5 shares at $435.0 per share)
        expected_shares = -5000 / 100 / 435.0
        assert abs(hedge_qty - expected_shares) < 0.01
        
        # Test with a portfolio that is within tolerance
        portfolio.get_portfolio_delta = lambda: 800  # Only +800 delta (0.8% of NLV)
        
        # Should return 0 as we're within tolerance (10% of NLV = 10,000 delta)
        hedge_qty = hedger._calculate_hedge_adjustment(portfolio, 435.0)
        assert hedge_qty == 0
    
    def test_calculate_hedge_adjustment_constant_mode(self):
        """Test calculating hedge adjustment amount in constant mode."""
        # Create a hedger with constant mode
        hedger = DeltaHedger({
            "mode": "constant",
            "target_delta": 2000,  # Target +2000 delta
            "delta_tolerance": 1000,     # 1000 delta tolerance
            "hedge_symbol": "SPY",
            "hedge_with_underlying": True
        })
        
        # Create portfolio with some delta exposure
        portfolio = Portfolio(initial_capital=100000)
        
        # Mock portfolio with delta of 4000 (above target+tolerance)
        portfolio.get_portfolio_delta = lambda: 4000
        
        # Calculate hedge adjustment
        hedge_qty = hedger._calculate_hedge_adjustment(portfolio, 435.0)
        
        # Should be negative to reduce delta
        assert hedge_qty < 0
        
        # Should hedge down to upper boundary (target + tolerance = 3000)
        expected_shares = -(4000 - 3000) / 100 / 435.0
        assert abs(hedge_qty - expected_shares) < 0.01
        
        # Test with a portfolio that is below lower boundary (target - tolerance = 1000)
        portfolio.get_portfolio_delta = lambda: 500
        
        # Should be positive to increase delta
        hedge_qty = hedger._calculate_hedge_adjustment(portfolio, 435.0)
        assert hedge_qty > 0
        
        # Should hedge up to lower boundary (target - tolerance = 1000)
        expected_shares = (1000 - 500) / 100 / 435.0
        assert abs(hedge_qty - expected_shares) < 0.01
        
        # Test with a portfolio that is within tolerance
        portfolio.get_portfolio_delta = lambda: 2500  # Within tolerance
        
        # Should return 0 as we're within tolerance
        hedge_qty = hedger._calculate_hedge_adjustment(portfolio, 435.0)
        assert hedge_qty == 0
    
    def test_boundary_approach(self):
        """Test the 'hedge to boundary' approach."""
        # Create a hedger with ratio mode
        hedger = DeltaHedger({
            "mode": "ratio",
            "target_delta_ratio": 0.0,  # Target neutral
            "delta_tolerance": 0.2,     # 20% tolerance
            "hedge_symbol": "SPY",
            "hedge_with_underlying": True
        })
        
        # Create portfolio with a large delta exposure exceeding the upper boundary
        portfolio = Portfolio(initial_capital=100000)
        portfolio.get_portfolio_delta = lambda: 25000  # +25000 delta
        portfolio.get_net_liquidation_value = lambda: 100000  # $100,000 NLV
        
        # Upper boundary is 0.0 + 0.2 = 0.2, which is 20,000 delta for a $100,000 portfolio
        # So we should hedge down to 20,000, not all the way to 0
        hedge_qty = hedger._calculate_hedge_adjustment(portfolio, 435.0)
        
        # Should be negative to reduce delta
        assert hedge_qty < 0
        
        # Calculate expected adjustment to reach upper boundary
        expected_shares = -(25000 - 20000) / 100 / 435.0
        assert abs(hedge_qty - expected_shares) < 0.01
        
        # Now test with delta below the lower boundary
        portfolio.get_portfolio_delta = lambda: -25000  # -25000 delta
        
        # Lower boundary is 0.0 - 0.2 = -0.2, which is -20,000 delta
        # So we should hedge up to -20,000, not all the way to 0
        hedge_qty = hedger._calculate_hedge_adjustment(portfolio, 435.0)
        
        # Should be positive to increase delta
        assert hedge_qty > 0
        
        # Calculate expected adjustment to reach lower boundary
        expected_shares = (-20000 - (-25000)) / 100 / 435.0
        assert abs(hedge_qty - expected_shares) < 0.01
    
    def test_hedging_with_options(self):
        """Test hedging with options instead of the underlying."""
        # Create a hedger that uses options
        hedger = DeltaHedger({
            "mode": "ratio",
            "target_delta_ratio": 0.0,  # Target neutral
            "delta_tolerance": 0.1,     # 10% tolerance
            "hedge_symbol": "SPY",
            "hedge_with_underlying": False,
            "hedge_option_delta": 0.7   # Use options with ~0.7 delta
        })
        
        # Create portfolio with some delta exposure
        portfolio = Portfolio(initial_capital=100000)
        portfolio.get_portfolio_delta = lambda: 5000  # +5000 delta
        portfolio.get_net_liquidation_value = lambda: 100000  # $100,000 NLV
        
        # Create mock option chain
        option_chain = [
            {"symbol": "SPY220318C430", "strike": 430, "delta": 0.65, "option_type": "CALL"},
            {"symbol": "SPY220318P440", "strike": 440, "delta": -0.35, "option_type": "PUT"},
            {"symbol": "SPY220318C420", "strike": 420, "delta": 0.75, "option_type": "CALL"},
            {"symbol": "SPY220318P450", "strike": 450, "delta": -0.25, "option_type": "PUT"}
        ]
        
        # Mock the find_hedge_option method to return a specific option
        hedger.find_hedge_option = lambda chain, target_delta: next(
            (opt for opt in chain if abs(opt["delta"] - target_delta) < 0.1), None
        )
        
        # Calculate hedge adjustment
        hedge_option = hedger.calculate_option_hedge(portfolio, 435.0, option_chain)
        
        # Should select a PUT option (negative delta) to offset positive portfolio delta
        assert hedge_option["option_type"] == "PUT"
        assert hedge_option["delta"] < 0
        
        # Quantity should offset most of the portfolio delta
        expected_quantity = int(5000 / (abs(hedge_option["delta"]) * 100))
        assert hedge_option["quantity"] == expected_quantity


if __name__ == "__main__":
    # This allows running the tests directly from this file
    pytest.main(["-v", __file__]) 