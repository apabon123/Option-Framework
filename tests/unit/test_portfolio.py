"""
Unit tests for the portfolio management module.
These tests validate position tracking and portfolio operations.
"""

import pytest
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the required modules
try:
    from core.portfolio import Portfolio, Position
except ImportError:
    pytest.skip("Core modules not available for testing", allow_module_level=True)


class TestPosition:
    """Tests for the Position class."""
    
    def test_position_initialization(self):
        """Test that positions are initialized correctly."""
        # Test stock position
        stock_pos = Position(
            symbol="AAPL",
            position_type="STOCK",
            quantity=100,
            entry_price=150.0,
            entry_date=datetime.now()
        )
        
        assert stock_pos.symbol == "AAPL"
        assert stock_pos.position_type == "STOCK"
        assert stock_pos.quantity == 100
        assert stock_pos.entry_price == 150.0
        
        # Test option position
        option_pos = Position(
            symbol="AAPL220121C00160000",  # Option symbol
            position_type="OPTION",
            quantity=10,
            entry_price=5.0,
            entry_date=datetime.now(),
            expiration=datetime(2022, 1, 21),
            strike=160.0,
            option_type="CALL"
        )
        
        assert option_pos.symbol == "AAPL220121C00160000"
        assert option_pos.position_type == "OPTION"
        assert option_pos.quantity == 10
        assert option_pos.entry_price == 5.0
        assert option_pos.expiration == datetime(2022, 1, 21)
        assert option_pos.strike == 160.0
        assert option_pos.option_type == "CALL"
    
    def test_position_market_value(self):
        """Test market value calculation for positions."""
        pos = Position(
            symbol="AAPL",
            position_type="STOCK",
            quantity=100,
            entry_price=150.0,
            entry_date=datetime.now()
        )
        
        # Calculate market value with current price
        market_value = pos.calculate_market_value(current_price=160.0)
        assert market_value == 100 * 160.0
        
        # Test with negative quantity (short position)
        short_pos = Position(
            symbol="AAPL",
            position_type="STOCK",
            quantity=-50,
            entry_price=150.0,
            entry_date=datetime.now()
        )
        
        short_market_value = short_pos.calculate_market_value(current_price=160.0)
        assert short_market_value == -50 * 160.0
    
    def test_position_pnl(self):
        """Test P&L calculations for positions."""
        pos = Position(
            symbol="AAPL",
            position_type="STOCK",
            quantity=100,
            entry_price=150.0,
            entry_date=datetime.now()
        )
        
        # Calculate unrealized P&L
        unrealized_pnl = pos.calculate_unrealized_pnl(current_price=160.0)
        assert unrealized_pnl == 100 * (160.0 - 150.0)
        
        # Calculate percentage P&L
        pct_pnl = pos.calculate_percentage_pnl(current_price=160.0)
        assert pct_pnl == ((160.0 - 150.0) / 150.0) * 100
        assert abs(pct_pnl - 6.67) < 0.01
        
        # Test short position P&L
        short_pos = Position(
            symbol="AAPL",
            position_type="STOCK",
            quantity=-50,
            entry_price=150.0,
            entry_date=datetime.now()
        )
        
        short_pnl = short_pos.calculate_unrealized_pnl(current_price=140.0)
        assert short_pnl == -50 * (140.0 - 150.0)
        assert short_pnl > 0  # Profit on short position when price decreases
    
    def test_option_expiration(self):
        """Test option expiration checks."""
        # Create an expired option
        expired_option = Position(
            symbol="AAPL220121C00160000",
            position_type="OPTION",
            quantity=10,
            entry_price=5.0,
            entry_date=datetime(2022, 1, 1),
            expiration=datetime(2022, 1, 21),
            strike=160.0,
            option_type="CALL"
        )
        
        # Create a current option
        future_date = datetime.now() + timedelta(days=30)
        current_option = Position(
            symbol="AAPL220221C00160000",
            position_type="OPTION",
            quantity=10,
            entry_price=5.0,
            entry_date=datetime.now(),
            expiration=future_date,
            strike=160.0,
            option_type="CALL"
        )
        
        assert expired_option.is_expired(as_of_date=datetime.now())
        assert not current_option.is_expired(as_of_date=datetime.now())
    
    def test_position_delta(self):
        """Test position delta calculation."""
        # Stock position should have delta of 1.0 (or -1.0 for short)
        stock_pos = Position(
            symbol="AAPL",
            position_type="STOCK",
            quantity=100,
            entry_price=150.0,
            entry_date=datetime.now()
        )
        
        # Option position with specified delta
        option_pos = Position(
            symbol="AAPL220121C00160000",
            position_type="OPTION",
            quantity=10,
            entry_price=5.0,
            entry_date=datetime.now(),
            expiration=datetime.now() + timedelta(days=30),
            strike=160.0,
            option_type="CALL",
            delta=0.5
        )
        
        assert stock_pos.calculate_position_delta() == 100.0
        assert option_pos.calculate_position_delta() == 10 * 0.5
        
        # Short option position
        short_option = Position(
            symbol="AAPL220121P00140000",
            position_type="OPTION",
            quantity=-5,
            entry_price=3.0,
            entry_date=datetime.now(),
            expiration=datetime.now() + timedelta(days=30),
            strike=140.0,
            option_type="PUT",
            delta=-0.3
        )
        
        assert short_option.calculate_position_delta() == -5 * (-0.3)


class TestPortfolio:
    """Tests for the Portfolio class."""
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio with multiple positions."""
        portfolio = Portfolio("Test Portfolio")
        
        # Add some stock positions
        portfolio.add_position(
            Position(
                symbol="AAPL",
                position_type="STOCK",
                quantity=100,
                entry_price=150.0,
                entry_date=datetime.now()
            )
        )
        
        portfolio.add_position(
            Position(
                symbol="MSFT",
                position_type="STOCK",
                quantity=50,
                entry_price=250.0,
                entry_date=datetime.now()
            )
        )
        
        # Add some option positions
        portfolio.add_position(
            Position(
                symbol="AAPL220121C00160000",
                position_type="OPTION",
                quantity=10,
                entry_price=5.0,
                entry_date=datetime.now(),
                expiration=datetime.now() + timedelta(days=30),
                strike=160.0,
                option_type="CALL",
                delta=0.5
            )
        )
        
        portfolio.add_position(
            Position(
                symbol="MSFT220121P00240000",
                position_type="OPTION",
                quantity=-5,
                entry_price=4.0,
                entry_date=datetime.now(),
                expiration=datetime.now() + timedelta(days=30),
                strike=240.0,
                option_type="PUT",
                delta=-0.4
            )
        )
        
        return portfolio
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        portfolio = Portfolio("Test Portfolio")
        assert portfolio.name == "Test Portfolio"
        assert len(portfolio.positions) == 0
    
    def test_add_position(self, sample_portfolio):
        """Test adding positions to portfolio."""
        assert len(sample_portfolio.positions) == 4
        
        # Try adding duplicate position (should merge)
        new_aapl = Position(
            symbol="AAPL",
            position_type="STOCK",
            quantity=50,
            entry_price=155.0,
            entry_date=datetime.now()
        )
        
        sample_portfolio.add_position(new_aapl)
        
        # Should still have 4 positions, but AAPL quantity should be updated
        assert len(sample_portfolio.positions) == 4
        aapl_pos = sample_portfolio.get_position_by_symbol("AAPL", "STOCK")
        assert aapl_pos.quantity == 150  # 100 + 50
        
        # Entry price should be weighted average
        expected_entry = (100 * 150.0 + 50 * 155.0) / 150
        assert abs(aapl_pos.entry_price - expected_entry) < 0.01
    
    def test_remove_position(self, sample_portfolio):
        """Test removing positions from portfolio."""
        initial_count = len(sample_portfolio.positions)
        
        # Remove a position
        sample_portfolio.remove_position("MSFT", "STOCK")
        
        assert len(sample_portfolio.positions) == initial_count - 1
        assert sample_portfolio.get_position_by_symbol("MSFT", "STOCK") is None
    
    def test_portfolio_value(self, sample_portfolio):
        """Test portfolio value calculation."""
        # Set current prices
        current_prices = {
            "AAPL": 160.0,
            "MSFT": 260.0,
            "AAPL220121C00160000": 7.0,
            "MSFT220121P00240000": 3.0
        }
        
        portfolio_value = sample_portfolio.calculate_total_value(current_prices)
        
        expected_value = (
            100 * 160.0 +  # AAPL stock
            50 * 260.0 +   # MSFT stock
            10 * 7.0 +     # AAPL calls
            -5 * 3.0       # Short MSFT puts
        )
        
        assert abs(portfolio_value - expected_value) < 0.01
    
    def test_portfolio_pnl(self, sample_portfolio):
        """Test portfolio P&L calculation."""
        # Set current prices
        current_prices = {
            "AAPL": 160.0,
            "MSFT": 260.0,
            "AAPL220121C00160000": 7.0,
            "MSFT220121P00240000": 3.0
        }
        
        portfolio_pnl = sample_portfolio.calculate_total_pnl(current_prices)
        
        expected_pnl = (
            100 * (160.0 - 150.0) +  # AAPL stock
            50 * (260.0 - 250.0) +   # MSFT stock
            10 * (7.0 - 5.0) +       # AAPL calls
            -5 * (3.0 - 4.0)         # Short MSFT puts
        )
        
        assert abs(portfolio_pnl - expected_pnl) < 0.01
    
    def test_portfolio_delta(self, sample_portfolio):
        """Test portfolio delta calculation."""
        portfolio_delta = sample_portfolio.calculate_portfolio_delta()
        
        expected_delta = (
            100 +           # AAPL stock
            50 +            # MSFT stock
            10 * 0.5 +      # AAPL calls
            -5 * (-0.4)     # Short MSFT puts
        )
        
        assert abs(portfolio_delta - expected_delta) < 0.01
    
    def test_portfolio_allocation(self, sample_portfolio):
        """Test portfolio allocation calculation."""
        # Set current prices
        current_prices = {
            "AAPL": 160.0,
            "MSFT": 260.0,
            "AAPL220121C00160000": 7.0,
            "MSFT220121P00240000": 3.0
        }
        
        allocation = sample_portfolio.calculate_allocation(current_prices)
        
        total_value = (
            100 * 160.0 +  # AAPL stock
            50 * 260.0 +   # MSFT stock
            10 * 7.0 +     # AAPL calls
            -5 * 3.0       # Short MSFT puts
        )
        
        expected_allocation = {
            "AAPL": (100 * 160.0) / total_value * 100,
            "MSFT": (50 * 260.0) / total_value * 100,
            "AAPL220121C00160000": (10 * 7.0) / total_value * 100,
            "MSFT220121P00240000": (-5 * 3.0) / total_value * 100
        }
        
        for symbol, alloc in allocation.items():
            assert abs(alloc - expected_allocation[symbol]) < 0.01
    
    def test_filter_positions(self, sample_portfolio):
        """Test filtering positions by type."""
        stocks = sample_portfolio.filter_positions_by_type("STOCK")
        options = sample_portfolio.filter_positions_by_type("OPTION")
        
        assert len(stocks) == 2
        assert len(options) == 2
        
        # Test filtering by symbol
        aapl_positions = sample_portfolio.filter_positions_by_symbol("AAPL")
        assert len(aapl_positions) == 2  # AAPL stock and AAPL option


if __name__ == "__main__":
    # This allows running the tests directly from this file
    pytest.main(["-v", __file__]) 