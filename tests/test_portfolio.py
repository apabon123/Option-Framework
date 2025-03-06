"""
Unit tests for Portfolio class
"""
import unittest
from datetime import datetime
import sys
import os
import pandas as pd
import logging

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.portfolio import Portfolio
from core.position import Position, OptionPosition


class TestPortfolio(unittest.TestCase):
    """Test the Portfolio class functionality"""
    
    def setUp(self):
        """Setup test environment"""
        # Configure logger
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger('test_logger')
        
        # Create a test portfolio
        self.portfolio = Portfolio(
            initial_capital=100000.0,
            max_position_size_pct=0.05,
            max_portfolio_delta=0.20,
            logger=self.logger
        )
        
        # Create test option data
        self.option_data = {
            'Strike': 100.0,
            'Type': 'call',
            'Expiration': datetime(2024, 12, 31),
            'MidPrice': 5.0,
            'Delta': 0.5,
            'Gamma': 0.05,
            'Theta': -0.1,
            'Vega': 0.2,
            'UnderlyingPrice': 105.0,
            'IV': 0.3,
            'DataDate': datetime(2024, 1, 1)
        }
    
    def test_add_position(self):
        """Test adding a position to the portfolio"""
        # Add a position
        position = self.portfolio.add_position(
            symbol="AAPL_240621C100",
            instrument_data=self.option_data,
            quantity=2,
            price=5.0,
            position_type='option',
            is_short=True
        )
        
        # Check that position was added correctly
        self.assertIn("AAPL_240621C100", self.portfolio.positions)
        self.assertEqual(position.contracts, 2)  # Using contracts, not quantity
        self.assertEqual(position.avg_entry_price, 5.0)
        
        # Check cash balance updated correctly (short option adds premium)
        # 2 contracts * $5.0 * 100 = $1,000 premium received
        self.assertEqual(self.portfolio.cash_balance, 100000.0 + 1000.0)
    
    def test_remove_position(self):
        """Test removing a position from the portfolio"""
        # Add a position
        self.portfolio.add_position(
            symbol="AAPL_240621C100",
            instrument_data=self.option_data,
            quantity=5,
            price=5.0,
            position_type='option',
            is_short=True
        )
        
        # Remove part of the position
        pnl = self.portfolio.remove_position(
            symbol="AAPL_240621C100",
            quantity=2,
            price=3.0  # Profit for short position
        )
        
        # Check position was updated correctly
        position = self.portfolio.positions["AAPL_240621C100"]
        self.assertEqual(position.contracts, 3)  # 5 - 2 = 3 remaining
        
        # Check PnL was calculated correctly
        # For short: (entry - exit) * quantity * 100
        # (5.0 - 3.0) * 2 * 100 = $400 profit
        self.assertEqual(pnl, 400.0)
        
        # Check cash balance was updated
        # Initial: 100000 + 5*5*100 = 102500
        # Buyback cost: 2*3*100 = 600
        # Final balance: 102500 - 600 = 101900
        self.assertEqual(self.portfolio.cash_balance, 101900.0)
    
    def test_get_portfolio_value(self):
        """Test portfolio value calculation"""
        # Add 2 positions
        self.portfolio.add_position(
            symbol="AAPL_240621C100",
            instrument_data=self.option_data,
            quantity=5,
            price=5.0,
            position_type='option',
            is_short=True
        )
        
        # Update position market data with new prices
        for symbol, position in self.portfolio.positions.items():
            updated_data = self.option_data.copy()
            updated_data['MidPrice'] = 3.0  # Price decreased (profit for short)
            position.update_market_data(updated_data)
        
        # Calculate expected portfolio value:
        # Cash: 100000 + 5*5*100 = 102500
        # Position value for shorts is negative: -(5*3*100) = -1500
        # Total: 102500 - 1500 = 101000
        expected_value = 102500.0 - 1500.0
        self.assertEqual(self.portfolio.get_portfolio_value(), expected_value)
    
    def test_update_portfolio_value(self):
        """Test the _update_portfolio_value method (fixed method)"""
        # Add a position
        self.portfolio.add_position(
            symbol="AAPL_240621C100",
            instrument_data=self.option_data,
            quantity=3,
            price=5.0,
            position_type='option',
            is_short=True
        )
        
        # Update market data for all positions
        for symbol, position in self.portfolio.positions.items():
            updated_data = self.option_data.copy()
            updated_data['MidPrice'] = 4.0
            position.update_market_data(updated_data)
        
        # Call the method we fixed
        self.portfolio._update_portfolio_value()
        
        # Check the calculated values
        self.assertEqual(self.portfolio.position_value, 3 * 4.0)
        self.assertEqual(self.portfolio.total_value, self.portfolio.cash_balance + 3 * 4.0)
        
        # Check equity history was updated
        self.assertEqual(len(self.portfolio.equity_history), 2)  # Initial + new entry


if __name__ == '__main__':
    unittest.main()