"""
Unit tests for Position and OptionPosition classes
"""
import unittest
from datetime import datetime
import sys
import os
import pandas as pd
import logging

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.position import Position, OptionPosition


class TestPosition(unittest.TestCase):
    """Test the Position class functionality"""
    
    def setUp(self):
        """Setup test environment"""
        # Configure logger
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger('test_logger')
        
        # Create test instrument data
        self.test_data = {
            'Strike': 100.0,
            'Type': 'call',
            'Expiration': datetime(2024, 12, 31),
            'MidPrice': 5.0,
            'Delta': 0.5,
            'Gamma': 0.05,
            'Theta': -0.1,
            'Vega': 0.2,
            'UnderlyingPrice': 105.0,
            'DataDate': datetime(2024, 1, 1)
        }
        
        # Create a test position
        self.position = Position(
            symbol="AAPL_240621C100",
            instrument_data=self.test_data,
            initial_contracts=0,
            is_short=True,
            logger=self.logger
        )
    
    def test_add_contracts(self):
        """Test adding contracts to a position"""
        # Add 2 contracts at $5.0
        self.position.add_contracts(2, 5.0)
        
        # Check that contracts were added correctly
        self.assertEqual(self.position.contracts, 2)
        self.assertEqual(self.position.avg_entry_price, 5.0)
        
        # Add 3 more contracts at $6.0
        self.position.add_contracts(3, 6.0)
        
        # Check that weighted average is correct (2*5 + 3*6)/5 = 5.6
        self.assertEqual(self.position.contracts, 5)
        self.assertAlmostEqual(self.position.avg_entry_price, 5.6)
    
    def test_remove_contracts(self):
        """Test removing contracts from a position"""
        # Add contracts first
        self.position.add_contracts(5, 5.0)
        self.assertEqual(self.position.contracts, 5)
        
        # Remove 2 contracts at $4.0 (a profit for short position)
        pnl = self.position.remove_contracts(2, 4.0)
        
        # Check that contracts were removed correctly
        self.assertEqual(self.position.contracts, 3)
        self.assertEqual(self.position.avg_entry_price, 5.0)  # Avg price stays the same
        
        # For a short position selling at 5.0 and buying back at 4.0, 
        # the profit is (5.0 - 4.0) * 2 * 100 = $200
        self.assertEqual(pnl, 200.0)
        self.assertEqual(self.position.realized_pnl, 200.0)
    
    def test_update_market_data(self):
        """Test updating position with market data"""
        # Add contracts
        self.position.add_contracts(3, 5.0)
        
        # Create updated market data with lower price (profit for short)
        updated_data = self.test_data.copy()
        updated_data['MidPrice'] = 3.0
        
        # Update market data
        self.position.update_market_data(updated_data)
        
        # Check that current price was updated
        self.assertEqual(self.position.current_price, 3.0)
        
        # For short position, profit is (entry - current) * contracts * 100
        # (5.0 - 3.0) * 3 * 100 = $600 profit
        self.assertEqual(self.position.unrealized_pnl, 600.0)
    
    def test_get_greeks(self):
        """Test getting Greeks from a position"""
        # Add contracts
        self.position.add_contracts(2, 5.0)
        
        # Update market data to ensure all Greeks are set
        self.position.update_market_data(self.test_data)
        
        # Get Greeks
        greeks = self.position.get_greeks()
        
        # For a short position, Greeks should be negative
        # Delta: -0.5 * 2 = -1.0
        self.assertAlmostEqual(greeks['delta'], -1.0)
        
        # Dollar delta: -0.5 * 2 * 105 * 100 = -$10,500
        self.assertAlmostEqual(greeks['dollar_delta'], -10500.0)
    
    def test_zero_entry_price_handling(self):
        """Test that operations with zero entry price don't cause errors"""
        # Create a position with zero entry price
        zero_price_position = Position(
            symbol="ZERO_PRICE",
            instrument_data=self.test_data,
            initial_contracts=0,
            is_short=True,
            logger=self.logger
        )
        
        # Set avg_entry_price to zero
        zero_price_position.avg_entry_price = 0
        zero_price_position.contracts = 1
        
        # Update with market data
        zero_price_position.update_market_data(self.test_data)
        
        # Test profit percentage calculation (similar to example_strategy.py)
        if zero_price_position.avg_entry_price <= 0:
            profit_pct = 0
        elif zero_price_position.is_short:
            profit_pct = (zero_price_position.avg_entry_price - zero_price_position.current_price) / zero_price_position.avg_entry_price
        else:
            profit_pct = (zero_price_position.current_price - zero_price_position.avg_entry_price) / zero_price_position.avg_entry_price
        
        # Should not raise exception and should return 0
        self.assertEqual(profit_pct, 0)


class TestOptionPosition(unittest.TestCase):
    """Test the OptionPosition class functionality"""
    
    def setUp(self):
        """Setup test environment"""
        # Configure logger
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger('test_logger')
        
        # Create test option data
        self.test_data = {
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
        
        # Create a test option position
        self.option = OptionPosition(
            option_symbol="AAPL_240621C100",
            option_data=self.test_data,
            initial_contracts=0,
            is_short=True,
            logger=self.logger
        )
    
    def test_option_specific_properties(self):
        """Test option-specific properties"""
        # Check that option-specific properties are set
        self.assertEqual(self.option.strike, 100.0)
        self.assertEqual(self.option.type, 'call')
        self.assertEqual(self.option.implied_volatility, 0.3)
    
    def test_is_itm(self):
        """Test in-the-money calculation"""
        # Call option with strike 100 and underlying at 105 is ITM
        self.assertTrue(self.option.is_itm())
        
        # Update to make it OTM
        data_otm = self.test_data.copy()
        data_otm['UnderlyingPrice'] = 95.0
        self.option.update_market_data(data_otm)
        
        # Now should be OTM
        self.assertFalse(self.option.is_itm())
        
        # Test with a put option
        put_data = self.test_data.copy()
        put_data['Type'] = 'put'
        put_data['UnderlyingPrice'] = 95.0
        
        put_option = OptionPosition(
            option_symbol="AAPL_240621P100",
            option_data=put_data,
            initial_contracts=0,
            is_short=True,
            logger=self.logger
        )
        
        # Put with strike 100 and underlying at 95 is ITM
        self.assertTrue(put_option.is_itm())
    
    def test_calculate_moneyness(self):
        """Test moneyness calculation"""
        # Moneyness for a call with underlying at 105 and strike at 100
        # is 105/100 = 1.05
        self.assertAlmostEqual(self.option.calculate_moneyness(), 1.05)
    
    def test_zero_entry_price_handling(self):
        """Test that operations with zero entry price don't cause errors"""
        # Create a position with zero entry price
        zero_price_option = OptionPosition(
            option_symbol="ZERO_PRICE",
            option_data=self.test_data,
            initial_contracts=0,
            is_short=True,
            logger=self.logger
        )
        
        # Set avg_entry_price to zero
        zero_price_option.avg_entry_price = 0
        zero_price_option.contracts = 1
        
        # Update with market data
        zero_price_option.update_market_data(self.test_data)
        
        # Test profit percentage calculation (similar to example_strategy.py)
        if zero_price_option.avg_entry_price <= 0:
            profit_pct = 0
        elif zero_price_option.is_short:
            profit_pct = (zero_price_option.avg_entry_price - zero_price_option.current_price) / zero_price_option.avg_entry_price
        else:
            profit_pct = (zero_price_option.current_price - zero_price_option.avg_entry_price) / zero_price_option.avg_entry_price
        
        # Should not raise exception and should return 0
        self.assertEqual(profit_pct, 0)


if __name__ == '__main__':
    unittest.main()