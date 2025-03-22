"""
Unit tests for the OptionDataManager.
These tests validate the loading, filtering, and processing of options data.
"""

import pytest
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the required modules
try:
    from data_managers.option_data_manager import OptionDataManager
except ImportError:
    pytest.skip("Data manager modules not available for testing", allow_module_level=True)


class TestOptionDataManager:
    """Tests for the OptionDataManager."""
    
    def test_initialization(self):
        """Test that the manager initializes correctly."""
        manager = OptionDataManager()
        assert manager is not None
        
        # Test with config
        config = {"timezone": "America/New_York", "cache_enabled": True}
        manager_with_config = OptionDataManager(config)
        assert manager_with_config is not None
    
    def test_load_data(self, sample_option_data, tmp_path):
        """Test loading data from file."""
        # Save sample data to a temporary file
        data_file = os.path.join(tmp_path, "test_options.csv")
        sample_option_data.to_csv(data_file, index=False)
        
        # Create manager and load data
        manager = OptionDataManager()
        result = manager.load_data(data_file)
        
        # Verify load was successful
        assert result is True
        
        # Check data was loaded correctly
        assert hasattr(manager, 'data')
        assert len(manager.data) > 0
        assert len(manager.data) == len(sample_option_data)
    
    def test_filter_chain(self, option_data_manager):
        """Test filtering option chain data."""
        # Apply filters for calls only
        calls = option_data_manager.filter_chain(option_type="CALL")
        
        # Verify all results are calls
        assert all(row['OptionType'] == 'CALL' for _, row in calls.iterrows())
        
        # Test filtering by strike
        otm_calls = option_data_manager.filter_chain(
            option_type="CALL",
            min_strike=440
        )
        
        # Verify all strikes are >= 440
        assert all(row['Strike'] >= 440 for _, row in otm_calls.iterrows())
        
        # Test filtering by days to expiry
        dte_filtered = option_data_manager.filter_chain(
            min_dte=20,
            max_dte=40
        )
        
        # Verify DTE is within range
        assert all(20 <= row['DTE'] <= 40 for _, row in dte_filtered.iterrows())
    
    def test_get_chain_for_symbol(self, option_data_manager):
        """Test getting option chain for a specific underlying symbol."""
        # Get chain for SPY
        spy_chain = option_data_manager.get_chain_for_symbol("SPY")
        
        # Verify all options have SPY as the underlying
        assert all(row['Underlying'] == 'SPY' for _, row in spy_chain.iterrows())
    
    def test_get_expirations(self, option_data_manager):
        """Test getting unique expiration dates."""
        expirations = option_data_manager.get_expirations()
        
        # Should return a list of datetime objects
        assert isinstance(expirations, list)
        assert all(isinstance(date, datetime) for date in expirations)
        
        # For our sample data, there should be at least one expiration
        assert len(expirations) >= 1
    
    def test_get_strikes(self, option_data_manager):
        """Test getting unique strike prices."""
        strikes = option_data_manager.get_strikes()
        
        # Should return a list of numeric strike prices
        assert isinstance(strikes, list)
        assert all(isinstance(strike, (int, float)) for strike in strikes)
        
        # For our sample data, there should be multiple strikes
        assert len(strikes) > 1
    
    # Additional tests as needed
    # def test_calculate_greeks(self, option_data_manager):
    #     """Test calculation of missing Greeks."""
    #     pass
    
    # def test_prepare_data_for_analysis(self, option_data_manager):
    #     """Test preparing data for analysis with various filters."""
    #     pass


if __name__ == "__main__":
    # This allows running the tests directly from this file
    pytest.main(["-v", __file__]) 