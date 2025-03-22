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
    
    def test_calculate_greeks(self, option_data_manager, sample_option_data, tmp_path):
        """Test calculation of missing Greeks."""
        # Create a modified version of sample data with some missing Greeks
        modified_data = sample_option_data.copy()
        # Remove some Greeks
        for i in range(0, len(modified_data), 3):  # Every third row
            modified_data.loc[i, 'Delta'] = None
            modified_data.loc[i, 'Gamma'] = None
        
        # Save to file
        data_file = os.path.join(tmp_path, "modified_options.csv")
        modified_data.to_csv(data_file, index=False)
        
        # Create manager and load data
        manager = OptionDataManager()
        manager.load_data(data_file)
        
        # Calculate missing Greeks
        filled_data = manager.calculate_greeks(
            underlying_price=435.0,
            risk_free_rate=0.02,
            calculation_date=datetime(2022, 2, 15)
        )
        
        # Check that all Greeks are now filled
        assert filled_data['Delta'].notna().all()
        assert filled_data['Gamma'].notna().all()
        
        # Check values are reasonable
        assert all(abs(delta) <= 1.0 for delta in filled_data['Delta'] if delta is not None)
        assert all(gamma >= 0 for gamma in filled_data['Gamma'] if gamma is not None)
        
        # For ATM options, delta should be close to 0.5 (calls) or -0.5 (puts)
        atm_calls = filled_data[(filled_data['Strike'] == 435) & (filled_data['OptionType'] == 'CALL')]
        atm_puts = filled_data[(filled_data['Strike'] == 435) & (filled_data['OptionType'] == 'PUT')]
        
        for _, row in atm_calls.iterrows():
            assert 0.4 <= row['Delta'] <= 0.6
        
        for _, row in atm_puts.iterrows():
            assert -0.6 <= row['Delta'] <= -0.4
    
    def test_prepare_data_for_analysis(self, option_data_manager):
        """Test preparing data for analysis with various filters."""
        # Test with various filters
        filtered_data = option_data_manager.prepare_data_for_analysis(
            option_type='CALL',
            min_strike=420,
            max_strike=450,
            min_dte=25,
            max_dte=35,
            min_delta=0.3,
            max_delta=0.7
        )
        
        # Verify filters were applied correctly
        assert all(row['OptionType'] == 'CALL' for _, row in filtered_data.iterrows())
        assert all(420 <= row['Strike'] <= 450 for _, row in filtered_data.iterrows())
        assert all(25 <= row['DTE'] <= 35 for _, row in filtered_data.iterrows())
        assert all(0.3 <= row['Delta'] <= 0.7 for _, row in filtered_data.iterrows())
        
        # Test with min bid price
        bid_filtered = option_data_manager.prepare_data_for_analysis(
            min_bid=2.0
        )
        
        assert all(row['Bid'] >= 2.0 for _, row in bid_filtered.iterrows())
        
        # Test sorting
        sorted_by_delta = option_data_manager.prepare_data_for_analysis(
            sort_by='Delta',
            ascending=False
        )
        
        # Verify data is sorted by Delta in descending order
        delta_values = sorted_by_delta['Delta'].tolist()
        assert delta_values == sorted(delta_values, reverse=True)
    
    def test_get_moneyness(self, option_data_manager):
        """Test calculating option moneyness."""
        # Add moneyness to the chain
        data_with_moneyness = option_data_manager.get_moneyness(underlying_price=435.0)
        
        # Verify moneyness column exists
        assert 'Moneyness' in data_with_moneyness.columns
        
        # Check moneyness calculation
        for _, row in data_with_moneyness.iterrows():
            if row['OptionType'] == 'CALL':
                expected_moneyness = 435.0 / row['Strike']
            else:  # PUT
                expected_moneyness = row['Strike'] / 435.0
                
            assert abs(row['Moneyness'] - expected_moneyness) < 0.001
        
        # ATM options should have moneyness close to 1.0
        atm_options = data_with_moneyness[data_with_moneyness['Strike'] == 435.0]
        for _, row in atm_options.iterrows():
            assert abs(row['Moneyness'] - 1.0) < 0.001


if __name__ == "__main__":
    # This allows running the tests directly from this file
    pytest.main(["-v", __file__]) 