"""
Unit tests for the options analysis module.
These tests validate the options pricing models and Greek calculations.
"""

import pytest
import os
import sys
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the required modules
try:
    from core.options_analysis import OptionPricer
except ImportError:
    pytest.skip("Core modules not available for testing", allow_module_level=True)


class TestOptionPricer:
    """Tests for the option pricing and analysis tools."""
    
    def test_initialization(self):
        """Test that the pricer initializes correctly."""
        pricer = OptionPricer()
        assert pricer is not None
    
    def test_black_scholes_call_pricing(self):
        """Test Black-Scholes pricing for call options."""
        pricer = OptionPricer()
        
        # Test an at-the-money call option
        call_price = pricer.black_scholes_price(
            option_type="CALL",
            underlying_price=100.0,
            strike=100.0,
            time_to_expiry=0.25,  # 3 months
            risk_free_rate=0.02,
            volatility=0.20
        )
        
        # Approximate expected price for these inputs
        # Using standard B-S formula, this should be around 5.05
        expected_call = 5.05
        assert abs(call_price - expected_call) < 0.1
        
        # Test an in-the-money call
        itm_call_price = pricer.black_scholes_price(
            option_type="CALL",
            underlying_price=110.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.02,
            volatility=0.20
        )
        
        # Should be more expensive than ATM call
        assert itm_call_price > call_price
        
        # Should include intrinsic value (underlying - strike = 10)
        assert itm_call_price > 10.0
        
        # Test an out-of-the-money call
        otm_call_price = pricer.black_scholes_price(
            option_type="CALL",
            underlying_price=90.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.02,
            volatility=0.20
        )
        
        # Should be cheaper than ATM call
        assert otm_call_price < call_price
        
        # Should be all time value (no intrinsic value)
        assert otm_call_price > 0.0
    
    def test_black_scholes_put_pricing(self):
        """Test Black-Scholes pricing for put options."""
        pricer = OptionPricer()
        
        # Test an at-the-money put option
        put_price = pricer.black_scholes_price(
            option_type="PUT",
            underlying_price=100.0,
            strike=100.0,
            time_to_expiry=0.25,  # 3 months
            risk_free_rate=0.02,
            volatility=0.20
        )
        
        # Approximate expected price for these inputs
        expected_put = 4.03  # Slightly less than ATM call due to interest rate
        assert abs(put_price - expected_put) < 0.1
        
        # Test an in-the-money put
        itm_put_price = pricer.black_scholes_price(
            option_type="PUT",
            underlying_price=90.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.02,
            volatility=0.20
        )
        
        # Should be more expensive than ATM put
        assert itm_put_price > put_price
        
        # Should include intrinsic value (strike - underlying = 10)
        assert itm_put_price > 10.0
        
        # Test an out-of-the-money put
        otm_put_price = pricer.black_scholes_price(
            option_type="PUT",
            underlying_price=110.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.02,
            volatility=0.20
        )
        
        # Should be cheaper than ATM put
        assert otm_put_price < put_price
        
        # Should be all time value (no intrinsic value)
        assert otm_put_price > 0.0
    
    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        pricer = OptionPricer()
        
        # Parameters
        S = 100.0  # Underlying price
        K = 100.0  # Strike
        t = 0.25   # Time to expiry
        r = 0.02   # Risk-free rate
        sigma = 0.20  # Volatility
        
        # Calculate prices
        call_price = pricer.black_scholes_price("CALL", S, K, t, r, sigma)
        put_price = pricer.black_scholes_price("PUT", S, K, t, r, sigma)
        
        # Put-Call parity: C - P = S - K*e^(-rt)
        parity_diff = call_price - put_price - (S - K * np.exp(-r * t))
        assert abs(parity_diff) < 0.01
    
    def test_implied_volatility_calculation(self):
        """Test calculating implied volatility from option prices."""
        pricer = OptionPricer()
        
        # First, get a price using a known volatility
        known_vol = 0.25
        price = pricer.black_scholes_price(
            option_type="CALL",
            underlying_price=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.02,
            volatility=known_vol
        )
        
        # Now, solve for the implied volatility
        implied_vol = pricer.calculate_implied_volatility(
            option_type="CALL",
            underlying_price=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.02,
            option_price=price
        )
        
        # The implied vol should match our known vol
        assert abs(implied_vol - known_vol) < 0.001
        
        # Test for a deep ITM option
        itm_price = pricer.black_scholes_price(
            option_type="CALL",
            underlying_price=120.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.02,
            volatility=known_vol
        )
        
        implied_vol_itm = pricer.calculate_implied_volatility(
            option_type="CALL",
            underlying_price=120.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.02,
            option_price=itm_price
        )
        
        assert abs(implied_vol_itm - known_vol) < 0.001
    
    def test_greek_calculations(self):
        """Test calculation of option Greeks."""
        pricer = OptionPricer()
        
        # Calculate all Greeks for a call option
        greeks = pricer.calculate_greeks(
            option_type="CALL",
            underlying_price=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.02,
            volatility=0.20
        )
        
        # Check that all Greeks are calculated
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "theta" in greeks
        assert "vega" in greeks
        assert "rho" in greeks
        
        # Verify properties of the Greeks for an ATM call
        # Delta should be approximately 0.5 for ATM call
        assert abs(greeks["delta"] - 0.5) < 0.1
        
        # Gamma should be positive
        assert greeks["gamma"] > 0
        
        # Theta should be negative (time decay)
        assert greeks["theta"] < 0
        
        # Vega should be positive (more valuable with higher vol)
        assert greeks["vega"] > 0
        
        # Rho should be positive for calls (more valuable with higher rates)
        assert greeks["rho"] > 0
        
        # Calculate Greeks for a put option
        put_greeks = pricer.calculate_greeks(
            option_type="PUT",
            underlying_price=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.02,
            volatility=0.20
        )
        
        # Verify properties of the Greeks for an ATM put
        # Delta should be approximately -0.5 for ATM put
        assert abs(put_greeks["delta"] + 0.5) < 0.1
        
        # Gamma should be positive and same as call gamma
        assert put_greeks["gamma"] > 0
        assert abs(put_greeks["gamma"] - greeks["gamma"]) < 0.001
        
        # Theta should be negative
        assert put_greeks["theta"] < 0
        
        # Vega should be positive and same as call vega
        assert put_greeks["vega"] > 0
        assert abs(put_greeks["vega"] - greeks["vega"]) < 0.001
        
        # Rho should be negative for puts
        assert put_greeks["rho"] < 0
    
    def test_delta_behavior(self):
        """Test that delta behaves correctly across different strikes."""
        pricer = OptionPricer()
        
        # Calculate delta for deep ITM, ATM, and deep OTM calls
        deep_itm_delta = pricer.calculate_greeks(
            option_type="CALL",
            underlying_price=120.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.02,
            volatility=0.20
        )["delta"]
        
        atm_delta = pricer.calculate_greeks(
            option_type="CALL",
            underlying_price=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.02,
            volatility=0.20
        )["delta"]
        
        deep_otm_delta = pricer.calculate_greeks(
            option_type="CALL",
            underlying_price=80.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.02,
            volatility=0.20
        )["delta"]
        
        # Delta should increase from OTM to ITM
        assert deep_otm_delta < atm_delta < deep_itm_delta
        
        # Delta for deep ITM call should approach 1.0
        assert deep_itm_delta > 0.8
        
        # Delta for deep OTM call should approach 0.0
        assert deep_otm_delta < 0.2


if __name__ == "__main__":
    # This allows running the tests directly from this file
    pytest.main(["-v", __file__]) 