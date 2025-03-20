#!/usr/bin/env python
"""
Test file to verify the fix for the hedge direction issue in SPAN margin calculation.
This test reproduces the specific scenario from the margin report where a long call
with a delta of 0.1954 should be hedged with a short stock position (negative delta).
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the root directory to the path so we can import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import required modules
from core.position import Position, OptionPosition
from core.margin import SPANMarginCalculator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('margin_test')

def test_long_call_with_hedge():
    """Test hedging a long call option with the proper direction stock position."""
    print("="*80)
    print("TESTING LONG CALL WITH STOCK HEDGE (REPRODUCING REPORTED ISSUE)")
    print("="*80)
    print()
    
    # Create the same scenario from the margin report
    call_option_data = {
        "UnderlyingSymbol": "SPY",
        "Strike": 498.0,
        "Type": "C",
        "Delta": 0.1954,  # Same delta as in the report
        "Gamma": 0.01,
        "Theta": -0.25,
        "Vega": 0.2,
        "underlying_price": 472.65  # Same underlying price as in the report
    }
    
    # Create an option position (long call with delta 0.1954)
    option_symbol = "SPY240328C00498000"  # Same option symbol as in the report
    call_position = OptionPosition(
        symbol=option_symbol,
        option_data=call_option_data,
        contracts=1,
        entry_price=2.99,  # Same price as in the report
        current_price=2.99,
        is_short=False  # LONG call
    )
    
    # Get the option delta
    option_delta = call_position.current_delta
    print(f"Option (Long Call) Delta: {option_delta:.4f}")
    
    # Calculate required stock hedge (same as in the report)
    hedge_delta = -option_delta  # Negative delta to offset positive call delta
    hedge_shares = int(round(hedge_delta * 100))  # Convert to shares
    underlying_price = call_option_data["underlying_price"]
    
    print(f"Hedge delta needed: {hedge_delta:.4f}")
    print(f"Shares to hedge 1 contract: {abs(hedge_shares)} at ${underlying_price:.2f}")
    
    # Create the hedge position with the CORRECT direction (short)
    # For a long call with positive delta, the hedge should be SHORT stock (is_short=True)
    hedge_is_short = hedge_delta < 0  # This should be True for a long call hedge
    
    hedge_position = Position(
        symbol="SPY",
        contracts=abs(hedge_shares),  # Use absolute number of shares
        entry_price=underlying_price,
        current_price=underlying_price,
        is_short=hedge_is_short  # Should be True for a short stock position
    )
    
    # Calculate stock position delta (negative if short)
    stock_delta = -hedge_position.contracts if hedge_position.is_short else hedge_position.contracts
    stock_delta_normalized = stock_delta / 100  # Normalize to option contract equivalents
    
    print(f"Stock Position: {hedge_position.contracts} shares, is_short={hedge_position.is_short}")
    print(f"Stock Delta: {stock_delta} shares ({stock_delta_normalized:.4f} option contracts equivalent)")
    
    # Create a SPAN margin calculator
    margin_calculator = SPANMarginCalculator(
        max_leverage=12.0,  # Same as in the report
        hedge_credit_rate=0.8,  # 80% credit for hedged positions
        logger=logger
    )
    
    # Calculate individual position margins
    option_margin = margin_calculator.calculate_position_margin(call_position)
    stock_margin = margin_calculator.calculate_position_margin(hedge_position)
    
    print(f"\nOption margin (unhedged): ${option_margin:.2f}")
    print(f"Standard hedge margin (25%): ${stock_margin:.2f}")
    print(f"Simple sum: ${option_margin + stock_margin:.2f}")
    
    # Calculate portfolio margin with both positions
    positions = {
        option_symbol: call_position,
        "SPY": hedge_position
    }
    
    portfolio_margin_result = margin_calculator.calculate_portfolio_margin(positions)
    portfolio_margin = portfolio_margin_result["total_margin"]
    hedging_benefit = portfolio_margin_result["hedging_benefits"]
    
    print(f"\nPortfolio margin with hedging: ${portfolio_margin:.2f}")
    print(f"Hedging benefit: ${hedging_benefit:.2f} ({hedging_benefit/(option_margin + stock_margin)*100:.1f}% reduction)")
    
    # Verify hedging benefit
    if hedging_benefit > 0:
        print("\n✅ HEDGING BENEFITS DETECTED: Test passed!")
    else:
        print("\n❌ NO HEDGING BENEFITS DETECTED: Test failed!")
    
    # Verify portfolio margin is less than sum of individual margins
    if portfolio_margin < (option_margin + stock_margin):
        print("✅ PORTFOLIO MARGIN IS REDUCED: Test passed!")
    else:
        print("❌ PORTFOLIO MARGIN IS NOT REDUCED: Test failed!")
    
    print()
    
    return {
        "option_margin": option_margin,
        "stock_margin": stock_margin,
        "simple_sum": option_margin + stock_margin,
        "portfolio_margin": portfolio_margin,
        "hedging_benefit": hedging_benefit
    }

def test_all_option_scenarios():
    """Test hedging for all option scenarios (long/short calls and puts)."""
    print("="*80)
    print("TESTING ALL OPTION SCENARIOS WITH PROPER HEDGING")
    print("="*80)
    print()
    
    # Test cases
    test_cases = [
        # Option type, Is short, Delta value, Expected hedge direction
        ("call", False, 0.50, True),   # Long call (+delta) -> Short stock (-delta)
        ("call", True, 0.50, False),   # Short call (-delta) -> Long stock (+delta)
        ("put", False, -0.50, False),  # Long put (-delta) -> Long stock (+delta)
        ("put", True, -0.50, True),    # Short put (+delta) -> Short stock (-delta)
    ]
    
    for idx, (option_type, is_short, delta_value, expected_hedge_is_short) in enumerate(test_cases):
        print(f"\nTest Case #{idx+1}: {'' if not is_short else 'Short'} {option_type.upper()} option")
        
        # Create option data
        option_data = {
            "UnderlyingSymbol": "SPY",
            "Strike": 450.0,
            "Type": option_type[0].upper(),  # C or P
            "Delta": delta_value,
            "Gamma": 0.01,
            "Theta": -0.25,
            "Vega": 0.2,
            "underlying_price": 450.0
        }
        
        # Create option position
        option_symbol = f"SPY240328{'C' if option_type == 'call' else 'P'}00450000"
        option_position = OptionPosition(
            symbol=option_symbol,
            option_data=option_data,
            contracts=1,
            entry_price=5.0,
            current_price=5.0,
            is_short=is_short
        )
        
        # Get the option delta - this should already account for position direction
        option_delta = option_position.current_delta
        
        # Calculate required hedge delta (opposite sign)
        hedge_delta = -option_delta
        hedge_shares = int(round(abs(hedge_delta) * 100))
        hedge_is_short = hedge_delta < 0
        
        print(f"  Option Delta: {option_delta:.4f}")
        print(f"  Required Hedge Delta: {hedge_delta:.4f}")
        print(f"  Hedge Direction: {'Short' if hedge_is_short else 'Long'}")
        print(f"  Expected Hedge Direction: {'Short' if expected_hedge_is_short else 'Long'}")
        
        # Verify hedge direction
        if hedge_is_short == expected_hedge_is_short:
            print("  ✅ HEDGE DIRECTION TEST PASSED")
        else:
            print("  ❌ HEDGE DIRECTION TEST FAILED")
        
        # Create hedge position with calculated direction
        hedge_position = Position(
            symbol="SPY",
            contracts=hedge_shares,
            entry_price=450.0,
            current_price=450.0,
            is_short=hedge_is_short
        )
        
        # Create margin calculator
        margin_calculator = SPANMarginCalculator(
            max_leverage=12.0,
            hedge_credit_rate=0.8
        )
        
        # Calculate margins
        option_margin = margin_calculator.calculate_position_margin(option_position)
        stock_margin = margin_calculator.calculate_position_margin(hedge_position)
        
        # Calculate portfolio margin
        positions = {
            option_symbol: option_position,
            "SPY": hedge_position
        }
        
        portfolio_result = margin_calculator.calculate_portfolio_margin(positions)
        portfolio_margin = portfolio_result["total_margin"]
        hedging_benefit = portfolio_result["hedging_benefits"]
        
        print(f"  Option Margin: ${option_margin:.2f}")
        print(f"  Stock Margin: ${stock_margin:.2f}")
        print(f"  Combined Margin: ${option_margin + stock_margin:.2f}")
        print(f"  Portfolio Margin: ${portfolio_margin:.2f}")
        print(f"  Hedging Benefit: ${hedging_benefit:.2f} ({hedging_benefit/(option_margin + stock_margin)*100:.1f}%)")
        
        # Verify hedging benefit
        if hedging_benefit > 0:
            print("  ✅ HEDGING BENEFITS DETECTED: Test passed!")
        else:
            print("  ❌ NO HEDGING BENEFITS DETECTED: Test failed!")
    
    print("\nAll Option Scenarios Test Complete")
    print()

if __name__ == "__main__":
    # Run the test that reproduces the issue
    print("\nRunning test to verify the hedge direction fix...\n")
    test_long_call_with_hedge()
    
    # Test all option scenarios
    print("\nRunning comprehensive tests for all option types...\n")
    test_all_option_scenarios() 