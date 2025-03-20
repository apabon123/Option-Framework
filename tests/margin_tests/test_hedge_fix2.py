#!/usr/bin/env python
"""
Test file to verify the fix for the hedge direction issue in SPAN margin calculation.
This test checks that the base MarginCalculator correctly delegates to SPANMarginCalculator
when calculating portfolio margin with hedging benefits.
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
from core.margin import MarginCalculator, SPANMarginCalculator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('margin_test')

def test_hedge_direction_delegation():
    """Test that the base MarginCalculator correctly delegates to SPANMarginCalculator."""
    print("="*80)
    print("TESTING BASE MARGIN CALCULATOR DELEGATION TO SPAN CALCULATOR")
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
    
    # Ensure option properties are correctly set
    underlying_price = call_option_data["underlying_price"]
    call_position.underlying_price = underlying_price
    call_position.current_delta = call_option_data["Delta"]
    call_position.current_gamma = call_option_data["Gamma"]
    call_position.current_theta = call_option_data["Theta"]
    call_position.current_vega = call_option_data["Vega"]
    call_position.option_type = "C"
    call_position.strike = call_option_data["Strike"]
    
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
    
    # Ensure position properties are set for the hedge position
    if hasattr(hedge_position, 'underlying_price'):
        hedge_position.underlying_price = underlying_price
    
    # Calculate stock position delta (negative if short)
    stock_delta = -hedge_position.contracts if hedge_position.is_short else hedge_position.contracts
    stock_delta_normalized = stock_delta / 100  # Normalize to option contract equivalents
    
    print(f"Stock Position: {hedge_position.contracts} shares, is_short={hedge_position.is_short}")
    print(f"Stock Delta: {stock_delta} shares ({stock_delta_normalized:.4f} option contracts equivalent)")
    
    # Create both types of margin calculators
    base_calculator = MarginCalculator(
        max_leverage=12.0,
        logger=logger
    )
    
    span_calculator = SPANMarginCalculator(
        max_leverage=12.0,  # Same as in the report
        hedge_credit_rate=0.8,  # 80% credit for hedged positions
        logger=logger
    )
    
    # Calculate individual position margins with SPAN
    option_margin = span_calculator.calculate_position_margin(call_position)
    stock_margin = span_calculator.calculate_position_margin(hedge_position)
    
    print(f"\nOption margin (unhedged): ${option_margin:.2f}")
    print(f"Standard hedge margin (25%): ${stock_margin:.2f}")
    print(f"Simple sum: ${option_margin + stock_margin:.2f}")
    
    # Calculate portfolio margin with both positions using SPAN directly
    positions = {
        option_symbol: call_position,
        "SPY": hedge_position
    }
    
    # Verify underlying prices are set correctly
    print("\nVerifying underlying prices:")
    print(f"Option position underlying price: ${call_position.underlying_price:.2f}")
    print(f"Stock position underlying price: ${hedge_position.underlying_price if hasattr(hedge_position, 'underlying_price') else 'N/A'}")
    
    # First test with the specialized SPAN calculator
    span_portfolio_result = span_calculator.calculate_portfolio_margin(positions)
    span_portfolio_margin = span_portfolio_result["total_margin"]
    span_hedging_benefit = span_portfolio_result["hedging_benefits"]
    
    print(f"\nSPAN Portfolio margin with hedging: ${span_portfolio_margin:.2f}")
    print(f"SPAN Hedging benefit: ${span_hedging_benefit:.2f} ({span_hedging_benefit/(option_margin + stock_margin)*100:.1f}% reduction)")
    
    # Now test with the base calculator that should delegate
    base_portfolio_result = base_calculator.calculate_portfolio_margin(positions)
    base_portfolio_margin = base_portfolio_result["total_margin"]
    base_hedging_benefit = base_portfolio_result["hedging_benefits"]
    
    print(f"\nBase Portfolio margin with delegation: ${base_portfolio_margin:.2f}")
    print(f"Base Hedging benefit: ${base_hedging_benefit:.2f} ({base_hedging_benefit/(option_margin + stock_margin)*100:.1f}% reduction)")
    
    # Verify the base calculator delegated correctly
    if base_hedging_benefit > 0 and abs(base_portfolio_margin - span_portfolio_margin) < 0.01:
        print("\n✅ BASE CALCULATOR DELEGATED CORRECTLY: Test passed!")
    else:
        print("\n❌ BASE CALCULATOR DELEGATION FAILED: Test failed!")
    
    # Verify portfolio margin is less than sum of individual margins
    if span_portfolio_margin < (option_margin + stock_margin):
        print("✅ PORTFOLIO MARGIN IS REDUCED: Test passed!")
    else:
        print("❌ PORTFOLIO MARGIN IS NOT REDUCED: Test failed!")
    
    print()
    
    return {
        "option_margin": option_margin,
        "stock_margin": stock_margin,
        "simple_sum": option_margin + stock_margin,
        "span_portfolio_margin": span_portfolio_margin,
        "span_hedging_benefit": span_hedging_benefit,
        "base_portfolio_margin": base_portfolio_margin,
        "base_hedging_benefit": base_hedging_benefit
    }

if __name__ == "__main__":
    # Run the test that verifies the delegation fix
    print("\nRunning test to verify the margin calculator delegation fix...\n")
    test_hedge_direction_delegation() 