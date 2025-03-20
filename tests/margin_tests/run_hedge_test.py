"""
Delta Hedging Test Script - Same Direction Positions Test

This script tests that the SPANMarginCalculator correctly identifies positions
with deltas in the same direction and doesn't apply any hedging benefits.
"""

import logging
import sys
import os
from pathlib import Path

# Get the absolute path of this script
script_path = Path(os.path.abspath(__file__)).resolve()
# Get the project root directory
project_root = script_path.parent.parent.parent

# Add the project root to the Python path
sys.path.insert(0, str(project_root))

from core.margin import SPANMarginCalculator
from core.position import OptionPosition, Position

# Configure root logger to suppress most messages
logging.getLogger().setLevel(logging.ERROR)

# Configure our test logger
logger = logging.getLogger('hedge_test')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

def test_same_direction_positions():
    """Test the absence of hedging benefits when positions have deltas in the same direction.
    
    This test verifies that portfolio margin calculation does not apply
    hedging benefits when an option position is paired with a stock position
    that has a delta with the same sign.
    """
    print("="*80)
    print("TESTING SAME DIRECTION POSITIONS (NO HEDGING BENEFITS)")
    print("="*80)
    print()
    
    # Create a put option position with negative delta
    put_option_data = {
        "UnderlyingSymbol": "SPY",
        "Strike": 480.0,
        "Type": "P",
        "Delta": -0.75,
        "Gamma": 0.01,
        "Theta": -0.5,
        "Vega": 0.2,
        "underlying_price": 480.0
    }
    
    # Create an option position (long put will have negative delta)
    option_symbol = "SPY240420P00480000"
    put_position = OptionPosition(
        symbol=option_symbol,
        option_data=put_option_data,
        contracts=1,
        entry_price=12.50,
        current_price=12.50,
        is_short=False  # LONG put
    )
    
    # Get the option delta
    option_delta = put_position.current_delta
    print(f"Option (Long Put) Delta: {option_delta:.2f}")
    
    # For SAME direction, we need a position with the SAME delta sign
    # If option delta is negative, stock should be short (also negative)
    # This is intentionally the WRONG way to hedge
    hedge_is_short = option_delta < 0
    hedge_position = Position(
        symbol="SPY",
        contracts=100,  # 100 shares to match 1 option contract
        entry_price=480.0,
        current_price=480.0,
        is_short=hedge_is_short
    )
    
    # Calculate stock position delta (positive if long, negative if short)
    stock_delta = -100 if hedge_is_short else 100  # 100 shares with sign based on direction
    print(f"Stock Position Delta: {stock_delta:.2f}")
    print(f"Stock Position is {'Short' if hedge_is_short else 'Long'}")
    
    # Calculate net delta (should be amplified when in same direction)
    net_delta = option_delta + stock_delta
    print(f"Net Delta: {net_delta:.2f}")
    
    # Create a calculator
    margin_calculator = SPANMarginCalculator(
        max_leverage=12,
        volatility_multiplier=1.0,
        initial_margin_percentage=0.1,
        maintenance_margin_percentage=0.07
    )
    
    # Calculate standalone margins
    put_margin = margin_calculator.calculate_position_margin(put_position)
    stock_margin = margin_calculator.calculate_position_margin(hedge_position)
    
    print(f"Put Option Margin: ${put_margin:.2f}")
    print(f"Stock Position Margin: ${stock_margin:.2f}")
    print(f"Total Standalone Margin: ${put_margin + stock_margin:.2f}")
    
    # Calculate portfolio margin with NO hedging benefits expected
    # Create a dictionary of positions for portfolio calculation
    positions = {
        option_symbol: put_position,
        "SPY": hedge_position
    }
    
    # Calculate portfolio margin
    portfolio_margin_result = margin_calculator.calculate_portfolio_margin(positions)
    total_margin = portfolio_margin_result["total_margin"]
    hedging_benefits = portfolio_margin_result.get("hedging_benefits", 0)
    
    print(f"Portfolio Margin: ${total_margin:.2f}")
    print(f"Hedging Benefits: ${hedging_benefits:.2f}")
    
    # Verify NO hedging benefits (same direction shouldn't provide benefits)
    if hedging_benefits == 0:
        print("✅ NO HEDGING BENEFITS DETECTED (CORRECT): Test passed!")
    else:
        print("❌ HEDGING BENEFITS DETECTED (INCORRECT): Test failed!")
    
    # Verify portfolio margin is approximately equal to sum of individual margins
    if abs(total_margin - (put_margin + stock_margin)) < 0.01:
        print("✅ PORTFOLIO MARGIN EQUALS STANDALONE SUM: Test passed!")
    else:
        print("❌ PORTFOLIO MARGIN DIFFERS FROM STANDALONE SUM: Test failed!")
    
    print()

def test_opposite_direction_positions():
    """Test the hedging benefits when positions have deltas in opposite directions.
    
    This test verifies that portfolio margin calculation correctly applies
    hedging benefits when an option position is paired with a stock position
    that has a delta with the opposite sign.
    """
    print("="*80)
    print("TESTING OPPOSITE DIRECTION POSITIONS (HEDGING BENEFITS)")
    print("="*80)
    print()
    
    # Create a put option position with negative delta
    put_option_data = {
        "UnderlyingSymbol": "SPY",
        "Strike": 480.0,
        "Type": "P",
        "Delta": -0.75,
        "Gamma": 0.01,
        "Theta": -0.5,
        "Vega": 0.2,
        "underlying_price": 480.0
    }
    
    # Create an option position (short put will have positive delta)
    option_symbol = "SPY240420P00480000"
    put_position = OptionPosition(
        symbol=option_symbol,
        option_data=put_option_data,
        contracts=1,
        entry_price=12.50,
        current_price=12.50,
        is_short=True
    )
    
    # Calculate the proper hedge delta - need OPPOSITE sign from option delta
    option_delta = put_position.current_delta
    print(f"Option (Short Put) Delta: {option_delta:.2f}")
    
    # For hedging, we need a position with the opposite delta sign
    # If option delta is positive, hedge should be short (negative delta)
    hedge_is_short = option_delta > 0
    hedge_position = Position(
        symbol="SPY",
        contracts=100,  # 100 shares to match 1 option contract
        entry_price=480.0,
        current_price=480.0,
        is_short=hedge_is_short
    )
    
    # Calculate stock position delta (positive if long, negative if short)
    stock_delta = -100 if hedge_is_short else 100  # 100 shares with sign based on direction
    print(f"Stock Position Delta: {stock_delta:.2f}")
    print(f"Stock Position is {'Short' if hedge_is_short else 'Long'}")
    
    # Calculate net delta (should be close to 0 if hedged properly)
    net_delta = option_delta + stock_delta
    print(f"Net Delta: {net_delta:.2f}")
    
    # Create a calculator
    margin_calculator = SPANMarginCalculator(
        max_leverage=12,
        volatility_multiplier=1.0,
        initial_margin_percentage=0.1,
        maintenance_margin_percentage=0.07
    )
    
    # Calculate standalone margins
    put_margin = margin_calculator.calculate_position_margin(put_position)
    stock_margin = margin_calculator.calculate_position_margin(hedge_position)
    
    print(f"Put Option Margin: ${put_margin:.2f}")
    print(f"Stock Position Margin: ${stock_margin:.2f}")
    print(f"Total Standalone Margin: ${put_margin + stock_margin:.2f}")
    
    # Calculate portfolio margin with hedging benefits
    # Create a dictionary of positions for portfolio calculation
    positions = {
        option_symbol: put_position,
        "SPY": hedge_position
    }
    
    # Calculate portfolio margin
    portfolio_margin_result = margin_calculator.calculate_portfolio_margin(positions)
    total_margin = portfolio_margin_result["total_margin"]
    hedging_benefits = portfolio_margin_result.get("hedging_benefits", 0)
    
    print(f"Portfolio Margin: ${total_margin:.2f}")
    print(f"Hedging Benefits: ${hedging_benefits:.2f}")
    
    # Verify hedging benefits
    if hedging_benefits > 0:
        print("✅ HEDGING BENEFITS DETECTED: Test passed!")
    else:
        print("❌ NO HEDGING BENEFITS DETECTED: Test failed!")
    
    # Verify portfolio margin is less than sum of individual margins
    if total_margin < (put_margin + stock_margin):
        print("✅ PORTFOLIO MARGIN IS REDUCED: Test passed!")
    else:
        print("❌ PORTFOLIO MARGIN IS NOT REDUCED: Test failed!")
    
    print()

def test_hedge_directions():
    """Test the correct hedge direction for different option types and positions.
    
    This test verifies that the proper hedge direction is determined for various
    option types (call/put) and positions (long/short).
    """
    print("="*80)
    print("TESTING HEDGE DIRECTION DETERMINATION")
    print("="*80)
    print()
    
    # Test cases with expected hedge directions
    test_cases = [
        # Option type, Is short, Original delta, Expected delta after position creation, Expected hedge direction
        ("call", True, 0.65, -0.65, False),   # Short call -> Long stock (opposite signs)
        ("call", False, 0.65, 0.65, True),    # Long call -> Short stock (opposite signs)
        ("put", True, -0.75, 0.75, True),     # Short put -> Short stock (opposite signs)
        ("put", False, -0.75, -0.75, False),  # Long put -> Long stock (opposite signs)
    ]
    
    for idx, (option_type, is_short, original_delta, expected_delta, expected_hedge_is_short) in enumerate(test_cases):
        print(f"Test Case #{idx+1}: {'' if not is_short else 'Short'} {option_type.upper()} option")
        
        # Create option data
        option_data = {
            "UnderlyingSymbol": "SPY",
            "Strike": 480.0,
            "Type": option_type[0].upper(),  # C or P
            "Delta": original_delta,
            "underlying_price": 480.0
        }
        
        # Create option position
        option_symbol = f"SPY240420{'C' if option_type == 'call' else 'P'}00480000"
        option_position = OptionPosition(
            symbol=option_symbol,
            option_data=option_data,
            contracts=1,
            entry_price=12.50,
            current_price=12.50,
            is_short=is_short
        )
        
        # Get the actual delta calculated by the position
        actual_delta = option_position.current_delta
        
        # Determine hedge direction based on the option delta sign
        # For proper hedging, we need the OPPOSITE sign for the hedge
        hedge_is_short = actual_delta > 0  # If option delta is positive, hedge should be short (negative delta)
        
        print(f"  Original Option Delta: {original_delta:.2f}")
        print(f"  Actual Option Delta after position creation: {actual_delta:.2f}")
        print(f"  Expected Delta after position creation: {expected_delta:.2f}")
        print(f"  Correct Hedge Direction: {'Short' if hedge_is_short else 'Long'}")
        print(f"  Expected Hedge Direction: {'Short' if expected_hedge_is_short else 'Long'}")
        
        if abs(actual_delta - expected_delta) < 0.01:
            print("  ✅ DELTA CALCULATION TEST PASSED")
        else:
            print(f"  ❌ DELTA CALCULATION TEST FAILED: Expected {expected_delta:.2f}, got {actual_delta:.2f}")
        
        if hedge_is_short == expected_hedge_is_short:
            print("  ✅ HEDGE DIRECTION TEST PASSED")
        else:
            print("  ❌ HEDGE DIRECTION TEST FAILED")
        
        print()
    
    print("Hedge Direction Tests Complete")
    print()

def run_hedge_tests():
    """Run a series of tests to verify the delta hedging functionality of the SPANMarginCalculator."""
    # Test 0: Verify hedge direction determination
    test_hedge_directions()
    
    # Test 1: Same-direction positions should not get hedging benefits
    test_same_direction_positions()
    
    # Test 2: Opposite-direction positions should get hedging benefits
    test_opposite_direction_positions()
    
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    # First test hedge direction determination
    test_hedge_directions()
    
    # Test same-direction positions (no hedging benefit expected)
    test_same_direction_positions()
    
    # Test opposite-direction positions (hedging benefit expected)
    test_opposite_direction_positions()
    
    print("="*80)
    print("ALL HEDGE TESTS COMPLETED")
    print("="*80) 