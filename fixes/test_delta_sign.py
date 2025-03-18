"""
Test script to validate delta sign handling for different option types

This script creates test option positions for all four scenarios:
1. Long Call  - Should have positive delta
2. Short Call - Should have negative delta
3. Long Put   - Should have negative delta
4. Short Put  - Should have positive delta

The test will print the calculated delta after applying the sign adjustment logic.
"""

import sys
import logging
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('delta_test')

# Import directly - we're running from the project root
from position import Position, OptionPosition
from position_inventory import PositionInventory

def create_option_position(symbol: str, option_type: str, delta: float, is_short: bool = False) -> OptionPosition:
    """Create an option position with the specified parameters"""
    # Create basic option data dictionary
    option_data = {
        'Symbol': symbol,
        'Type': option_type,
        'Delta': delta,
        'UnderlyingSymbol': 'SPY',
        'UnderlyingPrice': 480.0,
        'Strike': 470.0,
        'Expiration': datetime(2024, 4, 19)
    }
    
    # Create option position
    position = OptionPosition(
        symbol=symbol,
        option_data=option_data,
        contracts=1,
        entry_price=5.0,
        current_price=5.0,
        is_short=is_short,
        logger=logger
    )
    
    # Set option type and delta manually to ensure they're set correctly
    position.option_type = option_type
    position.current_delta = delta
    
    return position

def test_delta_sign():
    """Test delta sign handling for different option types"""
    # Create a position inventory
    inventory = PositionInventory(logger=logger)
    
    # Create test positions
    positions = [
        # Long Call - Delta is positive
        create_option_position("SPY240419C00470000", "C", 0.5, is_short=False),
        
        # Short Call - Delta should become negative
        create_option_position("SPY240419C00470000", "C", 0.5, is_short=True),
        
        # Long Put - Delta is negative
        create_option_position("SPY240419P00470000", "P", -0.5, is_short=False),
        
        # Short Put - Delta should remain negative (already negative for puts)
        create_option_position("SPY240419P00470000", "P", -0.5, is_short=True)
    ]
    
    # Add positions to inventory
    for i, position in enumerate(positions):
        # Use different symbols to avoid overwriting
        position.symbol = f"TEST_OPTION_{i}"
        inventory.add_position(position)
    
    # Print input values
    print("\n==== INPUT VALUES ====")
    for symbol, position in inventory.option_positions.items():
        print(f"{symbol}: Type={position.option_type}, Delta={position.current_delta}, Is Short={position.is_short}")
    
    # Manually apply the sign adjustment logic from get_option_delta()
    print("\n==== ADJUSTED DELTAS ====")
    for symbol, position in inventory.option_positions.items():
        position_delta = position.current_delta
        
        # Apply same logic as in get_option_delta
        if position.is_short:
            # For short positions, delta should be negative for calls
            if position_delta > 0:
                position_delta = -position_delta
        else:
            # For long positions, delta should be positive for calls
            # If it's already positive, don't change the sign
            if position_delta < 0 and hasattr(position, 'option_type') and position.option_type == "C":
                position_delta = -position_delta
        
        print(f"{symbol}: Type={position.option_type}, Adjusted Delta={position_delta}, Is Short={position.is_short}")
        
        # Update position delta with adjusted value for the final test
        position.current_delta = position_delta
    
    # Calculate overall option delta
    total_option_delta = inventory.get_option_delta()
    print(f"\nTotal option delta: {total_option_delta}")
    
    # Expected results:
    # 1. Long Call:   Delta should be positive (+0.5)
    # 2. Short Call:  Delta should be negative (-0.5)
    # 3. Long Put:    Delta should be negative (-0.5)
    # 4. Short Put:   Delta should be positive (+0.5)
    
    # Create a second test with raw delta values
    # for all four scenarios directly
    print("\n==== DIRECT DELTA TEST ====")
    scenarios = [
        {"name": "Long Call",  "type": "C", "raw_delta": 0.5,  "is_short": False, "expected": 0.5},
        {"name": "Short Call", "type": "C", "raw_delta": 0.5,  "is_short": True,  "expected": -0.5},
        {"name": "Long Put",   "type": "P", "raw_delta": -0.5, "is_short": False, "expected": -0.5},
        {"name": "Short Put",  "type": "P", "raw_delta": -0.5, "is_short": True,  "expected": 0.5}
    ]
    
    for scenario in scenarios:
        delta = scenario["raw_delta"]
        
        # Apply sign adjustment logic
        if scenario["is_short"]:
            # For short positions
            if delta > 0:  # positive delta (usually calls)
                delta = -delta
            elif delta < 0 and scenario["type"] == "P":  # negative delta (puts)
                delta = -delta  # flip for short puts
        else:
            # For long positions
            if delta < 0 and scenario["type"] == "C":  # negative delta for calls (unusual)
                delta = -delta
        
        print(f"{scenario['name']}: Raw={scenario['raw_delta']}, Adjusted={delta}, Expected={scenario['expected']}")

if __name__ == "__main__":
    test_delta_sign() 