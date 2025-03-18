"""
Simplified test script to validate delta sign handling for different option types

This script tests delta sign handling for all four scenarios:
1. Long Call  - Should have positive delta
2. Short Call - Should have negative delta 
3. Long Put   - Should have negative delta
4. Short Put  - Should have positive delta
"""

def test_delta_sign():
    # Test all four options scenarios
    scenarios = [
        {"name": "Long Call",  "type": "C", "raw_delta": 0.5,  "is_short": False, "expected": 0.5},
        {"name": "Short Call", "type": "C", "raw_delta": 0.5,  "is_short": True,  "expected": -0.5},
        {"name": "Long Put",   "type": "P", "raw_delta": -0.5, "is_short": False, "expected": -0.5},
        {"name": "Short Put",  "type": "P", "raw_delta": -0.5, "is_short": True,  "expected": 0.5}
    ]
    
    print("\n==== OPTION DELTA SIGN TEST ====")
    print("Testing all four option scenarios for proper delta sign handling")
    
    # Test current delta sign handling logic
    print("\n- Current Logic -")
    for scenario in scenarios:
        delta = scenario["raw_delta"]
        
        # Apply sign adjustment logic from PositionInventory.get_option_delta
        if scenario["is_short"]:
            # For short positions, delta should be negative for calls
            if delta > 0:
                delta = -delta
        else:
            # For long positions, delta should be positive for calls
            # If it's already positive, don't change the sign
            if delta < 0 and scenario["type"] == "C":
                delta = -delta
        
        # Check if result matches expected
        result = "✓" if delta == scenario["expected"] else "✗"
        print(f"{scenario['name']}: Raw={scenario['raw_delta']}, Adjusted={delta}, Expected={scenario['expected']} {result}")
    
    # Test improved delta sign handling logic
    print("\n- Improved Logic -")
    for scenario in scenarios:
        delta = scenario["raw_delta"]
        
        # Improved sign adjustment logic
        if scenario["is_short"]:
            # For short positions
            if delta > 0:  # positive delta (calls)
                delta = -delta
            elif delta < 0 and scenario["type"] == "P":  # negative delta (puts)
                delta = -delta  # flip for short puts
        else:
            # For long positions
            if delta < 0 and scenario["type"] == "C":  # negative delta for calls (unusual)
                delta = -delta
        
        # Check if result matches expected
        result = "✓" if delta == scenario["expected"] else "✗"
        print(f"{scenario['name']}: Raw={scenario['raw_delta']}, Adjusted={delta}, Expected={scenario['expected']} {result}")

if __name__ == "__main__":
    test_delta_sign() 