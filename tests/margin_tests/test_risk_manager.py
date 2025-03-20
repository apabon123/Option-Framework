#!/usr/bin/env python
"""
Test file to verify the improved reality check in the RiskManager
for SPAN margin calculation with hedging benefits.
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
from core.risk_manager import RiskManager
import logging
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('risk_manager_test')

def test_risk_manager_reality_check():
    """Test that the RiskManager reality check properly handles different margin scenarios."""
    print("="*80)
    print("TESTING RISK MANAGER MARGIN REALITY CHECK")
    print("="*80)
    print()
    
    # Load test configuration
    config = {
        'portfolio': {
            'max_leverage': 12,
            'max_position_size_pct': 0.25
        },
        'risk': {
            'rolling_window': 21,
            'target_z': 0,
            'min_z': -2.0,
            'min_investment': 0.25
        },
        'margin_management': {
            'margin_calculator_type': 'span',
            'margin_calculation_method': 'portfolio'
        }
    }
    
    # Create a RiskManager instance
    risk_manager = RiskManager(config, logger)
    
    # Create the same scenario from the margin report
    call_option_data = {
        "UnderlyingSymbol": "SPY",
        "Strike": 498.0,
        "Type": "C",
        "Delta": 0.1954,  # Same delta as in the report
        "Gamma": 0.01,
        "Theta": -0.25,
        "Vega": 0.2,
        "underlying_price": 472.65,  # Same underlying price as in the report
        "price": 2.99,
        "MidPrice": 2.99,
        "symbol": "SPY240328C00498000",
        "OptionSymbol": "SPY240328C00498000"
    }
    
    # Set up portfolio metrics
    portfolio_metrics = {
        'net_liquidation_value': 100000,
        'available_margin': 50000,
        'total_margin': 20000
    }
    
    # Calculate position size and check reality check behavior
    position_size = risk_manager.calculate_position_size(call_option_data, portfolio_metrics)
    
    print(f"Calculated position size: {position_size} contracts")
    print(f"Last margin per contract: ${risk_manager._last_margin_per_contract:.2f}")
    
    # Test with different delta values to see how the reality check behaves
    # Scenario 1: Opposite direction deltas (proper hedge)
    call_option_data["Delta"] = 0.5
    print("\nScenario 1: Option Delta = 0.5 (opposite of hedge)")
    position_size = risk_manager.calculate_position_size(call_option_data, portfolio_metrics)
    print(f"Calculated position size: {position_size} contracts")
    print(f"Margin per contract: ${risk_manager._last_margin_per_contract:.2f}")
    
    # Scenario 2: Same direction deltas (not a proper hedge)
    call_option_data["Delta"] = -0.5  # Same direction as hedge (short stock)
    print("\nScenario 2: Option Delta = -0.5 (same direction as hedge)")
    position_size = risk_manager.calculate_position_size(call_option_data, portfolio_metrics)
    print(f"Calculated position size: {position_size} contracts")
    print(f"Margin per contract: ${risk_manager._last_margin_per_contract:.2f}")
    
    # Scenario 3: Very low SPAN margin (should hit min threshold)
    call_option_data["Delta"] = 0.01  # Very low delta
    print("\nScenario 3: Option Delta = 0.01 (very low delta)")
    position_size = risk_manager.calculate_position_size(call_option_data, portfolio_metrics)
    print(f"Calculated position size: {position_size} contracts")
    print(f"Margin per contract: ${risk_manager._last_margin_per_contract:.2f}")
    
    print("\nReality check test completed.")
    print()

if __name__ == "__main__":
    # Run the test
    print("\nRunning test to verify the margin reality check in RiskManager...\n")
    test_risk_manager_reality_check() 