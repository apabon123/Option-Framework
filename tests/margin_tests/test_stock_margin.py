#!/usr/bin/env python
"""
Test file to verify stock margin calculation.
"""

import sys
import os
import logging

# Add the root directory to the path so we can import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import required modules
from core.position import Position
from core.margin import MarginCalculator, SPANMarginCalculator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def test_stock_margin():
    """Test stock margin calculation with MarginCalculator and SPANMarginCalculator."""
    print("="*80)
    print("TESTING STOCK MARGIN CALCULATION")
    print("="*80)
    print()
    
    # Create a stock position
    stock = Position(
        symbol="SPY",
        contracts=20,
        entry_price=472.65,
        current_price=472.65,
        is_short=True,
        logger=logger
    )
    
    # Calculate margin with base calculator
    base_calc = MarginCalculator(max_leverage=12.0, logger=logger)
    base_margin = base_calc.calculate_position_margin(stock)
    print(f"Base calculator stock margin: ${base_margin:.2f}")
    
    # Calculate margin with SPAN calculator
    span_calc = SPANMarginCalculator(max_leverage=12.0, logger=logger)
    span_margin = span_calc.calculate_position_margin(stock)
    print(f"SPAN calculator stock margin: ${span_margin:.2f}")
    
    return {
        "base_margin": base_margin,
        "span_margin": span_margin
    }

if __name__ == "__main__":
    test_stock_margin() 