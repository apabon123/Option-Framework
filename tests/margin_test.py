"""
Margin Calculation Test Script

This script tests the margin calculation to find where the contract multiplier issue is occurring.
"""

import logging
import sys
from core.margin import SPANMarginCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('margin_test')

def main():
    # Create a test instance of the SPANMarginCalculator
    span_calculator = SPANMarginCalculator(
        max_leverage=1.0,
        volatility_multiplier=1.0,
        initial_margin_percentage=0.1,  # 10% initial margin
        maintenance_margin_percentage=0.07,  # 7% maintenance margin
        hedge_credit_rate=0.8,  # 80% credit for hedged positions
        logger=logger
    )
    
    # Test with example values from the user's output
    option_price = 2.99  # Option price per share (SPY240328C00498000)
    underlying_price = 472.65  # Underlying price (SPY)
    
    # Run the test with detailed logging
    logger.info("Running margin calculation test with sample values")
    base_margin, scan_risk, manual_margin, actual_margin = span_calculator.test_option_margin_calculation(
        option_price=option_price,
        underlying_price=underlying_price,
        contracts=1  # Test with a single contract
    )
    
    # Summarize the results
    logger.info("Margin Calculation Test Results:")
    logger.info(f"  Option price per share: ${option_price:.2f}, per contract: ${option_price * 100:.2f}")
    logger.info(f"  Underlying price: ${underlying_price:.2f}")
    logger.info(f"  Base margin: ${base_margin:.2f}")
    logger.info(f"  Scan risk: ${scan_risk:.2f}")
    logger.info(f"  Manual margin calculation: ${manual_margin:.2f}")
    logger.info(f"  SPANMarginCalculator result: ${actual_margin:.2f}")
    
    if abs(actual_margin - manual_margin) > 0.01:
        ratio = actual_margin / manual_margin
        logger.warning(f"DISCREPANCY: Calculated margin differs from method result by a factor of {ratio:.4f}")
        
        # Check specific common issues
        if 0.009 < ratio < 0.011:
            logger.error("ISSUE IDENTIFIED: Method result is 1/100 of expected value")
            logger.error("This indicates the contract multiplier of 100 is missing in the calculation")
        elif 99 < ratio < 101:
            logger.error("ISSUE IDENTIFIED: Method result is 100x the expected value")
            logger.error("This indicates the contract multiplier of 100 is being applied twice")
    else:
        logger.info("No discrepancy found - margin calculation is correct")

if __name__ == "__main__":
    main() 