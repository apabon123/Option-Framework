#!/usr/bin/env python3
"""
Run all tests in the repository.

This script will run all tests in the repository, including margin tests and data tests.
Run this before making changes or pushing to GitHub to verify everything works.
"""

import sys
import os
from pathlib import Path

# Get absolute path of this script
SCRIPT_PATH = os.path.abspath(__file__)
# Get directory containing this script
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
# Get project root
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add project root to Python path
sys.path.insert(0, PROJECT_ROOT)

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_all_tests():
    """Run all tests in the repository."""
    print("=" * 80)
    print("Running all tests...")
    print("=" * 80)
    
    # Run margin calculator tests
    print("\nRunning Margin Calculator Tests...")
    from tests.margin_tests.margin_calculator_test import run_margin_tests
    run_margin_tests()
    print("\nMargin Calculator Tests completed.")
    
    # Run delta hedge tests
    print("\nRunning Delta Hedge Tests...")
    from tests.margin_tests.run_hedge_test import test_same_direction_positions
    test_same_direction_positions()
    print("\nDelta Hedge Tests completed.")
    
    # Run data tests
    print("\nRunning Data Tests...")
    from tests.data_tests.test_load_csv import test_load_csv
    test_load_csv()
    print("\nData Tests completed.")
    
    # Add more test imports and runs here
    # ...
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests() 