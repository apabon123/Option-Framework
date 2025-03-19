#!/usr/bin/env python
"""
Run Margin Calculator Test

This script runs the margin calculator test script with proper imports.
Run this from the project root directory.
"""

import sys
import os
from pathlib import Path

# Get the absolute path of this script
script_path = Path(__file__).resolve()
# Get the project root directory
project_root = script_path.parent

# Add the project root to the Python path
sys.path.insert(0, str(project_root))

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, 'output')
os.makedirs(output_dir, exist_ok=True)

# Import and run the tests
from tests.margin_calculator_test import run_margin_tests

if __name__ == "__main__":
    print("Running Margin Calculator Tests...")
    run_margin_tests()
    print("Margin Calculator Tests completed.") 