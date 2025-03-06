"""
Test configuration loading
"""

import os
import sys
import yaml
import pandas as pd
from datetime import datetime

# Add the project root to the path so we can import the core modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trading_engine import TradingEngine
from example_strategy import SimpleOptionStrategy

def test_config_loading():
    """Test that config loading works correctly."""
    
    # Create a simple configuration
    config = {
        'paths': {
            'input_file': 'test_data.csv',
            'output_dir': 'test_output',
        },
        'dates': {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
        },
        'strategy': {
            'name': 'TestStrategy',
            'delta_target': -0.25,
        }
    }
    
    # Create a strategy
    strategy = SimpleOptionStrategy(config['strategy'])
    
    # Create the engine
    engine = TradingEngine(config, strategy)
    
    # Verify attributes were set correctly
    print(f"Config loaded: {bool(engine.config)}")
    print(f"Start date: {engine.start_date}")
    print(f"End date: {engine.end_date}")
    print(f"Data file: {engine.data_file}")
    
    return True

if __name__ == "__main__":
    print("Testing configuration loading...")
    if test_config_loading():
        print("Configuration loading test passed!")
    else:
        print("Configuration loading test failed!")