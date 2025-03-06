"""
Configuration loader for theta engine

This module loads configuration from YAML file and provides functions
to access and manage configuration settings.
"""

import os
import yaml
import logging
from datetime import datetime
import pandas as pd
from typing import Dict, Any

def load_yaml_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (optional)
        
    Returns:
        dict: Configuration dictionary
    """
    # If config_path not provided, use default location
    if config_path is None:
        # Use the config folder by default
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'config.yaml')
    
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse dates if they're strings
    if 'dates' in config:
        if 'start_date' in config['dates'] and isinstance(config['dates']['start_date'], str):
            config['dates']['start_date'] = pd.to_datetime(config['dates']['start_date'])
        if 'end_date' in config['dates'] and isinstance(config['dates']['end_date'], str):
            config['dates']['end_date'] = pd.to_datetime(config['dates']['end_date'])
    
    return config

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        dict: Merged configuration dictionary
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override or add new key
            result[key] = value
    
    return result

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        dict: Default configuration dictionary
    """
    config = {
        # File paths
        'paths': {
            'input_file': "data/options_data.csv",
            'output_dir': "scenario_results",
            'trades_output_file': "trades.csv"
        },

        # Date range for backtesting
        'dates': {
            'start_date': pd.to_datetime("2024-01-01"),
            'end_date': pd.to_datetime("2024-12-31")
        },

        # Portfolio settings
        'portfolio': {
            'initial_capital': 100000,
            'max_leverage': 12,
            'max_nlv_percent': 1.0
        },

        # Risk management settings
        'risk': {
            'rolling_window': 21,
            'target_z': 0,
            'min_z': -2.0,
            'min_investment': 0.25,
            'short_window': 21,
            'medium_window': 63,
            'long_window': 95,
            'risk_scaling_window': "short"
        },

        # Strategy parameters
        'strategy': {
            'name': "ThetaEngine",
            'enable_hedging': True,
            'hedge_mode': "ratio",
            'hedge_with_underlying': True,
            'constant_portfolio_delta': 0.05,
            'hedge_target_ratio': 1.75,
            'hedge_symbol': "SPY",
            'days_to_expiry_min': 60,
            'days_to_expiry_max': 90,
            'is_short': True,
            'delta_target': -0.05,
            'profit_target': 0.65,
            'stop_loss_threshold': 2.5,
            'close_days_to_expiry': 14,
            'delta_tolerance': 1.5,
            'min_position_size': 1
        },

        # Trading parameters
        'trading': {
            'normal_spread': 0.60
        },

        # Margin management parameters
        'margin_management': {
            'margin_buffer_pct': 0.10,
            'negative_margin_threshold': -0.05,
            'rebalance_cooldown_days': 3,
            'forced_rebalance_threshold': -0.10,
            'max_position_reduction_pct': 0.25,
            'losing_position_max_reduction_pct': 0.40,
            'urgent_reduction_pct': 0.50
        }
    }
    
    return config

if __name__ == "__main__":
    # Test loading configuration
    try:
        config = load_yaml_config()
        print("Successfully loaded configuration:")
        print(f"Strategy name: {config['strategy']['name']}")
        print(f"Date range: {config['dates']['start_date']} to {config['dates']['end_date']}")
        print(f"Hedging enabled: {config['strategy']['enable_hedging']}")
        print(f"Underlying hedging: {config['strategy']['hedge_with_underlying']}")
    except Exception as e:
        print(f"Error loading configuration: {e}")