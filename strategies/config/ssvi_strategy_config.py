"""
Configuration for SSVI-based Relative Value Trading Strategy

This module provides default configurations and examples for the SSVI trading strategy.
Users can modify these configurations to adjust trading parameters, risk management,
and model settings.
"""

from typing import Dict, Any
import logging

# Default SSVI strategy configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    # Trading parameters
    'trade_params': {
        # Z-score threshold for identifying RV opportunities
        'zscore_threshold': 1.5,
        
        # Option filtering parameters
        'min_dte': 7,            # Minimum days to expiry
        'max_dte': 90,           # Maximum days to expiry
        'min_liquidity': 100,    # Minimum open interest
        'max_legs': 3,           # Maximum number of legs in a trade
        
        # Execution parameters
        'max_trades_per_update': 3,   # Maximum trades to execute per update
        'reversion_target': 0.0,      # Target Z-score for mean reversion
        'take_profit_zscore': 0.5,    # Take profit when Z-score reaches this level
        'stop_loss_zscore': 2.5,      # Stop loss when Z-score reaches this level
        'min_expected_return': 0.5,   # Minimum expected return (percent)
        'min_return_per_risk': 1.0,   # Minimum return per unit of risk
    },
    
    # Risk management parameters
    'risk_params': {
        'max_position_size': 10,       # Maximum contracts per position
        'max_delta_exposure': 100,     # Maximum delta exposure
        'max_vega_exposure': 1000,     # Maximum vega exposure
        'max_capital_per_trade': 0.05, # Maximum capital per trade (fraction)
        'delta_hedge_freq': 'daily',   # Delta hedging frequency (daily, hourly, etc.)
        'delta_hedge_threshold': 20,   # Delta hedging threshold
        'max_portfolio_risk': 0.02,    # Maximum portfolio risk (fraction)
        'correlation_adjustment': True, # Consider correlations for risk
    },
    
    # SSVI model parameters
    'model_params': {
        'fit_method': 'global_then_local',  # Fitting method
        'max_iterations': 1000,             # Maximum iterations for fitting
        'theta_bounds': (0.01, 1.0),        # Bounds for theta parameter
        'rho_bounds': (-0.99, 0.99),        # Bounds for rho parameter
        'eta_bounds': (0.01, 5.0),          # Bounds for eta parameter
        'lambda_bounds': (0.01, 5.0),       # Bounds for lambda parameter
        'parameter_penalty': 0.01,          # Penalty for parameter deviation
        'min_options_per_expiry': 5,        # Minimum options required per expiry
        'weight_by_delta': True,            # Weight fitting by delta
        'param_history_length': 30,         # Days of parameter history to keep
        'interpolation_method': 'cubic',    # Interpolation method
        'update_threshold': 0.1,            # Update SSVI if parameter change > threshold
    },
    
    # Logging parameters
    'logging': {
        'level': logging.INFO,
        'file': 'ssvi_strategy.log',
        'console': True,
        'trade_log': True,
        'signal_log': True,
    },
    
    # Backtesting parameters
    'backtest': {
        'initial_capital': 100000.0,
        'transaction_cost_percent': 0.0005,
        'slippage_percent': 0.001,
        'margin_requirement': 0.2,
        'interest_rate': 0.02,
        'market_impact_model': 'linear', 
    }
}

# Example configurations for different market regimes

# Configuration for high volatility markets
HIGH_VOL_CONFIG: Dict[str, Any] = {
    'trade_params': {
        **DEFAULT_CONFIG['trade_params'],
        'zscore_threshold': 2.0,       # Higher threshold in volatile markets
        'min_dte': 14,                 # Longer minimum DTE
        'max_dte': 60,                 # Shorter maximum DTE
        'min_expected_return': 1.0,    # Higher expected return
    },
    'risk_params': {
        **DEFAULT_CONFIG['risk_params'],
        'max_position_size': 5,        # Smaller position size
        'max_capital_per_trade': 0.03, # Less capital per trade
        'delta_hedge_freq': 'hourly',  # More frequent hedging
    },
    'model_params': {
        **DEFAULT_CONFIG['model_params'],
        'parameter_penalty': 0.02,     # Stronger parameter penalties
        'update_threshold': 0.05,      # More frequent updates
    },
}

# Configuration for low volatility markets
LOW_VOL_CONFIG: Dict[str, Any] = {
    'trade_params': {
        **DEFAULT_CONFIG['trade_params'],
        'zscore_threshold': 1.0,       # Lower threshold in calm markets
        'min_dte': 5,                  # Shorter minimum DTE
        'max_dte': 120,                # Longer maximum DTE
        'min_expected_return': 0.3,    # Lower expected return requirement
    },
    'risk_params': {
        **DEFAULT_CONFIG['risk_params'],
        'max_position_size': 15,       # Larger position size
        'max_capital_per_trade': 0.07, # More capital per trade
        'delta_hedge_freq': 'daily',   # Less frequent hedging
    },
    'model_params': {
        **DEFAULT_CONFIG['model_params'],
        'parameter_penalty': 0.005,    # Lighter parameter penalties
    },
}

# Configuration focused on butterflies and multi-leg strategies
MULTI_LEG_CONFIG: Dict[str, Any] = {
    'trade_params': {
        **DEFAULT_CONFIG['trade_params'],
        'max_legs': 4,                 # More legs allowed
        'min_liquidity': 50,           # Lower liquidity requirement
    },
    'risk_params': {
        **DEFAULT_CONFIG['risk_params'],
        'max_gamma_exposure': 500,     # Explicit gamma exposure limit
        'max_theta_exposure': -1000,   # Explicit theta exposure limit
    },
    'model_params': {
        **DEFAULT_CONFIG['model_params'],
        'weight_by_delta': False,      # Don't weight by delta
        'weight_by_vega': True,        # Weight by vega instead
    },
}

# Function to load configuration
def load_config(config_name: str = 'default') -> Dict[str, Any]:
    """
    Load a specific configuration by name.
    
    Args:
        config_name: Name of the configuration to load
        
    Returns:
        Dict containing the configuration
    """
    configs = {
        'default': DEFAULT_CONFIG,
        'high_vol': HIGH_VOL_CONFIG,
        'low_vol': LOW_VOL_CONFIG,
        'multi_leg': MULTI_LEG_CONFIG,
    }
    
    if config_name not in configs:
        logging.warning(f"Config '{config_name}' not found, using default")
        return DEFAULT_CONFIG
        
    return configs[config_name]
    
# Function to create a custom configuration by merging with defaults
def create_custom_config(custom_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a custom configuration by merging with defaults.
    
    Args:
        custom_params: Custom parameters to override defaults
        
    Returns:
        Dict containing the merged configuration
    """
    import copy
    import collections.abc
    
    def merge_dicts(d1, d2):
        """
        Recursively merge dictionaries.
        """
        result = copy.deepcopy(d1)
        
        for k, v2 in d2.items():
            if k in result and isinstance(result[k], dict) and isinstance(v2, dict):
                result[k] = merge_dicts(result[k], v2)
            else:
                result[k] = copy.deepcopy(v2)
                
        return result
    
    return merge_dicts(DEFAULT_CONFIG, custom_params) 