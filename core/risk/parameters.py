"""
Risk management parameters and configuration classes.

This module contains the parameter classes used for risk management,
including basic risk parameters and risk limits.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class RiskParameters:
    """Base configuration for risk management."""
    min_size: float = 0
    max_size: float = 100
    min_scalar: float = 0.0
    max_scalar: float = 1.0


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_size: float = 10
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.05
    position_limit_pct: float = 0.1
    concentration_limit: float = 0.2


@dataclass
class SharpeRiskParameters:
    """Parameters for Sharpe-based risk management"""
    min_sharpe: float = 0.5
    target_sharpe: float = 1.5
    lookback_days: int = 21
    window_days: int = 63
    risk_free_rate: float = 0.02
    risk_free_return: float = 0
    window_type: str = "medium"  # Use 'short', 'medium', or 'long' to match RiskMetrics windows


@dataclass
class VolatilityRiskParameters:
    """Parameters for volatility-based risk management"""
    target_volatility: float = 0.15
    lookback_days: int = 21
    min_variance_ratio: float = 0.5
    max_variance_ratio: float = 2.0
    window_days: int = 63
    window_type: str = "medium"  # Use 'short', 'medium', or 'long' to match RiskMetrics windows


@dataclass
class AdaptiveRiskParameters:
    """Parameters for adaptive risk management"""
    max_heat: float = 1.0
    cooldown_rate: float = 0.05
    heatup_rate: float = 0.02
    recovery_factor: float = 2.0
    window_days: int = 21
    window_type: str = "short"  # Use 'short', 'medium', or 'long' to match RiskMetrics windows


@dataclass
class CombinedRiskParameters:
    """Parameters for combined risk management approach"""
    weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])  # vol, sharpe, adaptive
    min_position_size: float = 0.0
    max_position_size: float = 1.0
    vol_params: VolatilityRiskParameters = field(default_factory=VolatilityRiskParameters)
    sharpe_params: SharpeRiskParameters = field(default_factory=SharpeRiskParameters)
    adaptive_params: AdaptiveRiskParameters = field(default_factory=AdaptiveRiskParameters)


def parse_risk_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse risk parameters from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of parsed risk parameters
    """
    risk_params = {}
    
    # Extract basic risk parameters
    risk_params['basic'] = RiskParameters(
        min_size=config.get('min_size', 0),
        max_size=config.get('max_size', 100),
        min_scalar=config.get('min_scalar', 0.0),
        max_scalar=config.get('max_scalar', 1.0)
    )
    
    # Extract risk limits
    risk_limits_config = config.get('risk_limits', {})
    risk_params['limits'] = RiskLimits(
        max_position_size=risk_limits_config.get('max_position_size', 10),
        max_daily_loss=risk_limits_config.get('max_daily_loss', 0.02),
        max_drawdown=risk_limits_config.get('max_drawdown', 0.05),
        position_limit_pct=risk_limits_config.get('position_limit_pct', 0.1),
        concentration_limit=risk_limits_config.get('concentration_limit', 0.2)
    )
    
    # Extract Sharpe risk parameters
    sharpe_config = config.get('sharpe', {})
    risk_params['sharpe'] = SharpeRiskParameters(
        min_sharpe=sharpe_config.get('min_sharpe', 0.5),
        target_sharpe=sharpe_config.get('target_sharpe', 1.5),
        lookback_days=sharpe_config.get('lookback_days', 21),
        window_days=sharpe_config.get('window_days', 63),
        risk_free_rate=sharpe_config.get('risk_free_rate', 0.02),
        risk_free_return=sharpe_config.get('risk_free_return', 0),
        window_type=sharpe_config.get('window_type', 'medium')
    )
    
    # Extract volatility risk parameters
    vol_config = config.get('volatility', {})
    risk_params['volatility'] = VolatilityRiskParameters(
        target_volatility=vol_config.get('target_volatility', 0.15),
        lookback_days=vol_config.get('lookback_days', 21),
        min_variance_ratio=vol_config.get('min_variance_ratio', 0.5),
        max_variance_ratio=vol_config.get('max_variance_ratio', 2.0),
        window_days=vol_config.get('window_days', 63),
        window_type=vol_config.get('window_type', 'medium')
    )
    
    # Extract adaptive risk parameters
    adaptive_config = config.get('adaptive', {})
    risk_params['adaptive'] = AdaptiveRiskParameters(
        max_heat=adaptive_config.get('max_heat', 1.0),
        cooldown_rate=adaptive_config.get('cooldown_rate', 0.05),
        heatup_rate=adaptive_config.get('heatup_rate', 0.02),
        recovery_factor=adaptive_config.get('recovery_factor', 2.0),
        window_days=adaptive_config.get('window_days', 21),
        window_type=adaptive_config.get('window_type', 'short')
    )
    
    # Extract combined risk parameters
    combined_config = config.get('combined', {})
    weights = combined_config.get('weights', [0.4, 0.3, 0.3])
    risk_params['combined'] = CombinedRiskParameters(
        weights=weights,
        min_position_size=combined_config.get('min_position_size', 0.0),
        max_position_size=combined_config.get('max_position_size', 1.0),
        vol_params=risk_params['volatility'],
        sharpe_params=risk_params['sharpe'],
        adaptive_params=risk_params['adaptive']
    )
    
    return risk_params 