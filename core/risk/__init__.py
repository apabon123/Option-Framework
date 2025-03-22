"""
Risk management module.

This module provides classes and utilities for risk management in trading strategies.
"""

from core.risk.metrics import RiskMetrics
from core.risk.manager import (
    RiskManager, 
    VolatilityTargetRiskManager,
    SharpeRatioRiskManager, 
    AdaptiveRiskManager,
    CombinedRiskManager
)
from core.risk.parameters import RiskParameters, RiskLimits
from core.risk.factory import RiskManagerFactory

__all__ = [
    'RiskMetrics',
    'RiskManager',
    'VolatilityTargetRiskManager',
    'SharpeRatioRiskManager',
    'AdaptiveRiskManager',
    'CombinedRiskManager',
    'RiskParameters',
    'RiskLimits',
    'RiskManagerFactory'
] 