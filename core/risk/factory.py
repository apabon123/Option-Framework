"""
Risk manager factory module.

This module provides a factory for creating risk manager instances.
"""

import logging
from typing import Dict, Any, Optional

from core.risk.metrics import RiskMetrics
from core.risk.manager import (
    RiskManager,
    VolatilityTargetRiskManager,
    SharpeRatioRiskManager,
    AdaptiveRiskManager,
    CombinedRiskManager
)
from core.risk.parameters import parse_risk_parameters


class RiskManagerFactory:
    """Factory for creating risk manager instances."""
    
    @staticmethod
    def create(config: Dict[str, Any], risk_metrics: RiskMetrics = None,
              logger: Optional[logging.Logger] = None) -> RiskManager:
        """
        Create a risk manager instance based on configuration.
        
        Args:
            config: Configuration dictionary
            risk_metrics: Risk metrics calculator instance
            logger: Logger instance
            
        Returns:
            RiskManager: Instance of a risk manager
        """
        logger = logger or logging.getLogger('risk_manager_factory')
        
        # Create risk metrics if not provided
        if risk_metrics is None:
            logger.info("Creating new RiskMetrics instance")
            risk_metrics = RiskMetrics(config, logger)
        
        # Get risk manager type from configuration
        risk_config = config.get('risk', {})
        risk_manager_type = risk_config.get('type', 'volatility').lower()
        
        # Parse risk parameters
        risk_params = parse_risk_parameters(risk_config)
        
        # Create appropriate risk manager
        logger.info(f"Creating risk manager of type: {risk_manager_type}")
        
        if risk_manager_type == 'volatility':
            return VolatilityTargetRiskManager(config, risk_metrics, logger)
        elif risk_manager_type == 'sharpe':
            return SharpeRatioRiskManager(config, risk_metrics, logger)
        elif risk_manager_type == 'adaptive':
            return AdaptiveRiskManager(config, risk_metrics, logger)
        elif risk_manager_type == 'combined':
            return CombinedRiskManager(config, risk_metrics, logger)
        else:
            logger.warning(f"Unknown risk manager type: {risk_manager_type}, using volatility")
            return VolatilityTargetRiskManager(config, risk_metrics, logger) 