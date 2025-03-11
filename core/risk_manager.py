"""
Risk Management Module

This module provides tools for position sizing, risk calculation, and portfolio exposure
management. It implements strategies for determining appropriate position sizes based
on portfolio metrics, risk parameters, and performance history.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np


class RiskManager:
    """
    Manages position sizing and risk calculations based on performance metrics.

    This class calculates risk scaling factors based on performance metrics,
    determines appropriate position sizes, and calculates margin requirements.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the RiskManager with configuration parameters"""
        self.config = config
        self.logger = logger or logging.getLogger('risk_manager')

        # Extract key risk parameters from portfolio section
        portfolio_config = config.get('portfolio', {})
        self.max_leverage = portfolio_config.get('max_leverage', 12)
        self.max_nlv_percent = portfolio_config.get('max_position_size_pct', 0.25)

        # Extract risk scaling parameters
        risk_config = config.get('risk', {})
        self.rolling_window = risk_config.get('rolling_window', 21)
        self.target_z = risk_config.get('target_z', 0)  # z-score at which full exposure is reached
        self.min_z = risk_config.get('min_z', -2.0)  # z-score for minimum exposure
        self.min_investment = risk_config.get('min_investment', 0.25)  # Minimum investment level

        # Track risk scaling history for analysis
        self.risk_scaling_history = []

        # Log initialization
        if self.logger:
            self.logger.info("[RiskManager] Initialized")
            self.logger.info(f"  Max Leverage: {self.max_leverage}x")
            self.logger.info(f"  Max NLV Percent: {self.max_nlv_percent:.2%}")
            self.logger.info(f"  Rolling Window: {self.rolling_window} days")

    def calculate_position_size(self, option_data: Dict[str, Any], portfolio_metrics: Dict[str, Any], risk_scaling: float = 1.0) -> int:
        """
        Calculate position size based on risk parameters with proper risk scaling and max_nlv_percent limit.

        Args:
            option_data: Option data with price information
            portfolio_metrics: Dictionary of portfolio metrics (NLV, available margin, etc.)
            risk_scaling: Risk scaling factor (default 1.0)

        Returns:
            int: Number of contracts to trade
        """
        net_liq = portfolio_metrics.get('net_liquidation_value', 0)
        available_margin = portfolio_metrics.get('available_margin', 0)
        current_margin = portfolio_metrics.get('total_margin', 0)

        # Handle NaN risk scaling
        if pd.isna(risk_scaling):
            self.logger.warning("[RiskManager] Risk scaling is NaN, using default value 1.0")
            risk_scaling = 1.0

        # Maximum margin allocation allowed based on risk scaling and current NLV
        max_margin_alloc = risk_scaling * net_liq

        # Calculate remaining margin capacity
        remaining_margin_capacity = max(max_margin_alloc - current_margin, 0)

        # Get option price and calculate margin per contract
        if hasattr(option_data, 'get') and not hasattr(option_data, 'iloc'):
            # This is a dictionary
            option_price = option_data.get('price', option_data.get('MidPrice', option_data.get('Ask', 0)))
            option_symbol = option_data.get('symbol', option_data.get('OptionSymbol', 'Unknown'))
        else:
            # This is a pandas Series
            option_price = option_data.get('MidPrice', option_data.get('Ask', 0))
            option_symbol = option_data.get('OptionSymbol', 'Unknown')

        # Make sure we have a valid price
        if option_price == 0:
            self.logger.warning(f"[RiskManager] Option price is 0 for {option_symbol}, cannot calculate position size")
            return 0

        # Calculate margin per contract
        margin_per_contract = option_price * 100 * self.max_leverage
        if margin_per_contract <= 0:
            self.logger.warning(
                f"[RiskManager] Invalid margin per contract for {option_symbol}: ${margin_per_contract:.2f}")
            return 0

        # Calculate maximum contracts based on remaining margin capacity
        capacity_max_contracts = int(remaining_margin_capacity / margin_per_contract) if margin_per_contract > 0 else 0

        # Also check against available margin (to avoid going negative)
        available_max_contracts = int(available_margin / margin_per_contract) if margin_per_contract > 0 else 0

        # Apply max_nlv_percent limit - maximum position size as percentage of NLV
        max_position_margin = net_liq * self.max_nlv_percent
        max_position_contracts = int(max_position_margin / margin_per_contract) if margin_per_contract > 0 else 0

        # Ensure all values are non-negative
        capacity_max_contracts = max(capacity_max_contracts, 0)
        available_max_contracts = max(available_max_contracts, 0)
        max_position_contracts = max(max_position_contracts, 0)

        # Take the most conservative (lowest) limit
        contracts = min(capacity_max_contracts, available_max_contracts, max_position_contracts)

        # Override when capacity exists but NLV percent would prevent any trading
        if (contracts == 0 and max_position_contracts == 0 and
                min(capacity_max_contracts, available_max_contracts) >= 1):
            contracts = 1
            self.logger.info(
                f"[Position Sizing] Override: Max NLV percent would yield 0 contracts, but capacity exists. Setting to 1 contract.")

        # Ensure at least the minimum position size if adding any contracts
        min_position_size = self.config.get('strategy', {}).get('min_position_size', 1)
        if contracts > 0 and contracts < min_position_size:
            contracts = min_position_size

        # Enhanced logging including the max_nlv_percent constraint
        self.logger.info(f"[Position Sizing] Option: {option_symbol}")
        self.logger.info(f"  Price: ${option_price:.2f}, Margin per contract: ${margin_per_contract:.2f}")
        self.logger.info(
            f"  NLV: ${net_liq:.2f}, Maximum Margin: ${max_margin_alloc:.2f}, Current Margin: ${current_margin:.2f}")
        self.logger.info(
            f"  Remaining Capacity: ${remaining_margin_capacity:.2f}, Available Margin: ${available_margin:.2f}")
        self.logger.info(
            f"  Max NLV Percent: {self.max_nlv_percent:.2%}, Position limit: {max_position_contracts} contracts")
        self.logger.info(
            f"  Capacity limit: {capacity_max_contracts} contracts, Available margin limit: {available_max_contracts} contracts")
        self.logger.info(f"  Risk scaling: {risk_scaling:.2f}")
        self.logger.info(f"  Final position size: {contracts} contracts")

        return contracts 