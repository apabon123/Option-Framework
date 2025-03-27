"""
Position Sizing Module

This module provides tools for position sizing based on margin requirements and portfolio metrics.
It implements strategies for determining appropriate position sizes based on available margin,
position limits, and other portfolio constraints.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np


class PositionSizer:
    """
    Manages position sizing based on margin requirements and portfolio constraints.

    This class calculates appropriate position sizes based on margin requirements,
    available capital, and position limits.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the PositionSizer with configuration parameters"""
        self.config = config
        self.logger = logger or logging.getLogger('position_sizer')

        # Extract key position sizing parameters from portfolio section
        portfolio_config = config.get('portfolio', {})
        self.max_leverage = portfolio_config.get('max_leverage', 12)
        self.max_nlv_percent = portfolio_config.get('max_position_size_pct', 0.25)

        # Track the most recently calculated margin per contract
        self._last_margin_per_contract = 0

        # Reference to the hedging manager - will be set by trading engine
        self.hedging_manager = None
        
        # Reference to risk scaler (if enabled)
        self.risk_scaler = None

        # Log initialization in a standardized format
        if self.logger:
            self.logger.info("=" * 40)
            self.logger.info("POSITION SIZER INITIALIZATION")
            self.logger.info(f"  Max leverage: {self.max_leverage}x")
            self.logger.info(f"  Max position size: {self.max_nlv_percent:.2%} of NLV")
            self.logger.info("=" * 40)

    def calculate_position_size(self, option_data: Dict[str, Any], portfolio_metrics: Dict[str, Any], risk_scaling: float = 1.0) -> int:
        """
        Calculate position size based on margin requirements and portfolio constraints.

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
            self.logger.warning("[PositionSizer] Risk scaling is NaN, using default value 1.0")
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
            underlying_price = option_data.get('UnderlyingPrice', 0)
            strike = option_data.get('Strike', 0)
            expiry = option_data.get('Expiration', None)
            delta = option_data.get('Delta', 0)
            option_type = 'C' if 'C' in option_symbol else 'P'
        else:
            # This is a pandas Series
            option_price = option_data.get('MidPrice', option_data.get('Ask', 0))
            option_symbol = option_data.get('OptionSymbol', 'Unknown')
            underlying_price = option_data.get('UnderlyingPrice', 0)
            strike = option_data.get('Strike', 0)
            expiry = option_data.get('Expiration', None)
            delta = option_data.get('Delta', 0)
            option_type = 'C' if 'C' in option_symbol else 'P'

        # Make sure we have a valid price
        if option_price == 0:
            self.logger.warning(f"[PositionSizer] Option price is 0 for {option_symbol}, cannot calculate position size")
            return 0

        # Initialize margin variables - always define them to avoid reference errors
        margin_config = self.config.get('margin_management', {})
        margin_calculation_method = margin_config.get('margin_calculation_method', 'simple')
        margin_calculator_type = margin_config.get('margin_calculator_type', 'span').lower()
        margin_calculator = None
        
        # Initialize margin_per_contract
        margin_per_contract = option_price * 100
        
        # Check if we have a pre-calculated margin value to use
        if 'margin_per_contract' in option_data and option_data['margin_per_contract'] > 0:
            margin_per_contract = option_data['margin_per_contract']
            hedging_benefits = option_data.get('hedge_benefit', 0)
            
            # Log that we're using pre-calculated values
            self.logger.info(f"[Position Sizing] Using pre-calculated margin value: ${margin_per_contract:.2f}")
            if hedging_benefits > 0:
                self.logger.info(f"[Position Sizing] Using pre-calculated hedge benefit: ${hedging_benefits:.2f}")
                
            # Store for reference by other components
            self._last_margin_per_contract = margin_per_contract
            return self._calculate_position_size_from_margin(
                margin_per_contract, net_liq, available_margin, remaining_margin_capacity, 
                option_symbol, option_price, risk_scaling, option_data
            )
        
        # HEDGED MARGIN CALCULATION
        # Use the hedging manager to calculate margin if available
        if self.hedging_manager and hasattr(self.hedging_manager, 'calculate_theoretical_margin'):
            self.logger.info(f"[Position Sizing] Using hedging manager for margin calculation with full hedging benefits")
            
            # Set a test quantity of 1 to calculate per-contract margin
            test_quantity = 1
            
            # Calculate theoretical margin with hedging
            margin_result = self.hedging_manager.calculate_theoretical_margin(option_data, test_quantity)
            
            if margin_result and 'total_margin' in margin_result:
                hedged_margin = margin_result['total_margin']
                hedging_benefits = margin_result.get('hedging_benefits', 0)
                
                # Convert to per-contract basis
                if test_quantity > 0:
                    margin_per_contract = hedged_margin / test_quantity
                
                # Log the results
                self.logger.info(f"[Position Sizing] Hedged margin calculation:")
                self.logger.info(f"  Total margin for {test_quantity} contract: ${hedged_margin:.2f}")
                self.logger.info(f"  Hedging benefits: ${hedging_benefits:.2f}")
                self.logger.info(f"  Margin per contract (with hedging): ${margin_per_contract:.2f}")
            else:
                self.logger.warning(f"[Position Sizing] Hedged margin calculation failed, falling back to traditional method")
                # Continue with traditional calculation below
        else:
            # TRADITIONAL MARGIN CALCULATION (Keep the existing approach as fallback)
            self.logger.info(f"[Position Sizing] Using traditional margin calculation approach")
            
            # Check if we have access to portfolio's margin calculator
            portfolio = portfolio_metrics.get('portfolio', None)
            use_portfolio_calculator = margin_config.get('use_portfolio_calculator', True)
            
            # Create a calculator based on configuration if we don't have one from the portfolio
            from core.margin import MarginCalculator, OptionMarginCalculator, SPANMarginCalculator
            
            if margin_calculator_type == 'span':
                self.logger.info(f"[Position Sizing] Creating new SPANMarginCalculator as specified in config")
                margin_calculator = SPANMarginCalculator(
                    max_leverage=self.max_leverage,
                    hedge_credit_rate=0.8,  # Standard hedge credit rate
                    logger=self.logger
                )
            elif margin_calculator_type == 'option':
                self.logger.info(f"[Position Sizing] Creating new OptionMarginCalculator as specified in config")
                margin_calculator = OptionMarginCalculator(
                    max_leverage=self.max_leverage,
                    logger=self.logger
                )
            else:  # Default to simple calculator
                self.logger.info(f"[Position Sizing] Creating new simple MarginCalculator as specified in config")
                margin_calculator = MarginCalculator(
                    max_leverage=self.max_leverage,
                    logger=self.logger
                )
        
        # Calculate margin per contract
        if margin_calculator and margin_calculation_method == 'portfolio':
            # Create a temporary position object for margin calculation
            from core.position import OptionPosition
            from core.margin import SPANMarginCalculator  # Import the SPAN calculator explicitly
            
            # Prepare option data dictionary
            option_data_dict = {
                'Type': option_type,
                'Strike': strike,
                'Expiration': expiry,
                'UnderlyingPrice': underlying_price
            }
            
            # Create option position with 1 contract for margin calculation
            temp_position = OptionPosition(
                symbol=option_symbol,
                option_data=option_data_dict,
                contracts=1,
                entry_price=option_price,
                current_price=option_price,
                is_short=True,  # Assuming short positions for selling options
                logger=self.logger
            )
            
            # Ensure underlying price is properly set - it might be missing from the position after creation
            if temp_position.underlying_price == 0 and underlying_price > 0:
                self.logger.warning(f"[Margin Trace] Fixing missing underlying price: ${underlying_price:.2f}")
                temp_position.underlying_price = underlying_price
            
            # Set greeks if available
            if 'Delta' in option_data:
                temp_position.current_delta = option_data.get('Delta', 0)
                temp_position.current_gamma = option_data.get('Gamma', 0)
                temp_position.current_theta = option_data.get('Theta', 0)
                temp_position.current_vega = option_data.get('Vega', 0)
                temp_position.implied_volatility = option_data.get('ImpliedVolatility', 0.3)
            
            # CRITICAL LOGGING - Will be displayed prominently
            self.logger.warning("======= MARGIN CALCULATION BREAKDOWN =======")
            self.logger.warning(f"Option: {option_symbol}, Price: ${option_price:.2f}")
            self.logger.warning(f"Underlying: ${underlying_price:.2f}")
            
            # Calculate margin per contract
            if hasattr(margin_calculator, 'calculate_option_margin'):
                margin_result = margin_calculator.calculate_option_margin(
                    temp_position, underlying_price=underlying_price
                )
                temp_position_margin = margin_result.get('total_margin', option_price * 100)
                
                # Log the margin breakdown from SPAN calculator if available
                if 'margin_breakdown' in margin_result:
                    breakdown = margin_result['margin_breakdown']
                    for key, value in breakdown.items():
                        self.logger.warning(f"  {key}: ${value:.2f}")
                
                # Calculate the implied leverage
                implied_leverage = (underlying_price * 100) / temp_position_margin if temp_position_margin > 0 else 0
                self.logger.warning(f"  Implied leverage: {implied_leverage:.2f}x")
                
                # Get the combined margin per contract (with hedging if applicable)
                combined_margin_per_contract = temp_position_margin
                
                # Adjust for hedging benefits if available
                if self.hedging_manager and hasattr(self.hedging_manager, 'calculate_hedge_benefits'):
                    hedge_benefits = self.hedging_manager.calculate_hedge_benefits(
                        option_symbol, 1, temp_position.current_delta, option_price
                    )
                    
                    if hedge_benefits > 0:
                        self.logger.warning(f"  Hedge benefits: ${hedge_benefits:.2f}")
                        combined_margin_per_contract = max(temp_position_margin - hedge_benefits, 0)
                        
                self.logger.warning(f"  Final margin per contract: ${combined_margin_per_contract:.2f}")
            else:
                # Fall back to simple calculation
                combined_margin_per_contract = option_price * 100 * self.max_leverage
                self.logger.warning(f"  Using simple calculation: ${combined_margin_per_contract:.2f}")
            
            # For position sizing, use the hedged margin
            margin_per_contract = combined_margin_per_contract
            
            # Calculate combined margin for expected position size (based on capacity limit)
            capacity_estimate = int(remaining_margin_capacity / margin_per_contract) if margin_per_contract > 0 else 0
            self.logger.warning(f"Total position margin for {capacity_estimate} contracts (with SPAN): ${margin_per_contract * capacity_estimate:.2f}")
            self.logger.warning("=============================================")
        else:
            # Fall back to the simple calculation
            # Check if we have a pre-calculated margin from trading_engine
            if 'margin_per_contract' in option_data and option_data['margin_per_contract'] > 0:
                margin_per_contract = option_data['margin_per_contract']
                self.logger.info(f"[Position Sizing] Using pre-calculated SPAN margin: ${margin_per_contract:.2f} per contract")
                
                # If we have hedge benefit information, log it
                if 'hedge_benefit' in option_data:
                    self.logger.info(f"[Position Sizing] Pre-calculated hedge benefit: ${option_data['hedge_benefit']:.2f}")
            else:
                # Fall back to simple calculation
                margin_per_contract = option_price * 100 * self.max_leverage
                self.logger.info(f"[Position Sizing] Using simple margin calculation: ${margin_per_contract:.2f} per contract")
        
        if margin_per_contract <= 0:
            self.logger.warning(
                f"[PositionSizer] Invalid margin per contract for {option_symbol}: ${margin_per_contract:.2f}")
            return 0
            
        # Store the last calculated margin_per_contract for reference by other components
        self._last_margin_per_contract = margin_per_contract

        # Use helper method for position sizing calculation
        return self._calculate_position_size_from_margin(
            margin_per_contract, net_liq, available_margin, remaining_margin_capacity,
            option_symbol, option_price, risk_scaling, option_data
        )

    def _calculate_position_size_from_margin(self, margin_per_contract: float, net_liq: float, 
                                           available_margin: float, remaining_margin_capacity: float,
                                           option_symbol: str, option_price: float, risk_scaling: float = 1.0,
                                           option_data: Dict[str, Any] = None) -> int:
        """
        Calculate position size when margin per contract is already known.
        
        Args:
            margin_per_contract: Pre-calculated margin per contract
            net_liq: Net liquidation value
            available_margin: Available margin
            remaining_margin_capacity: Remaining margin capacity
            option_symbol: Option symbol
            option_price: Option price
            risk_scaling: Risk scaling factor
            option_data: Optional option data dictionary for enhanced logging
            
        Returns:
            int: Number of contracts to trade
        """
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
        self.logger.info(f"  Price: ${option_price:.2f}, Margin per contract: ${margin_per_contract:.2f} (hedged)")
        self.logger.info(f"  Risk scaling factor: {risk_scaling:.2f}")
        self.logger.info(f"  Net liquidation value: ${net_liq:.2f}")
        self.logger.info(f"  Available margin: ${available_margin:.2f}")
        self.logger.info(f"  Remaining margin capacity: ${remaining_margin_capacity:.2f}")
        self.logger.info(f"  Max contracts by capacity: {capacity_max_contracts}")
        self.logger.info(f"  Max contracts by available margin: {available_max_contracts}")
        self.logger.info(f"  Max contracts by NLV percent ({self.max_nlv_percent:.2%}): {max_position_contracts}")
        self.logger.info(f"  Final position size: {contracts} contracts")

        # Additional logging for hedge requirements if available
        if option_data and 'Delta' in option_data and option_data['Delta'] != 0:
            delta = option_data['Delta']
            option_quantity = contracts
            
            # Calculate hedge shares needed
            hedge_shares = -1 * delta * option_quantity * 100  # 100 shares per contract
            
            # If we have an underlying price, calculate the hedge value
            underlying_price = option_data.get('UnderlyingPrice', 0)
            if underlying_price > 0:
                hedge_value = abs(hedge_shares) * underlying_price
                self.logger.info(f"  Hedge requirement: {abs(hedge_shares):.0f} shares (${hedge_value:.2f})")
                
                # Estimate hedge margin (typically 50% for stocks)
                hedge_margin = hedge_value * 0.5  # 50% for long stock, conservative estimate
                total_margin = (margin_per_contract * contracts) + hedge_margin
                self.logger.info(f"  Total estimated margin (with hedge): ${total_margin:.2f}")

        return contracts
        
    # Additional utility methods can be added here 