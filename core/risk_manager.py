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
        
        # Track the most recently calculated margin per contract
        self._last_margin_per_contract = 0

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
            underlying_price = option_data.get('UnderlyingPrice', 0)
            strike = option_data.get('Strike', 0)
            expiry = option_data.get('Expiration', None)
            option_type = 'C' if 'C' in option_symbol else 'P'
        else:
            # This is a pandas Series
            option_price = option_data.get('MidPrice', option_data.get('Ask', 0))
            option_symbol = option_data.get('OptionSymbol', 'Unknown')
            underlying_price = option_data.get('UnderlyingPrice', 0)
            strike = option_data.get('Strike', 0)
            expiry = option_data.get('Expiration', None)
            option_type = 'C' if 'C' in option_symbol else 'P'

        # Make sure we have a valid price
        if option_price == 0:
            self.logger.warning(f"[RiskManager] Option price is 0 for {option_symbol}, cannot calculate position size")
            return 0

        # Check if we have access to portfolio's margin calculator for a more accurate calculation
        portfolio = portfolio_metrics.get('portfolio', None)
        margin_calculator = None
        margin_config = self.config.get('margin_management', {})
        margin_calculation_method = margin_config.get('margin_calculation_method', 'simple')
        margin_calculator_type = margin_config.get('margin_calculator_type', 'span').lower()
        use_portfolio_calculator = margin_config.get('use_portfolio_calculator', True)
        
        # Initialize margin_per_contract to a safe default
        margin_per_contract = option_price * 100
        
        if portfolio and hasattr(portfolio, 'margin_calculator') and use_portfolio_calculator:
            margin_calculator = portfolio.margin_calculator
            self.logger.info(f"[Position Sizing] Using portfolio margin calculator ({margin_calculation_method} method)")

            # Check if we have the correct calculator type based on configuration
            from core.margin import MarginCalculator, OptionMarginCalculator, SPANMarginCalculator
            
            if margin_calculator_type == 'span' and not isinstance(margin_calculator, SPANMarginCalculator):
                self.logger.info(f"[Position Sizing] Converting to SPANMarginCalculator as specified in config")
                margin_calculator = SPANMarginCalculator(
                    max_leverage=self.max_leverage,
                    hedge_credit_rate=0.8,  # Standard hedge credit rate
                    logger=self.logger
                )
            elif margin_calculator_type == 'option' and not isinstance(margin_calculator, OptionMarginCalculator):
                self.logger.info(f"[Position Sizing] Converting to OptionMarginCalculator as specified in config")
                margin_calculator = OptionMarginCalculator(
                    max_leverage=self.max_leverage,
                    logger=self.logger
                )
            elif margin_calculator_type == 'simple' and (
                isinstance(margin_calculator, SPANMarginCalculator) or 
                isinstance(margin_calculator, OptionMarginCalculator)
            ):
                self.logger.info(f"[Position Sizing] Converting to simple MarginCalculator as specified in config")
                margin_calculator = MarginCalculator(
                    max_leverage=self.max_leverage,
                    logger=self.logger
                )
        else:
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
            margin_per_contract = margin_calculator.calculate_position_margin(temp_position)
            
            # Add detailed trace logs for margin diagnosis
            self.logger.warning(f"[Margin Trace] Raw margin received from calculator: ${margin_per_contract:.4f}")
            self.logger.warning(f"[Margin Trace] Option price × 100: ${option_price * 100:.4f}")
            self.logger.warning(f"[Margin Trace] Ratio of margin to option price × 100: {margin_per_contract / (option_price * 100):.4f}")
            
            # Check if margin is suspiciously low (less than premium)
            if margin_per_contract < option_price * 100:
                self.logger.warning(f"[Margin Trace] Margin may be missing contract multiplier: ${margin_per_contract:.2f} vs option premium × 100: ${option_price * 100:.2f}")
                # Only apply multiplier if margin is EXTREMELY low (less than premium)
                if margin_per_contract < option_price:
                    self.logger.warning(f"[Margin Trace] Margin less than option premium, applying adjustment: ${margin_per_contract:.2f} → ${margin_per_contract * 100:.2f}")
                    margin_per_contract = margin_per_contract * 100
                
            # Ensure margin is never less than premium for short options (regulatory requirement)
            if margin_per_contract < option_price * 100:
                self.logger.warning(f"[Margin Trace] Adjusted margin still below option premium. Setting to option premium * 100: ${option_price * 100:.2f}")
                margin_per_contract = max(margin_per_contract, option_price * 100)
            
            # Calculate option delta and hedge details
            delta = temp_position.current_delta if hasattr(temp_position, 'current_delta') else option_data.get('Delta', 0)
            hedge_delta = -delta  # Opposite sign to the option delta
            hedge_shares = abs(hedge_delta) * 100  # 100 shares per contract
            
            # Log option and hedge delta info
            self.logger.warning(f"Option delta: {delta:.4f}, Hedge delta needed: {hedge_delta:.4f}")
            self.logger.warning(f"Shares to hedge 1 contract: {hedge_shares:.1f} at ${underlying_price:.2f}")
            
            # Calculate hedge margin (if needed)
            from core.position import Position
            
            # Create a temporary hedge position with proper attributes
            temp_hedge_position = Position(
                symbol=option_data.get('UnderlyingSymbol', 'SPY'),
                contracts=int(hedge_shares),  # Ensure this is an integer
                entry_price=underlying_price,
                current_price=underlying_price,
                is_short=hedge_delta < 0,  # Short if hedge delta is negative 
                position_type='stock',  # Explicitly set to stock
                logger=self.logger
            )
            
            # Ensure underlying_price is also set for the stock position (technically not needed but for consistency)
            if hasattr(temp_hedge_position, 'underlying_price'):
                temp_hedge_position.underlying_price = underlying_price
            
            # Set correct delta for the hedge position - this should be set based on is_short
            # For stock positions: delta = shares (positive if long, negative if short)
            temp_hedge_position.current_delta = -int(hedge_shares) if temp_hedge_position.is_short else int(hedge_shares)
            
            # Calculate the hedge margin using SPAN calculator
            hedge_margin = margin_calculator.calculate_position_margin(temp_hedge_position)
            self.logger.warning(f"Option margin (unhedged): ${margin_per_contract:.2f}")
            self.logger.warning(f"Standard hedge margin (25%): ${hedge_margin:.2f}")
            self.logger.warning(f"Simple sum: ${margin_per_contract + hedge_margin:.2f}")
            
            # Create a positions dictionary for portfolio margin calculation
            positions_dict = {
                temp_position.symbol: temp_position,
                temp_hedge_position.symbol: temp_hedge_position
            }
            
            # Calculate the proper portfolio margin for the combined position
            try:
                # Use calculate_portfolio_margin from span_calculator to ensure proper hedging benefits
                span_result = margin_calculator.calculate_portfolio_margin(positions_dict)
                
                # Handle both dictionary and float return types from calculate_portfolio_margin
                if isinstance(span_result, dict):
                    span_margin = span_result.get('total_margin', 0)
                    self.logger.warning(f"SPAN margin calculation returned dictionary with total_margin: ${span_margin:.2f}")
                else:
                    span_margin = span_result
                    self.logger.warning(f"SPAN margin calculation returned float value: ${span_margin:.2f}")
                
                # Verify result is reasonable - it should be less than the sum of individual margins
                # but not drastically less (typical offsets range from 10-30%)
                simple_combined = margin_per_contract + hedge_margin
                reasonable_min = simple_combined * 0.65  # Allow up to 35% offset
                reasonable_max = simple_combined * 1.1   # Allow up to 10% increase
                
                # Check if positions are actually hedging (opposite delta signs)
                offsetting_positions = (temp_position.current_delta * temp_hedge_position.current_delta < 0)
                
                # Calculate minimum reasonable margin (premium plus buffer)
                min_premium_margin = option_price * 100 * 1.1  # Premium + 10% buffer
                
                # Calculate maximum reasonable margin (sum with moderate offset)
                max_reasonable_margin = simple_combined * 0.8  # Allow up to 20% offset
                
                # For non-short options, premium floor doesn't apply
                if not temp_position.is_short:
                    min_premium_margin = 0
                
                # First verify if positions are actually hedging each other
                if not offsetting_positions:
                    # Positions are adding risk, no reduction should apply
                    self.logger.warning(f"Positions have same-direction deltas, not applying hedge benefit")
                    combined_margin_per_contract = simple_combined
                else:
                    # Positions are hedging, apply normal SPAN calculation with proper checks
                    # Minimum margin should be the greater of: premium buffer or max position margin * 0.5
                    position_min = max(margin_per_contract, hedge_margin) * 0.5
                    final_min = max(min_premium_margin, position_min)
                    
                    if span_margin < final_min:
                        # Result seems too aggressive, use more conservative estimate
                        self.logger.warning(f"SPAN margin ${span_margin:.2f} below minimum threshold ${final_min:.2f}")
                        combined_margin_per_contract = final_min
                    elif span_margin > max_reasonable_margin and span_margin > simple_combined:
                        # Result is higher than simple sum which is unusual for hedged positions
                        self.logger.warning(f"SPAN margin ${span_margin:.2f} above maximum threshold ${max_reasonable_margin:.2f}")
                        combined_margin_per_contract = simple_combined
                    else:
                        # SPAN result is reasonable, use it
                        self.logger.warning(f"Using SPAN portfolio margin: ${span_margin:.2f} ("+
                                        f"{(1 - span_margin/simple_combined):.1%} offset from simple sum)")
                        combined_margin_per_contract = span_margin
            except Exception as e:
                # If SPAN calculation fails, fall back to conservative approach
                self.logger.warning(f"Error calculating SPAN margin: {str(e)}")
                combined_margin_per_contract = (margin_per_contract + hedge_margin) * 0.85
                self.logger.warning(f"Using fallback margin calculation: ${combined_margin_per_contract:.2f} (15% offset)")
            
            # Calculate total hedge value for reporting
            hedge_value = hedge_shares * underlying_price
            
            # Log detailed margin breakdown
            self.logger.warning(f"Hedge value: ${hedge_value:.2f}, Standard hedge margin (25%): ${hedge_margin:.2f}")
            self.logger.warning(f"Margin per contract (UNHEDGED SPAN): ${margin_per_contract:.2f}")
            self.logger.warning(f"Margin per contract (HEDGED SPAN): ${combined_margin_per_contract:.2f}")
            
            # For position sizing, use the hedged margin
            margin_per_contract = combined_margin_per_contract
            
            # Calculate combined margin for expected position size (based on capacity limit)
            capacity_estimate = int(remaining_margin_capacity / margin_per_contract) if margin_per_contract > 0 else 0
            self.logger.warning(f"Total position margin for {capacity_estimate} contracts (with SPAN): ${margin_per_contract * capacity_estimate:.2f}")
            self.logger.warning("=============================================")
        else:
            # Fall back to the simple calculation
            margin_per_contract = option_price * 100 * self.max_leverage
            self.logger.info(f"[Position Sizing] Using simple margin calculation: ${margin_per_contract:.2f} per contract")
        
        if margin_per_contract <= 0:
            self.logger.warning(
                f"[RiskManager] Invalid margin per contract for {option_symbol}: ${margin_per_contract:.2f}")
            return 0
            
        # Store the last calculated margin_per_contract for reference by other components
        self._last_margin_per_contract = margin_per_contract

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
        self.logger.info(f"  Price: ${option_price:.2f}, Margin per contract: ${margin_per_contract:.2f} (unhedged)")
        
        # Estimate hedge requirements and margin impact
        # For short calls, delta is typically positive
        # For short puts, delta is typically negative
        delta = option_data.get('Delta', 0)
        hedge_delta = -delta  # Opposite direction of option delta
        hedge_shares = abs(hedge_delta) * 100  # 100 shares per contract
        hedge_value = hedge_shares * underlying_price
        hedge_margin_rate = 0.25  # Typical margin rate for long equity
        estimated_hedge_margin = hedge_value * hedge_margin_rate
        
        self.logger.info(f"  Hedge estimation (per contract):")
        self.logger.info(f"    Option delta: {delta:.4f}, Hedge delta: {hedge_delta:.4f}")
        self.logger.info(f"    Hedge shares: {hedge_shares:.0f} of underlying at ${underlying_price:.2f}")
        self.logger.info(f"    Hedge value: ${hedge_value:.2f}, Estimated hedge margin: ${estimated_hedge_margin:.2f}")
        self.logger.info(f"    Total estimated margin (with hedge): ${margin_per_contract + estimated_hedge_margin:.2f}")
        
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
        
        # Estimate total position margin (for informational purposes)
        total_option_margin = contracts * margin_per_contract
        total_hedge_margin = contracts * estimated_hedge_margin
        self.logger.info(f"  Estimated total margin (informational):")
        self.logger.info(f"    Option position ({contracts} contracts): ${total_option_margin:.2f}")
        self.logger.info(f"    Hedge position: ${total_hedge_margin:.2f}")
        self.logger.info(f"    Combined (before portfolio offsets): ${total_option_margin + total_hedge_margin:.2f}")
        self.logger.info(f"    Note: Actual margin will be less due to portfolio margin offsets")

        return contracts 