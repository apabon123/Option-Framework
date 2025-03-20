"""
Margin Calculation Module

This module provides tools for calculating margin requirements for different
types of positions, including options, futures, and stocks. It includes support
for both standard margin calculations and SPAN margin simulations.
"""

import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
from .position import Position, OptionPosition


class MarginCalculator:
    """
    Base margin calculator for simple margin calculations based on leverage.
    """

    def __init__(self, max_leverage: float = 1.0, logger: Optional[logging.Logger] = None):
        """
        Initialize the margin calculator.

        Args:
            max_leverage: Maximum leverage allowed (default: 1.0 = no leverage)
            logger: Logger instance
        """
        self.max_leverage = max_leverage
        self.logger = logger or logging.getLogger('trading')

    def calculate_position_margin(self, position: Position) -> float:
        """
        Calculate margin requirement for a position with better error handling.

        Args:
            position: Position to calculate margin for

        Returns:
            float: Margin requirement in dollars
        """
        # Check if this is a base class instance and we should delegate to a specialized calculator
        if self.__class__ == MarginCalculator and type(self).__name__ == 'MarginCalculator':
            # For proper SPAN margin calculations, delegate to SPANMarginCalculator
            if hasattr(position, 'option_type') or getattr(position, 'is_option', False):
                # Only import here to avoid circular imports
                from .margin import SPANMarginCalculator
                if hasattr(self, 'logger'):
                    self.logger.info(f"[Margin] Using specialized SPANMarginCalculator for option position margin")
                
                # Create a SPAN calculator with default parameters
                span_calculator = SPANMarginCalculator(
                    max_leverage=getattr(self, 'max_leverage', 1.0),
                    hedge_credit_rate=0.8,  # Standard hedge credit rate
                    logger=getattr(self, 'logger', None)
                )
                
                # Delegate to the SPAN calculator for option positions
                return span_calculator.calculate_position_margin(position)
                
        # Validate position object
        if not hasattr(position, 'contracts'):
            if self.logger:
                self.logger.warning(f"Invalid position object in calculate_position_margin: {type(position)}")
            return 0

        if position.contracts <= 0:
            return 0

        # For null positions, return zero margin
        if not hasattr(position, 'current_price') or position.current_price is None:
            return 0

        # For stock/ETF positions, use RegT margin requirements (25% of position value)
        if not hasattr(position, 'option_type') and not getattr(position, 'is_option', False):
            position_value = position.current_price * position.contracts
            reg_t_margin = max(position_value * 0.25, 2000 if position.contracts >= 100 else position_value * 0.25)
            
            if self.logger:
                self.logger.debug(f"[Margin] Stock/ETF position {position.symbol}: {position.contracts} shares")
                self.logger.debug(f"  Position value calculation: {position.current_price:.2f} × {position.contracts} = ${position_value:.2f}")
                self.logger.debug(f"  RegT margin calculation: max(${position_value:.2f} × 0.25, ${2000 if position.contracts >= 100 else position_value * 0.25:.2f}) = ${reg_t_margin:.2f}")
                self.logger.debug(f"  Final stock/ETF margin: ${reg_t_margin:.2f}")
            
            return reg_t_margin
            
        # Basic margin calculation for options
        leverage = getattr(self, 'max_leverage', 1.0)

        # Use either avg_entry_price or current_price
        position_price = position.current_price
        if hasattr(position, 'avg_entry_price') and position.avg_entry_price is not None:
            position_price = position.avg_entry_price

        # Log starting values
        if self.logger:
            self.logger.debug(f"[Margin] Beginning calculation for {position.symbol}")
            self.logger.debug(f"  Position type: {type(position).__name__}")
            self.logger.debug(f"  Contracts: {position.contracts}")
            self.logger.debug(f"  Price: ${position_price:.4f}")
            self.logger.debug(f"  Is short: {position.is_short}")

        # For options, multiply by 100 (contract multiplier)
        contract_multiplier = 100 if hasattr(position, 'option_type') else 1
        initial_margin = position_price * position.contracts * contract_multiplier / leverage

        # Adjust margin for unrealized PnL if position is short
        adjusted_margin = initial_margin
        if hasattr(position, 'unrealized_pnl') and position.is_short:
            adjusted_margin = initial_margin + position.unrealized_pnl

        # Log the margin calculation
        if self.logger:
            self.logger.debug(f"[Margin] Position {position.symbol}: {position.contracts} contracts")
            self.logger.debug(f"  Contract multiplier: {contract_multiplier}")
            self.logger.debug(f"  Initial margin calculation: {position_price:.4f} × {position.contracts} × {contract_multiplier} / {leverage:.2f} = ${initial_margin:.2f}")
            if hasattr(position, 'unrealized_pnl') and position.is_short:
                self.logger.debug(f"  Adjusted for unrealized PnL (${position.unrealized_pnl:.2f}): ${adjusted_margin:.2f}")
            self.logger.debug(f"  Final margin: ${adjusted_margin:.2f}")

        return max(adjusted_margin, 0)  # Ensure non-negative margin

    def calculate_portfolio_margin(self, positions_or_portfolio):
        """
        Calculate margin for a portfolio of positions, with safety checks.
        
        This improved version handles both dictionary of positions and Portfolio objects.
        
        Args:
            positions_or_portfolio: Dictionary of positions by symbol or Portfolio object
            
        Returns:
            dict: Dictionary with total margin and margin by position
        """
        # Check if this is a base class instance and there's a subclass implementation
        # If we're using a base MarginCalculator instance but a child class is expected
        if self.__class__ == MarginCalculator and type(self).__name__ == 'MarginCalculator':
            # Check if SPAN margin calculator should be used based on configuration
            from .margin import SPANMarginCalculator
            if hasattr(self, 'logger'):
                self.logger.info(f"[Margin] Using base MarginCalculator, but child class may be needed. Checking if SPAN is required.")
            
            # Create a SPAN margin calculator with default parameters
            span_calculator = SPANMarginCalculator(
                max_leverage=getattr(self, 'max_leverage', 1.0),
                hedge_credit_rate=0.8,  # Standard hedge credit rate
                logger=getattr(self, 'logger', None)
            )
            
            if hasattr(self, 'logger'):
                self.logger.info(f"[Margin] Delegating to SPANMarginCalculator for proper hedge benefits")
            
            # Delegate to the SPAN calculator
            return span_calculator.calculate_portfolio_margin(positions_or_portfolio)
            
        # Handle the case where a Portfolio object is passed instead of a dictionary
        positions = {}
        
        # Check if positions_or_portfolio is a Portfolio object
        if hasattr(positions_or_portfolio, 'positions'):
            positions = positions_or_portfolio.positions
        # Check if it's already a dictionary
        elif isinstance(positions_or_portfolio, dict):
            positions = positions_or_portfolio
        else:
            # Return zero margin if input is invalid
            if self.logger:
                self.logger.warning(f"Invalid input to calculate_portfolio_margin: {type(positions_or_portfolio)}")
            return {'total_margin': 0, 'margin_by_position': {}}
        
        if not positions:
            return {'total_margin': 0, 'margin_by_position': {}}
        
        # Calculate margin for each position with error handling
        margin_by_position = {}
        total_margin = 0
        
        for symbol, position in positions.items():
            try:
                # Ensure position is a valid Position object
                if not hasattr(position, 'contracts'):
                    if self.logger:
                        self.logger.warning(f"Invalid position object for {symbol}: {type(position)}")
                    continue
                    
                position_margin = self.calculate_position_margin(position)
                margin_by_position[symbol] = position_margin
                total_margin += position_margin
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error calculating margin for {symbol}: {e}")
                # Continue processing other positions
                continue
        
        # Log the portfolio margin
        if self.logger:
            self.logger.debug(f"[Margin] Portfolio of {len(positions)} positions")
            self.logger.debug(f"  Total margin: ${total_margin:.2f}")
        
        return {
            'total_margin': total_margin,
            'margin_by_position': margin_by_position,
            'hedging_benefits': 0  # Base calculator doesn't provide hedging benefits
        }

    def calculate_total_margin(self, positions: Dict[str, Position]) -> float:
        """
        Calculate total margin requirement for all positions.

        Args:
            positions: Dictionary of positions by symbol

        Returns:
            float: Total margin requirement in dollars
        """
        result = self.calculate_portfolio_margin(positions)
        return result.get('total_margin', 0)


class OptionMarginCalculator(MarginCalculator):
    """
    Margin calculator specialized for options with more accurate margin 
    calculations based on option type and moneyness.
    """

    def __init__(
            self,
            max_leverage: float = 1.0,
            otm_margin_multiplier: float = 0.8,  # Lower margin for OTM options
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the option margin calculator.

        Args:
            max_leverage: Maximum leverage allowed
            otm_margin_multiplier: Multiplier for OTM options (reduces margin requirements)
            logger: Logger instance
        """
        super().__init__(max_leverage, logger)
        self.otm_margin_multiplier = otm_margin_multiplier

    def calculate_position_margin(self, position: Position) -> float:
        """
        Calculate margin requirement for an option position with OTM discount.

        Args:
            position: Option position to calculate margin for

        Returns:
            float: Margin requirement in dollars
        """
        # Validate position
        if position.contracts <= 0:
            return 0

        # If not an option position, use the base class implementation
        if not isinstance(position, OptionPosition):
            return super().calculate_position_margin(position)

        # Log the input position details
        if self.logger:
            self.logger.debug(f"[Margin] Option position {position.symbol}: {position.contracts} contracts")
            self.logger.debug(f"  Option type: {position.option_type if hasattr(position, 'option_type') else 'unknown'}")
            self.logger.debug(f"  Price: ${position.current_price:.4f}")
            self.logger.debug(f"  Contract multiplier: 100")
        
        # Basic margin calculation with contract multiplier
        initial_margin = position.current_price * position.contracts * 100 * self.max_leverage

        # Apply OTM discount
        is_otm = not position.is_itm()
        if is_otm:
            initial_margin *= self.otm_margin_multiplier
            if self.logger:
                self.logger.debug(f"  OTM discount applied: {self.otm_margin_multiplier:.2f} × ${initial_margin / self.otm_margin_multiplier:.2f} = ${initial_margin:.2f}")

        # Adjust for unrealized PnL for short positions (losses increase margin)
        if position.is_short and position.unrealized_pnl < 0:
            adjusted_margin = initial_margin - position.unrealized_pnl  # Negative PnL becomes positive addition
        else:
            adjusted_margin = initial_margin

        # Calculate option premium (price * contracts * contract multiplier)
        option_premium = position.current_price * position.contracts * 100

        # For short options, ensure margin is never less than the option premium
        if position.is_short and adjusted_margin < option_premium:
            if self.logger:
                self.logger.warning(f"  WARNING: Calculated margin (${adjusted_margin:.2f}) is less than option premium (${option_premium:.2f})")
                self.logger.info(f"  Setting margin to option premium: ${option_premium:.2f}")
            adjusted_margin = option_premium

        # Log the calculation steps
        if self.logger:
            self.logger.debug(f"  Initial margin calculation: {position.current_price:.4f} × {position.contracts} × 100 × {self.max_leverage:.2f} = ${initial_margin:.2f}")
            if position.is_short and position.unrealized_pnl < 0:
                self.logger.debug(f"  Unrealized PnL adjustment: ${-position.unrealized_pnl:.2f}")
            self.logger.debug(f"  Final margin: ${adjusted_margin:.2f}")

        return max(adjusted_margin, 0)  # Ensure non-negative margin

    # We inherit calculate_portfolio_margin and calculate_total_margin from the base class


class SPANMarginCalculator(MarginCalculator):
    """
    Margin calculator for portfolio margin using the SPAN methodology.
    
    This calculator implements a simplified version of the SPAN (Standard Portfolio 
    Analysis of Risk) methodology used by major clearing houses for margin requirements.
    It accounts for:
    
    1. Price risk (delta and gamma)
    2. Time decay risk (theta)
    3. Volatility risk (vega)
    4. Correlation between positions in the same underlying
    5. Offsets between hedged positions
    
    It is designed for option portfolios with proper Greek calculations.
    """

    def __init__(
            self,
            max_leverage: float = 1.0,
            volatility_multiplier: float = 1.0,  # Scenario stress multiplier
            correlation_matrix: Optional[pd.DataFrame] = None,  # For correlations between underlyings
            initial_margin_percentage: float = 0.1,  # Initial margin as percentage of notional
            maintenance_margin_percentage: float = 0.07,  # Maintenance margin as percentage of notional
            hedge_credit_rate: float = 0.8,  # Credit rate for hedged positions (0-1)
            price_move_pct: float = 0.05,  # Price move scenario (5% by default)
            vol_shift_pct: float = 0.3,  # Volatility shift scenario (30% by default)
            gamma_scaling_factor: float = 0.3,  # Scaling factor for gamma effects 
            min_scan_risk_percentage: float = 0.25,  # Minimum scan risk as percentage of premium
            max_margin_to_premium_ratio: float = 20.0,  # Cap on margin-to-premium ratio
            otm_scaling_enabled: bool = True,  # Whether to scale margin for OTM options
            otm_minimum_scaling: float = 0.1,  # Minimum scaling for far OTM options (10%)
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the SPAN margin calculator with configurable parameters.
        
        Args:
            max_leverage: Maximum leverage allowed (default: 1.0 = no leverage)
            volatility_multiplier: Multiplier for volatility stress scenarios (1.0 = no stress)
            correlation_matrix: DataFrame with correlation coefficients between underlyings
            initial_margin_percentage: Initial margin as percentage of notional value (0.1 = 10%)
            maintenance_margin_percentage: Maintenance margin as percentage of notional (0.07 = 7%)
            hedge_credit_rate: Credit rate for hedged positions (0.8 = 80% credit)
            price_move_pct: Percentage price move for risk scenarios (0.05 = 5%)
            vol_shift_pct: Volatility shift for risk scenarios (0.3 = 30%)
            gamma_scaling_factor: Scaling factor for gamma risk (0.3 = 30%)
            min_scan_risk_percentage: Minimum scan risk as percentage of premium (0.25 = 25%)
            max_margin_to_premium_ratio: Cap on margin-to-premium ratio (20.0 = 20x premium)
            otm_scaling_enabled: Whether to scale margin for OTM options
            otm_minimum_scaling: Minimum scaling for far OTM options (0.1 = 10%)
            logger: Logger instance
        """
        super().__init__(max_leverage, logger)
        self.volatility_multiplier = volatility_multiplier
        self.correlation_matrix = correlation_matrix
        self.initial_margin_percentage = initial_margin_percentage
        self.maintenance_margin_percentage = maintenance_margin_percentage
        self.hedge_credit_rate = min(max(hedge_credit_rate, 0), 1)  # Ensure between 0-1
        
        # Store new parameters
        self.price_move_pct = price_move_pct
        self.vol_shift_pct = vol_shift_pct
        self.gamma_scaling_factor = gamma_scaling_factor
        self.min_scan_risk_percentage = min_scan_risk_percentage
        self.max_margin_to_premium_ratio = max_margin_to_premium_ratio
        self.otm_scaling_enabled = otm_scaling_enabled
        self.otm_minimum_scaling = otm_minimum_scaling
        
        # Log initialization parameters
        if self.logger:
            self.logger.info(f"[SPAN Margin] Initialized SPANMarginCalculator with parameters:")
            self.logger.info(f"  max_leverage: {max_leverage:.2f}")
            self.logger.info(f"  volatility_multiplier: {volatility_multiplier:.2f}")
            self.logger.info(f"  initial_margin_percentage: {initial_margin_percentage:.4f} ({initial_margin_percentage*100:.2f}%)")
            self.logger.info(f"  maintenance_margin_percentage: {maintenance_margin_percentage:.4f} ({maintenance_margin_percentage*100:.2f}%)")
            self.logger.info(f"  hedge_credit_rate: {hedge_credit_rate:.2f}")
            self.logger.info(f"  price_move_pct: {price_move_pct:.2f} ({price_move_pct*100:.0f}%)")
            self.logger.info(f"  vol_shift_pct: {vol_shift_pct:.2f} ({vol_shift_pct*100:.0f}%)")
            self.logger.info(f"  gamma_scaling_factor: {gamma_scaling_factor:.2f}")
            self.logger.info(f"  min_scan_risk_percentage: {min_scan_risk_percentage:.2f} ({min_scan_risk_percentage*100:.0f}%)")
            self.logger.info(f"  max_margin_to_premium_ratio: {max_margin_to_premium_ratio:.1f}x")
            self.logger.info(f"  otm_scaling_enabled: {otm_scaling_enabled}")
            self.logger.info(f"  otm_minimum_scaling: {otm_minimum_scaling:.2f} ({otm_minimum_scaling*100:.0f}%)")
            
            # Warn if the initial_margin_percentage seems too low
            if initial_margin_percentage < 0.01:
                self.logger.warning(f"[SPAN Margin] WARNING: initial_margin_percentage is very low ({initial_margin_percentage:.4f})")
                self.logger.warning(f"[SPAN Margin] This may result in unexpectedly low margin requirements")
            # Warn if the initial_margin_percentage seems abnormally high
            elif initial_margin_percentage > 0.5:
                self.logger.warning(f"[SPAN Margin] WARNING: initial_margin_percentage is unusually high ({initial_margin_percentage:.4f})")
                self.logger.warning(f"[SPAN Margin] This may result in extremely high margin requirements")

    def _calculate_scan_risk(self, position: Position) -> float:
        """
        Calculate scan risk (price and volatility risk) for a position.
        
        Args:
            position: Position to calculate scan risk for
            
        Returns:
            float: Scan risk amount (dollar value)
        """
        # If position is not an option, return a basic margin calculation
        if not isinstance(position, OptionPosition):
            return position.calculate_margin_requirement(self.max_leverage)
        
        # Get position details
        option_price = position.current_price
        if option_price <= 0:
            option_price = position.avg_entry_price

        # For short positions, we're looking at potential losses
        is_short = position.is_short
        is_call = position.option_type.upper() in ['C', 'CALL']
        is_put = position.option_type.upper() in ['P', 'PUT']
        
        # Get option Greeks
        delta = position.current_delta
        gamma = position.current_gamma
        vega = position.current_vega
        underlying_price = position.underlying_price
        otm_amount = 0
        
        # Log option details for debugging
        if self.logger:
            self.logger.debug(f"[Margin] Option type: {position.option_type}, Is short: {is_short}")
            self.logger.debug(f"[Margin] Delta: {delta:.4f}, Gamma: {gamma:.6f}, Vega: {vega:.4f}")
            self.logger.debug(f"[Margin] Underlying price: {underlying_price:.2f}")
            self.logger.debug(f"[Margin] Strike price: {position.strike:.2f}")

        # Ensure delta has the correct sign: 
        # - Long call: positive delta
        # - Short call: negative delta
        # - Long put: negative delta
        # - Short put: positive delta (for margin calculation)
        # This is for CALCULATION only - not the stored value
        calc_delta = delta
        
        # Ensure the delta sign is correct for all option types
        if is_short and is_call and delta > 0:
            # Short call should have negative delta
            calc_delta = -delta
            if self.logger:
                self.logger.debug(f"[Margin] Flipping delta sign for short call: {delta:.4f} -> {calc_delta:.4f}")
        elif is_short and is_put and delta < 0:
            # Short put should have positive delta for risk calculation
            calc_delta = abs(delta)
            if self.logger:
                self.logger.debug(f"[Margin] Flipping delta sign for short put: {delta:.4f} -> {calc_delta:.4f}")
        elif not is_short and is_call and delta < 0:
            # Long call should have positive delta
            calc_delta = abs(delta)
            if self.logger:
                self.logger.debug(f"[Margin] Flipping delta sign for long call: {delta:.4f} -> {calc_delta:.4f}")
        elif not is_short and is_put and delta > 0:
            # Long put should have negative delta
            calc_delta = -delta
            if self.logger:
                self.logger.debug(f"[Margin] Flipping delta sign for long put: {delta:.4f} -> {calc_delta:.4f}")
                
        # Log the delta we're using for calculation
        if self.logger:
            self.logger.debug(f"[Margin] Using delta for calculation: {calc_delta:.4f}")

        # Calculate price move risk scenarios
        price_up_scenario = underlying_price * (1 + self.price_move_pct)
        price_down_scenario = underlying_price * (1 - self.price_move_pct)
        
        # Calculate the absolute price move amount (for gamma calculation)
        price_move_amount = underlying_price * self.price_move_pct
        
        # Calculate delta risk component (price move effect)
        delta_risk = calc_delta * price_move_amount
        
        # Calculate gamma risk component (convexity effect)
        # For a price move of p, gamma effect is approximately 0.5 * gamma * p^2
        gamma_risk = 0.5 * gamma * (price_move_amount ** 2)
        
        # Scale gamma risk to avoid overweighting
        gamma_risk = gamma_risk * self.gamma_scaling_factor

        # Calculate volatility risk component
        # For a price move of p, vega effect is approximately vega * p
        vega_risk = vega * price_move_amount

        # Combine all risk components
        # SPAN typically takes the worst case scenario across 16 scenarios
        # We're simplifying by using the sum of absolute effects as a conservative approach
        total_risk = abs(delta_risk) + abs(gamma_risk) + abs(vega_risk)
        
        # Calculate option premium for comparison
        option_premium = position.current_price * position.contracts * 100
        
        # Sanity check: if scan risk is suspiciously low compared to premium, apply a correction
        min_scan_risk_pct = getattr(self, 'min_scan_risk_percentage', 0.25)  # Default to 25% if not configured
        if total_risk < option_premium * min_scan_risk_pct:
            if self.logger:
                self.logger.warning(f"[Scan Risk] WARNING: Calculated scan risk (${total_risk:.2f}) is suspiciously low compared to premium (${option_premium:.2f})")
                self.logger.warning(f"[Scan Risk] Applying minimum scan risk of {min_scan_risk_pct*100:.0f}% of premium: ${option_premium * min_scan_risk_pct:.2f}")
            total_risk = max(total_risk, option_premium * min_scan_risk_pct)
        
        if self.logger:
            self.logger.info(f"[Scan Risk] Final scan risk for {position.symbol}: ${total_risk:.2f}")
        
        return total_risk

    def calculate_position_margin(self, position: Position) -> float:
        """
        Calculate margin requirement for a position using SPAN methodology.

        Args:
            position: Position to calculate margin for

        Returns:
            float: Margin requirement in dollars
        """
        if position.contracts <= 0:
            return 0

        # Log the input position details at the very beginning
        if self.logger:
            self.logger.info(f"[Margin] Starting SPAN margin calculation for {position.symbol}")
            self.logger.info(f"  Position type: {type(position).__name__}")
            self.logger.info(f"  Contracts: {position.contracts}")
            self.logger.info(f"  Current price: ${position.current_price:.4f}")
            self.logger.info(f"  Is short: {position.is_short}")
            
            # Log additional option-specific properties if available
            if hasattr(position, 'option_type'):
                self.logger.info(f"  Option type: {position.option_type}")
                self.logger.info(f"  Strike price: ${position.strike:.2f}")
                self.logger.info(f"  Underlying price: ${position.underlying_price:.2f}")
                if hasattr(position, 'current_delta'):
                    self.logger.info(f"  Delta: {position.current_delta:.4f}")

        # For stock/ETF positions used in hedging
        if not isinstance(position, OptionPosition):
            # For ETFs and stocks, use the RegT margin requirement (higher of 25% of position value or $2000)
            position_value = position.current_price * position.contracts
            reg_t_margin = max(position_value * 0.25, 2000 if position.contracts >= 100 else position_value * 0.25)

            if self.logger:
                self.logger.info(f"  Stock/ETF position {position.symbol}: {position.contracts} shares")
                self.logger.info(f"  Position value calculation: {position.current_price:.2f} × {position.contracts} = ${position_value:.2f}")
                self.logger.info(f"  RegT margin calculation: max(${position_value:.2f} × 0.25, ${2000 if position.contracts >= 100 else position_value * 0.25:.2f}) = ${reg_t_margin:.2f}")
                self.logger.info(f"  Final stock/ETF margin: ${reg_t_margin:.2f}")

            return reg_t_margin

        # For option positions
        # Log all the parameters that will be used in the calculation
        if self.logger:
            self.logger.info(f"  ===== Option Margin Calculation Steps =====")
            self.logger.info(f"  Option symbol: {position.symbol}")
            self.logger.info(f"  Option price per share: ${position.current_price:.4f}")
            self.logger.info(f"  Contract multiplier: 100 (standard for equity options)")
            self.logger.info(f"  Total premium per contract: ${position.current_price * 100:.2f}")
            self.logger.info(f"  Number of contracts: {position.contracts}")
            self.logger.info(f"  Total option value: ${position.current_price * position.contracts * 100:.2f}")
            self.logger.info(f"  Underlying price: ${position.underlying_price:.2f}")
            self.logger.info(f"  Initial margin percentage: {self.initial_margin_percentage:.2%}")
        
        # Calculate option premium with contract multiplier properly applied
        contract_multiplier = 100  # Standard for equity options
        option_premium = position.current_price * position.contracts * contract_multiplier
        
        if self.logger:
            self.logger.info(f"  Option premium calculation: {position.current_price:.4f} × {position.contracts} × {contract_multiplier} = ${option_premium:.2f}")
        
        # Calculate notional value (underlying price * contracts * 100 shares per contract)
        notional_value = position.underlying_price * position.contracts * contract_multiplier
        
        # Add a safety check for zero underlying price
        if position.underlying_price <= 0 and self.logger:
            self.logger.warning(f"[Margin] WARNING: Zero or negative underlying price detected for {position.symbol}")
            # Try to estimate a reasonable underlying price from option data
            if hasattr(position, 'strike') and position.strike > 0:
                estimated_price = position.strike
                self.logger.warning(f"[Margin] Using strike price as estimate: ${estimated_price:.2f}")
                notional_value = estimated_price * position.contracts * contract_multiplier
        
        if self.logger:
            self.logger.info(f"  Notional calculation: {position.underlying_price:.2f} × {position.contracts} × {contract_multiplier} = ${notional_value:.2f}")

        # ENHANCEMENT: Scale the margin percentage based on moneyness and delta
        # For far OTM options, we'll use a reduced margin percentage
        adjusted_margin_percentage = self.initial_margin_percentage
        
        # Check if we have delta information - use it to scale the margin
        if hasattr(position, 'current_delta') and hasattr(position, 'strike'):
            # Get the absolute value of delta (0 to 1)
            abs_delta = abs(position.current_delta)
            
            # Calculate moneyness: how far ITM or OTM the option is
            if position.option_type.upper() in ['C', 'CALL']:
                moneyness = (position.underlying_price / position.strike) - 1
            elif position.option_type.upper() in ['P', 'PUT']:
                moneyness = 1 - (position.underlying_price / position.strike)
            else:
                # Default to 0 if option type is unknown
                moneyness = 0
                
            # Log moneyness info
            if self.logger:
                self.logger.info(f"  Option moneyness: {moneyness:.2%}")
                self.logger.info(f"  Absolute delta: {abs_delta:.4f}")
            
            # Scale the margin percentage based on moneyness and delta
            # Far OTM options (negative moneyness) get reduced margin
            if self.otm_scaling_enabled and moneyness < -0.05:  # More than 5% OTM
                # The deeper OTM and smaller delta, the lower the margin percentage
                # For very deep OTM options, this could be as low as 10% of the original
                scaling_factor = min(1.0, max(self.otm_minimum_scaling, abs_delta * 5))  # 0.1 to 1.0 scaling
                adjusted_margin_percentage *= scaling_factor
                
                if self.logger:
                    self.logger.info(f"  Far OTM option detected. Scaling margin by factor: {scaling_factor:.2f}")
                    self.logger.info(f"  Adjusted margin percentage: {adjusted_margin_percentage:.2%}")
        
        # Calculate initial margin based on adjusted notional
        base_margin = notional_value * adjusted_margin_percentage
        
        if self.logger:
            self.logger.info(f"  Base margin calculation: ${notional_value:.2f} × {adjusted_margin_percentage:.4f} = ${base_margin:.2f}")

        # Calculate scan risk (risk of market move)
        scan_risk = self._calculate_scan_risk(position)
        
        if self.logger:
            self.logger.info(f"  Scan risk from _calculate_scan_risk: ${scan_risk:.2f}")

        # Use the maximum of base margin and scan risk
        margin = max(base_margin, scan_risk)
        
        if self.logger:
            self.logger.info(f"  Initial margin (max of base_margin and scan_risk): ${margin:.2f}")
        
        # Ensure margin is never less than option premium for short options
        if position.is_short:
            old_margin = margin
            margin = max(margin, option_premium)
            
            if self.logger:
                if margin > old_margin:
                    self.logger.warning(f"  Short option premium (${option_premium:.2f}) is higher than calculated margin (${old_margin:.2f})")
                    self.logger.warning(f"  Setting margin to option premium: ${margin:.2f}")
                else:
                    self.logger.info(f"  Short option premium: ${option_premium:.2f} (margin is already higher)")

        # ENHANCEMENT: Apply a cap on the margin-to-premium ratio
        # Industry practice typically has a maximum ratio, especially for OTM options
        if option_premium > 0:
            margin_to_premium_ratio = margin / option_premium
            
            if margin_to_premium_ratio > self.max_margin_to_premium_ratio:
                capped_margin = option_premium * self.max_margin_to_premium_ratio
                
                if self.logger:
                    self.logger.warning(f"  Margin-to-premium ratio too high: {margin_to_premium_ratio:.2f}x")
                    self.logger.warning(f"  Capping margin at {self.max_margin_to_premium_ratio:.1f}x premium: ${capped_margin:.2f}")
                
                margin = capped_margin

        # Check if computed margin is suspiciously low compared to premium
        if margin < option_premium * 0.5 and option_premium > 0:
            if self.logger:
                self.logger.warning(f"  WARNING: Calculated margin (${margin:.2f}) is much lower than premium (${option_premium:.2f})")
                self.logger.warning(f"  This may indicate a missing contract multiplier (100x) in the calculation")
            # Apply the contract multiplier to the margin if it appears to be missing
            corrected_margin = margin * contract_multiplier
            if self.logger:
                self.logger.warning(f"  Automatically correcting: ${margin:.2f} → ${corrected_margin:.2f}")
            margin = corrected_margin
        
        if self.logger:
            if option_premium > 0:
                self.logger.info(f"  Final margin-to-premium ratio: {margin/option_premium:.2f}x")
            self.logger.info(f"  Final margin for {position.symbol}: ${margin:.2f}")
            self.logger.info(f"  ===== End of Margin Calculation =====")

        return margin

    def _extract_underlying_symbol(self, position: Position) -> str:
        """
        Extract the underlying symbol from a position.

        Args:
            position: Position to extract underlying from

        Returns:
            str: Underlying symbol
        """
        if isinstance(position, OptionPosition):
            # Try to extract from symbol using common patterns
            symbol = position.symbol

            # For standard option symbols like "SPY240621C00410000"
            if len(symbol) > 10:
                # Attempt to extract underlying ticker
                # This is a simplified approach - real option symbols might need more complex parsing
                import re
                match = re.match(r'([A-Z]+)', symbol)
                if match:
                    return match.group(1)

            # If we can't extract from symbol, return first 3-5 chars as best guess
            return symbol[:min(5, len(symbol))]

        # For stock/ETF positions, just return the symbol
        return position.symbol

    def calculate_portfolio_margin(self, positions: Dict[str, Position]) -> Dict[str, Any]:
        """
        Calculate portfolio margin for a collection of positions.
        Applies hedging benefits for offsetting positions in the same underlying.
        
        Args:
            positions (dict): Dictionary of positions keyed by symbol
            
        Returns:
            dict: Dictionary with margin amounts and hedging benefits
        """
        if not positions:
            return {"total_margin": 0, "margin_by_position": {}, "hedging_benefits": 0}
        
        # Calculate standalone margin for each position
        standalone_margin = 0
        margin_by_position = {}
        
        for symbol, position in positions.items():
            position_margin = self.calculate_position_margin(position)
            standalone_margin += position_margin
            margin_by_position[symbol] = position_margin
            if self.logger:
                self.logger.debug(f"[Margin] {symbol}: Initial margin=${position_margin:.2f}")
        
        if self.logger:
            self.logger.debug(f"[Margin] Portfolio of {len(positions)} positions")
        
        # Group positions by underlying
        positions_by_underlying = {}
        for symbol, position in positions.items():
            underlying = None
            if hasattr(position, 'underlying') and position.underlying:
                underlying = position.underlying
            else:
                underlying = symbol  # Use symbol as underlying for non-options
            
            if underlying not in positions_by_underlying:
                positions_by_underlying[underlying] = []
            
            positions_by_underlying[underlying].append(symbol)
        
        if self.logger:
            self.logger.debug("[Margin] Grouped positions by underlying:")
            for underlying, symbols in positions_by_underlying.items():
                self.logger.debug(f"  {underlying}: {symbols}")
        
        # Calculate delta-based hedging offsets for each underlying group
        total_hedging_benefit = 0
        
        for underlying, symbols in positions_by_underlying.items():
            if len(symbols) <= 1:
                continue  # Need at least 2 positions for hedging
            
            # Calculate position deltas for this underlying
            position_deltas = {}
            net_delta = 0
            total_absolute_delta = 0
            
            for symbol in symbols:
                position = positions[symbol]
                
                # Calculate delta based on position type
                if isinstance(position, OptionPosition):
                    # Get the option delta value - sign is already flipped for short positions
                    # in the OptionPosition constructor
                    delta_value = position.current_delta
                    
                    # Convert to total delta for the position
                    position_delta = delta_value * position.contracts * 100  # 100 shares per contract
                    
                    if self.logger:
                        self.logger.debug(f"  Added delta for {symbol} (Option): {position_delta:.2f}")
                else:
                    # For stocks/ETFs, each share has a delta of 1.0
                    # CRITICAL FIX: Do NOT multiply by 100 for stock positions!
                    # Each share already has a delta of 1.0
                    position_delta = position.contracts
                    
                    # Apply sign based on position direction
                    if position.is_short:
                        position_delta = -position_delta
                    
                    if self.logger:
                        self.logger.debug(f"  Using stock delta for {symbol}: {position_delta:.2f}")
                        self.logger.debug(f"  Added delta for {symbol} (Stock/ETF): {position_delta:.2f}")
                
                position_deltas[symbol] = position_delta
                net_delta += position_delta
                total_absolute_delta += abs(position_delta)
            
            if self.logger:
                self.logger.debug(f"[Margin] Position deltas for {underlying} group:")
                for symbol, delta in position_deltas.items():
                    position_type = "Option" if isinstance(positions[symbol], OptionPosition) else "Stock/ETF"
                    self.logger.debug(f"  {symbol} ({position_type}): Delta={delta:.2f}")
                self.logger.debug(f"  Net delta: {net_delta:.2f}")
                self.logger.debug(f"  Total absolute delta: {total_absolute_delta:.2f}")
            
            # Check if positions are in opposite directions - if net delta is less than total absolute delta
            # that means there's some offsetting happening
            if abs(net_delta) < total_absolute_delta:
                # Calculate delta hedge quality (how well the positions offset)
                # A perfect hedge would have net_delta = 0 and delta_hedge_quality = 1.0
                delta_hedge_quality = 1.0 - (abs(net_delta) / total_absolute_delta)
                
                # Apply offset factor based on hedge quality
                offset_factor = self.hedge_credit_rate * delta_hedge_quality
                
                # Calculate the margin for this group
                group_margin = sum(margin_by_position[symbol] for symbol in symbols)
                group_hedging_benefit = group_margin * offset_factor
                
                if self.logger:
                    self.logger.debug(f"  [Margin] Partial delta hedge ({delta_hedge_quality*100:.1f}% effective) - applying partial offset")
                    self.logger.debug(f"[Margin] Hedging benefit for {underlying} group: ${group_hedging_benefit:.2f}")
                    self.logger.debug(f"  Net delta: {net_delta:.2f}, Total absolute delta: {total_absolute_delta:.2f}")
                    self.logger.debug(f"  Delta hedge quality: {delta_hedge_quality*100:.1f}% effective")
                    self.logger.debug(f"  Offset factor: {offset_factor:.2f}")
                    self.logger.debug(f"  Standalone margin: ${group_margin:.2f}")
                    self.logger.debug(f"  Reduced margin: ${group_margin - group_hedging_benefit:.2f}")
                
                total_hedging_benefit += group_hedging_benefit
            else:
                if self.logger:
                    self.logger.debug(f"[Margin] No hedging benefit for {underlying} group - all positions in same direction")
                    self.logger.debug(f"  Net delta: {net_delta:.2f}")
                    self.logger.debug(f"  Standalone margin: ${sum(margin_by_position[symbol] for symbol in symbols):.2f}")
        
        # Apply the total hedging benefit
        portfolio_margin = standalone_margin - total_hedging_benefit
        
        if self.logger:
            self.logger.info("[Margin] Portfolio margin calculation complete")
            self.logger.info(f"  Total margin: ${portfolio_margin:.2f}")
            self.logger.info(f"  Total hedging benefits: ${total_hedging_benefit:.2f}")
        
        return {
            "total_margin": portfolio_margin,
            "margin_by_position": margin_by_position,
            "hedging_benefits": total_hedging_benefit
        }

    def test_option_margin_calculation(self, option_price: float, underlying_price: float, contracts: int = 1):
        """
        Test the SPAN margin calculation for an option position with pre-specified parameters.
        
        This is a utility function to verify the margin calculation is working as expected,
        by creating a synthetic position and calculating its margin.
        
        Args:
            option_price: Option price per share
            underlying_price: Underlying price
            contracts: Number of contracts to test
            
        Returns:
            float: Calculated margin value
        """
        # Create a synthetic option position with standard parameters
        from .position import OptionPosition
        
        # Create the test position
        test_position = OptionPosition(
            symbol=f"TEST_OPTION",
            contracts=contracts,
            entry_price=option_price,
            is_short=True,
            position_type='option',
            option_data={
                'underlying_price': underlying_price,
                'strike': underlying_price * 0.95,  # 5% OTM put
                'option_type': 'P'
            },
            logger=self.logger
        )
        
        # Set option_type field (required for proper margin calculation)
        test_position.option_type = 'P'  # Put option
        
        # Set test values for greeks (reasonable approximations)
        test_position.current_delta = -0.2  # Typical for OTM put
        test_position.current_gamma = 0.01
        test_position.current_theta = 0.5
        test_position.current_vega = 0.2
        test_position.underlying_price = underlying_price
        test_position.strike = underlying_price * 0.95
        
        # Step 1: Calculate notional value
        notional_value = underlying_price * contracts * 100
        
        if self.logger:
            self.logger.warning(f"[MARGIN TEST] Step 1: Calculate notional value")
            self.logger.warning(f"[MARGIN TEST]   Formula: underlying_price × contracts × 100")
            self.logger.warning(f"[MARGIN TEST]   ${underlying_price:.2f} × {contracts} × 100 = ${notional_value:.2f}")
        
        # Step 2: Calculate initial margin (base margin)
        base_margin = notional_value * self.initial_margin_percentage
        
        if self.logger:
            self.logger.warning(f"[MARGIN TEST] Step 2: Calculate base margin")
            self.logger.warning(f"[MARGIN TEST]   Formula: notional_value × initial_margin_percentage")
            self.logger.warning(f"[MARGIN TEST]   ${notional_value:.2f} × {self.initial_margin_percentage:.4f} = ${base_margin:.2f}")
        
        # Step 3: Calculate scan risk
        # Use 5% price move to match _calculate_scan_risk (changed from 15%)
        price_move_pct = 0.05
        price_move = underlying_price * price_move_pct
        
        # First-order approximation using delta
        delta_effect = test_position.current_delta * price_move * contracts * 100
        
        # Second-order approximation using gamma with scaling factor
        gamma_scaling_factor = 0.3  # Match the scaling factor used in _calculate_scan_risk
        gamma_effect = 0.5 * test_position.current_gamma * (price_move ** 2) * contracts * 100 * gamma_scaling_factor
        
        # Total risk is the sum of delta and gamma effects
        scan_risk = abs(delta_effect + gamma_effect * self.volatility_multiplier)
        
        if self.logger:
            self.logger.warning(f"[MARGIN TEST] Step 3: Calculate scan risk")
            self.logger.warning(f"[MARGIN TEST]   Price move: {price_move_pct:.0%} of ${underlying_price:.2f} = ${price_move:.2f}")
            self.logger.warning(f"[MARGIN TEST]   Delta effect: {test_position.current_delta:.4f} × ${price_move:.2f} × {contracts} × 100 = ${delta_effect:.2f}")
            self.logger.warning(f"[MARGIN TEST]   Gamma effect: 0.5 × {test_position.current_gamma:.6f} × ${price_move:.2f}² × {contracts} × 100 × {gamma_scaling_factor} = ${gamma_effect:.2f}")
            self.logger.warning(f"[MARGIN TEST]   Total scan risk: |${delta_effect:.2f} + ${gamma_effect * self.volatility_multiplier:.2f}| = ${scan_risk:.2f}")
        
        # Step 4: Choose the higher of base margin and scan risk
        margin = max(base_margin, scan_risk)
        
        if self.logger:
            self.logger.warning(f"[MARGIN TEST] Step 4: Choose maximum margin")
            self.logger.warning(f"[MARGIN TEST]   Max of base margin (${base_margin:.2f}) and scan risk (${scan_risk:.2f}) = ${margin:.2f}")
        
        # Step 5: Check against option premium (for short options)
        option_premium = option_price * contracts * 100
        
        if margin < option_premium:
            if self.logger:
                self.logger.warning(f"[MARGIN TEST] Step 5: Compare to option premium")
                self.logger.warning(f"[MARGIN TEST]   Option premium: ${option_price:.2f} × {contracts} × 100 = ${option_premium:.2f}")
                self.logger.warning(f"[MARGIN TEST]   Margin (${margin:.2f}) is less than premium, using premium: ${option_premium:.2f}")
            margin = option_premium
        
        # Calculate margin using the normal method
        calculated_margin = self.calculate_position_margin(test_position)
        
        if self.logger:
            self.logger.warning(f"[MARGIN TEST] Comparison: ")
            self.logger.warning(f"[MARGIN TEST]   Manual calculation: ${margin:.2f}")
            self.logger.warning(f"[MARGIN TEST]   calculate_position_margin: ${calculated_margin:.2f}")
            
            # Verify the results match (or explain discrepancy)
            if abs(margin - calculated_margin) > 0.01:
                self.logger.warning(f"[MARGIN TEST]   DISCREPANCY DETECTED: ${abs(margin - calculated_margin):.2f} difference")
                self.logger.warning(f"[MARGIN TEST]   Check calculation steps in calculate_position_margin method")
            else:
                self.logger.warning(f"[MARGIN TEST]   Results match within tolerance")
        
        return calculated_margin

    def test_delta_hedged_offset_factor(self, option_delta: float, option_price: float, stock_shares: int) -> float:
        """
        Test the offset factor calculation for a delta-hedged portfolio.
        
        This method creates a synthetic test portfolio with an option position and a stock
        position that hedges the option's delta exposure. It then calculates the portfolio
        margin with hedging benefits to verify that well-hedged positions receive
        appropriate margin reductions.
        
        Args:
            option_delta: Delta of the option position (e.g., 0.75 for a call)
            option_price: Price of the option per share
            stock_shares: Number of shares to hedge with (should be close to delta * 100 for a perfect hedge)
            
        Returns:
            float: The calculated offset factor (higher is better, max 0.8)
        """
        from .position import OptionPosition, Position
        
        if self.logger:
            self.logger.info(f"[DELTA HEDGE TEST] Testing delta-hedged portfolio:")
            self.logger.info(f"  Option: 1 contract, Delta: {option_delta:.2f}, Price: ${option_price:.2f}")
            self.logger.info(f"  Stock: {stock_shares} shares ({'long' if option_delta < 0 else 'short' if option_delta > 0 else 'neutral'})")
        
        # Create a synthetic option position
        underlying_price = 480.0  # Standard test price
        option_position = OptionPosition(
            symbol=f"TEST_OPTION",
            contracts=1,
            entry_price=option_price,
            is_short=True,  # Short option
            position_type='option',
            option_data={
                'underlying_price': underlying_price,
                'strike': underlying_price - 10,  # ITM for call
                'option_type': 'C' if option_delta > 0 else 'P'
            },
            logger=self.logger
        )
        
        # Set required option fields
        option_position.option_type = 'C' if option_delta > 0 else 'P'
        option_position.current_delta = option_delta
        option_position.current_gamma = 0.01
        option_position.current_theta = 0.5
        option_position.current_vega = 0.2
        option_position.underlying_price = underlying_price
        option_position.strike = underlying_price - 10 if option_delta > 0 else underlying_price + 10
        option_position.current_price = option_price
        
        # Create a stock position that offsets the option's delta
        # For a short call (positive delta), we need long stock
        # For a short put (negative delta), we need short stock
        stock_position = Position(
            symbol="TEST_STOCK",
            contracts=stock_shares,
            entry_price=underlying_price,
            is_short=(option_delta < 0),  # Short stock if option has negative delta
            position_type='stock',
            logger=self.logger
        )
        stock_position.current_price = underlying_price
        
        # Calculate standalone margins
        option_margin = self.calculate_position_margin(option_position)
        stock_margin = self.calculate_position_margin(stock_position)
        
        if self.logger:
            self.logger.info(f"[DELTA HEDGE TEST] Standalone margin requirements:")
            self.logger.info(f"  Option margin: ${option_margin:.2f}")
            self.logger.info(f"  Stock margin: ${stock_margin:.2f}")
            self.logger.info(f"  Total standalone margin: ${option_margin + stock_margin:.2f}")
        
        # Create a test portfolio
        portfolio = {
            'TEST_OPTION': option_position,
            'TEST_STOCK': stock_position
        }
        
        # Calculate portfolio margin with hedging benefits
        portfolio_margin_result = self.calculate_portfolio_margin(portfolio)
        total_portfolio_margin = portfolio_margin_result['total_margin']
        hedging_benefits = portfolio_margin_result['hedging_benefits']
        
        # Calculate the effective offset factor from the results
        standalone_margin = option_margin + stock_margin
        if standalone_margin > 0 and hedging_benefits > 0:
            effective_offset_factor = hedging_benefits / (standalone_margin * self.hedge_credit_rate)
        else:
            effective_offset_factor = 0.0
        
        if self.logger:
            self.logger.info(f"[DELTA HEDGE TEST] Portfolio margin calculation results:")
            self.logger.info(f"  Standalone margin: ${standalone_margin:.2f}")
            self.logger.info(f"  Portfolio margin: ${total_portfolio_margin:.2f}")
            self.logger.info(f"  Hedging benefits: ${hedging_benefits:.2f} ({(hedging_benefits/standalone_margin*100):.1f}% reduction)")
            self.logger.info(f"  Effective offset factor: {effective_offset_factor:.2f}")
            
            # Calculate option delta and stock delta for comparison
            option_delta_exposure = option_delta * 100  # One contract = 100 shares equivalent
            stock_delta_exposure = stock_shares * (1 if not stock_position.is_short else -1)
            
            self.logger.info(f"[DELTA HEDGE TEST] Delta exposure analysis:")
            self.logger.info(f"  Option delta exposure: {option_delta_exposure:.2f}")
            self.logger.info(f"  Stock delta exposure: {stock_delta_exposure:.2f}")
            self.logger.info(f"  Net delta exposure: {option_delta_exposure + stock_delta_exposure:.2f}")
            
            perfect_hedge = abs(option_delta_exposure) == abs(stock_delta_exposure) and ((option_delta_exposure < 0) != (stock_delta_exposure < 0))
            good_hedge = abs(option_delta_exposure + stock_delta_exposure) / (abs(option_delta_exposure) + abs(stock_delta_exposure)) < 0.1
            
            if perfect_hedge:
                self.logger.info(f"[DELTA HEDGE TEST] This is a PERFECT hedge!")
            elif good_hedge:
                self.logger.info(f"[DELTA HEDGE TEST] This is a GOOD hedge (within 10% of perfect)")
            else:
                self.logger.info(f"[DELTA HEDGE TEST] This is a PARTIAL hedge")
        
        return effective_offset_factor

    @classmethod
    def from_config(cls, config_path: str = 'config/config.yaml', logger: Optional[logging.Logger] = None):
        """
        Create a SPANMarginCalculator instance from configuration file.
        
        Args:
            config_path: Path to the configuration YAML file
            logger: Optional logger instance
            
        Returns:
            SPANMarginCalculator: An initialized margin calculator with parameters from config
        """
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except (FileNotFoundError, yaml.YAMLError) as e:
            if logger:
                logger.error(f"Failed to load config from {config_path}: {e}")
            # Use default config if file not found or invalid
            config = {"margin": {"span": {}}}
        
        # Extract span margin configuration
        span_config = config.get("margin", {}).get("span", {})
        
        # Get parameter values with defaults
        max_leverage = span_config.get("max_leverage", 1.0)
        volatility_multiplier = span_config.get("volatility_multiplier", 1.0)
        initial_margin_percentage = span_config.get("initial_margin_percentage", 0.1)
        maintenance_margin_percentage = span_config.get("maintenance_margin_percentage", 0.07)
        hedge_credit_rate = span_config.get("hedge_credit_rate", 0.8)
        price_move_pct = span_config.get("price_move_pct", 0.05)
        vol_shift_pct = span_config.get("vol_shift_pct", 0.3)
        gamma_scaling_factor = span_config.get("gamma_scaling_factor", 0.3)
        min_scan_risk_percentage = span_config.get("min_scan_risk_percentage", 0.25)
        max_margin_to_premium_ratio = span_config.get("max_margin_to_premium_ratio", 20.0)
        otm_scaling_enabled = span_config.get("otm_scaling_enabled", True)
        otm_minimum_scaling = span_config.get("otm_minimum_scaling", 0.1)
        
        # Log the loaded configuration
        if logger:
            logger.info(f"[SPAN Margin] Loading configuration from {config_path}")
            logger.info(f"[SPAN Margin] Using parameters:")
            for key, value in span_config.items():
                logger.info(f"  {key}: {value}")
        
        # Create calculator instance with parameters from config
        return cls(
            max_leverage=max_leverage,
            volatility_multiplier=volatility_multiplier,
            initial_margin_percentage=initial_margin_percentage,
            maintenance_margin_percentage=maintenance_margin_percentage,
            hedge_credit_rate=hedge_credit_rate,
            price_move_pct=price_move_pct,
            vol_shift_pct=vol_shift_pct,
            gamma_scaling_factor=gamma_scaling_factor,
            min_scan_risk_percentage=min_scan_risk_percentage,
            max_margin_to_premium_ratio=max_margin_to_premium_ratio,
            otm_scaling_enabled=otm_scaling_enabled,
            otm_minimum_scaling=otm_minimum_scaling,
            logger=logger
        )