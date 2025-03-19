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

        # Basic margin calculation
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
            'margin_by_position': margin_by_position
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
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the SPAN margin calculator.
        
        Args:
            max_leverage: Maximum leverage allowed
            volatility_multiplier: Multiplier for volatility stress scenarios
            correlation_matrix: DataFrame with correlation coefficients between underlyings
            initial_margin_percentage: Initial margin as percentage of notional value
            maintenance_margin_percentage: Maintenance margin as percentage of notional
            hedge_credit_rate: Credit rate for hedged positions (0-1)
            logger: Logger instance
        """
        super().__init__(max_leverage, logger)
        self.volatility_multiplier = volatility_multiplier
        self.correlation_matrix = correlation_matrix
        self.initial_margin_percentage = initial_margin_percentage
        self.maintenance_margin_percentage = maintenance_margin_percentage
        self.hedge_credit_rate = min(max(hedge_credit_rate, 0), 1)  # Ensure between 0-1
        
        # Log initialization parameters
        if self.logger:
            self.logger.info(f"[SPAN Margin] Initialized SPANMarginCalculator with parameters:")
            self.logger.info(f"  max_leverage: {max_leverage:.2f}")
            self.logger.info(f"  volatility_multiplier: {volatility_multiplier:.2f}")
            self.logger.info(f"  initial_margin_percentage: {initial_margin_percentage:.4f} ({initial_margin_percentage*100:.2f}%)")
            self.logger.info(f"  maintenance_margin_percentage: {maintenance_margin_percentage:.4f} ({maintenance_margin_percentage*100:.2f}%)")
            self.logger.info(f"  hedge_credit_rate: {hedge_credit_rate:.2f}")
            
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
        Calculate the scan risk component of SPAN margin.
        
        This simulates the worst-case scenario across 16 standard price and
        volatility scenarios to determine the risk of market moves.
        
        Args:
            position: Position to calculate scan risk for
            
        Returns:
            float: Scan risk in dollars
        """
        # Log that we're starting the scan risk calculation
        if self.logger:
            self.logger.info(f"[Scan Risk] Starting scan risk calculation for {position.symbol}")
            self.logger.info(f"[Scan Risk] Position: {position.contracts} contracts, Price: ${position.current_price:.4f}")
            
            # Log additional option details if available
            if hasattr(position, 'underlying_price'):
                self.logger.info(f"[Scan Risk] Underlying price: ${position.underlying_price:.2f}")
            if hasattr(position, 'current_delta'):
                self.logger.info(f"[Scan Risk] Delta: {position.current_delta:.4f}")
            if hasattr(position, 'current_gamma'):
                self.logger.info(f"[Scan Risk] Gamma: {position.current_gamma:.6f}")
        
        # For non-option positions, use a simplified approach
        if not isinstance(position, OptionPosition):
            # For stocks/ETFs, use a fixed percentage of position value
            position_value = position.current_price * position.contracts
            scan_risk = position_value * 0.05  # 5% market risk (changed from 15% for consistency)
            
            if self.logger:
                self.logger.info(f"[Scan Risk] Stock/ETF position, simplified calculation: ${position_value:.2f} × 0.05 = ${scan_risk:.2f}")
            
            return scan_risk
        
        # For options, we need to calculate based on Greeks
        # First, validate that we have the required Greek values
        if not hasattr(position, 'current_delta') or not hasattr(position, 'current_gamma'):
            # Log the missing Greeks
            if self.logger:
                self.logger.warning(f"[Scan Risk] Missing Greeks for {position.symbol}, using simplified calculation")
                
            # If missing Greeks, use a conservative approach based on option value
            option_value = position.current_price * position.contracts * 100
            scan_risk = option_value * 0.25  # 25% of option value
            
            if self.logger:
                self.logger.info(f"[Scan Risk] Simplified calculation due to missing Greeks: ${option_value:.2f} × 0.25 = ${scan_risk:.2f}")
                
            return scan_risk
        
        # Calculate scan risk based on a simulated price move of 5% of the underlying price
        # This is a more reasonable 1-day price move compared to the previous 15%
        price_move_pct = 0.05  # Changed from 0.15 to 0.05 (5% is more aligned with industry standards)
        underlying_price = position.underlying_price if hasattr(position, 'underlying_price') and position.underlying_price > 0 else 100.0
        price_move = underlying_price * price_move_pct
        
        if self.logger:
            self.logger.info(f"[Scan Risk] Price move calculation: {price_move_pct:.0%} of ${underlying_price:.2f} = ${price_move:.2f}")
        
        # First-order approximation using delta
        delta = position.current_delta
        # Ensure we are using the contract multiplier (100) in delta calculations
        contract_multiplier = 100
        delta_effect = delta * price_move * position.contracts * contract_multiplier
        
        if self.logger:
            self.logger.info(f"[Scan Risk] Delta effect calculation: {delta:.4f} × ${price_move:.2f} × {position.contracts} × {contract_multiplier} = ${delta_effect:.2f}")
        
        # Second-order approximation using gamma
        gamma = position.current_gamma
        # Apply a gamma scaling factor to prevent gamma from dominating the calculation
        gamma_scaling_factor = 0.3  # Scale gamma impact to align with industry standards
        gamma_effect = 0.5 * gamma * (price_move ** 2) * position.contracts * contract_multiplier * gamma_scaling_factor
        
        if self.logger:
            self.logger.info(f"[Scan Risk] Gamma effect calculation: 0.5 × {gamma:.6f} × ${price_move:.2f}² × {position.contracts} × {contract_multiplier} × {gamma_scaling_factor} = ${gamma_effect:.2f}")
        
        # Apply volatility multiplier to gamma effect to account for market stress
        gamma_effect_with_vol = gamma_effect * self.volatility_multiplier
        
        if self.logger and self.volatility_multiplier != 1.0:
            self.logger.info(f"[Scan Risk] Gamma effect with volatility multiplier: ${gamma_effect:.2f} × {self.volatility_multiplier:.2f} = ${gamma_effect_with_vol:.2f}")
        
        # The total scan risk is the absolute value of the sum of delta and gamma effects
        scan_risk = abs(delta_effect + gamma_effect_with_vol)
        
        # Calculate option premium for comparison
        option_premium = position.current_price * position.contracts * contract_multiplier
        
        # Sanity check: if scan risk is suspiciously low compared to premium, apply a correction
        if scan_risk < option_premium * 0.25:  # If scan risk is less than 25% of premium
            if self.logger:
                self.logger.warning(f"[Scan Risk] WARNING: Calculated scan risk (${scan_risk:.2f}) is suspiciously low compared to premium (${option_premium:.2f})")
                self.logger.warning(f"[Scan Risk] Applying minimum scan risk of 25% of premium: ${option_premium * 0.25:.2f}")
            scan_risk = max(scan_risk, option_premium * 0.25)  # At least 25% of premium
        
        if self.logger:
            self.logger.info(f"[Scan Risk] Combined risk: |${delta_effect:.2f} + ${gamma_effect_with_vol:.2f}| = ${scan_risk:.2f}")
            self.logger.info(f"[Scan Risk] Final scan risk for {position.symbol}: ${scan_risk:.2f}")
        
        return scan_risk

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
            if moneyness < -0.05:  # More than 5% OTM
                # The deeper OTM and smaller delta, the lower the margin percentage
                # For very deep OTM options, this could be as low as 10% of the original
                scaling_factor = min(1.0, max(0.1, abs_delta * 5))  # 0.1 to 1.0 scaling
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
            max_acceptable_ratio = 20.0  # Cap at 20x the premium
            
            if margin_to_premium_ratio > max_acceptable_ratio:
                capped_margin = option_premium * max_acceptable_ratio
                
                if self.logger:
                    self.logger.warning(f"  Margin-to-premium ratio too high: {margin_to_premium_ratio:.2f}x")
                    self.logger.warning(f"  Capping margin at {max_acceptable_ratio:.1f}x premium: ${capped_margin:.2f}")
                
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
        Calculate SPAN margin for a portfolio with delta hedging benefits.

        This enhanced implementation:
        1. Groups positions by underlying
        2. Calculates net delta exposure per underlying
        3. Applies margin offsets for hedged positions
        4. Accounts for inter-product correlation

        Args:
            positions: Dictionary of positions by symbol

        Returns:
            dict: Dictionary with total margin, margin by position, and hedging benefits
        """
        if not positions:
            return {'total_margin': 0, 'margin_by_position': {}, 'hedging_benefits': 0}
            
        # Validate input type
        if not isinstance(positions, dict):
            if self.logger:
                self.logger.error(f"Invalid input to calculate_portfolio_margin: {type(positions)}")
            raise ValueError(f"Expected dictionary of positions, got {type(positions)}")

        # Step 1: Calculate standalone margin for each position
        margin_by_position = {}
        for symbol, position in positions.items():
            position_margin = self.calculate_position_margin(position)
            margin_by_position[symbol] = position_margin
            
            if self.logger:
                self.logger.debug(f"[Margin] {position.symbol}: Initial margin=${position_margin:.2f}")
                self.logger.debug(f"  Final margin: ${position_margin:.2f}")

        # Step 2: Group positions by underlying for delta netting
        positions_by_underlying = {}
        for symbol, position in positions.items():
            # For option positions, extract the underlying symbol
            if hasattr(position, 'option_data') and position.option_data and 'UnderlyingSymbol' in position.option_data:
                underlying = position.option_data['UnderlyingSymbol']
            elif hasattr(position, 'underlying_symbol'):
                underlying = position.underlying_symbol
            else:
                # Use our extraction method as fallback
                underlying = self._extract_underlying_symbol(position)
                
            # For stock positions, use their own symbol as underlying
            if not hasattr(position, 'option_type') or not position.option_type:
                # This is a stock/ETF position - it's its own underlying
                underlying = symbol
                
            if underlying not in positions_by_underlying:
                positions_by_underlying[underlying] = []
            positions_by_underlying[underlying].append(symbol)

        # Log the grouping
        if self.logger:
            self.logger.debug(f"[Margin] Portfolio of {len(positions)} positions")
            self.logger.debug(f"[Margin] Grouped positions by underlying:")
            for underlying, symbols in positions_by_underlying.items():
                self.logger.debug(f"  {underlying}: {symbols}")

        # Step 3: Calculate delta offsets for each underlying group
        margin_with_offsets = {}
        hedging_benefits = 0
        
        for underlying, position_symbols in positions_by_underlying.items():
            # Calculate net delta for this underlying group
            net_delta = 0
            standalone_margin = 0
            
            # Track positions in this group
            group_positions = {}
            
            for symbol in position_symbols:
                position = positions[symbol]
                standalone_margin += margin_by_position[symbol]
                
                # Add to group positions
                group_positions[symbol] = position
                
                # Calculate delta in absolute terms (contracts * delta)
                if hasattr(position, 'current_delta'):
                    # For options with delta
                    if position.is_short:
                        position_delta = -position.current_delta * position.contracts * 100
                    else:
                        position_delta = position.current_delta * position.contracts * 100
                    net_delta += position_delta
                elif not hasattr(position, 'option_type') or not position.option_type:
                    # For stock/ETF positions, delta is just number of shares
                    position_delta = -position.contracts if position.is_short else position.contracts
                    net_delta += position_delta
            
            # Enhanced hedging benefit logic
            # Apply hedging benefit whenever there are multiple positions in a group
            if len(position_symbols) > 1:
                # Calculate total absolute delta for the group
                total_abs_delta = sum(
                    abs(positions[s].current_delta * positions[s].contracts * 100) 
                    if hasattr(positions[s], 'current_delta') 
                    else abs(positions[s].contracts) 
                    for s in position_symbols
                )
                
                # Calculate offset factor - how well positions offset each other
                # When net_delta is close to 0, positions are well-hedged
                if total_abs_delta > 0:
                    offset_factor = 1.0 - abs(net_delta) / total_abs_delta
                else:
                    offset_factor = 0.0
                    
                # Ensure a minimum offset for diversified positions
                # This provides some benefit even for imperfect hedges
                offset_factor = max(0.2, offset_factor)
                
                # Cap the offset factor to avoid excessive reductions
                offset_factor = min(0.8, offset_factor)
                
                # Apply the hedge credit using our offset factor
                reduced_margin = standalone_margin * (1 - self.hedge_credit_rate * offset_factor)
                
                # Track the hedging benefit
                benefit = standalone_margin - reduced_margin
                hedging_benefits += benefit
                
                if self.logger:
                    self.logger.debug(f"[Margin] Hedging benefit for {underlying} group: ${benefit:.2f}")
                    self.logger.debug(f"  Net delta: {net_delta:.2f}, Total absolute delta: {total_abs_delta:.2f}")
                    self.logger.debug(f"  Offset factor: {offset_factor:.2f}")
                    self.logger.debug(f"  Standalone margin: ${standalone_margin:.2f}")
                    self.logger.debug(f"  Reduced margin: ${reduced_margin:.2f}")
                
                # Store the reduced margin for this group
                for symbol in position_symbols:
                    # Distribute the reduced margin proportionally
                    original = margin_by_position[symbol]
                    proportion = original / standalone_margin if standalone_margin > 0 else 0
                    margin_with_offsets[symbol] = reduced_margin * proportion
            else:
                # No hedging benefit for single positions
                for symbol in position_symbols:
                    margin_with_offsets[symbol] = margin_by_position[symbol]
        
        # Step 4: Calculate total margin with all offsets
        total_margin = sum(margin_with_offsets.values())
        
        # Prepare the result
        result = {
            'total_margin': total_margin,
            'margin_by_position': margin_with_offsets,
            'hedging_benefits': hedging_benefits
        }
        
        if self.logger:
            self.logger.info(f"[Margin] Portfolio margin calculation complete")
            self.logger.info(f"  Total margin: ${total_margin:.2f}")
            self.logger.info(f"  Total hedging benefits: ${hedging_benefits:.2f}")
        
        return result

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