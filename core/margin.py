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
        # Store direct reference to the logging manager if passed
        self.logging_manager = None
        if logger and hasattr(logger, 'component_log_levels'):
            # This is likely our custom LoggingManager's logger
            self.logging_manager = logger

    def _should_log(self, level_name: str) -> bool:
        """
        Check if a message should be logged based on margin log level setting.
        
        Args:
            level_name: Name of log level ('minimal', 'standard', 'verbose', 'debug')
            
        Returns:
            bool: Whether the message should be logged
        """
        # For now, just return True to allow all logging (workaround for issue)
        return True
        
        # The code below is disabled temporarily due to comparison issues
        """
        # If we don't have a logger, don't log
        if not self.logger:
            return False
            
        # Check if we have a component log level from our LoggingManager
        component_level = None
        
        # If we have a direct reference to the logging manager with component levels
        if hasattr(self, 'logging_manager') and self.logging_manager:
            if hasattr(self.logging_manager, 'component_log_levels'):
                component_level = self.logging_manager.component_log_levels.get('margin', 'standard')
        
        # If we don't have a component level, fall back to standard logging levels
        if component_level is None:
            log_level_map = {
                'minimal': logging.WARNING,   # Always show
                'standard': logging.INFO,     # Normal operation
                'verbose': logging.DEBUG,     # Detailed operation
                'debug': logging.DEBUG        # Debug details
            }
            return self.logger.isEnabledFor(log_level_map.get(level_name, logging.INFO))
        
        # Map between level names and their priority (higher number = more verbose)
        level_priority = {
            "minimal": 0,   # Most important, always show (least verbose)
            "standard": 1,  # Normal operation
            "verbose": 2,   # Detailed operation
            "debug": 3      # Debug details (most verbose)
        }
        
        # Get priority numbers - ensure they are integers
        current_level_priority = level_priority.get(component_level, 1)  # Default to standard
        message_level_priority = level_priority.get(level_name, 1)       # Default to standard
        
        # Safety check to prevent type comparison errors
        if not isinstance(current_level_priority, int) or not isinstance(message_level_priority, int):
            # If we encounter a type error, log a warning and default to allowing the log
            print(f"WARNING: Invalid log level type comparison: {type(current_level_priority)} vs {type(message_level_priority)}")
            return True
        
        # Log if message priority is less than or equal to the current level priority
        return message_level_priority <= current_level_priority
        """
    
    def log_minimal(self, message: str) -> None:
        """
        Log a minimal message that should always be shown.
        
        Args:
            message: Message to log
        """
        if self.logger and self._should_log('minimal'):
            self.logger.info(f"[Margin] {message}")
            
    def log_standard(self, message: str) -> None:
        """
        Log a standard message for normal operations.
        
        Args:
            message: Message to log
        """
        if self.logger and self._should_log('standard'):
            self.logger.info(f"[Margin] {message}")
            
    def log_verbose(self, message: str) -> None:
        """
        Log a verbose message with detailed calculation steps.
        
        Args:
            message: Message to log
        """
        if self.logger and self._should_log('verbose'):
            self.logger.debug(f"[Margin] {message}")
            
    def log_debug(self, message: str) -> None:
        """
        Log a debug message with implementation details.
        
        Args:
            message: Message to log
        """
        if self.logger and self._should_log('debug'):
            self.logger.debug(f"[Margin] {message}")

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
        
        # Log the start of portfolio margin calculation - always show this
        self.log_minimal(f"Portfolio calculation for {len(positions)} positions")
        
        for symbol, position in positions.items():
            position_margin = self.calculate_position_margin(position)
            standalone_margin += position_margin
            margin_by_position[symbol] = position_margin
            # Use standard level for per-position margin amounts
            self.log_standard(f"{symbol}: Initial margin=${position_margin:.2f}")
        
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
        
        # Log grouped positions at verbose level - these details are usually not needed
        self.log_verbose("Grouped positions by underlying:")
        for underlying, symbols in positions_by_underlying.items():
            self.log_verbose(f"  {underlying}: {symbols}")
        
        # Calculate hedging benefits
        total_hedging_benefit = 0
        
        for underlying, symbols in positions_by_underlying.items():
            # Skip if only one position for this underlying (no hedging benefit)
            if len(symbols) <= 1:
                continue
            
            # Calculate potential hedging benefit
            underlying_positions = [positions[symbol] for symbol in symbols]
            underlying_margin = sum(margin_by_position[symbol] for symbol in symbols)
            
            # Check for delta neutrality (approximate)
            net_delta = 0
            total_delta_exposure = 0
            
            # Calculate net delta consistently 
            for position in underlying_positions:
                # Track position delta for debugging
                position_delta = 0
                
                if isinstance(position, OptionPosition):
                    # For options, delta is per share, we need to multiply by 100 for the contract
                    # This is the absolute value in dollars of the option delta
                    current_delta = position.current_delta if hasattr(position, 'current_delta') else 0
                    # The number of contracts
                    contracts = position.contracts if hasattr(position, 'contracts') else 0
                    # The underlying price
                    underlying_price = position.underlying_price if hasattr(position, 'underlying_price') else 0
                    
                    if current_delta != 0 and contracts != 0 and underlying_price != 0:
                        position_delta = current_delta * underlying_price * contracts * 100
                        self.log_verbose(f"Option position {position.symbol}: Delta={current_delta:.4f}, Contracts={contracts}, Price=${underlying_price:.2f}, Delta dollars=${position_delta:.2f}")
                    else:
                        self.log_verbose(f"Option position {position.symbol}: Incomplete data for delta calculation")
                        continue
                else:
                    # For stocks, calculate delta based on direction and number of shares
                    direction = -1 if getattr(position, 'is_short', False) else 1
                    shares = position.contracts if hasattr(position, 'contracts') else 0
                    price = position.current_price if hasattr(position, 'current_price') else 0
                    
                    if shares != 0 and price != 0:
                        position_delta = shares * direction * price
                        self.log_verbose(f"Stock position {position.symbol}: Shares={shares}, Direction={direction}, Price=${price:.2f}, Delta dollars=${position_delta:.2f}")
                    else:
                        self.log_verbose(f"Stock position {position.symbol}: Incomplete data for delta calculation")
                        continue
                
                net_delta += position_delta
                total_delta_exposure += abs(position_delta)
            
            # Log detailed delta hedging calculations
            self.log_minimal(f"Delta hedging calculation for {underlying}:")
            self.log_minimal(f"  Net delta: ${net_delta:.2f}")
            self.log_minimal(f"  Total delta exposure: ${total_delta_exposure:.2f}")
            self.log_minimal(f"  Underlying margin: ${underlying_margin:.2f}")
            
            # Calculate hedge credit based on delta neutrality
            hedge_credit = 0
            
            # Fix: Prevent division by zero and ensure we're using absolute values for calculations
            delta_hedge_ratio = 0.0
            if total_delta_exposure > 0 and underlying_margin > 0:
                # Calculate neutrality as ratio of net to gross exposure
                delta_hedge_ratio = 1.0 - (abs(net_delta) / total_delta_exposure)
                self.log_minimal(f"  Delta hedge ratio calculation: 1.0 - (|{net_delta:.2f}| / {total_delta_exposure:.2f}) = {delta_hedge_ratio:.4f}")
                
                # Ensure ratio is between 0 and 1
                delta_hedge_ratio = min(1.0, max(0.0, delta_hedge_ratio))
                # Apply the hedge credit
                hedge_credit_rate = getattr(self, 'hedge_credit_rate', 0.5)
                hedge_credit = underlying_margin * delta_hedge_ratio * hedge_credit_rate
                self.log_minimal(f"  Hedge credit: {underlying_margin:.2f} × {delta_hedge_ratio:.4f} × {hedge_credit_rate:.2f} = ${hedge_credit:.2f}")
                
                # Cap the hedging benefit to a maximum percentage of the underlying margin
                max_hedge_benefit = underlying_margin * 0.8
                if hedge_credit > max_hedge_benefit:
                    self.log_minimal(f"  Capping hedge credit to 80% of margin: ${max_hedge_benefit:.2f}")
                    hedge_credit = min(hedge_credit, max_hedge_benefit)
                
                # Ensure hedge_credit is never negative (which would increase margin)
                hedge_credit = max(0, hedge_credit)
            
            # Track total hedging benefit
            total_hedging_benefit += hedge_credit
            
            # Log hedging benefit at standard level
            self.log_standard(f"Hedge benefit for {underlying}: ${hedge_credit:.2f} ({delta_hedge_ratio*100:.1f}% offset)")
        
        # Calculate portfolio margin as standalone margin minus hedging benefits
        portfolio_margin = standalone_margin - total_hedging_benefit
        
        # Always log the final margin calculation
        self.log_minimal(f"Final portfolio margin: ${portfolio_margin:.2f}")
        self.log_minimal(f"Hedging benefits: ${total_hedging_benefit:.2f}")
        
        # Prepare the results
        result = {
            "total_margin": portfolio_margin,
            "margin_by_position": margin_by_position,
            "hedging_benefits": total_hedging_benefit
        }
        
        # Log a summary
        self.log_margin_summary(positions, result)
        
        return result

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
            logger: Optional[logging.Logger] = None,
            params: Optional[Dict[str, Any]] = None
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
            params: Dictionary containing all parameters (overrides individual parameters)
        """
        super().__init__(max_leverage, logger)
        
        # If params is provided, override individual parameters
        if params:
            max_leverage = params.get('max_leverage', max_leverage)
            volatility_multiplier = params.get('volatility_multiplier', volatility_multiplier)
            initial_margin_percentage = params.get('initial_margin_percentage', initial_margin_percentage)
            maintenance_margin_percentage = params.get('maintenance_margin_percentage', maintenance_margin_percentage)
            hedge_credit_rate = params.get('hedge_credit_rate', hedge_credit_rate)
            price_move_pct = params.get('price_move_pct', price_move_pct)
            vol_shift_pct = params.get('vol_shift_pct', vol_shift_pct)
            gamma_scaling_factor = params.get('gamma_scaling_factor', gamma_scaling_factor)
            min_scan_risk_percentage = params.get('min_scan_risk_percentage', min_scan_risk_percentage)
            max_margin_to_premium_ratio = params.get('max_margin_to_premium_ratio', max_margin_to_premium_ratio)
            otm_scaling_enabled = params.get('otm_scaling_enabled', otm_scaling_enabled)
            otm_minimum_scaling = params.get('otm_minimum_scaling', otm_minimum_scaling)
            
            # If correlation_matrix is provided in params, try to convert it to a DataFrame
            if 'correlation_matrix' in params:
                correlation_matrix_data = params.get('correlation_matrix')
                if isinstance(correlation_matrix_data, dict):
                    try:
                        correlation_matrix = pd.DataFrame(correlation_matrix_data)
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Could not convert correlation_matrix from params to DataFrame: {e}")
                
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
        Calculate the scan risk (price movement risk) for a position.
        
        Args:
            position: Position to calculate scan risk for
            
        Returns:
            float: Scan risk amount in dollars
        """
        # Check if we have the required position attributes
        required_attrs = ['contracts', 'current_price', 'underlying_price']
        if not all(hasattr(position, attr) for attr in required_attrs):
            missing = [attr for attr in required_attrs if not hasattr(position, attr)]
            self.log_minimal(f"Missing attributes for scan risk: {missing}")
            # Default to a simple calculation based on premium
            premium = position.current_price * position.contracts * 100
            return premium * 0.5  # 50% of premium as default risk
        
        # Calculate price move based on configured percentage
        price_move = position.underlying_price * self.price_move_pct
        
        # Zero risk for long options (accounted for in portfolio valuation)
        if not position.is_short:
            return 0
        
        # Calculate delta effect (first-order price move impact)
        delta_effect = 0
        if hasattr(position, 'current_delta'):
            # For short options, delta is in terms of option price change per $1 move in underlying
            # Compute the dollar impact of the price move
            delta_effect = position.current_delta * price_move * position.contracts * 100
            self.log_verbose(f"Delta effect calculation:")
            self.log_verbose(f"  Delta: {position.current_delta:.4f}")
            self.log_verbose(f"  Price move: ${price_move:.2f}")
            self.log_verbose(f"  Contracts: {position.contracts}")
            self.log_verbose(f"  Delta effect: ${delta_effect:.2f}")
        
        # Calculate gamma effect (second-order price move impact)
        gamma_effect = 0
        if hasattr(position, 'current_gamma'):
            # Gamma measures the rate of change of delta per $1 move in the underlying
            # 0.5 * gamma * (price move)^2 gives the second-order approximation
            gamma_effect = 0.5 * position.current_gamma * (price_move ** 2) * position.contracts * 100
            
            # Apply scaling factor to avoid overestimating gamma effects
            gamma_effect *= self.gamma_scaling_factor
            
            self.log_verbose(f"Gamma effect calculation:")
            self.log_verbose(f"  Gamma: {position.current_gamma:.6f}")
            self.log_verbose(f"  Price move squared: ${price_move**2:.2f}")
            self.log_verbose(f"  Scaling factor: {self.gamma_scaling_factor:.2f}")
            self.log_verbose(f"  Gamma effect: ${gamma_effect:.2f}")
        
        # Calculate volatility effect (vega impact from volatility shifts)
        vega_effect = 0
        if hasattr(position, 'current_vega'):
            # Vega measures option price change per 1% change in implied volatility
            # Compute the dollar impact of the vol shift
            vol_shift = self.vol_shift_pct * self.volatility_multiplier
            vega_effect = position.current_vega * vol_shift * position.contracts * 100
            
            self.log_verbose(f"Volatility effect calculation:")
            self.log_verbose(f"  Vega: {position.current_vega:.6f}")
            self.log_verbose(f"  Vol shift: {vol_shift*100:.1f}%")
            self.log_verbose(f"  Vega effect: ${vega_effect:.2f}")
        
        # Calculate theta effect (time decay over short period)
        theta_effect = 0
        if hasattr(position, 'current_theta'):
            # For most margin calculations, we don't include favorable theta decay
            # Only account for theta if it's unfavorable (typical for long options)
            if position.current_theta > 0 and position.is_short:
                theta_days = 1  # Consider 1 day of adverse movement
                theta_effect = position.current_theta * theta_days * position.contracts * 100
                
                self.log_verbose(f"Theta effect calculation:")
                self.log_verbose(f"  Theta: ${position.current_theta:.6f}")
                self.log_verbose(f"  Days considered: {theta_days}")
                self.log_verbose(f"  Theta effect: ${theta_effect:.2f}")
        
        # Sum all effects for total risk
        # For short positions, positive effects are unfavorable (increase risk)
        # For long positions, negative effects are unfavorable (increase risk)
        total_risk = abs(delta_effect) + abs(gamma_effect) + abs(vega_effect) + abs(theta_effect)
        
        # Apply volatility multiplier to add an extra safety margin in high-vol environments
        total_risk *= self.volatility_multiplier
        
        # Log summary of the scan risk calculation
        self.log_standard(f"Scan risk components for {position.symbol}:")
        self.log_standard(f"  Delta risk: ${abs(delta_effect):.2f}")
        self.log_standard(f"  Gamma risk: ${abs(gamma_effect):.2f}")
        self.log_standard(f"  Vega risk: ${abs(vega_effect):.2f}")
        self.log_standard(f"  Theta risk: ${abs(theta_effect):.2f}")
        self.log_standard(f"  Total scan risk: ${total_risk:.2f}")
        
        return total_risk

    def calculate_position_margin(self, position: Position) -> float:
        """
        Calculate SPAN margin for a position.
        
        Implements a more sophisticated margin calculation that considers:
        - Maximum price move scenarios
        - Volatility shift scenarios
        - Delta and gamma effects
        - Time decay effects
        
        Args:
            position: Position to calculate margin for
            
        Returns:
            float: Margin requirement in dollars
        """
        # Check if position is an option
        is_option = hasattr(position, 'option_type') or getattr(position, 'is_option', False)
        
        # For non-option positions, delegate to parent class
        if not is_option:
            return super().calculate_position_margin(position)
        
        # Log position info at standard level
        self.log_standard(f"Starting SPAN margin calculation for {position.symbol}")
        
        # Check for required attributes
        if not all(hasattr(position, attr) for attr in ['contracts', 'current_price', 'underlying_price']):
            required_attrs = ['contracts', 'current_price', 'underlying_price']
            missing = [attr for attr in required_attrs if not hasattr(position, attr)]
            self.log_minimal(f"Missing attributes for SPAN calculation: {missing}")
            # Fall back to basic calculation
            return super().calculate_position_margin(position)
        
        # Calculate option premium
        premium = position.current_price * position.contracts * 100  # 100 shares per contract
        
        # Verbose logging for option details
        self.log_verbose(f"Option details:")
        self.log_verbose(f"  Type: {position.option_type}")
        self.log_verbose(f"  Strike: ${position.strike:.2f}")
        self.log_verbose(f"  Underlying price: ${position.underlying_price:.2f}")
        self.log_verbose(f"  Current price per share: ${position.current_price:.4f}")
        self.log_verbose(f"  Number of contracts: {position.contracts}")
        self.log_verbose(f"  Premium: ${premium:.2f}")
        
        # Get contract multiplier
        contract_multiplier = 100
        
        # Calculate notional value
        notional_value = position.underlying_price * position.contracts * contract_multiplier
        
        # Step 1: Calculate scan risk (max likely loss from market move)
        scan_risk = self._calculate_scan_risk(position)
        
        # Step 2: Calculate base risk (minimum required margin based on price and leverage)
        # Initial margin is 10% of notional by default
        initial_margin = notional_value * self.initial_margin_percentage / self.max_leverage
        
        # Step 3: Determine if premium-based minimum should apply
        min_premium_margin = premium * 1.0  # 100% of premium as minimum for long options
        if position.is_short:
            min_premium_margin = premium * 1.5  # 150% of premium for short options
        
        # Step 4: Apply OTM scaling if enabled
        if self.otm_scaling_enabled and hasattr(position, 'strike'):
            # Calculate out-of-the-money amount
            if position.option_type.lower() == 'call':
                otm_amount = max(0, position.strike - position.underlying_price)
                moneyness = position.underlying_price / position.strike if position.strike > 0 else 0
            else:  # Put
                otm_amount = max(0, position.underlying_price - position.strike)
                moneyness = position.strike / position.underlying_price if position.underlying_price > 0 else 0
            
            # Calculate scaling factor based on moneyness
            # Deep OTM options get reduced margin requirements
            otm_scaling = max(self.otm_minimum_scaling, min(1.0, moneyness))
            
            # Apply OTM scaling to scan risk and initial margin
            scan_risk *= otm_scaling
            initial_margin *= otm_scaling
            
            # Log OTM scaling details at verbose level
            self.log_verbose(f"OTM scaling:")
            self.log_verbose(f"  OTM amount: ${otm_amount:.2f}")
            self.log_verbose(f"  Moneyness: {moneyness:.4f}")
            self.log_verbose(f"  Applied scaling factor: {otm_scaling:.4f}")
        
        # Step 5: Ensure minimum scan risk
        min_scan_risk = premium * self.min_scan_risk_percentage
        if scan_risk < min_scan_risk:
            self.log_verbose(f"Raising scan risk to minimum value:")
            self.log_verbose(f"  Original scan risk: ${scan_risk:.2f}")
            self.log_verbose(f"  Minimum scan risk: ${min_scan_risk:.2f} ({self.min_scan_risk_percentage*100:.0f}% of premium)")
            scan_risk = min_scan_risk
        
        # Step 6: Calculate total SPAN margin
        span_margin = max(initial_margin, scan_risk)
        
        # Step 7: Cap margin for extremely expensive options
        max_margin = premium * self.max_margin_to_premium_ratio
        if span_margin > max_margin:
            self.log_verbose(f"Capping margin at maximum ratio to premium:")
            self.log_verbose(f"  Original margin: ${span_margin:.2f}")
            self.log_verbose(f"  Maximum margin: ${max_margin:.2f} ({self.max_margin_to_premium_ratio}x premium)")
            span_margin = max_margin
        
        # Apply the margin treatment based on whether position is long or short
        if not position.is_short:
            # For long options, the maximum potential loss is the premium paid
            # But that's already accounted for in portfolio valuation, so zero margin
            self.log_standard(f"Long option position - margin requirement: $0.00")
            return 0
        
        # For short options, apply the calculated margin
        # Log final margin calculation at standard level
        self.log_standard(f"Final margin for {position.symbol}: ${span_margin:.2f}")
        self.log_standard(f"Components: Scan risk=${scan_risk:.2f}, Initial margin=${initial_margin:.2f}")
        
        # More detailed debug information
        self.log_debug(f"Margin calculation factors:")
        self.log_debug(f"  Underlying price: ${position.underlying_price:.2f}")
        self.log_debug(f"  Initial margin %: {self.initial_margin_percentage:.4f}")
        self.log_debug(f"  Max leverage: {self.max_leverage:.2f}")
        self.log_debug(f"  Volatility multiplier: {self.volatility_multiplier:.2f}")
        
        return span_margin

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

    def log_margin_summary(self, positions: Dict[str, Position], margin_result: Dict[str, Any]) -> None:
        """
        Log a concise summary of margin calculations.
        Always shown regardless of log level setting.
        
        Args:
            positions: Dictionary of positions keyed by symbol
            margin_result: Result dictionary from calculate_portfolio_margin
        """
        if not self.logger:
            return
            
        total_positions = len(positions)
        total_margin = margin_result.get('total_margin', 0)
        hedging_benefits = margin_result.get('hedging_benefits', 0)
        
        # Calculate total value of positions
        total_value = 0
        for position in positions.values():
            if hasattr(position, 'position_value'):
                total_value += position.position_value
            elif hasattr(position, 'current_price') and hasattr(position, 'contracts'):
                multiplier = 100 if hasattr(position, 'option_type') else 1
                total_value += position.current_price * position.contracts * multiplier
                
        # Calculate margin utilization if we have a total value
        margin_utilization = (total_margin / total_value) * 100 if total_value > 0 else 0
        
        # Make this stand out in the logs
        self.log_minimal("======= MARGIN CALCULATION SUMMARY =======")
        self.log_minimal(f"Total positions: {total_positions}")
        self.log_minimal(f"Total margin: ${total_margin:.2f}")
        self.log_minimal(f"Hedging benefits: ${hedging_benefits:.2f}")
        self.log_minimal(f"Position value: ${total_value:.2f}")
        self.log_minimal(f"Margin utilization: {margin_utilization:.2f}%")
        self.log_minimal("=========================================")
    
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
        
        # Log the start of portfolio margin calculation - always show this
        self.log_minimal(f"Portfolio calculation for {len(positions)} positions")
        
        for symbol, position in positions.items():
            position_margin = self.calculate_position_margin(position)
            standalone_margin += position_margin
            margin_by_position[symbol] = position_margin
            # Use standard level for per-position margin amounts
            self.log_standard(f"{symbol}: Initial margin=${position_margin:.2f}")
        
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
        
        # Log grouped positions at verbose level - these details are usually not needed
        self.log_verbose("Grouped positions by underlying:")
        for underlying, symbols in positions_by_underlying.items():
            self.log_verbose(f"  {underlying}: {symbols}")
        
        # Calculate hedging benefits
        total_hedging_benefit = 0
        
        for underlying, symbols in positions_by_underlying.items():
            # Skip if only one position for this underlying (no hedging benefit)
            if len(symbols) <= 1:
                continue
            
            # Calculate potential hedging benefit
            underlying_positions = [positions[symbol] for symbol in symbols]
            underlying_margin = sum(margin_by_position[symbol] for symbol in symbols)
            
            # Check for delta neutrality (approximate)
            net_delta = 0
            total_delta_exposure = 0
            
            # Calculate net delta consistently 
            for position in underlying_positions:
                # Track position delta for debugging
                position_delta = 0
                
                if isinstance(position, OptionPosition):
                    # For options, delta is per share, we need to multiply by 100 for the contract
                    # This is the absolute value in dollars of the option delta
                    current_delta = position.current_delta if hasattr(position, 'current_delta') else 0
                    # The number of contracts
                    contracts = position.contracts if hasattr(position, 'contracts') else 0
                    # The underlying price
                    underlying_price = position.underlying_price if hasattr(position, 'underlying_price') else 0
                    
                    if current_delta != 0 and contracts != 0 and underlying_price != 0:
                        position_delta = current_delta * underlying_price * contracts * 100
                        self.log_verbose(f"Option position {position.symbol}: Delta={current_delta:.4f}, Contracts={contracts}, Price=${underlying_price:.2f}, Delta dollars=${position_delta:.2f}")
                    else:
                        self.log_verbose(f"Option position {position.symbol}: Incomplete data for delta calculation")
                        continue
                else:
                    # For stocks, calculate delta based on direction and number of shares
                    direction = -1 if getattr(position, 'is_short', False) else 1
                    shares = position.contracts if hasattr(position, 'contracts') else 0
                    price = position.current_price if hasattr(position, 'current_price') else 0
                    
                    if shares != 0 and price != 0:
                        position_delta = shares * direction * price
                        self.log_verbose(f"Stock position {position.symbol}: Shares={shares}, Direction={direction}, Price=${price:.2f}, Delta dollars=${position_delta:.2f}")
                    else:
                        self.log_verbose(f"Stock position {position.symbol}: Incomplete data for delta calculation")
                        continue
                
                net_delta += position_delta
                total_delta_exposure += abs(position_delta)
            
            # Log detailed delta hedging calculations
            self.log_minimal(f"Delta hedging calculation for {underlying}:")
            self.log_minimal(f"  Net delta: ${net_delta:.2f}")
            self.log_minimal(f"  Total delta exposure: ${total_delta_exposure:.2f}")
            self.log_minimal(f"  Underlying margin: ${underlying_margin:.2f}")
            
            # Calculate hedge credit based on delta neutrality
            hedge_credit = 0
            
            # Fix: Prevent division by zero and ensure we're using absolute values for calculations
            delta_hedge_ratio = 0.0
            if total_delta_exposure > 0 and underlying_margin > 0:
                # Calculate neutrality as ratio of net to gross exposure
                delta_hedge_ratio = 1.0 - (abs(net_delta) / total_delta_exposure)
                self.log_minimal(f"  Delta hedge ratio calculation: 1.0 - (|{net_delta:.2f}| / {total_delta_exposure:.2f}) = {delta_hedge_ratio:.4f}")
                
                # Ensure ratio is between 0 and 1
                delta_hedge_ratio = min(1.0, max(0.0, delta_hedge_ratio))
                # Apply the hedge credit
                hedge_credit_rate = getattr(self, 'hedge_credit_rate', 0.5)
                hedge_credit = underlying_margin * delta_hedge_ratio * hedge_credit_rate
                self.log_minimal(f"  Hedge credit: {underlying_margin:.2f} × {delta_hedge_ratio:.4f} × {hedge_credit_rate:.2f} = ${hedge_credit:.2f}")
                
                # Cap the hedging benefit to a maximum percentage of the underlying margin
                max_hedge_benefit = underlying_margin * 0.8
                if hedge_credit > max_hedge_benefit:
                    self.log_minimal(f"  Capping hedge credit to 80% of margin: ${max_hedge_benefit:.2f}")
                    hedge_credit = min(hedge_credit, max_hedge_benefit)
                
                # Ensure hedge_credit is never negative (which would increase margin)
                hedge_credit = max(0, hedge_credit)
            
            # Track total hedging benefit
            total_hedging_benefit += hedge_credit
            
            # Log hedging benefit at standard level
            self.log_standard(f"Hedge benefit for {underlying}: ${hedge_credit:.2f} ({delta_hedge_ratio*100:.1f}% offset)")
        
        # Calculate portfolio margin as standalone margin minus hedging benefits
        portfolio_margin = standalone_margin - total_hedging_benefit
        
        # Always log the final margin calculation
        self.log_minimal(f"Final portfolio margin: ${portfolio_margin:.2f}")
        self.log_minimal(f"Hedging benefits: ${total_hedging_benefit:.2f}")
        
        # Prepare the results
        result = {
            "total_margin": portfolio_margin,
            "margin_by_position": margin_by_position,
            "hedging_benefits": total_hedging_benefit
        }
        
        # Log a summary
        self.log_margin_summary(positions, result)
        
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