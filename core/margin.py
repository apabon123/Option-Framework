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
        Calculate margin requirement for a single position.

        Args:
            position: Position to calculate margin for

        Returns:
            float: Margin requirement in dollars
        """
        if position.contracts <= 0:
            return 0

        # Basic margin calculation: position value * leverage
        initial_margin = position.avg_entry_price * position.contracts * 100 / self.max_leverage

        # Adjust margin for unrealized PnL (both gains and losses)
        # For short positions, losses increase margin, gains reduce it
        adjusted_margin = initial_margin + position.unrealized_pnl if position.is_short else initial_margin

        # Log the margin calculation
        if self.logger:
            self.logger.debug(f"[Margin] Position {position.symbol}: {position.contracts} contracts")
            self.logger.debug(f"  Initial margin: ${initial_margin:.2f}")
            self.logger.debug(f"  Adjusted margin: ${adjusted_margin:.2f}")

        return max(adjusted_margin, 0)  # Ensure non-negative margin

    def calculate_portfolio_margin(self, positions: Dict[str, Position]) -> Dict[str, Any]:
        """
        Calculate margin for a portfolio of positions.

        Args:
            positions: Dictionary of positions by symbol

        Returns:
            dict: Dictionary with total margin and margin by position
        """
        if not positions:
            return {'total_margin': 0, 'margin_by_position': {}}

        # Calculate margin for each position
        margin_by_position = {}
        total_margin = 0

        for symbol, position in positions.items():
            position_margin = self.calculate_position_margin(position)
            margin_by_position[symbol] = position_margin
            total_margin += position_margin

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
        if position.contracts <= 0:
            return 0

        # Basic margin calculation based on entry price (initial margin)
        initial_margin = position.avg_entry_price * position.contracts * 100 / self.max_leverage

        # Apply OTM discount if the position is a proper OptionPosition
        if isinstance(position, OptionPosition):
            is_otm = not position.is_itm()
            if is_otm:
                initial_margin *= self.otm_margin_multiplier
                if self.logger:
                    self.logger.debug(f"[Margin] OTM discount applied to {position.symbol}")

        # Adjust for unrealized PnL for short positions (losses increase margin)
        # Only add losses, don't reduce for gains (conservative approach)
        if position.is_short and position.unrealized_pnl < 0:
            adjusted_margin = initial_margin - position.unrealized_pnl  # Negative PnL becomes positive addition
        else:
            adjusted_margin = initial_margin

        # Log the calculation steps
        if self.logger:
            self.logger.debug(f"[Margin] {position.symbol}: Initial margin=${initial_margin:.2f}")
            if position.is_short and position.unrealized_pnl < 0:
                self.logger.debug(f"  Unrealized PnL adjustment: ${-position.unrealized_pnl:.2f}")
            self.logger.debug(f"  Final margin: ${adjusted_margin:.2f}")

        return max(adjusted_margin, 0)  # Ensure non-negative margin

    # We inherit calculate_portfolio_margin and calculate_total_margin from the base class


class SPANMarginCalculator(MarginCalculator):
    """
    Margin calculator implementing a simplified SPAN (Standard Portfolio Analysis of Risk)
    margin model for option portfolios with enhanced support for delta hedging.

    This implementation focuses on proper risk netting between options and their
    underlying securities, with special handling for delta-hedged positions.

    SPAN is the industry standard for calculating margin for options and futures,
    providing capital efficiency through risk-based portfolio margining with delta offsets.
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
        Initialize the SPAN margin calculator with enhanced delta hedging support.

        Args:
            max_leverage: Maximum leverage allowed
            volatility_multiplier: Scenario stress multiplier for volatility
            correlation_matrix: Matrix of correlations between underlyings (for margin offsets)
            initial_margin_percentage: Initial margin as percentage of notional value
            maintenance_margin_percentage: Maintenance margin as percentage of notional value
            hedge_credit_rate: Credit rate for properly hedged positions (0-1)
            logger: Logger instance
        """
        super().__init__(max_leverage, logger)
        self.volatility_multiplier = volatility_multiplier
        self.correlation_matrix = correlation_matrix
        self.initial_margin_percentage = initial_margin_percentage
        self.maintenance_margin_percentage = maintenance_margin_percentage
        self.hedge_credit_rate = min(max(hedge_credit_rate, 0), 1)  # Ensure between 0-1

    def _calculate_scan_risk(self, position: OptionPosition) -> float:
        """
        Calculate scan risk for a position (core of SPAN calculation).

        Args:
            position: Option position

        Returns:
            float: Scan risk amount in dollars
        """
        # In a full SPAN implementation, this would simulate multiple price/vol scenarios
        # For simplicity, we'll use a basic approximation based on delta and gamma

        # Simulate a price move of 15% of the underlying price
        price_move_pct = 0.15
        price_move = position.underlying_price * price_move_pct

        # First-order approximation using delta
        delta_effect = position.current_delta * price_move * position.contracts * 100

        # Second-order approximation using gamma
        gamma_effect = 0.5 * position.current_gamma * (price_move ** 2) * position.contracts * 100

        # Total risk is the sum of delta and gamma effects
        scan_risk = abs(delta_effect + gamma_effect * self.volatility_multiplier)

        return scan_risk

    def calculate_position_margin(self, position: Position) -> float:
        """
        Calculate SPAN margin for a single position.

        Args:
            position: Position to calculate margin for

        Returns:
            float: Margin requirement in dollars
        """
        if position.contracts <= 0:
            return 0

        # For stock/ETF positions used in hedging
        if not isinstance(position, OptionPosition):
            # For ETFs and stocks, use the RegT margin requirement (higher of 25% of position value or $2000)
            position_value = position.current_price * position.contracts
            reg_t_margin = max(position_value * 0.25, 2000)

            if self.logger:
                self.logger.debug(f"[SPAN Margin] Stock/ETF position {position.symbol}: {position.contracts} shares")
                self.logger.debug(f"  Position value: ${position_value:.2f}")
                self.logger.debug(f"  RegT margin: ${reg_t_margin:.2f}")

            return reg_t_margin

        # For option positions
        # Calculate notional value
        notional_value = position.underlying_price * position.contracts * 100

        # Calculate initial margin based on notional
        base_margin = notional_value * self.initial_margin_percentage

        # Calculate scan risk (risk of market move)
        scan_risk = self._calculate_scan_risk(position)

        # Use the maximum of base margin and scan risk
        margin = max(base_margin, scan_risk)

        # Log the calculation
        if self.logger:
            self.logger.debug(f"[SPAN Margin] Option position {position.symbol}: {position.contracts} contracts")
            self.logger.debug(f"  Notional value: ${notional_value:.2f}")
            self.logger.debug(f"  Base margin: ${base_margin:.2f}")
            self.logger.debug(f"  Scan risk: ${scan_risk:.2f}")
            self.logger.debug(f"  Final margin: ${margin:.2f}")

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

    def calculate_total_margin(self, positions: Dict[str, Position]) -> float:
        """
        Calculate total margin for all positions with hedging benefits.

        Args:
            positions: Dictionary of positions by symbol

        Returns:
            float: Total margin requirement in dollars
        """
        result = self.calculate_portfolio_margin(positions)
        return result.get('total_margin', 0)

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

        # Step 1: Calculate standalone margin for each position
        margin_by_position = {}
        for symbol, position in positions.items():
            position_margin = self.calculate_position_margin(position)
            margin_by_position[symbol] = position_margin

        # Step 2: Group positions by underlying for delta netting
        positions_by_underlying = {}
        for symbol, position in positions.items():
            underlying = self._extract_underlying_symbol(position)
            if underlying not in positions_by_underlying:
                positions_by_underlying[underlying] = []
            positions_by_underlying[underlying].append(symbol)

        # Log the grouping
        if self.logger:
            self.logger.debug(f"[SPAN Margin] Grouped positions by underlying:")
            for underlying, symbols in positions_by_underlying.items():
                self.logger.debug(f"  {underlying}: {symbols}")

        # Step 3: Calculate net deltas and apply hedging benefits
        total_standalone_margin = sum(margin_by_position.values())
        total_hedging_benefit = 0

        # For each underlying, calculate net delta exposure and apply margin offsets
        net_margin_by_underlying = {}

        for underlying, symbols in positions_by_underlying.items():
            # Skip underlyings with only one position (no hedging benefit)
            if len(symbols) <= 1:
                continue

            # Calculate net delta for this underlying
            net_delta = 0
            total_underlying_margin = 0

            for symbol in symbols:
                position = positions[symbol]
                delta = 0

                if isinstance(position, OptionPosition):
                    # For options, use delta * contracts * 100 * underlying_price
                    delta = position.current_delta * position.contracts * 100
                    # Reverse sign for short positions
                    if position.is_short:
                        delta = -delta
                else:
                    # For stocks/ETFs, delta is simply the number of shares
                    delta = position.contracts
                    # Reverse sign for short positions
                    if position.is_short:
                        delta = -delta

                net_delta += delta
                total_underlying_margin += margin_by_position[symbol]

            # Calculate dollar value of net delta exposure
            net_delta_dollars = abs(net_delta)

            # Standalone margin is the sum of all position margins
            standalone_margin = total_underlying_margin

            # Apply hedging benefit based on how well hedged the position is
            hedge_ratio = 1.0 - min(
                net_delta_dollars / sum(abs(positions[s].current_delta * positions[s].contracts * 100)
                                        if isinstance(positions[s], OptionPosition) else
                                        abs(positions[s].contracts)
                                        for s in symbols), 1.0)

            # Apply the hedge credit
            hedging_benefit = standalone_margin * hedge_ratio * self.hedge_credit_rate

            # Calculate net margin after hedging benefit
            net_margin = standalone_margin - hedging_benefit

            # Add to total hedging benefit
            total_hedging_benefit += hedging_benefit

            # Store net margin for this underlying
            net_margin_by_underlying[underlying] = {
                'standalone_margin': standalone_margin,
                'net_delta': net_delta,
                'hedge_ratio': hedge_ratio,
                'hedging_benefit': hedging_benefit,
                'net_margin': net_margin
            }

            # Log the calculation
            if self.logger:
                self.logger.debug(f"[SPAN Margin] Underlying {underlying}:")
                self.logger.debug(f"  Standalone margin: ${standalone_margin:.2f}")
                self.logger.debug(f"  Net delta: {net_delta:.2f}")
                self.logger.debug(f"  Hedge ratio: {hedge_ratio:.2%}")
                self.logger.debug(f"  Hedging benefit: ${hedging_benefit:.2f}")
                self.logger.debug(f"  Net margin: ${net_margin:.2f}")

        # Step 4: Apply inter-product correlation benefits (if correlation matrix is provided)
        inter_product_benefit = 0
        if self.correlation_matrix is not None and len(net_margin_by_underlying) > 1:
            # This would calculate margin benefits from correlation between different underlyings
            # For simplicity, we'll apply a basic approximation
            underlyings = list(net_margin_by_underlying.keys())

            # For each pair of underlyings, check correlation and apply benefit
            for i in range(len(underlyings)):
                for j in range(i + 1, len(underlyings)):
                    u1 = underlyings[i]
                    u2 = underlyings[j]

                    # Get correlation if available
                    correlation = 0.5  # Default medium correlation
                    if u1 in self.correlation_matrix.index and u2 in self.correlation_matrix.columns:
                        correlation = self.correlation_matrix.loc[u1, u2]

                    # Only apply benefit for positive correlation
                    if correlation > 0:
                        # Calculate benefit based on correlation and opposing delta positions
                        u1_delta = net_margin_by_underlying[u1]['net_delta']
                        u2_delta = net_margin_by_underlying[u2]['net_delta']

                        # If deltas are in opposite directions, can provide diversification benefit
                        if u1_delta * u2_delta < 0:
                            # Benefit is proportional to correlation and the smaller of the two exposures
                            smaller_exposure = min(abs(u1_delta), abs(u2_delta))
                            benefit = smaller_exposure * correlation * 0.5  # 50% max benefit
                            inter_product_benefit += benefit

                            if self.logger:
                                self.logger.debug(f"[SPAN Margin] Inter-product benefit between {u1} and {u2}:")
                                self.logger.debug(f"  Correlation: {correlation:.2f}")
                                self.logger.debug(f"  Benefit: ${benefit:.2f}")

        # Step 5: Calculate final portfolio margin
        total_margin = total_standalone_margin - total_hedging_benefit - inter_product_benefit

        # Ensure margin doesn't go below minimum threshold
        # In real SPAN, there are minimum floor requirements
        total_margin = max(total_margin, max(margin_by_position.values()) * 0.5)

        if self.logger:
            self.logger.debug(f"[SPAN Margin] Portfolio summary:")
            self.logger.debug(f"  Standalone margin: ${total_standalone_margin:.2f}")
            self.logger.debug(f"  Hedging benefit: ${total_hedging_benefit:.2f}")
            self.logger.debug(f"  Inter-product benefit: ${inter_product_benefit:.2f}")
            self.logger.debug(f"  Final portfolio margin: ${total_margin:.2f}")

        return {
            'total_margin': total_margin,
            'margin_by_position': margin_by_position,
            'hedging_benefits': total_hedging_benefit,
            'inter_product_benefits': inter_product_benefit,
            'net_margin_by_underlying': net_margin_by_underlying
        }