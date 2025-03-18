"""
Margin Management Module

This module provides tools for managing portfolio margin, ensuring positions
stay within defined margin limits, and implements rebalancing strategies
when margin thresholds are exceeded.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import math
import copy

class PortfolioRebalancer:
    """
    Handles portfolio rebalancing based on margin constraints.
    
    This class monitors portfolio margin usage and implements rebalancing
    strategies when thresholds are exceeded, focusing on:
    1. Rebalancing at 90% margin usage, bringing it down to 80%
    2. Considering hedging impact for new positions
    3. Using a structured approach to determine which positions to trim
    """
    
    def __init__(
        self, 
        portfolio,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the portfolio rebalancer.
        
        Args:
            portfolio: Portfolio instance
            config: Configuration dictionary
            logger: Logger instance
        """
        self.portfolio = portfolio
        self.logger = logger or logging.getLogger('trading_engine')
        
        # Store the config for reference
        self.config = config
        self.margin_config = config.get('margin_management', {})
        
        # Initialize margin thresholds
        self.high_margin_threshold = self.margin_config.get('high_margin_threshold', 0.90)  # Rebalance at 90% usage
        self.target_margin_threshold = self.margin_config.get('target_margin_threshold', 0.80)  # Target 80% usage
        self.margin_buffer_pct = self.margin_config.get('margin_buffer_pct', 0.10)  # 10% buffer
        
        # Margin calculation settings
        self.margin_calculation_method = self.margin_config.get('margin_calculation_method', 'simple')
        self.use_portfolio_calculator = self.margin_config.get('use_portfolio_calculator', True)
        
        # Reference to the portfolio's margin calculator (if available and enabled)
        self.margin_calculator = None
        if self.use_portfolio_calculator and hasattr(self.portfolio, 'margin_calculator'):
            self.margin_calculator = self.portfolio.margin_calculator
            if self.logger:
                self.logger.info(f"[PortfolioRebalancer] Using portfolio's margin calculator: {type(self.margin_calculator).__name__}")
        
        # Rebalance cooldown settings
        self.rebalance_cooldown_days = self.margin_config.get('rebalance_cooldown_days', 3)
        self.last_rebalance_date = None
        
        # Position reduction settings
        self.max_position_reduction_pct = self.margin_config.get('max_position_reduction_pct', 0.25)  # Max 25% reduction per position
        self.losing_position_max_reduction_pct = self.margin_config.get('losing_position_max_reduction_pct', 0.40)  # Up to 40% for losing positions
        self.urgent_reduction_pct = self.margin_config.get('urgent_reduction_pct', 0.50)  # Up to 50% in urgent situations
        
        # Initialize tracking variables
        self.rebalance_history = []
        
        # Log initialization
        self.logger.info("[PortfolioRebalancer] Initialized with margin management settings:")
        self.logger.info(f"  High margin threshold: {self.high_margin_threshold:.0%} (will trigger rebalancing)")
        self.logger.info(f"  Target margin threshold: {self.target_margin_threshold:.0%} (target after rebalancing)")
        self.logger.info(f"  Margin buffer: {self.margin_buffer_pct:.0%}")
        self.logger.info(f"  Rebalance cooldown: {self.rebalance_cooldown_days} days")
        self.logger.info(f"  Position reduction limits: normal={self.max_position_reduction_pct:.0%}, losing={self.losing_position_max_reduction_pct:.0%}, urgent={self.urgent_reduction_pct:.0%}")
    
    def analyze_margin_status(self, current_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Analyze current margin status and determine if rebalancing is needed.
        
        Args:
            current_date: Current trading date
            
        Returns:
            dict: Analysis results including margin metrics and rebalance needs
        """
        # Get current portfolio metrics
        portfolio_value = self.portfolio.get_portfolio_value()
        margin_requirement = self.portfolio.calculate_margin_requirement()
        
        # Calculate margin utilization
        margin_utilization = margin_requirement / portfolio_value if portfolio_value > 0 else 1.0
        
        # Determine if we need rebalancing
        needs_rebalancing = margin_utilization >= self.high_margin_threshold
        
        # Check for cooldown period
        in_cooldown = False
        days_since_rebalance = None
        
        if current_date and self.last_rebalance_date:
            days_since_rebalance = (current_date - self.last_rebalance_date).days
            in_cooldown = days_since_rebalance < self.rebalance_cooldown_days
            
            if in_cooldown and needs_rebalancing:
                self.logger.info(f"[PortfolioRebalancer] In rebalance cooldown period: day {days_since_rebalance} of {self.rebalance_cooldown_days}")
                cooldown_end = self.last_rebalance_date + pd.Timedelta(days=self.rebalance_cooldown_days)
                self.logger.info(f"[PortfolioRebalancer] Cooldown ends: {cooldown_end.strftime('%Y-%m-%d')}")
        
        # Final determination considering cooldown
        needs_rebalancing = needs_rebalancing and not in_cooldown
        
        # Calculate target margin reduction if needed
        target_margin = portfolio_value * self.target_margin_threshold
        margin_to_reduce = margin_requirement - target_margin if needs_rebalancing else 0
        
        # Log margin status
        self.logger.info("[PortfolioRebalancer] Margin Status Analysis:")
        self.logger.info(f"  Portfolio Value: ${portfolio_value:.2f}")
        self.logger.info(f"  Margin Requirement: ${margin_requirement:.2f}")
        self.logger.info(f"  Margin Utilization: {margin_utilization:.2%}")
        self.logger.info(f"  High Threshold: {self.high_margin_threshold:.2%}")
        self.logger.info(f"  Target Threshold: {self.target_margin_threshold:.2%}")
        
        if needs_rebalancing:
            self.logger.info(f"  Rebalancing needed - Margin to reduce: ${margin_to_reduce:.2f}")
        elif margin_utilization >= self.high_margin_threshold and in_cooldown:
            self.logger.info(f"  Rebalancing needed but in cooldown period ({days_since_rebalance}/{self.rebalance_cooldown_days} days)")
        else:
            self.logger.info(f"  No rebalancing needed - within acceptable margin limits")
        
        return {
            'portfolio_value': portfolio_value,
            'margin_requirement': margin_requirement,
            'margin_utilization': margin_utilization,
            'needs_rebalancing': needs_rebalancing,
            'margin_to_reduce': margin_to_reduce,
            'target_margin': target_margin,
            'in_cooldown': in_cooldown,
            'days_since_rebalance': days_since_rebalance
        }
    
    def can_add_position_with_hedge(self, position_data: Dict[str, Any], hedge_delta: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a new position can be added considering its margin impact along with any necessary hedging.
        
        Args:
            position_data: Position data including symbol, contracts, price, etc.
            hedge_delta: Delta to hedge (opposite of position delta)
            
        Returns:
            Tuple of (can_add, details)
        """
        # Get current portfolio metrics
        portfolio_value = self.portfolio.get_portfolio_value()
        current_margin = self.portfolio.calculate_margin_requirement()
        
        # Extract basic position information
        symbol = position_data.get('symbol', '')
        contracts = position_data.get('contracts', 0)
        price = position_data.get('price', 0)
        
        # Initialize margin variables
        position_margin = 0
        hedge_margin = 0
        total_additional_margin = 0
        
        # Debug the configuration
        self.logger.info(f"[PortfolioRebalancer] Margin calculation configuration:")
        self.logger.info(f"  Calculation method: {self.margin_calculation_method}")
        self.logger.info(f"  Use portfolio calculator: {self.use_portfolio_calculator}")
        self.logger.info(f"  Portfolio has margin calculator: {hasattr(self.portfolio, 'margin_calculator')}")
        if hasattr(self.portfolio, 'margin_calculator'):
            self.logger.info(f"  Portfolio margin calculator type: {type(self.portfolio.margin_calculator).__name__}")
        self.logger.info(f"  Rebalancer has margin calculator: {self.margin_calculator is not None}")
        if self.margin_calculator:
            self.logger.info(f"  Rebalancer margin calculator type: {type(self.margin_calculator).__name__}")
        
        # SIMPLE APPROACH: Use position_data's margin_per_contract for margin calculation
        # This will be more reliable than using portfolio's margin calculator which may have issues
        if 'margin_per_contract' in position_data:
            margin_per_contract = position_data.get('margin_per_contract', 0)
            self.logger.info(f"[PortfolioRebalancer] Using margin_per_contract from position data: ${margin_per_contract:.2f}")
            
            # Calculate position margin
            position_margin = margin_per_contract * contracts
            
            # Estimate hedge margin based on delta
            hedge_shares = abs(hedge_delta) * 100
            underlying_price = position_data.get('underlying_price', 0)
            if underlying_price <= 0 and 'instrument_data' in position_data:
                underlying_price = position_data['instrument_data'].get('UnderlyingPrice', 0)
            
            hedge_margin_rate = 0.25  # 25% margin requirement for index ETFs
            hedge_margin = hedge_shares * underlying_price * hedge_margin_rate
            
            # Total additional margin required
            total_additional_margin = position_margin + hedge_margin
        # FALLBACK: Use a simple estimation based on price and contracts
        else:
            self.logger.info(f"[PortfolioRebalancer] Using simplified margin calculation")
            
            # For option positions, estimate margin as a percentage of notional value
            notional_value = price * contracts * 100  # 100 shares per contract
            
            # Use different margin rates based on position type
            is_short = position_data.get('is_short', False)
            if is_short:
                # Higher margin for short options (20-40% of notional)
                margin_rate = 0.40
            else:
                # Full value for long options (100% of notional)
                margin_rate = 1.0
            
            position_margin = notional_value * margin_rate
            
            # Estimate hedge margin based on delta
            hedge_shares = abs(hedge_delta) * 100
            underlying_price = position_data.get('underlying_price', 0)
            if underlying_price <= 0 and 'instrument_data' in position_data:
                underlying_price = position_data['instrument_data'].get('UnderlyingPrice', 0)
            
            hedge_margin_rate = 0.25  # 25% margin requirement for index ETFs
            hedge_margin = hedge_shares * underlying_price * hedge_margin_rate
            
            # Total additional margin required
            total_additional_margin = position_margin + hedge_margin
        
        # Calculate new margin utilization if position is added
        new_margin = current_margin + total_additional_margin
        new_utilization = new_margin / portfolio_value if portfolio_value > 0 else 1.0
        
        # Determine if position can be added
        can_add = new_utilization < self.high_margin_threshold
        
        # Store the result with detailed information
        details = {
            'current_margin': current_margin,
            'position_margin': position_margin,
            'hedge_margin': hedge_margin,
            'total_additional_margin': total_additional_margin,
            'new_margin': new_margin,
            'current_utilization': current_margin / portfolio_value if portfolio_value > 0 else 1.0,
            'new_utilization': new_utilization,
            'threshold': self.high_margin_threshold,
            'calculation_method': 'simplified' if 'margin_per_contract' not in position_data else 'position_data'
        }
        
        # Log the analysis
        self.logger.info(f"[PortfolioRebalancer] Analyzing margin impact of new position:")
        self.logger.info(f"  Symbol: {symbol}, Contracts: {contracts}")
        self.logger.info(f"  Position margin: ${position_margin:.2f}")
        self.logger.info(f"  Expected hedge delta: {hedge_delta:.2f}")
        self.logger.info(f"  Estimated hedge margin: ${hedge_margin:.2f}")
        self.logger.info(f"  Total additional margin: ${total_additional_margin:.2f}")
        self.logger.info(f"  Current margin: ${current_margin:.2f} ({details['current_utilization']:.2%})")
        self.logger.info(f"  New margin: ${new_margin:.2f} ({new_utilization:.2%})")
        self.logger.info(f"  High threshold: {self.high_margin_threshold:.2%}")
        self.logger.info(f"  Can add position: {can_add}")
        self.logger.info(f"  Calculation method: {details['calculation_method']}")
        
        return can_add, details
    
    def rebalance_portfolio(self, current_date: datetime, market_data_by_symbol: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Rebalance the portfolio to maintain margin requirements within thresholds.
        
        Args:
            current_date: The current date
            market_data_by_symbol: Market data keyed by symbol
            
        Returns:
            Dict[str, Any]: Dictionary with rebalance results
        """
        if not self.portfolio:
            self.logger.warning("No portfolio provided for rebalancing")
            return {'rebalanced': False}
        
        # Log the current positions
        self._log_positions_table()
        
        # Get current margin status
        portfolio_value = self.portfolio.get_portfolio_value()
        current_margin = self.calculate_total_margin_requirement()
        margin_utilization = (current_margin / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        self.logger.info(f"[PortfolioRebalancer] Margin Status Analysis:")
        self.logger.info(f"  Portfolio Value: ${portfolio_value:.2f}")
        self.logger.info(f"  Margin Requirement: ${current_margin:.2f}")
        self.logger.info(f"  Margin Utilization: {margin_utilization:.2f}%")
        self.logger.info(f"  High Threshold: {self.high_margin_threshold * 100:.2f}%")
        self.logger.info(f"  Target Threshold: {self.target_margin_threshold * 100:.2f}%")
        
        # If utilization is below high threshold, no rebalancing needed
        if margin_utilization <= self.high_margin_threshold * 100:
            self.logger.info(f"No rebalancing needed - Margin utilization {margin_utilization:.2f}% is within acceptable limits")
            return {'rebalanced': False}
        
        # Calculate how much margin we need to reduce
        target_margin = portfolio_value * self.target_margin_threshold
        margin_to_reduce = current_margin - target_margin
        
        self.logger.info(f"  Rebalancing needed - Margin to reduce: ${margin_to_reduce:.2f}")
        
        # Get position data with margin info
        position_data = self._get_positions_with_margin()
        
        # Log the positions before rebalancing
        self.logger.info("\n[PortfolioRebalancer] Positions Before Rebalancing:\n")
        self._log_positions_table()
        
        # Sort positions by unrealized P&L percent (worst first)
        position_data.sort(key=lambda x: x['pnl_pct'])
        
        # Plan closures
        closure_plan = []
        remaining_margin_to_reduce = margin_to_reduce
        
        for position_info in position_data:
            symbol = position_info['symbol']
            position = self.portfolio.positions[symbol]
            
            # Calculate how many contracts to close
            position_margin = position_info['margin']
            if position_margin <= 0:
                continue
            
            # We close at most all contracts/shares
            max_contracts_to_close = position.contracts
            
            # Start by suggesting to close half the position, rounded up
            contracts_to_close = min(
                max_contracts_to_close,
                math.ceil(max_contracts_to_close * 0.5)
            )
            
            # If that's not enough, increase to what we need
            margin_per_contract = position_margin / max_contracts_to_close if max_contracts_to_close > 0 else 0
            if margin_per_contract * contracts_to_close < remaining_margin_to_reduce and contracts_to_close < max_contracts_to_close:
                # Calculate exactly how many contracts we need to close to meet our margin reduction target
                contracts_needed = math.ceil(remaining_margin_to_reduce / margin_per_contract) if margin_per_contract > 0 else 0
                contracts_to_close = min(max_contracts_to_close, contracts_needed)
            
            margin_freed = margin_per_contract * contracts_to_close
            
            # If we're closing this position
            if contracts_to_close > 0:
                close_pct = (contracts_to_close / max_contracts_to_close) * 100
                
                self.logger.info(f"[PortfolioRebalancer] Position {symbol}: Plan to close {contracts_to_close} of {max_contracts_to_close} contracts ({close_pct:.0f}%)")
                self.logger.info(f"  Margin per contract: ${margin_per_contract:.2f}")
                self.logger.info(f"  Margin freed: ${margin_freed:.2f}")
                self.logger.info(f"  Unrealized P&L: ${position.unrealized_pnl:.2f} ({position_info['pnl_pct']:.1f}%)")
                
                remaining_margin_to_reduce -= margin_freed
                self.logger.info(f"  Remaining margin to reduce: ${remaining_margin_to_reduce:.2f}")
                
                closure_plan.append({
                    'symbol': symbol,
                    'contracts_to_close': contracts_to_close,
                    'margin_freed': margin_freed
                })
                
                # If we've reduced enough margin, stop
                if remaining_margin_to_reduce <= 0:
                    break
        
        # Execute position closures
        self.logger.info("\n[PortfolioRebalancer] Executing Position Closures:\n")
        closed_positions = []
        total_margin_freed = 0
        total_realized_pnl = 0
        
        for closure in closure_plan:
            symbol = closure['symbol']
            contracts_to_close = closure['contracts_to_close']
            
            position = self.portfolio.positions[symbol]
            
            # Skip if no market data for this symbol
            if symbol not in market_data_by_symbol:
                self.logger.warning(f"  Warning: No market data available for {symbol}, skipping closure")
                continue
            
            # Store original entry price for reporting
            original_entry_price = position.avg_entry_price
            
            # Close the position
            self.logger.info(f"[PortfolioRebalancer] Closing {contracts_to_close} contracts of {symbol}")
            
            # Let the portfolio handle the position closure
            pnl = self.portfolio.remove_position(
                symbol,
                contracts_to_close,
                market_data_by_symbol[symbol].get('MidPrice', 0),
                reason="Margin Rebalance"
            )
            total_realized_pnl += pnl
            
            # Calculate actual margin freed
            margin_freed = closure['margin_freed']
            total_margin_freed += margin_freed
            
            # Log the closure
            current_price = market_data_by_symbol[symbol].get('MidPrice', 0)
            self.logger.info(f"[PortfolioRebalancer] Closed {contracts_to_close} contracts of {symbol}")
            self.logger.info(f"  Entry Price: ${original_entry_price:.2f}, Exit Price: ${current_price:.2f}")
            self.logger.info(f"  PnL: ${pnl:.2f}")
            self.logger.info(f"  Margin Freed: ${margin_freed:.2f}")
            
            closed_positions.append({
                'symbol': symbol,
                'contracts_closed': contracts_to_close,
                'margin_freed': margin_freed,
                'realized_pnl': pnl
            })
        
        # Summary after rebalancing
        self.logger.info("\n[PortfolioRebalancer] Rebalancing Summary:")
        self.logger.info(f"  Total Positions Affected: {len(closed_positions)}")
        self.logger.info(f"  Total Margin Freed: ${total_margin_freed:.2f}")
        self.logger.info(f"  Total Realized PnL: ${total_realized_pnl:.2f}")
        
        # Recalculate margin metrics after rebalancing
        new_margin_requirement = self.calculate_total_margin_requirement()
        new_utilization = new_margin_requirement / portfolio_value if portfolio_value > 0 else 0
        
        self.logger.info(f"  Original Margin Requirement: ${current_margin:.2f}")
        self.logger.info(f"  New Margin Requirement: ${new_margin_requirement:.2f}")
        self.logger.info(f"  New Margin Utilization: {new_utilization:.2%}")
        self.logger.info(f"  Target Threshold: {self.target_margin_threshold * 100:.2f}%")
        
        # Update last rebalance date
        self.last_rebalance_date = current_date
        
        # Add to history for tracking
        self.rebalance_history.append({
            'date': current_date,
            'positions_closed': len(closed_positions),
            'margin_freed': total_margin_freed,
            'realized_pnl': total_realized_pnl,
            'initial_utilization': margin_utilization,
            'final_utilization': new_utilization
        })
        
        # Return results as a dictionary
        return {
            'rebalanced': True,
            'positions_closed': len(closed_positions),
            'margin_freed': total_margin_freed,
            'realized_pnl': total_realized_pnl,
            'initial_utilization': margin_utilization / 100,  # Convert from percentage to decimal
            'final_utilization': new_utilization  # Already in decimal
        }
    
    def _get_positions_with_margin(self) -> List[Dict[str, Any]]:
        """
        Get all positions with margin information.
        
        Returns:
            list: List of position data dictionaries with margin information
        """
        position_data = []
        
        for symbol, position in self.portfolio.positions.items():
            # Calculate margin
            position_margin = position.calculate_margin_requirement(1.0)  # Use basic margin without leverage
            
            # Calculate PnL percentage
            if position.avg_entry_price > 0:
                if position.is_short:
                    pnl_pct = (position.avg_entry_price - position.current_price) / position.avg_entry_price * 100
                else:
                    pnl_pct = (position.current_price - position.avg_entry_price) / position.avg_entry_price * 100
            else:
                pnl_pct = 0
                
            # Get days to expiry if available
            dte = getattr(position, 'days_to_expiry', 0)
            
            position_data.append({
                'symbol': symbol,
                'position': position,
                'margin': position_margin,
                'dte': dte,
                'pnl_pct': pnl_pct,
                'delta': position.current_delta,
                'unrealized_pnl': position.unrealized_pnl
            })
            
        return position_data
    
    def _prioritize_positions_for_reduction(self, positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort positions based on our reduction strategy.
        
        Our strategy prioritizes:
        1. Losing positions first (largest negative PnL)
        2. For positions with similar PnL, sort by DTE (lowest first)
        3. For positions with similar PnL and DTE, sort by absolute delta (lowest first)
        
        Args:
            positions: List of position data dictionaries
            
        Returns:
            list: Sorted positions for reduction
        """
        return sorted(positions, key=lambda item: (
            item['unrealized_pnl'] if item['unrealized_pnl'] < 0 else 9999999,  # Sort negative PnL first
            item['dte'],  # Then by DTE for positions with similar PnL
            abs(item['delta'])  # Then by absolute delta (lower delta first)
        ))
    
    def _calculate_max_reduction_percentage(self, position_info: Dict[str, Any], margin_utilization: float) -> float:
        """
        Calculate the maximum percentage of a position to close.
        
        Args:
            position_info: Position information
            margin_utilization: Current margin utilization
            
        Returns:
            float: Maximum percentage to close (0.0-1.0)
        """
        # Default to standard max reduction
        max_pct_to_close = self.max_position_reduction_pct
        
        # For positions in loss, we might want to close a larger percentage
        if position_info['unrealized_pnl'] < 0:
            position = position_info['position']
            pnl_severity = abs(position_info['unrealized_pnl']) / (
                position.avg_entry_price * position.contracts * 100)
                
            # Increase max closure percentage based on loss severity
            if pnl_severity > 0.1:  # If losing more than 10% of position value
                losing_max_reduction = self.losing_position_max_reduction_pct
                max_pct_to_close = min(losing_max_reduction, pnl_severity * 2)
        
        # For extremely high margin utilization, allow more aggressive reduction
        if margin_utilization > 0.95:  # If over 95% utilized
            max_pct_to_close = self.urgent_reduction_pct
            
        return max_pct_to_close
    
    def _log_positions_table(self) -> None:
        """
        Log a formatted table of current positions.
        """
        if not self.portfolio:
            return

        rows = []
        headers = ["Symbol", "Contracts", "Entry", "Current", "PnL", "PnL %", "Margin", "DTE", "Delta"]
        
        for symbol, position in self.portfolio.positions.items():
            position_margin = position.calculate_margin_requirement(1.0)
            
            # Calculate P&L percentage
            pnl_pct = 0
            if position.avg_entry_price > 0:
                pnl_pct = (position.unrealized_pnl / (position.avg_entry_price * position.contracts * 100)) * 100 if hasattr(position, 'avg_entry_price') else 0
            
            # For stocks, use a different calculation
            if hasattr(position, 'position_type') and position.position_type == 'stock':
                pnl_pct = (position.current_price / position.avg_entry_price - 1) * 100
            
            # Get days to expiry if available
            dte = getattr(position, 'days_to_expiry', 0)
            
            rows.append([
                symbol,
                position.contracts,
                f"${position.avg_entry_price:.2f}",
                f"${position.current_price:.2f}",
                f"${position.unrealized_pnl:.2f}",
                f"{pnl_pct:.1f}%",
                f"${position_margin:.2f}",
                f"{dte}",
                f"{position.current_delta:.3f}"
            ])
        
        # Log the positions table
        self.logger.info("\nPositions Table:")
        self.logger.info("-" * 130)
        self.logger.info("|" + "|".join([f"{header:<16}" for header in headers]) + "|")
        self.logger.info("-" * 130)
        for row in rows:
            self.logger.info("|" + "|".join([f"{item:<16}" for item in row]) + "|")
        self.logger.info("-" * 130)

    def calculate_total_margin_requirement(self) -> float:
        """
        Calculate the total margin requirement for the portfolio.
        Sums up margin requirements for each position.
        
        Returns:
            float: Total margin requirement
        """
        if not self.portfolio:
            return 0.0
            
        total_margin = 0.0
        
        for position in self.portfolio.positions.values():
            if hasattr(position, 'calculate_margin_requirement'):
                margin = position.calculate_margin_requirement(1.0)  # Use basic margin without leverage
                total_margin += margin
                
        return total_margin 