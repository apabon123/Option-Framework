"""
Risk Scaling Module

This module provides functionality for risk-based position size scaling.
It implements various methods to adjust position sizes based on performance metrics,
market conditions, and risk parameters.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class RiskScaler:
    """
    Base class for risk scaling functionality.
    
    This class provides risk-based position size scaling using various methods 
    such as Sharpe ratio, volatility targeting, and adaptive methods.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the RiskScaler with configuration parameters"""
        self.config = config
        self.logger = logger or logging.getLogger('risk_scaler')

        # Extract risk scaling parameters from config
        risk_config = config.get('risk_scaling', {})
        
        # Determine if risk scaling is enabled (default to True for backward compatibility)
        self.enabled = risk_config.get('enabled', True)
        
        # Set scaling method 
        self.scaling_method = risk_config.get('method', 'sharpe')
        
        # Extract common parameters
        self.rolling_window = risk_config.get('rolling_window', 21)
        self.target_z = risk_config.get('target_z', 0)  # z-score at which full exposure is reached
        self.min_z = risk_config.get('min_z', -2.0)  # z-score for minimum exposure
        self.min_investment = risk_config.get('min_investment', 0.25)  # Minimum investment level

        # Load method-specific parameters
        self.sharpe_params = risk_config.get('sharpe', {})
        self.volatility_params = risk_config.get('volatility', {})
        self.adaptive_params = risk_config.get('adaptive', {})
        self.combined_params = risk_config.get('combined', {})

        # Track risk scaling history for analysis
        self.risk_scaling_history = []

        # Log initialization in a standardized format
        if self.logger:
            self.logger.info("=" * 40)
            self.logger.info("RISK SCALER INITIALIZATION")
            self.logger.info(f"  Risk scaling: {'Enabled' if self.enabled else 'Disabled'}")
            if self.enabled:
                self.logger.info(f"  Scaling method: {self.scaling_method}")
                self.logger.info(f"  Rolling window: {self.rolling_window} days")
                self.logger.info(f"  Target Z-score: {self.target_z:.2f}")
                self.logger.info(f"  Min Z-score: {self.min_z:.2f}")
                self.logger.info(f"  Min investment level: {self.min_investment:.2%}")
            self.logger.info("=" * 40)

    def calculate_risk_scaling(self, returns: Union[pd.Series, List[Dict[str, Any]], Dict[str, float]]) -> float:
        """
        Calculate risk scaling factor based on the configured method.
        
        Args:
            returns: Series, list, or dictionary of returns
            
        Returns:
            float: Risk scaling factor between min_scaling and max_scaling
        """
        # If risk scaling is disabled, return 1.0
        if not self.enabled:
            return 1.0
            
        # Convert returns to pandas Series if needed
        if isinstance(returns, list):
            # Convert list of dictionaries to Series
            if returns and isinstance(returns[0], dict):
                returns_series = pd.Series([entry['return'] for entry in returns])
            else:
                self.logger.warning("[RiskScaler] Invalid returns format, using neutral scaling (1.0)")
                return 1.0
        elif isinstance(returns, dict):
            # Convert dictionary to Series
            if returns:
                returns_series = pd.Series(list(returns.values()), index=list(returns.keys()))
            else:
                self.logger.info("[RiskScaler] Empty returns dictionary, using neutral scaling (1.0)")
                return 1.0
        else:
            returns_series = returns
            
        # Return 1.0 if we have no returns data
        if hasattr(returns_series, 'empty') and returns_series.empty:
            self.logger.info("[RiskScaler] No returns data, using neutral scaling (1.0)")
            return 1.0
        elif len(returns_series) == 0:
            self.logger.info("[RiskScaler] No returns data, using neutral scaling (1.0)")
            return 1.0
            
        # Select the appropriate method
        if self.scaling_method == 'sharpe':
            risk_scaling = self._calculate_sharpe_scaling(returns_series)
        elif self.scaling_method == 'volatility':
            risk_scaling = self._calculate_volatility_scaling(returns_series)
        elif self.scaling_method == 'adaptive':
            risk_scaling = self._calculate_adaptive_scaling(returns_series)
        elif self.scaling_method == 'combined':
            risk_scaling = self._calculate_combined_scaling(returns_series)
        else:
            # Default to neutral scaling
            risk_scaling = 1.0
            self.logger.warning(f"[RiskScaler] Unknown scaling method '{self.scaling_method}', using neutral scaling (1.0)")
            
        # Store risk scaling history
        self.risk_scaling_history.append({
            'date': pd.Timestamp.now(),
            'scaling': risk_scaling,
            'method': self.scaling_method
        })
            
        return risk_scaling
        
    def _calculate_sharpe_scaling(self, returns: pd.Series) -> float:
        """
        Calculate risk scaling factor based on Sharpe ratio z-score.
        
        Args:
            returns: Series of returns
            
        Returns:
            float: Risk scaling factor
        """
        # Handle different window sizes based on available data
        if len(returns) < self.rolling_window:
            # If we have fewer observations than window, use expanding window
            self.logger.debug(
                f"[Risk Scaling] Using expanding window (n={len(returns)}) instead of rolling window ({self.rolling_window})")

            # Calculate annualized mean and standard deviation
            mean_val = returns.mean() * 252  # Annualize
            std_val = returns.std() * np.sqrt(252)  # Annualize

            # Calculate current Sharpe ratio
            current_sharpe = mean_val / std_val if std_val != 0 else np.nan

            # Calculate historical Sharpe series using expanding window
            sharpe_series = returns.expanding(min_periods=1).apply(
                lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() != 0 else np.nan
            )
        else:
            # Use rolling window as configured
            self.logger.debug(f"[Risk Scaling] Using {self.rolling_window}-day rolling window")

            # Calculate rolling statistics
            rolling_mean = returns.rolling(self.rolling_window).mean() * 252
            rolling_std = returns.rolling(self.rolling_window).std() * np.sqrt(252)

            # Calculate Sharpe ratio series
            sharpe_series = rolling_mean / rolling_std.replace(0, np.nan)

            # Get current Sharpe ratio (most recent value)
            current_sharpe = sharpe_series.iloc[-1] if not sharpe_series.empty else np.nan

        # Calculate historical statistics for z-score
        hist_mean = sharpe_series.mean()
        hist_std = sharpe_series.std()
        nan_count = sharpe_series.isna().sum()

        # Calculate z-score and scaling factor
        if pd.isna(hist_std) or hist_std == 0:
            # If we can't calculate a proper z-score, use neutral scaling
            z_score = 0
            scaling = 1.0
            self.logger.info("[Risk Scaling] Cannot calculate valid z-score, using neutral scaling (1.0)")
        else:
            # Calculate z-score
            z_score = (current_sharpe - hist_mean) / hist_std

            # Apply scaling logic based on z-score
            if z_score >= self.target_z:
                # Above target z-score - full exposure
                scaling = 1.0
            elif z_score <= self.min_z:
                # Below minimum z-score - minimum exposure
                scaling = self.min_investment
            else:
                # Between min and target z-scores - linear scaling
                scaling = self.min_investment + (z_score - self.min_z) / (self.target_z - self.min_z) * (
                            1 - self.min_investment)

        # Format values for logging
        curr_sharpe_str = f"{current_sharpe:.2f}" if not pd.isna(current_sharpe) else "N/A"
        hist_mean_str = f"{hist_mean:.2f}" if not pd.isna(hist_mean) else "N/A"
        hist_std_str = f"{hist_std:.2f}" if not pd.isna(hist_std) else "N/A"
        z_str = f"{z_score:.2f}" if not pd.isna(z_score) else "N/A"

        # Log risk scaling calculation
        self.logger.info(f"[Risk Scaling] Current Sharpe: {curr_sharpe_str}, "
                         f"Hist Mean: {hist_mean_str}, "
                         f"Std: {hist_std_str}, "
                         f"NaN Count: {nan_count}, "
                         f"z: {z_str} => Scaling: {scaling:.2f}")

        # Show interpretation of the scaling factor
        if scaling < 0.5:
            self.logger.info(f"  [Risk Scaling] Low scaling factor ({scaling:.2f}) - reducing position sizing")
        elif scaling >= 0.9:
            self.logger.info(f"  [Risk Scaling] High scaling factor ({scaling:.2f}) - normal position sizing")
        else:
            self.logger.info(f"  [Risk Scaling] Moderate scaling factor ({scaling:.2f}) - cautious position sizing")

        return scaling
        
    def _calculate_volatility_scaling(self, returns: pd.Series) -> float:
        """
        Calculate risk scaling factor based on realized volatility.
        
        Args:
            returns: Series of returns
            
        Returns:
            float: Risk scaling factor
        """
        # Default target volatility
        target_vol = self.volatility_params.get('target_volatility', 0.15)  # 15% annualized vol
        
        # Calculate realized volatility
        if len(returns) < self.rolling_window:
            # Use expanding window for small samples
            realized_vol = returns.std() * np.sqrt(252)  # Annualize
        else:
            # Use rolling window
            realized_vol = returns.rolling(self.rolling_window).std().iloc[-1] * np.sqrt(252)
            
        # Calculate volatility ratio
        vol_ratio = target_vol / realized_vol if realized_vol > 0 else 1.0
        
        # Apply constraints
        min_ratio = self.volatility_params.get('min_variance_ratio', 0.5)
        max_ratio = self.volatility_params.get('max_variance_ratio', 2.0)
        
        # Constrain scaling factor
        scaling = np.clip(vol_ratio, min_ratio, max_ratio)
        
        # Log details
        self.logger.info(f"[Risk Scaling] Target Vol: {target_vol:.2%}, Realized Vol: {realized_vol:.2%}")
        self.logger.info(f"[Risk Scaling] Vol Ratio: {vol_ratio:.2f}, Scaling: {scaling:.2f}")
        
        return scaling
        
    def _calculate_adaptive_scaling(self, returns: pd.Series) -> float:
        """
        Calculate risk scaling factor based on recent performance and heat level.
        
        Args:
            returns: Series of returns
            
        Returns:
            float: Risk scaling factor
        """
        # Get parameters
        max_heat = self.adaptive_params.get('max_heat', 1.0)
        cooldown_rate = self.adaptive_params.get('cooldown_rate', 0.05)
        heatup_rate = self.adaptive_params.get('heatup_rate', 0.02)
        
        # Get current heat level or initialize
        if not hasattr(self, 'heat_level'):
            self.heat_level = 0.0
            self.last_peak = 1.0
            self.last_trough = 1.0
        
        # Get most recent return
        most_recent_return = returns.iloc[-1] if not returns.empty else 0.0
        
        # Update heat level based on recent return
        if most_recent_return > 0:
            # Positive return, reduce heat
            self.heat_level = max(0.0, self.heat_level - cooldown_rate)
        else:
            # Negative return, increase heat
            self.heat_level = min(max_heat, self.heat_level + heatup_rate)
            
        # Calculate drawdown if we have history
        if len(returns) > 1:
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Update peak and trough
            current_value = cum_returns.iloc[-1]
            self.last_peak = max(self.last_peak, current_value)
            self.last_trough = min(self.last_trough, current_value)
            
            # Calculate drawdown
            current_drawdown = 1 - (current_value / self.last_peak)
            
            # Calculate recovery ratio
            recovery_ratio = 1 - (current_drawdown / 0.1)  # 10% max drawdown
        else:
            recovery_ratio = 1.0
        
        # Clip recovery ratio
        recovery_ratio = np.clip(recovery_ratio, 0.1, 1.0)
        
        # Calculate heat scaling (inverse of heat level)
        heat_scaling = 1.0 - self.heat_level
        
        # Apply recovery ratio with a factor
        recovery_factor = self.adaptive_params.get('recovery_factor', 2.0)
        recovery_scaling = recovery_ratio * recovery_factor
        
        # Combine scalings with weighting
        scaling = 0.7 * heat_scaling + 0.3 * recovery_scaling
        
        # Clip to reasonable range
        scaling = np.clip(scaling, 0.1, 1.0)
        
        # Log details
        self.logger.info(f"[Risk Scaling] Heat Level: {self.heat_level:.2f}, Heat Scaling: {heat_scaling:.2f}")
        self.logger.info(f"[Risk Scaling] Recovery Ratio: {recovery_ratio:.2f}, Recovery Scaling: {recovery_scaling:.2f}")
        self.logger.info(f"[Risk Scaling] Final Scaling: {scaling:.2f}")
        
        return scaling
        
    def _calculate_combined_scaling(self, returns: pd.Series) -> float:
        """
        Calculate risk scaling factor using a weighted combination of methods.
        
        Args:
            returns: Series of returns
            
        Returns:
            float: Risk scaling factor
        """
        # Get weights (default to equal weights)
        weights = self.combined_params.get('weights', [0.33, 0.33, 0.34])
        
        # Make sure we have 3 weights that sum to 1.0
        if len(weights) != 3 or abs(sum(weights) - 1.0) > 0.01:
            self.logger.warning("[RiskScaler] Invalid weights for combined scaling, using equal weights")
            weights = [0.33, 0.33, 0.34]
            
        vol_weight, sharpe_weight, adaptive_weight = weights
        
        # Calculate individual scalings
        vol_scaling = self._calculate_volatility_scaling(returns)
        sharpe_scaling = self._calculate_sharpe_scaling(returns)
        adaptive_scaling = self._calculate_adaptive_scaling(returns)
        
        # Calculate weighted average
        scaling = (vol_weight * vol_scaling +
                   sharpe_weight * sharpe_scaling +
                   adaptive_weight * adaptive_scaling)
                   
        # Clip to reasonable range
        scaling = np.clip(scaling, 0.1, 2.0)
        
        # Log combined results
        self.logger.info(f"[Risk Scaling] Combined Scaling Breakdown:")
        self.logger.info(f"  Volatility: {vol_scaling:.2f} (weight={vol_weight:.2f})")
        self.logger.info(f"  Sharpe: {sharpe_scaling:.2f} (weight={sharpe_weight:.2f})")
        self.logger.info(f"  Adaptive: {adaptive_scaling:.2f} (weight={adaptive_weight:.2f})")
        self.logger.info(f"  Combined: {scaling:.2f}")
        
        return scaling 