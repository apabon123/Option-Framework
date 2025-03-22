"""
Risk manager implementations.

This module provides implementations of risk managers for dynamic position sizing
and risk management in trading strategies.
"""

import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from core.risk.metrics import RiskMetrics
from core.risk.parameters import (
    RiskParameters, RiskLimits, SharpeRiskParameters,
    VolatilityRiskParameters, AdaptiveRiskParameters, CombinedRiskParameters
)


class RiskManager(ABC):
    """Abstract base class for risk management."""
    
    def __init__(self, config: Dict[str, Any], risk_metrics: RiskMetrics,
                logger: Optional[logging.Logger] = None):
        """
        Initialize the risk manager.
        
        Args:
            config: Configuration dictionary
            risk_metrics: Risk metrics calculator
            logger: Logger instance
        """
        self.config = config
        self.risk_metrics = risk_metrics
        self.logger = logger or logging.getLogger('risk_manager')
        
        # Get risk parameters from config
        risk_config = config.get('risk', {})
        risk_limits_config = risk_config.get('risk_limits', {})
        
        # Initialize basic parameters and limits
        self.risk_limits = RiskLimits(**risk_limits_config)
        
        # Initialize risk tracking
        self.equity_history = []
        self.timestamp_history = []
        self.position_exposure_history = []
        self.risk_scaling_history = []
        
        # Set contract specification if available
        self.contract_spec = self.config.get('contract_spec', {})
        
        self.logger.info("Risk manager initialized")
        self.logger.info(f"Risk limits: max_position={self.risk_limits.max_position_size}, "
                     f"max_daily_loss={self.risk_limits.max_daily_loss:.2%}")
    
    def calculate_position_size(self, data: Dict[str, Any], portfolio_metrics: Dict[str, Any]) -> float:
        """
        Calculate position size based on risk parameters and market data.
        
        This implementation provides a base position size without risk scaling.
        Specific risk managers should override this to apply their risk models.
        
        Args:
            data: Market data and strategy parameters
            portfolio_metrics: Portfolio metrics
            
        Returns:
            float: Position size in number of contracts/shares
        """
        # Extract key metrics
        price = data.get('price', 0)
        signal_strength = data.get('strength', 1.0)
        
        # Get current equity
        current_equity = portfolio_metrics.get('equity', 100000)
        
        # Calculate fixed base position size - no risk limits applied
        margin_per_contract = self.contract_spec.get('margin', price * 0.5)
        maximum_contracts = current_equity / margin_per_contract
        
        # Cap at maximum position size from config
        max_contracts = min(
            maximum_contracts * 0.95,  # 95% of equity-based limit
            self.risk_limits.max_position_size
        )
        
        # Calculate base position size using signal strength
        base_position_size = max_contracts * signal_strength
        
        # Apply no further risk scaling in the base implementation
        final_position_size = base_position_size
        
        # Log details
        self.logger.debug(f"Base position calculation:")
        self.logger.debug(f"  Price: ${price:.2f}")
        self.logger.debug(f"  Margin per contract: ${margin_per_contract:.2f}")
        self.logger.debug(f"  Equity: ${current_equity:.2f}")
        self.logger.debug(f"  Max contracts: {max_contracts:.2f}")
        self.logger.debug(f"  Signal strength: {signal_strength:.2f}")
        self.logger.debug(f"  Final position size: {final_position_size:.2f}")
        
        return final_position_size
    
    @abstractmethod
    def calculate_risk_scaling(self, returns: pd.Series) -> Tuple[float, float, float]:
        """
        Calculate risk scaling factor based on performance metrics.
        
        Args:
            returns: Series of returns with DatetimeIndex
            
        Returns:
            tuple: (risk_scaling, metric_value, z_score)
        """
        pass
    
    def initialize_risk_tracking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initialize risk tracking columns in DataFrame."""
        # Copy dataframe
        df = df.copy()
        
        # Initialize risk columns
        risk_columns = {
            'position': 0.0,  # Base position size
            'risk_scaling': 1.0,  # Risk-based position scaling
            'risk_adjusted_size': 0.0,  # Risk-adjusted position size
        }
        
        # Add columns if they don't exist
        for col, default in risk_columns.items():
            if col not in df.columns:
                df[col] = default
        
        return df
    
    def apply_risk_scaling(self, df: pd.DataFrame, returns: pd.Series,
                          risk_metrics: pd.DataFrame) -> pd.Series:
        """
        Calculate position size multipliers based on risk metrics.
        
        Args:
            df: DataFrame with position data
            returns: Series of returns
            risk_metrics: DataFrame with risk metrics
            
        Returns:
            pd.Series: Series of risk-adjusted position sizes
        """
        # Get risk scaling factors
        risk_scaling, _, _ = self.calculate_risk_scaling(returns)
        
        # Apply risk scaling to position sizes
        return df['position'] * risk_scaling
    
    def check_risk_limits(self, df: pd.DataFrame) -> List[str]:
        """Check risk limits based on base positions only."""
        violations = []
        
        # Check position size limit (percentage of equity)
        position_exposure = df['position'] * self.contract_spec.get('margin', 1.0) / df.get('equity', 100000)
        
        # Check if any positions exceed position size limit
        if (position_exposure > self.risk_limits.position_limit_pct).any():
            violations.append(
                f"Position size exceeds {self.risk_limits.position_limit_pct:.1%} "
                f"of equity limit"
            )
        
        # Check concentration limit
        if (position_exposure > self.risk_limits.concentration_limit).any():
            violations.append(
                f"Position concentration exceeds {self.risk_limits.concentration_limit:.1%} limit"
            )
        
        # Check drawdown limit
        max_daily_loss_pct = abs(df['returns'][df['returns'] < 0].min()) if 'returns' in df else 0
        if max_daily_loss_pct > self.risk_limits.max_daily_loss:
            violations.append(
                f"Daily loss of {max_daily_loss_pct:.1%} exceeds "
                f"{self.risk_limits.max_daily_loss:.1%} limit"
            )
        
        return violations
    
    def update_risk_metrics(self, equity: float, timestamp: pd.Timestamp) -> None:
        """Update risk tracking metrics."""
        self.equity_history.append(equity)
        self.timestamp_history.append(timestamp)
    
    def get_risk_summary(self) -> dict:
        """Get current risk metrics summary."""
        # Return empty dictionary if no data
        if not self.equity_history:
            return {}
        
        # Calculate current metrics
        current_equity = self.equity_history[-1]
        
        # Calculate maximum allowed position size
        margin_max = (current_equity * self.risk_limits.position_limit_pct) / self.contract_spec.get('margin', 1.0)
        
        # Calculate position concentration limit
        concentration_max = (current_equity * self.risk_limits.concentration_limit) / \
                            self.contract_spec.get('margin', 1.0)
        
        # Take the minimum of all limits
        max_position = min(margin_max, concentration_max, self.risk_limits.max_position_size)
        
        return {
            'equity': current_equity,
            'max_position': max_position,
            'position_limit_pct': self.risk_limits.position_limit_pct,
            'max_daily_loss': self.risk_limits.max_daily_loss
        }
    
    def calculate_exposure_metrics(self, position_size: float, margin_per_contract: float,
                                 equity: float) -> Dict[str, float]:
        """Calculate exposure and risk metrics for a position."""
        margin_requirement = position_size * margin_per_contract
        exposure_pct = margin_requirement / equity if equity > 0 else 0
        
        return {
            'position_size': position_size,
            'margin_requirement': margin_requirement,
            'exposure_pct': exposure_pct
        }
    
    def apply_position_limits(self, position_size: float, equity: float) -> float:
        """Apply position limits based on risk parameters."""
        # Calculate limits
        equity_based_limit = equity * self.risk_limits.position_limit_pct
        margin_per_contract = self.contract_spec.get('margin', 1.0)
        
        # Calculate maximum position size
        max_position = min(
            equity_based_limit / margin_per_contract,
            self.risk_limits.max_position_size,
            position_size
        )
        
        return max_position


class VolatilityTargetRiskManager(RiskManager):
    """
    Volatility-based risk manager that scales positions to target a specific volatility.
    
    This risk manager adjusts position sizes based on recent volatility to maintain
    a consistent risk profile over time.
    """
    
    def __init__(self, config: Dict[str, Any], risk_metrics: RiskMetrics,
                logger: Optional[logging.Logger] = None):
        """Initialize the volatility-based risk manager."""
        super().__init__(config, risk_metrics, logger)
        
        # Get volatility parameters
        risk_config = config.get('risk', {})
        vol_config = risk_config.get('volatility', {})
        
        # Create volatility parameters object
        self.vol_params = VolatilityRiskParameters(
            target_volatility=vol_config.get('target_volatility', 0.15),
            lookback_days=vol_config.get('lookback_days', 21),
            min_variance_ratio=vol_config.get('min_variance_ratio', 0.5),
            max_variance_ratio=vol_config.get('max_variance_ratio', 2.0),
            window_days=vol_config.get('window_days', 63),
            window_type=vol_config.get('window_type', 'medium')
        )
        
        self.logger.info("Volatility-based risk manager initialized")
        self.logger.info(f"Target volatility: {self.vol_params.target_volatility:.1%}")
        self.logger.info(f"Lookback days: {self.vol_params.lookback_days}")
        self.logger.info(f"Variance ratio limits: [{self.vol_params.min_variance_ratio}, "
                     f"{self.vol_params.max_variance_ratio}]")
    
    def calculate_risk_scaling(self, returns: pd.Series) -> Tuple[float, float, float]:
        """
        Calculate risk scaling factor based on recent volatility.
        
        Args:
            returns: Series of returns with DatetimeIndex
            
        Returns:
            tuple: (risk_scaling, recent_vol, z_score)
        """
        # Validate returns series
        if returns.empty:
            return 1.0, 0.0, 0.0
        
        # Get window parameters
        window_type = self.vol_params.window_type
        window = self.risk_metrics.get_window_size(window_type)
        min_periods = self.risk_metrics.get_min_periods(window_type)
        target_vol = self.vol_params.target_volatility
        
        # Check if we have enough data
        if len(returns) < min_periods:
            self.logger.warning(
                f"Insufficient data for volatility calculation: {len(returns)} < {min_periods}"
            )
            return 1.0, 0.0, 0.0
        
        # Calculate recent volatility (recent window)
        recent_window = min(self.vol_params.lookback_days, len(returns))
        recent_returns = returns.iloc[-recent_window:]
        recent_vol = recent_returns.std() * np.sqrt(252)  # Annualized
        
        # Calculate historical volatility (longer window)
        historical_vol = returns.std() * np.sqrt(252)  # Annualized
        
        # If historical volatility is zero, use recent volatility
        if historical_vol == 0:
            historical_vol = recent_vol
        
        # If both are zero, use default scaling of 1.0
        if recent_vol == 0 and historical_vol == 0:
            return 1.0, 0.0, 0.0
        
        # Calculate variance ratio
        variance_ratio = (recent_vol / historical_vol) ** 2 if historical_vol > 0 else 1.0
        
        # Calculate z-score of the recent volatility
        historical_vols = returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(252)
        z_score = self.risk_metrics.calculate_z_score(recent_vol, historical_vols)
        
        # Adjust for extreme values
        z_score = np.clip(z_score, -3, 3)
        
        # Calculate volatility-based scaling
        if recent_vol > 0:
            vol_scaling = target_vol / recent_vol
        else:
            vol_scaling = 1.0
        
        # Apply variance ratio limits
        min_ratio = self.vol_params.min_variance_ratio
        max_ratio = self.vol_params.max_variance_ratio
        variance_ratio = np.clip(variance_ratio, min_ratio, max_ratio)
        
        # Apply variance ratio adjustment
        if variance_ratio > 1.0:
            # Reduce position size for high variance ratio
            vol_scaling /= np.sqrt(variance_ratio)
        
        # Calculate final risk scaling
        risk_scaling = np.clip(vol_scaling, 0.1, 2.0)
        
        # Log details
        self.logger.debug(f"Volatility risk scaling calculation:")
        self.logger.debug(f"  Recent volatility: {recent_vol:.2%}")
        self.logger.debug(f"  Historical volatility: {historical_vol:.2%}")
        self.logger.debug(f"  Target volatility: {target_vol:.2%}")
        self.logger.debug(f"  Variance ratio: {variance_ratio:.2f}")
        self.logger.debug(f"  Z-score: {z_score:.2f}")
        self.logger.debug(f"  Risk scaling: {risk_scaling:.2f}")
        
        return risk_scaling, recent_vol, z_score
    
    def calculate_position_size(self, data: Dict[str, Any], portfolio_metrics: Dict[str, Any]) -> float:
        """
        Calculate position size with volatility-based risk scaling.
        
        Args:
            data: Market data and strategy parameters
            portfolio_metrics: Portfolio metrics
            
        Returns:
            float: Position size in number of contracts/shares
        """
        # Get base position size from parent class
        base_size = super().calculate_position_size(data, portfolio_metrics)
        
        # Get returns series from portfolio metrics
        returns = portfolio_metrics.get('returns_series')
        
        # Apply volatility-based risk scaling if returns are available
        if returns is not None and not returns.empty:
            risk_scaling, _, _ = self.calculate_risk_scaling(returns)
            scaled_size = base_size * risk_scaling
            
            self.logger.debug(f"Volatility-based position sizing:")
            self.logger.debug(f"  Base size: {base_size:.2f}")
            self.logger.debug(f"  Risk scaling: {risk_scaling:.2f}")
            self.logger.debug(f"  Scaled size: {scaled_size:.2f}")
            
            return scaled_size
        else:
            self.logger.debug(f"No returns data available for risk scaling, using base size")
            return base_size


class SharpeRatioRiskManager(RiskManager):
    """
    Sharpe ratio-based risk manager that scales positions based on risk-adjusted returns.
    
    This risk manager adjusts position sizes based on the Sharpe ratio to reward
    strategies with better risk-adjusted performance.
    """
    
    def __init__(self, config: Dict[str, Any], risk_metrics: RiskMetrics,
                logger: Optional[logging.Logger] = None):
        """Initialize the Sharpe ratio-based risk manager."""
        super().__init__(config, risk_metrics, logger)
        
        # Get Sharpe parameters
        risk_config = config.get('risk', {})
        sharpe_config = risk_config.get('sharpe', {})
        
        # Create Sharpe parameters object
        self.sharpe_params = SharpeRiskParameters(
            min_sharpe=sharpe_config.get('min_sharpe', 0.5),
            target_sharpe=sharpe_config.get('target_sharpe', 1.5),
            lookback_days=sharpe_config.get('lookback_days', 21),
            window_days=sharpe_config.get('window_days', 63),
            risk_free_rate=sharpe_config.get('risk_free_rate', 0.02),
            window_type=sharpe_config.get('window_type', 'medium')
        )
        
        self.logger.info("Sharpe ratio-based risk manager initialized")
        self.logger.info(f"Min Sharpe: {self.sharpe_params.min_sharpe:.2f}")
        self.logger.info(f"Target Sharpe: {self.sharpe_params.target_sharpe:.2f}")
        self.logger.info(f"Lookback days: {self.sharpe_params.lookback_days}")
        self.logger.info(f"Risk-free rate: {self.sharpe_params.risk_free_rate:.2%}")
    
    def calculate_risk_scaling(self, returns: pd.Series) -> Tuple[float, float, float]:
        """
        Calculate risk scaling factor based on Sharpe ratio.
        
        Args:
            returns: Series of returns with DatetimeIndex
            
        Returns:
            tuple: (risk_scaling, sharpe_ratio, z_score)
        """
        # Validate returns series
        if returns.empty:
            return 1.0, 0.0, 0.0
        
        # Get window parameters
        window_type = self.sharpe_params.window_type
        window = self.risk_metrics.get_window_size(window_type)
        min_periods = self.risk_metrics.get_min_periods(window_type)
        
        # Check if we have enough data
        if len(returns) < min_periods:
            self.logger.warning(
                f"Insufficient data for Sharpe calculation: {len(returns)} < {min_periods}"
            )
            return 1.0, 0.0, 0.0
        
        # Calculate recent Sharpe ratio
        recent_window = min(self.sharpe_params.lookback_days, len(returns))
        recent_returns = returns.iloc[-recent_window:]
        recent_metrics = self.risk_metrics.calculate_metrics(recent_returns, window_type='short')
        recent_sharpe = recent_metrics.get('sharpe_ratio', 0.0)
        
        # Calculate z-score of recent Sharpe ratio
        roll_returns = returns.rolling(window=window, min_periods=min_periods)
        roll_mean = roll_returns.mean() * 252  # Annualized return
        roll_std = roll_returns.std() * np.sqrt(252)  # Annualized volatility
        roll_sharpe = (roll_mean - self.sharpe_params.risk_free_rate) / roll_std
        roll_sharpe = roll_sharpe.dropna()
        
        # Calculate z-score
        z_score = self.risk_metrics.calculate_z_score(recent_sharpe, roll_sharpe)
        
        # Clip z-score to reasonable range
        z_score = np.clip(z_score, -3, 3)
        
        # Calculate Sharpe-based scaling
        min_sharpe = self.sharpe_params.min_sharpe
        target_sharpe = self.sharpe_params.target_sharpe
        
        # Linear scaling from min_sharpe to target_sharpe
        if recent_sharpe <= min_sharpe:
            sharpe_scaling = 0.1  # Minimum scaling for poor Sharpe
        elif recent_sharpe >= target_sharpe:
            sharpe_scaling = 1.0  # Full scaling for good Sharpe
        else:
            # Linear interpolation
            sharpe_scaling = 0.1 + 0.9 * (recent_sharpe - min_sharpe) / (target_sharpe - min_sharpe)
        
        # Calculate final risk scaling
        risk_scaling = np.clip(sharpe_scaling, 0.1, 2.0)
        
        # Log details
        self.logger.debug(f"Sharpe risk scaling calculation:")
        self.logger.debug(f"  Recent Sharpe: {recent_sharpe:.2f}")
        self.logger.debug(f"  Min Sharpe: {min_sharpe:.2f}")
        self.logger.debug(f"  Target Sharpe: {target_sharpe:.2f}")
        self.logger.debug(f"  Z-score: {z_score:.2f}")
        self.logger.debug(f"  Risk scaling: {risk_scaling:.2f}")
        
        return risk_scaling, recent_sharpe, z_score
    
    def calculate_position_size(self, data: Dict[str, Any], portfolio_metrics: Dict[str, Any]) -> float:
        """
        Calculate position size with Sharpe-based risk scaling.
        
        Args:
            data: Market data and strategy parameters
            portfolio_metrics: Portfolio metrics
            
        Returns:
            float: Position size in number of contracts/shares
        """
        # Get base position size from parent class
        base_size = super().calculate_position_size(data, portfolio_metrics)
        
        # Get returns series from portfolio metrics
        returns = portfolio_metrics.get('returns_series')
        
        # Apply Sharpe-based risk scaling if returns are available
        if returns is not None and not returns.empty:
            risk_scaling, sharpe_ratio, _ = self.calculate_risk_scaling(returns)
            scaled_size = base_size * risk_scaling
            
            self.logger.debug(f"Sharpe-based position sizing:")
            self.logger.debug(f"  Base size: {base_size:.2f}")
            self.logger.debug(f"  Sharpe ratio: {sharpe_ratio:.2f}")
            self.logger.debug(f"  Risk scaling: {risk_scaling:.2f}")
            self.logger.debug(f"  Scaled size: {scaled_size:.2f}")
            
            return scaled_size
        else:
            self.logger.debug(f"No returns data available for risk scaling, using base size")
            return base_size


class AdaptiveRiskManager(RiskManager):
    """
    Adaptive risk manager that adjusts position sizes based on recent performance.
    
    This risk manager decreases position sizes after losses and gradually
    increases them after successful trades to manage drawdowns.
    """
    
    def __init__(self, config: Dict[str, Any], risk_metrics: RiskMetrics,
                logger: Optional[logging.Logger] = None):
        """Initialize the adaptive risk manager."""
        super().__init__(config, risk_metrics, logger)
        
        # Get adaptive parameters
        risk_config = config.get('risk', {})
        adaptive_config = risk_config.get('adaptive', {})
        
        # Create adaptive parameters object
        self.adaptive_params = AdaptiveRiskParameters(
            max_heat=adaptive_config.get('max_heat', 1.0),
            cooldown_rate=adaptive_config.get('cooldown_rate', 0.05),
            heatup_rate=adaptive_config.get('heatup_rate', 0.02),
            recovery_factor=adaptive_config.get('recovery_factor', 2.0),
            window_days=adaptive_config.get('window_days', 21),
            window_type=adaptive_config.get('window_type', 'short')
        )
        
        # Initialize heat level
        self.heat_level = 0.0
        self.last_return = 0.0
        
        self.logger.info("Adaptive risk manager initialized")
        self.logger.info(f"Max heat: {self.adaptive_params.max_heat:.2f}")
        self.logger.info(f"Cooldown rate: {self.adaptive_params.cooldown_rate:.2%}")
        self.logger.info(f"Heatup rate: {self.adaptive_params.heatup_rate:.2%}")
        self.logger.info(f"Recovery factor: {self.adaptive_params.recovery_factor:.2f}")
    
    def calculate_risk_scaling(self, returns: pd.Series) -> Tuple[float, float, float]:
        """
        Calculate risk scaling factor based on recent performance and heat level.
        
        Args:
            returns: Series of returns with DatetimeIndex
            
        Returns:
            tuple: (risk_scaling, heat_level, recovery_ratio)
        """
        # Validate returns series
        if returns.empty:
            return 1.0, self.heat_level, 0.0
        
        # Get window parameters
        window_type = self.adaptive_params.window_type
        window = self.risk_metrics.get_window_size(window_type)
        min_periods = self.risk_metrics.get_min_periods(window_type)
        
        # Get the most recent return
        most_recent_return = returns.iloc[-1] if not returns.empty else 0.0
        
        # Check if we have a new return to process
        if most_recent_return != self.last_return:
            # Update heat level based on the return
            if most_recent_return < 0:
                # Increase heat level for losses (faster heat-up for bigger losses)
                heat_increase = self.adaptive_params.cooldown_rate * abs(most_recent_return) * 20
                self.heat_level = min(self.heat_level + heat_increase, self.adaptive_params.max_heat)
                
                self.logger.debug(f"Heat level increased to {self.heat_level:.2f} "
                               f"after loss of {most_recent_return:.2%}")
            else:
                # Decrease heat level for gains (faster cool-down for bigger gains)
                heat_decrease = self.adaptive_params.heatup_rate * most_recent_return * 10
                self.heat_level = max(self.heat_level - heat_decrease, 0.0)
                
                self.logger.debug(f"Heat level decreased to {self.heat_level:.2f} "
                               f"after gain of {most_recent_return:.2%}")
            
            # Update last return
            self.last_return = most_recent_return
        
        # Calculate drawdown
        if len(returns) >= min_periods:
            metrics = self.risk_metrics.calculate_metrics(returns, window_type)
            current_drawdown = metrics.get('current_drawdown', 0.0)
            max_drawdown = metrics.get('max_drawdown', 0.0)
            
            # Calculate recovery ratio (how far we are from max drawdown)
            recovery_ratio = 1.0
            if max_drawdown > 0:
                # 1.0 = fully recovered, 0.0 = at max drawdown
                recovery_ratio = 1.0 - (current_drawdown / max_drawdown)
        else:
            recovery_ratio = 1.0
        
        # Calculate risk scaling based on heat level and recovery
        heat_scaling = 1.0 - (self.heat_level / self.adaptive_params.max_heat)
        recovery_scaling = recovery_ratio * self.adaptive_params.recovery_factor
        
        # Combine scalings with some weighting
        risk_scaling = 0.7 * heat_scaling + 0.3 * recovery_scaling
        
        # Clip to reasonable range
        risk_scaling = np.clip(risk_scaling, 0.1, 1.0)
        
        # Log details
        self.logger.debug(f"Adaptive risk scaling calculation:")
        self.logger.debug(f"  Heat level: {self.heat_level:.2f}")
        self.logger.debug(f"  Heat scaling: {heat_scaling:.2f}")
        self.logger.debug(f"  Recovery ratio: {recovery_ratio:.2f}")
        self.logger.debug(f"  Recovery scaling: {recovery_scaling:.2f}")
        self.logger.debug(f"  Risk scaling: {risk_scaling:.2f}")
        
        return risk_scaling, self.heat_level, recovery_ratio
    
    def calculate_position_size(self, data: Dict[str, Any], portfolio_metrics: Dict[str, Any]) -> float:
        """
        Calculate position size with adaptive risk scaling.
        
        Args:
            data: Market data and strategy parameters
            portfolio_metrics: Portfolio metrics
            
        Returns:
            float: Position size in number of contracts/shares
        """
        # Get base position size from parent class
        base_size = super().calculate_position_size(data, portfolio_metrics)
        
        # Get returns series from portfolio metrics
        returns = portfolio_metrics.get('returns_series')
        
        # Apply adaptive risk scaling if returns are available
        if returns is not None and not returns.empty:
            risk_scaling, heat_level, _ = self.calculate_risk_scaling(returns)
            scaled_size = base_size * risk_scaling
            
            self.logger.debug(f"Adaptive position sizing:")
            self.logger.debug(f"  Base size: {base_size:.2f}")
            self.logger.debug(f"  Heat level: {heat_level:.2f}")
            self.logger.debug(f"  Risk scaling: {risk_scaling:.2f}")
            self.logger.debug(f"  Scaled size: {scaled_size:.2f}")
            
            return scaled_size
        else:
            self.logger.debug(f"No returns data available for risk scaling, using base size")
            return base_size


class CombinedRiskManager(RiskManager):
    """
    Combined risk manager that uses multiple risk models with weighted averaging.
    
    This risk manager calculates position sizes using a weighted combination
    of volatility, Sharpe ratio, and adaptive approaches for more robust sizing.
    """
    
    def __init__(self, config: Dict[str, Any], risk_metrics: RiskMetrics,
                logger: Optional[logging.Logger] = None):
        """Initialize the combined risk manager."""
        super().__init__(config, risk_metrics, logger)
        
        # Get combined parameters
        risk_config = config.get('risk', {})
        combined_config = risk_config.get('combined', {})
        
        # Create sub-managers
        self.vol_manager = VolatilityTargetRiskManager(config, risk_metrics, logger)
        self.sharpe_manager = SharpeRatioRiskManager(config, risk_metrics, logger)
        self.adaptive_manager = AdaptiveRiskManager(config, risk_metrics, logger)
        
        # Create combined parameters object
        self.combined_params = CombinedRiskParameters(
            weights=combined_config.get('weights', [0.4, 0.3, 0.3]),
            min_position_size=combined_config.get('min_position_size', 0.0),
            max_position_size=combined_config.get('max_position_size', 1.0),
            vol_params=self.vol_manager.vol_params,
            sharpe_params=self.sharpe_manager.sharpe_params,
            adaptive_params=self.adaptive_manager.adaptive_params
        )
        
        self.logger.info("Combined risk manager initialized")
        self.logger.info(f"Weights: vol={self.combined_params.weights[0]:.2f}, "
                     f"sharpe={self.combined_params.weights[1]:.2f}, "
                     f"adaptive={self.combined_params.weights[2]:.2f}")
    
    def calculate_risk_scaling(self, returns: pd.Series) -> Tuple[float, float, float]:
        """
        Calculate risk scaling factor using a weighted combination of methods.
        
        Args:
            returns: Series of returns with DatetimeIndex
            
        Returns:
            tuple: (risk_scaling, weighted_score, None)
        """
        # Get weights
        vol_weight, sharpe_weight, adaptive_weight = self.combined_params.weights
        
        # Get scalings from each manager
        vol_scaling, vol_metric, _ = self.vol_manager.calculate_risk_scaling(returns)
        sharpe_scaling, sharpe_metric, _ = self.sharpe_manager.calculate_risk_scaling(returns)
        adaptive_scaling, adaptive_metric, _ = self.adaptive_manager.calculate_risk_scaling(returns)
        
        # Calculate weighted average
        weighted_scaling = (
            vol_weight * vol_scaling +
            sharpe_weight * sharpe_scaling +
            adaptive_weight * adaptive_scaling
        )
        
        # Calculate weighted score for logging
        weighted_score = (
            vol_weight * vol_metric +
            sharpe_weight * sharpe_metric +
            adaptive_weight * adaptive_metric
        )
        
        # Clip to reasonable range
        risk_scaling = np.clip(weighted_scaling, 0.1, 2.0)
        
        # Log details
        self.logger.debug(f"Combined risk scaling calculation:")
        self.logger.debug(f"  Volatility scaling: {vol_scaling:.2f} (weight={vol_weight:.2f})")
        self.logger.debug(f"  Sharpe scaling: {sharpe_scaling:.2f} (weight={sharpe_weight:.2f})")
        self.logger.debug(f"  Adaptive scaling: {adaptive_scaling:.2f} (weight={adaptive_weight:.2f})")
        self.logger.debug(f"  Weighted scaling: {weighted_scaling:.2f}")
        self.logger.debug(f"  Final risk scaling: {risk_scaling:.2f}")
        
        return risk_scaling, weighted_score, None
    
    def calculate_position_size(self, data: Dict[str, Any], portfolio_metrics: Dict[str, Any]) -> float:
        """
        Calculate position size with combined risk scaling.
        
        Args:
            data: Market data and strategy parameters
            portfolio_metrics: Portfolio metrics
            
        Returns:
            float: Position size in number of contracts/shares
        """
        # Get base position size from parent class
        base_size = super().calculate_position_size(data, portfolio_metrics)
        
        # Get returns series from portfolio metrics
        returns = portfolio_metrics.get('returns_series')
        
        # Apply combined risk scaling if returns are available
        if returns is not None and not returns.empty:
            # Calculate individual position sizes first
            vol_size = self.vol_manager.calculate_position_size(data, portfolio_metrics)
            sharpe_size = self.sharpe_manager.calculate_position_size(data, portfolio_metrics)
            adaptive_size = self.adaptive_manager.calculate_position_size(data, portfolio_metrics)
            
            # Get weights
            vol_weight, sharpe_weight, adaptive_weight = self.combined_params.weights
            
            # Calculate weighted average position size
            weighted_size = (
                vol_weight * vol_size +
                sharpe_weight * sharpe_size +
                adaptive_weight * adaptive_size
            )
            
            # Apply overall scaling limits
            min_size = self.combined_params.min_position_size * base_size
            max_size = self.combined_params.max_position_size * base_size
            final_size = np.clip(weighted_size, min_size, max_size)
            
            self.logger.debug(f"Combined position sizing:")
            self.logger.debug(f"  Base size: {base_size:.2f}")
            self.logger.debug(f"  Volatility size: {vol_size:.2f} (weight={vol_weight:.2f})")
            self.logger.debug(f"  Sharpe size: {sharpe_size:.2f} (weight={sharpe_weight:.2f})")
            self.logger.debug(f"  Adaptive size: {adaptive_size:.2f} (weight={adaptive_weight:.2f})")
            self.logger.debug(f"  Weighted size: {weighted_size:.2f}")
            self.logger.debug(f"  Final size: {final_size:.2f}")
            
            return final_size
        else:
            self.logger.debug(f"No returns data available for risk scaling, using base size")
            return base_size 