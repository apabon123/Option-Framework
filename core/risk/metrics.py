"""
Risk metrics calculation module.

This module provides the RiskMetrics class for calculating various
risk and performance metrics for trading strategies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime


class RiskMetrics:
    """Calculates and manages risk metrics for trading strategies."""
    
    def __init__(self, config: Dict[str, Any] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the risk metrics calculator.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger('risk_metrics')
        
        # Default configuration
        self.config = config or {}
        metrics_config = self.config.get('risk_metrics', {})
        
        # Configure windows for different timeframes
        self.short_window = metrics_config.get('short_window', 21)  # ~1 month
        self.medium_window = metrics_config.get('medium_window', 63)  # ~3 months
        self.long_window = metrics_config.get('long_window', 252)  # ~1 year
        
        # Set default risk-free rate for Sharpe calculations
        self.risk_free_rate = metrics_config.get('risk_free_rate', 0.02)  # 2% annual
        
        # Configure required data points for statistical validity
        self.min_periods = {
            'short': max(5, min(10, self.short_window // 2)),
            'medium': max(10, min(30, self.medium_window // 3)),
            'long': max(20, min(60, self.long_window // 4)),
        }
        
        # Enable or disable specific metrics
        self.enable_drawdown = metrics_config.get('enable_drawdown', True)
        self.enable_var = metrics_config.get('enable_var', True)
        self.enable_autocorr = metrics_config.get('enable_autocorr', True)
        
        # Configure calculation parameters
        self.var_quantile = metrics_config.get('var_quantile', 0.05)  # 5% VaR
        self.annualization_factor = metrics_config.get('annualization_factor', 252)  # Trading days in a year
        
        # Validate configuration
        self._validate_config()
        
        self.logger.debug("RiskMetrics initialized with configuration:")
        self.logger.debug(f"  Short window: {self.short_window} days")
        self.logger.debug(f"  Medium window: {self.medium_window} days")
        self.logger.debug(f"  Long window: {self.long_window} days")
        self.logger.debug(f"  Risk-free rate: {self.risk_free_rate:.2%}")
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        # Validate window sizes
        if not 5 <= self.short_window <= 252:
            self.logger.warning(f"Short window outside recommended range: {self.short_window}")
        if not 20 <= self.medium_window <= 756:
            self.logger.warning(f"Medium window outside recommended range: {self.medium_window}")
        if not 60 <= self.long_window <= 1260:
            self.logger.warning(f"Long window outside recommended range: {self.long_window}")
        
        # Validate relative window sizes
        if not self.short_window < self.medium_window < self.long_window:
            self.logger.warning("Window sizes should be increasing: short < medium < long")
        
        # Validate risk-free rate
        if not 0 <= self.risk_free_rate <= 1:
            raise ValueError(f"Risk-free rate must be between 0 and 1: {self.risk_free_rate}")
    
    def get_window_size(self, window_type: str) -> int:
        """
        Get the window size based on the window type.
        
        Args:
            window_type: Window type ('short', 'medium', or 'long')
        
        Returns:
            Window size in days
        """
        if window_type == 'short':
            return self.short_window
        elif window_type == 'medium':
            return self.medium_window
        elif window_type == 'long':
            return self.long_window
        else:
            self.logger.warning(f"Unknown window type: {window_type}, using medium")
            return self.medium_window
    
    def get_min_periods(self, window_type: str) -> int:
        """
        Get the minimum number of periods for calculations.
        
        Args:
            window_type: Window type ('short', 'medium', or 'long')
        
        Returns:
            Minimum number of periods
        """
        return self.min_periods.get(window_type, self.min_periods['medium'])
    
    def calculate_metrics(self, returns: pd.Series, window_type: str = 'medium') -> Dict[str, float]:
        """
        Calculate risk metrics for a series of returns.
        
        Args:
            returns: Series of returns with DatetimeIndex
            window_type: Window type ('short', 'medium', or 'long')
        
        Returns:
            Dictionary of risk metrics
        """
        # Validate returns data
        self._validate_returns(returns, window_type)
        
        # Get window parameters
        window = self.get_window_size(window_type)
        min_periods = self.get_min_periods(window_type)
        
        # Initialize metrics
        metrics = {}
        
        # Calculate basic return statistics
        metrics['return'] = self._calculate_return(returns)
        metrics['annualized_return'] = self._annualize_return(returns)
        
        # Calculate rolling metrics
        roll_returns = returns.rolling(window=window, min_periods=min_periods)
        metrics['volatility'] = self._calculate_volatility(returns, roll_returns)
        metrics['annualized_volatility'] = metrics['volatility'] * np.sqrt(self.annualization_factor)
        
        # Calculate Sharpe ratio
        excess_returns = roll_returns.mean() * self.annualization_factor - self.risk_free_rate
        sharpe = excess_returns / (roll_returns.std() * np.sqrt(self.annualization_factor))
        metrics['sharpe_ratio'] = sharpe.mean()
        
        # Calculate Sortino ratio if we have enough data
        if len(returns) >= min_periods:
            downside_returns = returns[returns < 0]
            if not downside_returns.empty:
                downside_deviation = downside_returns.std() * np.sqrt(self.annualization_factor)
                if downside_deviation > 0:
                    excess_return = metrics['annualized_return'] - self.risk_free_rate
                    metrics['sortino_ratio'] = excess_return / downside_deviation
                else:
                    metrics['sortino_ratio'] = np.nan
            else:
                metrics['sortino_ratio'] = np.inf  # No negative returns
        else:
            metrics['sortino_ratio'] = np.nan
        
        # Calculate drawdown if enabled
        if self.enable_drawdown:
            drawdown_stats = self._calculate_drawdown(returns)
            metrics.update(drawdown_stats)
        
        # Calculate Value at Risk if enabled
        if self.enable_var and len(returns) >= min_periods:
            metrics['var'] = self._calculate_var(returns)
            metrics['cvar'] = self._calculate_cvar(returns)
        
        # Calculate autocorrelation if enabled
        if self.enable_autocorr and len(returns) >= min_periods:
            metrics['autocorrelation'] = self._calculate_autocorrelation(returns)
        
        # Calculate additional metrics
        metrics['win_rate'] = self._calculate_win_rate(returns)
        metrics['profit_factor'] = self._calculate_profit_factor(returns)
        
        # Calculate Calmar ratio if we have max drawdown
        if 'max_drawdown' in metrics and metrics['max_drawdown'] > 0:
            metrics['calmar_ratio'] = metrics['annualized_return'] / metrics['max_drawdown']
        else:
            metrics['calmar_ratio'] = np.nan
        
        return metrics
    
    def _validate_returns(self, returns: pd.Series, window_type: str) -> None:
        """
        Validate returns data for risk metric calculations.
        
        Args:
            returns: Series of returns
            window_type: Window type for validation
        """
        # Ensure returns is a Series
        if not isinstance(returns, pd.Series):
            self.logger.error("Returns must be a pandas Series")
            raise TypeError("Returns must be a pandas Series")
        
        # Check for empty data
        if returns.empty:
            self.logger.error("Returns series is empty")
            raise ValueError("Returns series is empty")
        
        # Check for DatetimeIndex
        if not isinstance(returns.index, pd.DatetimeIndex):
            self.logger.warning("Returns index is not DatetimeIndex, calculations may be affected")
        
        # Check for missing values
        missing = returns.isna().sum()
        if missing > 0:
            self.logger.warning(f"Returns contain {missing} missing values")
        
        # Check for sufficient data points
        min_periods = self.get_min_periods(window_type)
        if len(returns) < min_periods:
            self.logger.warning(
                f"Returns series has only {len(returns)} points, "
                f"which is less than the minimum recommended {min_periods}"
            )
    
    def _calculate_return(self, returns: pd.Series) -> float:
        """
        Calculate the total return from a series of returns.
        
        Args:
            returns: Series of returns
        
        Returns:
            Total return as a decimal
        """
        return (1 + returns).prod() - 1
    
    def _annualize_return(self, returns: pd.Series) -> float:
        """
        Calculate the annualized return from a series of returns.
        
        Args:
            returns: Series of returns
        
        Returns:
            Annualized return as a decimal
        """
        if len(returns) < 2:
            return returns.mean() * self.annualization_factor
        
        total_return = self._calculate_return(returns)
        days = (returns.index[-1] - returns.index[0]).days
        if days <= 0:
            return total_return * self.annualization_factor
        
        years = days / 365.25
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_volatility(self, returns: pd.Series, roll_returns: pd.core.window.rolling.Rolling) -> float:
        """
        Calculate the volatility of returns.
        
        Args:
            returns: Series of returns
            roll_returns: Rolling window of returns
        
        Returns:
            Volatility as a decimal
        """
        return roll_returns.std().mean()
    
    def _calculate_drawdown(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate drawdown statistics.
        
        Args:
            returns: Series of returns
        
        Returns:
            Dictionary of drawdown metrics
        """
        # Initialize metrics
        metrics = {}
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.cummax()
        
        # Calculate drawdown
        drawdown = (cum_returns / running_max) - 1
        
        # Calculate max drawdown
        metrics['max_drawdown'] = abs(drawdown.min()) if not drawdown.empty else 0
        
        # Calculate current drawdown
        metrics['current_drawdown'] = abs(drawdown.iloc[-1]) if not drawdown.empty else 0
        
        # Calculate average drawdown
        metrics['avg_drawdown'] = abs(drawdown[drawdown < 0].mean()) if not drawdown[drawdown < 0].empty else 0
        
        # Calculate drawdown duration statistics
        if not drawdown.empty:
            # Identify start of drawdowns (when drawdown goes from 0 to negative)
            start_dd = (drawdown < 0) & (drawdown.shift(1) >= 0)
            
            # Identify end of drawdowns (when drawdown goes from negative to 0)
            end_dd = (drawdown >= 0) & (drawdown.shift(1) < 0)
            
            # Add the current drawdown end if we're in a drawdown
            if drawdown.iloc[-1] < 0:
                end_dd.iloc[-1] = True
            
            # Identify drawdown periods
            start_dates = returns.index[start_dd]
            end_dates = returns.index[end_dd]
            
            # Calculate durations (in trading days)
            if len(start_dates) > 0 and len(end_dates) > 0:
                # Handle case where the last drawdown is ongoing
                if len(start_dates) > len(end_dates):
                    end_dates = end_dates.append(pd.DatetimeIndex([returns.index[-1]]))
                
                # Calculate durations
                durations = [(end - start).days for start, end in zip(start_dates, end_dates)]
                
                if durations:
                    metrics['max_drawdown_duration'] = max(durations)
                    metrics['avg_drawdown_duration'] = sum(durations) / len(durations)
                else:
                    metrics['max_drawdown_duration'] = 0
                    metrics['avg_drawdown_duration'] = 0
            else:
                metrics['max_drawdown_duration'] = 0
                metrics['avg_drawdown_duration'] = 0
        else:
            metrics['max_drawdown_duration'] = 0
            metrics['avg_drawdown_duration'] = 0
        
        return metrics
    
    def _calculate_var(self, returns: pd.Series) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
        
        Returns:
            VaR as a decimal
        """
        return abs(returns.quantile(self.var_quantile))
    
    def _calculate_cvar(self, returns: pd.Series) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            returns: Series of returns
        
        Returns:
            CVaR as a decimal
        """
        var = self._calculate_var(returns)
        return abs(returns[returns <= -var].mean()) if not returns[returns <= -var].empty else var
    
    def _calculate_autocorrelation(self, returns: pd.Series, lag: int = 1) -> float:
        """
        Calculate autocorrelation of returns.
        
        Args:
            returns: Series of returns
            lag: Lag for autocorrelation
        
        Returns:
            Autocorrelation coefficient
        """
        if len(returns) <= lag:
            return np.nan
        
        # Calculate autocorrelation
        return returns.autocorr(lag)
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """
        Calculate win rate (percentage of positive returns).
        
        Args:
            returns: Series of returns
        
        Returns:
            Win rate as a decimal
        """
        if returns.empty:
            return np.nan
        
        return (returns > 0).mean()
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """
        Calculate profit factor (gross profits / gross losses).
        
        Args:
            returns: Series of returns
        
        Returns:
            Profit factor
        """
        if returns.empty:
            return np.nan
        
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        
        return gross_profits / gross_losses if gross_losses != 0 else np.inf
    
    def daily_metrics(self, returns: pd.Series) -> pd.DataFrame:
        """
        Calculate rolling metrics for each day using the short, medium, and long windows.
        
        Args:
            returns: Series of returns
        
        Returns:
            DataFrame with daily metrics
        """
        self._validate_returns(returns, 'short')
        
        # Initialize daily metrics DataFrame
        daily_metrics = pd.DataFrame(index=returns.index)
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        daily_metrics['cum_return'] = cum_returns
        
        # Calculate metrics for each window
        for window_type in ['short', 'medium', 'long']:
            window = self.get_window_size(window_type)
            min_periods = self.get_min_periods(window_type)
            
            # Skip if we don't have enough data for this window
            if len(returns) < min_periods:
                continue
            
            # Calculate rolling metrics
            roll_returns = returns.rolling(window=window, min_periods=min_periods)
            
            # Add basic metrics
            daily_metrics[f'{window_type}_return'] = roll_returns.mean() * self.annualization_factor
            daily_metrics[f'{window_type}_vol'] = roll_returns.std() * np.sqrt(self.annualization_factor)
            
            # Calculate rolling Sharpe ratio
            excess_ret = daily_metrics[f'{window_type}_return'] - self.risk_free_rate
            daily_metrics[f'{window_type}_sharpe'] = excess_ret / daily_metrics[f'{window_type}_vol']
            
            # Calculate rolling drawdown
            roll_cum_ret = (1 + roll_returns).cumprod()
            roll_max = roll_cum_ret.rolling(window=window, min_periods=min_periods).max()
            daily_metrics[f'{window_type}_drawdown'] = (roll_cum_ret / roll_max - 1).abs()
        
        return daily_metrics
    
    def calculate_z_score(self, metric_value: float, metric_series: pd.Series) -> float:
        """
        Calculate Z-score for a metric.
        
        Args:
            metric_value: Current value of the metric
            metric_series: Historical series of the metric
        
        Returns:
            Z-score
        """
        if metric_series.empty or metric_series.isna().all():
            return 0
        
        # Calculate mean and standard deviation
        mean = metric_series.mean()
        std = metric_series.std()
        
        # Handle zero standard deviation
        if std == 0:
            return 0
        
        # Calculate Z-score
        return (metric_value - mean) / std 