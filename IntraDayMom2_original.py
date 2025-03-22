"""
IntraDayMom2.py - Part 1: Infrastructure
Intraday Momentum Strategy Implementation
Based on the paper: Beat the Market - An Effective Intraday Momentum Strategy
"""
from __future__ import annotations  # Enable postponed evaluation of annotations
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List, Set, Union, TYPE_CHECKING
from datetime import datetime, time
from collections import defaultdict
import yaml
import logging.handlers
import pytz
import sys
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
import quantstats as qs
import logging
import pickle
import warnings
from dataclasses import dataclass, field

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _validate_returns(returns: pd.Series, component_name: str, logger: logging.Logger) -> None:
    """Helper to validate and log return statistics."""
    if returns is None:
        raise ValueError(f"{component_name}: Returns cannot be None")

    if not isinstance(returns, pd.Series):
        raise TypeError(f"{component_name}: Returns must be a pandas Series")

    if returns.empty:
        raise ValueError(f"{component_name}: Returns series is empty")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError(f"{component_name}: Returns index must be a DatetimeIndex")

    if returns.index.tz is None:
        raise ValueError(f"{component_name}: Returns index must be timezone-aware")

    # Log statistics
    logger.debug(f"\n{component_name} Returns Statistics:")
    logger.debug(f"Shape: {returns.shape}")
    logger.debug(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    logger.debug(f"Mean: {returns.mean():.4%}")
    logger.debug(f"Std: {returns.std():.4%}")
    logger.debug(f"Min: {returns.min():.4%}")
    logger.debug(f"Max: {returns.max():.4%}")
    logger.debug(f"Sample (first 5):\n{returns.head()}")

    # Flag extreme values
    extreme_returns = returns[abs(returns) > 1.0]
    if not extreme_returns.empty:
        logger.warning(f"{component_name}: Found {len(extreme_returns)} returns > 100%")
        logger.warning(f"Extreme returns:\n{extreme_returns}")

class DataManagerError(Exception):
    """Base exception class for DataManager errors"""
    pass

class DataLoadError(DataManagerError):
    """Exception raised for errors during data loading"""
    pass

class DataValidationError(DataManagerError):
    """Exception raised for data validation failures"""
    pass

class TimeZoneError(DataManagerError):
    """Exception raised for timezone-related errors"""
    pass


class ExitReason(Enum):
    """Enumeration of possible trade exit reasons"""
    MARKET_CLOSE = "MARKET_CLOSE"
    VWAP_STOP = "VWAP_STOP"
    BOUNDARY_STOP = "BOUNDARY_STOP"


@dataclass
class StrategyParameters:
    """Data class for strategy parameters"""
    lookback_days: int
    volatility_multiplier: float
    min_holding_period: pd.Timedelta = pd.Timedelta(minutes=1)
    entry_times: List[int] = None  # List of valid entry minute marks (e.g., [0, 30])

    def __post_init__(self):
        """Set default entry times if none provided"""
        if self.entry_times is None:
            self.entry_times = [0, 30]  # Default to trading on hour and half hour

@dataclass
class ContractSpecification:
    """Data class for contract specifications"""
    symbol: str  # Add this line
    tick_size: float
    multiplier: float
    margin: float
    market_open: time
    market_close: time
    last_entry: time

@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    direction: int
    pnl: float
    costs: float
    exit_reason: str
    base_size: float
    final_size: float
    base_return: float  # Added field
    final_return: float  # Added field
    strategy_name: str
    symbol: str
    contract_spec: ContractSpecification

@dataclass
class TransactionCosts:
    """Configuration for transaction costs"""
    commission_rate: float
    slippage_rate: float
    min_commission: float = 0.0
    fixed_costs: float = 0.0

@dataclass
class RiskParameters:
    """Base configuration for risk management."""
    min_size: float
    max_size: float
    min_scalar: float
    max_scalar: float

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_size: float
    max_daily_loss: float
    max_drawdown: float
    position_limit_pct: float  # Max position size as % of equity
    concentration_limit: float  # Max exposure to single instrument

@dataclass
class VolatilityParams:
    """Volatility-based sizing parameters"""
    target_volatility: float
    estimation_window: int
    min_scaling: float
    max_scaling: float
    adaptation_rate: float  # Added this field
    vol_target_range: Tuple[float, float] = (0.10, 0.20)  # Added with default

@dataclass
class SharpeParams:
    """Parameters for Sharpe-based risk management"""
    target_sharpe: float
    target_volatility: float
    min_scaling: float
    max_scaling: float
    adaptation_rate: float = 0.1
    min_trades: int = 5
    risk_free_rate: float = 0.02
    target_range: Tuple[float, float] = (0.5, 2.0)
    window_type: str = "medium"  # Use 'short', 'medium', or 'long' to match RiskMetrics windows

@dataclass
class AdaptiveParams:
    """Parameters for adaptive risk management"""
    base_volatility: float
    regime_window: int
    adaptation_rate: float
    min_scaling: float
    max_scaling: float
    vol_target_range: Tuple[float, float] = (0.10, 0.20)
    regime_thresholds: Tuple[float, float] = (0.8, 1.2)  # Added for regime detection

class MetricType(Enum):
    """Types of metrics that can be calculated."""
    ROLLING = "rolling"
    SUMMARY = "summary"

@dataclass
class MetricResult:
    value: Union[pd.DataFrame, Dict[str, float]]
    calculation_time: pd.Timestamp
    metric_type: MetricType
    input_rows: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class BaseReturns:
    """Class representing base strategy returns before position sizing."""
    returns: pd.Series
    metrics: Optional[Union[pd.DataFrame, MetricResult]] = None
    summary_metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Validate returns data and log statistics."""
        logger = logging.getLogger(__name__)
        _validate_returns(self.returns, "BaseReturns", logger)

        # Log additional debug info
        logger.debug("BaseReturns - First 5 values:\n%s", self.returns.head())
        logger.debug("BaseReturns - Summary: min=%s, max=%s, mean=%s",
                    self.returns.min(), self.returns.max(), self.returns.mean())
    @property
    def start_date(self) -> pd.Timestamp:
        """Get the start date of returns data."""
        return self.returns.index.min()

    @property
    def end_date(self) -> pd.Timestamp:
        """Get the end date of returns data."""
        return self.returns.index.max()

    @property
    def trading_days(self) -> int:
        """Get the number of trading days."""
        return len(self.returns)

    def calculate_metrics(self, risk_metrics: RiskMetrics) -> None:
        """Calculate metrics using validated returns data."""
        if len(self.returns) == 0:
            return

        self.metrics = risk_metrics.calculate_metrics(
            self.returns,
            metric_type=MetricType.ROLLING,
            caller="BaseReturns"
        )

        self.summary_metrics = risk_metrics.calculate_metrics(
            self.returns,
            metric_type=MetricType.SUMMARY,
            caller="BaseReturns"
        )

    def subset(self, start_date: Optional[pd.Timestamp] = None,
               end_date: Optional[pd.Timestamp] = None) -> 'BaseReturns':
        """Create a new BaseReturns object with a subset of the data."""
        mask = pd.Series(True, index=self.returns.index)
        if start_date:
            mask &= self.returns.index >= start_date
        if end_date:
            mask &= self.returns.index <= end_date

        return BaseReturns(returns=self.returns[mask])


@dataclass
class LeveredReturns:
    """Class representing returns after position sizing."""
    position_sizes: pd.Series
    base_returns: BaseReturns
    metrics: Optional[Union[pd.DataFrame, 'MetricResult']] = None
    summary_metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Validate data and calculate levered returns."""
        if not isinstance(self.position_sizes, pd.Series):
            raise TypeError("position_sizes must be a pandas Series")

        if not isinstance(self.base_returns, BaseReturns):
            raise TypeError("base_returns must be a BaseReturns instance")

        # Align position sizes with returns index
        self.position_sizes = self.position_sizes.reindex(
            self.base_returns.returns.index
        ).fillna(1.0)

    @property
    def returns(self) -> pd.Series:
        import logging
        logger = logging.getLogger(__name__)
        # Debug: print the first few values of base returns and position sizes
        logger.debug("LeveredReturns - Base returns (head):\n%s", self.base_returns.returns.head())
        logger.debug("LeveredReturns - Position sizes (head):\n%s", self.position_sizes.head())

        # Compute the levered equity curve by compounding the scaled base returns.
        levered_equity = (1 + self.base_returns.returns * self.position_sizes).cumprod()
        logger.debug("LeveredReturns - Levered equity (head):\n%s", levered_equity.head())

        # Then derive daily returns as the percentage change in the equity curve.
        levered_returns = levered_equity.pct_change().fillna(0)
        logger.debug("LeveredReturns - Levered returns (head):\n%s", levered_returns.head())

        return levered_returns


    def calculate_metrics(self, risk_metrics: 'RiskMetrics') -> None:
        """Calculate metrics using the levered returns."""
        levered_returns = self.returns

        if len(levered_returns) == 0:
            return

        self.metrics = risk_metrics.calculate_metrics(
            levered_returns,
            metric_type=MetricType.ROLLING,
            caller="LeveredReturns"
        )

        self.summary_metrics = risk_metrics.calculate_metrics(
            levered_returns,
            metric_type=MetricType.SUMMARY,
            caller="LeveredReturns"
        )

@dataclass
class RegimeConfig:
    """Configuration for regime detection"""
    enabled: bool = True
    method: str = "volatility"  # "volatility", "trend", or "combined"
    lookback: int = 252
    vol_threshold: float = 1.5  # Multiplier for regime change detection
    trend_threshold: float = 0.5  # Z-score for trend detection

class Config:
    """Centralized configuration management optimized for futures/options"""

    def __init__(self, config_file='config.yaml'):
        # Load configurations from YAML file
        config_path = Path(config_file)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Basic configurations
        self.input_files = config.get('input_files', {})
        self.output_paths = config.get('output_paths', {})
        self.strategy_params = config.get('strategy_params', {})
        self.contract_specs = config.get('contract_specs', {})
        self.trading_hours = config.get('trading_hours', {})
        self.transaction_costs = config.get('transaction_costs', {})
        self.initial_equity = config.get('initial_equity', 100000)
        self.timezone = config.get('timezone', 'US/Central')

        # Risk and position management configurations
        self.risk_metrics = config.get('risk_metrics', {})
        self.position_params = config.get('position_params', {})
        self.adaptive_params = config.get('adaptive_params', {})
        self.volatility_params = config.get('volatility_params', {})
        self.sharpe_params = config.get('sharpe_params', {})
        self.risk_limits = config.get('risk_limits', {})
        self.risk_params = config.get('risk_params', {'risk_manager_type': 'volatility'})  # Added this line

        # Analysis parameters
        self.symbol = config.get('symbol')
        self.days_to_analyze = config.get('days_to_analyze', 252)
        self.lookback_buffer = config.get('lookback_buffer', 63)

        # Validate and process configurations
        self._validate_parameters()
        self._process_trading_hours()

    def get_risk_manager_config(self) -> Dict:
        """Get risk manager configuration."""
        return {
            'type': self.risk_params.get('risk_manager_type', 'volatility'),  # Default to volatility
            'volatility_params': self.volatility_params,
            'sharpe_params': self.sharpe_params,
            'adaptive_params': self.adaptive_params,
            'risk_limits': self.risk_limits,
            'combined_weights': self.risk_params.get('combined_weights', [0.4, 0.3, 0.3])  # Added this line
        }

    def _validate_parameters(self):
        """Validate critical configuration parameters"""
        if not self.input_files:
            raise ValueError("Input files are not specified in the configuration.")
        if not self.output_paths:
            raise ValueError("Output paths are not specified in the configuration.")
        if self.initial_equity <= 0:
            raise ValueError("Initial equity must be a positive number.")
        if self.symbol is None:
            raise ValueError("Trading symbol must be specified in configuration.")
        if self.symbol not in self.input_files:
            raise ValueError(f"No input file specified for symbol '{self.symbol}'")

        # Ensure all file paths are valid
        for symbol, path in self.input_files.items():
            if not Path(path).is_file():
                raise FileNotFoundError(f"Input file for symbol '{symbol}' not found at {path}")

        # Ensure output directories exist or create them
        for symbol, path in self.output_paths.items():
            output_dir = Path(path)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Validate contract specifications
        for symbol, specs in self.contract_specs.items():
            required_specs = ['tick_size', 'multiplier', 'margin']
            for spec in required_specs:
                if spec not in specs:
                    raise ValueError(f"Contract specification '{spec}' missing for symbol '{symbol}'")

        # Validate trading hours
        for symbol, hours in self.trading_hours.items():
            required_hours = ['market_open', 'market_close', 'last_entry']
            for hour in required_hours:
                if hour not in hours:
                    raise ValueError(f"Trading hour '{hour}' missing for symbol '{symbol}'")

        # Validate risk parameters
        if not self.risk_metrics:
            raise ValueError("Risk metrics configuration is missing")
        if not self.risk_limits:
            raise ValueError("Risk limits configuration is missing")

    def _process_trading_hours(self):
        """Convert trading hour strings to time objects and store timezone"""
        try:
            self.timezone = pytz.timezone(self.timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            raise ValueError(f"Invalid timezone specified: {self.timezone}")

        for symbol, hours in self.trading_hours.items():
            processed_hours = {}
            for key in ['market_open', 'market_close', 'last_entry']:
                if isinstance(hours[key], str):
                    try:
                        # Store as time object
                        processed_hours[key] = datetime.strptime(hours[key], '%H:%M').time()
                    except ValueError:
                        raise ValueError(
                            f"Invalid time format for '{key}' in trading hours for symbol '{symbol}'. "
                            f"Expected format 'HH:MM'."
                        )
            self.trading_hours[symbol] = processed_hours

    def get_symbol_specs(self, symbol: str) -> Dict:
        """Get all specifications for a given symbol"""
        if symbol not in self.contract_specs:
            raise ValueError(f"No specifications found for symbol '{symbol}'")

        return {
            'contract_specs': self.contract_specs[symbol],
            'trading_hours': self.trading_hours[symbol],
            'input_file': self.input_files.get(symbol),
            'output_path': self.output_paths.get(symbol)
        }


@dataclass
class TradingResults:
    """Contains all results from a strategy execution."""

    def __init__(self,
                 symbol: str,
                 strategy_name: str,
                 base_trades: List[Trade],
                 final_trades: List[Trade],
                 trade_metrics: List[Dict],
                 daily_performance: pd.DataFrame,
                 execution_data: pd.DataFrame,
                 config: Config,
                 contract_spec: ContractSpecification,
                 timestamp: Optional[pd.Timestamp] = None,
                 base_returns: Optional[BaseReturns] = None,
                 levered_returns: Optional[LeveredReturns] = None):
        """
        Initialize TradingResults with strategy execution data.
        """
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.base_trades = base_trades
        self.final_trades = final_trades
        self.trade_metrics = trade_metrics
        self.daily_performance = daily_performance
        self.execution_data = execution_data
        self.config = config
        self.contract_spec = contract_spec
        self.timestamp = timestamp or pd.Timestamp.now()

        # Initialize risk metrics calculator
        self.risk_metrics = RiskMetrics(config)

        # Store or initialize returns objects
        self.base_returns = base_returns
        self.levered_returns = levered_returns

        # Initialize returns objects if not provided
        if self.base_returns is None and self.daily_performance is not None:
            if 'base_returns' in self.daily_performance.columns:
                self.base_returns = BaseReturns(
                    returns=self.daily_performance['base_returns']
                )

        if (self.levered_returns is None and self.daily_performance is not None
                and self.base_returns is not None):
            if 'position_size' in self.daily_performance.columns:
                levered_returns = (self.daily_performance['base_returns'] *
                                   self.daily_performance['position_size'])
                self.levered_returns = LeveredReturns(
                    returns=levered_returns,
                    position_sizes=self.daily_performance['position_size'],
                    base_returns=self.base_returns
                )

        # Calculate metrics
        self.calculate_all_metrics()

    def calculate_all_metrics(self) -> None:
        """Calculate all metrics for both base and levered returns."""
        if self.base_returns:
            self.base_returns.calculate_metrics(self.risk_metrics)
        if self.levered_returns:
            self.levered_returns.calculate_metrics(self.risk_metrics)

    @property
    def metrics(self) -> Dict:
        """Return calculated performance metrics combined from all sources."""
        metrics = {}

        # Trade-based metrics
        if self.base_trades:
            metrics.update({
                'base_total_trades': len(self.base_trades),
                'base_win_rate': sum(1 for t in self.base_trades if t.pnl > t.costs) / len(self.base_trades),
                'base_total_pnl': sum(t.pnl for t in self.base_trades),
                'base_total_costs': sum(t.costs for t in self.base_trades)
            })

        if self.final_trades:
            metrics.update({
                'final_total_trades': len(self.final_trades),
                'final_win_rate': sum(1 for t in self.final_trades if t.pnl > t.costs) / len(self.final_trades),
                'final_total_pnl': sum(t.pnl for t in self.final_trades),
                'final_total_costs': sum(t.costs for t in self.final_trades)
            })

        # Get metrics from returns objects
        if self.base_returns and self.base_returns.summary_metrics:
            metrics.update({f'base_{k}': v for k, v in self.base_returns.summary_metrics.items()})

        if self.levered_returns and self.levered_returns.summary_metrics:
            metrics.update({f'levered_{k}': v for k, v in self.levered_returns.summary_metrics.items()})

        return metrics

    def get_metric(self, metric_name: str, returns_type: str = 'levered') -> pd.Series:
        """
        Get a specific metric time series.

        Args:
            metric_name (str): Name of the metric to retrieve
            returns_type (str): Either 'base' or 'levered'

        Returns:
            pd.Series: Time series of the requested metric
        """
        returns_obj = self.base_returns if returns_type == 'base' else self.levered_returns

        if returns_obj is None or returns_obj.metrics is None:
            raise ValueError(f"No metrics available for returns_type '{returns_type}'")

        metrics_df = returns_obj.metrics.value if isinstance(returns_obj.metrics, MetricResult) else returns_obj.metrics

        if metric_name not in metrics_df.columns:
            raise KeyError(f"Metric '{metric_name}' not found for {returns_type} returns")

        return metrics_df[metric_name]

    def get_returns(self, returns_type: str = 'levered') -> pd.Series:
        """Get returns series of specified type."""
        returns_obj = self.base_returns if returns_type == 'base' else self.levered_returns
        return returns_obj.returns if returns_obj else pd.Series()

    def get_position_sizes(self) -> pd.Series:
        """Get position sizes series."""
        return (self.levered_returns.position_sizes
                if self.levered_returns
                else pd.Series())


class LoggingConfig:
    """Centralized logging configuration for the trading system."""
    _instance = None
    _initialized = False
    _loggers = set()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggingConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.log_dir = Path("logs")
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create separate log files for different purposes
            self.main_log = self.log_dir / f"trading_system_{self.timestamp}.log"
            self.trade_log = self.log_dir / f"trades_{self.timestamp}.log"
            self.debug_log = self.log_dir / f"debug_{self.timestamp}.log"

            self.config = self._load_config()
            self._initialized = True

    def _load_config(self) -> Dict:
        """Load logging configuration from yaml file or return default config."""
        try:
            with open('logging_config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                'log_levels': {
                    'root': 'DEBUG',  # Changed from INFO to DEBUG
                    'data_manager': 'DEBUG',  # Changed to DEBUG for debugging
                    'risk_manager': 'DEBUG',
                    'strategy': 'DEBUG',
                    'trading_system': 'DEBUG',
                    'trade_execution': 'DEBUG'
                },
                'formatters': {
                    'detailed': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'simple': '%(levelname)s - %(message)s',
                    'trade': '%(asctime)s - %(levelname)s - TRADE - %(message)s'
                },
                'rotation': {
                    'when': 'midnight',
                    'interval': 1,
                    'backupCount': 7
                }
            }

    def setup(self) -> None:
        """Set up logging configuration with separate handlers for different log types."""
        if hasattr(self, '_setup_complete'):
            return

        self.log_dir.mkdir(exist_ok=True)

        # Configure root logger - Explicitly set to DEBUG
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Make sure this is DEBUG
        root_logger.handlers.clear()

        # Create formatters
        formatters = {
            'detailed': logging.Formatter(self.config['formatters']['detailed']),
            'simple': logging.Formatter(self.config['formatters']['simple']),
            'trade': logging.Formatter(self.config['formatters']['trade'])
        }

        # Main log file handler (INFO and above)
        main_handler = logging.handlers.TimedRotatingFileHandler(
            self.main_log,
            when=self.config['rotation']['when'],
            interval=self.config['rotation']['interval'],
            backupCount=self.config['rotation']['backupCount']
        )
        main_handler.setFormatter(formatters['detailed'])
        main_handler.setLevel(logging.INFO)

        # Trade log file handler (specific for trade executions)
        trade_handler = logging.handlers.TimedRotatingFileHandler(
            self.trade_log,
            when=self.config['rotation']['when'],
            interval=self.config['rotation']['interval'],
            backupCount=self.config['rotation']['backupCount']
        )
        trade_handler.setFormatter(formatters['trade'])
        trade_handler.addFilter(lambda record: 'TRADE' in record.getMessage())
        trade_handler.setLevel(logging.INFO)

        # Debug log file handler
        debug_handler = logging.handlers.TimedRotatingFileHandler(
            self.debug_log,
            when=self.config['rotation']['when'],
            interval=self.config['rotation']['interval'],
            backupCount=self.config['rotation']['backupCount']
        )
        debug_handler.setFormatter(formatters['detailed'])
        debug_handler.setLevel(logging.DEBUG)

        # Console handler (reduced output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatters['simple'])
        console_handler.setLevel(logging.INFO)
        # Add filter to exclude trade execution details from console
        console_handler.addFilter(
            lambda record: 'TRADE' not in record.getMessage() or record.levelno >= logging.WARNING
        )

        # Add handlers to root logger
        root_logger.addHandler(main_handler)
        root_logger.addHandler(trade_handler)
        root_logger.addHandler(debug_handler)
        root_logger.addHandler(console_handler)

        self._setup_complete = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a module-level logger with proper configuration.
        """
        # Initialize singleton if not already done
        if not cls._initialized:
            cls()  # Initialize if not already done

        logger = logging.getLogger(name)

        # Only configure if this logger hasn't been seen before
        if name not in cls._loggers:
            # Get module name (last part of the dotted path)
            module_name = name.split('.')[-1]

            # Set level from config if available for this module
            if module_name in cls._instance.config['log_levels']:
                logger.setLevel(cls._instance.config['log_levels'][module_name])

            # Add to set of configured loggers
            cls._loggers.add(name)

            # Only log initialization for the root logger
            if name == '__main__':
                logger.info("Logging system initialized")

        return logger



class DataManager:
    """
    DataManager handles raw data operations with comprehensive validation and cleaning.
    Focuses on providing clean, validated market data ready for strategy implementation.
    """

    # Class-level constants for validation
    REQUIRED_COLUMNS = {
        'TimeStamp', 'Open', 'High', 'Low', 'Close',
        'UpVolume', 'DownVolume'
    }

    REQUIRED_SPECS = {
        'tick_size', 'multiplier', 'margin'
    }

    def __init__(self, config: Config):
        """Initialize DataManager with configuration."""
        self.logger = LoggingConfig.get_logger(__name__)
        self.logger.info("Initializing DataManager")
        self.config = config
        self._verify_config()
        self._data_cache = {}

    def prepare_data_for_analysis(self, symbol: str, days_to_analyze: int,
                                  lookback_buffer: int) -> pd.DataFrame:
        """Prepare clean market data for analysis."""
        try:
            cache_key = f"{symbol}_{days_to_analyze}_{lookback_buffer}"
            if cache_key in self._data_cache:
                self.logger.info("Using cached data")
                return self._data_cache[cache_key].copy()

            self.logger.info(f"\nPreparing data for {symbol}")
            df = self.load_and_validate_data(
                self.config.input_files[symbol],
                self.config.contract_specs[symbol]
            )

            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.info("Converting index to DatetimeIndex")
                df.index = pd.to_datetime(df.index)

            # Calculate analysis period dates
            start_date = self._get_start_date(df, days_to_analyze, lookback_buffer)
            trading_start = self._get_trading_start_date(df, days_to_analyze, lookback_buffer)
            data_start = df.index[0].date()

            self.logger.info(f"Data period: {data_start} to {df.index[-1].date()}")
            self.logger.info(f"Analysis start: {start_date}")
            self.logger.info(f"Trading start: {trading_start}")

            df = self._calculate_basic_market_data(df)
            df = self._filter_analysis_period(df, days_to_analyze, lookback_buffer)
            df = self._add_trading_markers(df, symbol)
            self._validate_prepared_data(df)

            self._data_cache[cache_key] = df.copy()
            return df

        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}")
            raise

    def _calculate_daily_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily values with proper error handling."""
        try:
            self.logger.debug("Calculating daily values")

            # Create a copy with timezone-naive index for resampling
            df_naive = df.copy()
            df_naive.index = df_naive.index.tz_localize(None)

            # Calculate daily values using proper grouping
            daily_values = df_naive.groupby(df_naive.index.date).agg({
                'Open': 'first',
                'Close': 'last'
            })

            # Rename columns
            daily_values = daily_values.rename(columns={
                'Open': 'day_open',
                'Close': 'day_close'
            })

            # Calculate previous day's close properly
            daily_values['prev_close'] = daily_values['day_close'].shift(1)

            return daily_values

        except Exception as e:
            self.logger.error("Daily value calculation failed", exc_info=True)
            raise

    def load_and_validate_data(self, file_path: str, contract_spec: Dict) -> pd.DataFrame:
        """Load and validate market data with comprehensive error handling."""
        try:
            self._validate_file(file_path)
            df = self._load_data(file_path)
            self._validate_columns(df)

            # Process timestamps and timezones
            df = self._process_timestamp_index(df)
            df = self._handle_timezones(df)

            # Clean and validate data
            df = self._validate_ohlc_data(df)
            df = self._remove_price_anomalies(df)

            # Add contract specifications
            df = self._add_contract_specs(df, contract_spec)

            return df

        except Exception as e:
            self.logger.error("Data loading and validation failed", exc_info=True)
            raise

    def _verify_config(self) -> None:
        """Verify configuration parameters required for data management."""
        try:
            if not hasattr(self.config, 'input_files') or not self.config.input_files:
                raise ValueError("No input files specified in configuration")

            # Verify file paths exist
            for symbol, path in self.config.input_files.items():
                if not Path(path).exists():
                    self.logger.warning(f"Data file for {symbol} not found at: {path}")

            # Check timezone configuration
            if not hasattr(self.config, 'timezone'):
                self.logger.warning("No timezone specified in configuration, will use UTC")

            self.logger.debug(
                f"Configuration verified - "
                f"Symbols: {list(self.config.input_files.keys())}, "
                f"Timezone: {getattr(self.config, 'timezone', 'UTC')}"
            )

        except Exception as e:
            self.logger.error("Configuration verification failed", exc_info=True)
            raise

    def _validate_prepared_data(self, df: pd.DataFrame) -> None:
        """Validate the prepared dataset."""
        try:
            required_columns = {
                'Open', 'High', 'Low', 'Close',
                'volume', 'vwap', 'day_open', 'prev_close',
                'minute_of_day', 'move_from_open'
            }

            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Validate data quality
            null_counts = df[list(required_columns)].isnull().sum()
            if null_counts.any():
                self.logger.warning("Found null values:")
                for col, count in null_counts[null_counts > 0].items():
                    self.logger.warning(f"  {col}: {count} null values")

            # Validate numerical ranges
            if (df['minute_of_day'] < 0).any() or (df['minute_of_day'] >= 1440).any():
                raise ValueError("Invalid minute_of_day values detected")

            if (df['move_from_open'] < 0).any():
                raise ValueError("Negative move_from_open values detected")

            self.logger.info("Data validation completed successfully")

        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise

    def _validate_file(self, file_path: str) -> None:
        """Validate file existence and format."""
        if not Path(file_path).exists():
            msg = f"Data file not found: {file_path}"
            self.logger.error(msg)
            raise DataLoadError(msg)

        if not file_path.endswith('.csv'):
            self.logger.warning(f"File {file_path} is not a CSV file")

    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file with error handling and type validation."""
        try:
            df = pd.read_csv(
                file_path,
                parse_dates=['TimeStamp'],
                dtype={
                    'Open': 'float64',
                    'High': 'float64',
                    'Low': 'float64',
                    'Close': 'float64',
                    'UpVolume': 'float64',
                    'DownVolume': 'float64'
                }
            )
            self.logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
            return df

        except Exception as e:
            msg = f"Failed to load data: {str(e)}"
            self.logger.error(msg)
            raise DataLoadError(msg)

    def _validate_config(self) -> None:
        """Validate configuration parameters required for data management."""
        try:
            if not hasattr(self.config, 'contract_specs') or not self.config.contract_specs:
                raise ValueError("No contract specifications found in configuration")

            for symbol, specs in self.config.contract_specs.items():
                # Check required specifications
                required_specs = {'tick_size', 'multiplier', 'margin',
                                  'market_open', 'market_close', 'last_entry'}
                missing_specs = required_specs - set(specs.keys())
                if missing_specs:
                    raise ValueError(f"Missing specifications for {symbol}: {missing_specs}")

                # Validate trading hours format
                for time_field in ['market_open', 'market_close', 'last_entry']:
                    time_str = specs[time_field]
                    try:
                        if isinstance(time_str, str):
                            datetime.strptime(time_str, '%H:%M')
                    except ValueError:
                        raise ValueError(
                            f"Invalid time format for {time_field} in {symbol}: {time_str}. "
                            f"Expected format: 'HH:MM'"
                        )

            # Verify timezone
            if not hasattr(self.config, 'timezone'):
                self.logger.warning("No timezone specified in configuration, will use UTC")

            self.logger.debug(
                f"Configuration verified - "
                f"Symbols: {list(self.config.contract_specs.keys())}, "
                f"Timezone: {getattr(self.config, 'timezone', 'UTC')}"
            )

        except Exception as e:
            self.logger.error("Configuration verification failed", exc_info=True)
            raise

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate required columns exist in DataFrame."""
        missing_columns = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_columns:
            msg = f"Missing required columns: {missing_columns}"
            self.logger.error(msg)
            raise DataValidationError(msg)

    def _process_timestamp_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate timestamp index."""
        try:
            df = df.set_index('TimeStamp')
            df.index = pd.to_datetime(df.index)

            # Ensure index is sorted and unique
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()

            duplicates = df.index.duplicated()
            if duplicates.any():
                self.logger.warning(f"Removing {duplicates.sum()} duplicate timestamps")
                df = df[~duplicates]

            return df

        except Exception as e:
            msg = f"Failed to process timestamp index: {str(e)}"
            self.logger.error(msg, exc_info=True)
            raise DataValidationError(msg)

    def _create_daily_index(self, dates: List[datetime.date], input_tz: str) -> pd.DatetimeIndex:
        """Create daily index while preserving timezone."""
        return pd.DatetimeIndex([
            pd.Timestamp(date) for date in dates
        ]).tz_localize(input_tz)

    def _handle_timezones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle timezone conversion - the ONLY place where timezone conversion should occur."""
        try:
            # Get configured timezone from config
            target_tz = getattr(self.config, 'timezone', 'UTC')

            # If data has no timezone, assume UTC
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')

            # Convert to target timezone
            df.index = df.index.tz_convert(target_tz)

            self.logger.info(f"Converted data timezone to {target_tz}")
            return df

        except Exception as e:
            msg = f"Timezone conversion failed: {str(e)}"
            self.logger.error(msg, exc_info=True)
            raise TimeZoneError(msg)

    def _validate_ohlc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLC data."""
        df = df.copy()

        # Track invalid conditions
        invalid_rows = pd.DataFrame(index=df.index)
        invalid_rows['high_low'] = df['High'] < df['Low']
        invalid_rows['open_range'] = (df['Open'] > df['High']) | (df['Open'] < df['Low'])
        invalid_rows['close_range'] = (df['Close'] > df['High']) | (df['Close'] < df['Low'])

        # Log issues found
        issues_found = {
            'High < Low': invalid_rows['high_low'].sum(),
            'Open outside range': invalid_rows['open_range'].sum(),
            'Close outside range': invalid_rows['close_range'].sum()
        }

        if any(issues_found.values()):
            self.logger.warning(
                f"Found {sum(issues_found.values())} OHLC inconsistencies"
            )
            df = self._fix_ohlc_issues(df, invalid_rows)

        return df

    def _fix_ohlc_issues(self, df: pd.DataFrame, invalid_rows: pd.DataFrame) -> pd.DataFrame:
        """Fix OHLC data issues."""
        # Fix High/Low inversions
        if invalid_rows['high_low'].any():
            mask = invalid_rows['high_low']
            df.loc[mask, ['High', 'Low']] = df.loc[mask, ['Low', 'High']].values

        # Fix Open/Close values
        for col, mask in [('Open', invalid_rows['open_range']),
                          ('Close', invalid_rows['close_range'])]:
            if mask.any():
                df.loc[mask, col] = df.loc[mask].apply(
                    lambda x: np.clip(x[col], x['Low'], x['High']), axis=1
                )

        return df

    def _remove_price_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove anomalous price movements."""
        returns = df['Close'].pct_change()

        # Calculate rolling statistics
        roll_std = returns.rolling(window=20, min_periods=1).std()

        # Identify extreme moves (>5 std dev)
        outliers = abs(returns) > (5 * roll_std)

        if outliers.any():
            self.logger.warning(f"Removing {outliers.sum()} anomalous price movements")
            df = df[~outliers]

        return df

    def _add_contract_specs(self, df: pd.DataFrame, contract_spec: Dict) -> pd.DataFrame:
        """Add contract specifications with validation."""
        missing_specs = self.REQUIRED_SPECS - set(contract_spec.keys())
        if missing_specs:
            msg = f"Missing contract specifications: {missing_specs}"
            self.logger.error(msg)
            raise DataValidationError(msg)

        for spec, value in contract_spec.items():
            df[spec] = value

        return df

    def _calculate_basic_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic market data columns from raw data."""
        try:
            df = df.copy()

            # Get market hours from config for the symbol
            symbol_specs = self.config.contract_specs[self.config.symbol]
            market_open = datetime.strptime(symbol_specs['market_open'], '%H:%M').time()
            market_close = datetime.strptime(symbol_specs['market_close'], '%H:%M').time()

            self.logger.info(f"Processing market data between {market_open} and {market_close}")

            # Add minute of day
            df['minute_of_day'] = df.index.hour * 60 + df.index.minute

            # Create market filters
            df['trading_date'] = df.index.date
            market_open_mask = (df.index.time == market_open)
            market_close_mask = (df.index.time == market_close)
            market_hours = (
                    (df.index.time >= market_open) &
                    (df.index.time <= market_close)
            )

            # Get market open prices for each day
            opens = df[market_open_mask].groupby('trading_date')['Open'].first()
            df['day_open'] = df['trading_date'].map(opens)

            # Get market close prices for each day
            closes = df[market_close_mask].groupby('trading_date')['Close'].last()

            # Calculate previous close
            df['prev_close'] = df['trading_date'].map(closes.shift(1))

            # Fill first day's prev_close with its open
            first_date = df['trading_date'].iloc[0]
            if first_date in opens.index:
                df.loc[df['trading_date'] == first_date, 'prev_close'] = opens[first_date]

            # Calculate move from open (only during market hours)
            df['move_from_open'] = np.nan
            df.loc[market_hours, 'move_from_open'] = (
                    abs(df.loc[market_hours, 'Close'] - df.loc[market_hours, 'day_open']) /
                    df.loc[market_hours, 'day_open']
            )

            # Calculate volume and VWAP
            if all(col in df.columns for col in ['UpVolume', 'DownVolume']):
                df['volume'] = df['UpVolume'] + df['DownVolume']

                # Calculate VWAP only for market hours
                df['vwap'] = np.nan
                market_data = df[market_hours].copy()
                volume = market_data['volume']
                price_volume = market_data['Close'] * volume
                df.loc[market_hours, 'vwap'] = (
                        price_volume.groupby(market_data['trading_date']).cumsum() /
                        volume.groupby(market_data['trading_date']).cumsum()
                )
            else:
                self.logger.warning("Volume columns not found, using Close price for VWAP")
                df['volume'] = 0
                df['vwap'] = df['Close']

            # Clean up
            df = df.drop('trading_date', axis=1)

            # Log statistics
            valid_market_hours = market_hours.sum()
            self.logger.info(f"Processed {valid_market_hours:,} market hours bars")
            self.logger.info("Calculated basic market data columns (market hours only):")
            for col in ['minute_of_day', 'day_open', 'prev_close', 'move_from_open', 'volume', 'vwap']:
                market_hours_count = df.loc[market_hours, col].notna().sum()
                self.logger.info(f"  - {col}: {market_hours_count:,} valid values during market hours")

            return df

        except Exception as e:
            self.logger.error(f"Failed to calculate basic market data: {str(e)}")
            raise

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP while maintaining compatibility with strategy."""
        try:
            df = df.copy()
            df['volume'] = df['UpVolume'] + df['DownVolume']

            if (df['volume'] < 0).any():
                raise DataValidationError("Negative volume values detected")

            # Calculate VWAP using original method for compatibility
            df['vwap'] = (df['Close'] * df['volume']).groupby(df.index.date).cumsum() / \
                         df['volume'].groupby(df.index.date).cumsum()
            df['vwap'].fillna(df['Close'], inplace=True)

            return df

        except Exception as e:
            msg = f"VWAP calculation failed: {str(e)}"
            self.logger.error(msg, exc_info=True)
            raise DataValidationError(msg)

    def _filter_analysis_period(self, df: pd.DataFrame, days_to_analyze: int,
                                lookback_buffer: int) -> pd.DataFrame:
        """Filter data to analysis period."""
        try:
            # Convert dates to Series for proper handling
            trading_days = pd.Series(df.index.date).unique()

            # Calculate start dates
            start_idx = max(len(trading_days) - (days_to_analyze + lookback_buffer), 0)
            trading_start_idx = min(start_idx + lookback_buffer, len(trading_days) - 1)

            start_date = trading_days[start_idx]
            trading_start = trading_days[trading_start_idx]

            # Filter data
            df = df[df.index.date >= start_date].copy()
            df['is_trading_period'] = df.index.date >= trading_start

            return df

        except Exception as e:
            self.logger.error(f"Period filtering failed: {str(e)}")
            raise

    def _add_trading_markers(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add trading session markers."""
        try:
            # Get trading hours from contract specifications
            contract_spec = self.config.contract_specs[symbol]

            # Convert string times to time objects if they aren't already
            market_open = (contract_spec['market_open']
                           if isinstance(contract_spec['market_open'], time)
                           else datetime.strptime(contract_spec['market_open'], '%H:%M').time())

            market_close = (contract_spec['market_close']
                            if isinstance(contract_spec['market_close'], time)
                            else datetime.strptime(contract_spec['market_close'], '%H:%M').time())

            last_entry = (contract_spec['last_entry']
                          if isinstance(contract_spec['last_entry'], time)
                          else datetime.strptime(contract_spec['last_entry'], '%H:%M').time())

            df = df.copy()

            # Add trading session markers
            df['trading_hour'] = (
                    (df.index.time >= market_open) &
                    (df.index.time <= market_close)
            )
            df['pre_market'] = df.index.time < market_open
            df['post_market'] = df.index.time > market_close
            df['can_enter'] = (
                    (df.index.time >= market_open) &
                    (df.index.time <= last_entry)
            )

            # Log trading hours info
            self.logger.info(f"\nTrading Hours for {symbol}:")
            self.logger.info(f"Market Open: {market_open}")
            self.logger.info(f"Market Close: {market_close}")
            self.logger.info(f"Last Entry: {last_entry}")

            return df

        except Exception as e:
            self.logger.error(f"Adding trading markers failed: {str(e)}")
            raise

    def _get_start_date(self, df: pd.DataFrame, days_to_analyze: int,
                        lookback_buffer: int) -> datetime.date:
        """Calculate analysis start date including lookback period."""
        trading_days = pd.Series(df.index.date).unique()
        start_idx = max(len(trading_days) - (days_to_analyze + lookback_buffer), 0)
        return trading_days[start_idx]

    def _get_trading_start_date(self, df: pd.DataFrame, days_to_analyze: int,
                                lookback_buffer: int) -> datetime.date:
        """Calculate actual trading start date after lookback period."""
        trading_days = pd.Series(df.index.date).unique()
        start_idx = max(len(trading_days) - (days_to_analyze + lookback_buffer), 0)
        return trading_days[min(start_idx + lookback_buffer, len(trading_days) - 1)]

    def _log_data_preparation_summary(self, df: pd.DataFrame) -> None:
        """Log summary of prepared data."""
        try:
            self.logger.info("\nPrepared Data Summary:")
            self.logger.info(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
            self.logger.info(f"Total rows: {len(df):,}")

            # Convert to Series before getting unique dates
            unique_dates = pd.Series(df.index.date).unique()
            self.logger.info(f"Trading days: {len(unique_dates)}")
            self.logger.info(f"Trading hours: {df['trading_hour'].sum():,}")
            self.logger.info(f"Analysis period rows: {df['is_trading_period'].sum():,}")

            # Log data quality statistics
            self._log_data_quality_stats(df)

        except Exception as e:
            self.logger.error(f"Failed to log data summary: {str(e)}")

    def _log_data_quality_stats(self, df: pd.DataFrame) -> None:
        """Log data quality statistics."""
        try:
            self.logger.info("\nData Quality Statistics:")

            # Check for missing values
            null_counts = df.isnull().sum()
            if null_counts.any():
                self.logger.info("\nMissing Values:")
                for col, count in null_counts[null_counts > 0].items():
                    self.logger.info(f"  {col}: {count:,} ({count / len(df):.2%})")

            # Check for price anomalies
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                stats = df[col].describe()
                self.logger.info(f"\n{col} Statistics:")
                self.logger.info(f"  Mean: {stats['mean']:.2f}")
                self.logger.info(f"  Std: {stats['std']:.2f}")
                self.logger.info(f"  Min: {stats['min']:.2f}")
                self.logger.info(f"  Max: {stats['max']:.2f}")

            # Check for zero or negative prices
            for col in price_cols:
                zero_count = (df[col] <= 0).sum()
                if zero_count > 0:
                    self.logger.warning(
                        f"Found {zero_count:,} zero or negative values in {col}"
                    )

            # Volume statistics
            if 'volume' in df.columns:
                vol_stats = df['volume'].describe()
                self.logger.info("\nVolume Statistics:")
                self.logger.info(f"  Mean: {vol_stats['mean']:.0f}")
                self.logger.info(f"  Std: {vol_stats['std']:.0f}")
                self.logger.info(f"  Max: {vol_stats['max']:.0f}")
                zero_volume = (df['volume'] == 0).sum()
                if zero_volume > 0:
                    self.logger.warning(
                        f"Found {zero_volume:,} zero volume bars"
                    )

            # Price consistency checks
            inconsistent = (
                    (df['High'] < df['Low']) |
                    (df['Open'] > df['High']) |
                    (df['Open'] < df['Low']) |
                    (df['Close'] > df['High']) |
                    (df['Close'] < df['Low'])
            ).sum()

            if inconsistent > 0:
                self.logger.warning(
                    f"Found {inconsistent:,} bars with inconsistent OHLC values"
                )

            # Trading hours coverage
            if 'trading_hour' in df.columns:
                trading_bars = df['trading_hour'].sum()
                coverage = trading_bars / len(df)
                self.logger.info(f"\nTrading Hours Coverage: {coverage:.2%}")

        except Exception as e:
            self.logger.error(f"Failed to log data quality stats: {str(e)}")

    def _log_data_summary(self, df: pd.DataFrame) -> None:
        """Log detailed data statistics."""
        try:
            self.logger.info("\nProcessed Data Summary:")
            self.logger.info(f"Total rows: {len(df):,}")

            # Price statistics
            for col in ['Open', 'High', 'Low', 'Close']:
                stats = df[col].describe()
                self.logger.info(
                    f"{col}: mean={stats['mean']:.2f}, "
                    f"std={stats['std']:.2f}"
                )

            # Volume statistics
            if 'volume' in df.columns:
                vol_stats = df['volume'].describe()
                self.logger.info(
                    f"Volume: mean={vol_stats['mean']:.0f}, "
                    f"std={vol_stats['std']:.0f}"
                )

            # Data quality checks
            null_counts = df.isnull().sum()
            if null_counts.any():
                self.logger.warning(
                    f"\nNull values found:\n{null_counts[null_counts > 0]}"
                )

            # Trading hours coverage
            if 'trading_hour' in df.columns:
                trading_hours = df['trading_hour'].sum()
                coverage = trading_hours / len(df) * 100
                self.logger.info(
                    f"\nTrading hours coverage: {trading_hours:,} "
                    f"({coverage:.1f}% of data)"
                )

        except Exception as e:
            self.logger.error(f"Failed to log data summary: {str(e)}")

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._data_cache.clear()
        self.logger.info("Data cache cleared")


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    This class defines the interface for trading strategies, focusing on signal generation
    and exit conditions. Trade execution and position sizing are handled by TradeManager
    and RiskManager respectively.
    """

    def __init__(self, name: str, config: Config):
        """
        Initialize strategy with basic parameters.

        Args:
            name: Strategy name
            config: System configuration containing all required parameters
        """
        self.logger = LoggingConfig.get_logger(f"{__name__}.{name}")
        self.name = name
        self.config = config
        self.contract_spec = self._get_contract_spec()
        self.logger.info(f"Initialized {name} strategy")

    @staticmethod
    def _create_timezone_aware_index(df: pd.DataFrame, dates: List[datetime.date]) -> pd.DatetimeIndex:
        """
        Create timezone-aware index preserving DataFrame's timezone.

        This is the same implementation as in TradeManager to maintain consistency
        across the system.

        Args:
            df: DataFrame with timezone-aware index
            dates: List of dates to create index for

        Returns:
            DatetimeIndex with preserved timezone

        Raises:
            ValueError: If input DataFrame lacks timezone information
        """
        if df.index.tz is None:
            raise ValueError("Input DataFrame must have timezone-aware index")

        return pd.DatetimeIndex([
            pd.Timestamp(date).tz_localize(None).tz_localize(df.index.tz)
            for date in dates
        ])

    def _get_contract_spec(self) -> ContractSpecification:
        """Get contract specification from config."""
        spec_data = self.config.contract_specs[self.config.symbol]
        return ContractSpecification(
            symbol=self.config.symbol,
            tick_size=spec_data['tick_size'],
            multiplier=spec_data['multiplier'],
            margin=spec_data['margin'],
            market_open=self._parse_time(spec_data['market_open']),
            market_close=self._parse_time(spec_data['market_close']),
            last_entry=self._parse_time(spec_data['last_entry'])
        )

    def _parse_time(self, time_str: Union[str, time]) -> time:
        """Parse time string to time object."""
        if isinstance(time_str, time):
            return time_str
        return datetime.strptime(time_str, '%H:%M').time()

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on strategy logic.

        Args:
            df: DataFrame with market data

        Returns:
            DataFrame with added signal column (-1 for short, 0 for neutral, 1 for long)
        """
        pass

    @abstractmethod
    def check_exit_conditions(self, bar: pd.Series, position: int, entry_price: float,
                              current_time: pd.Timestamp) -> Tuple[bool, str]:
        """
        Check if current position should be exited.

        Args:
            bar: Current price bar data
            position: Current position (-1 for short, 1 for long)
            entry_price: Position entry price
            current_time: Current bar timestamp

        Returns:
            Tuple of (should_exit: bool, exit_reason: str)
        """
        pass

    def aggregate_to_daily(self, df_base: pd.DataFrame, df_levered: pd.DataFrame,
                           base_trades: List[Trade], final_trades: List[Trade]) -> pd.DataFrame:
        """
        Aggregate intraday data to daily metrics.

        This method aggregates intraday trading data into daily summary metrics,
        including price data, position metrics, P&L, and performance statistics.
        All datetime handling preserves the timezone from the input data.

        Args:
            df_base: DataFrame with base strategy data
            df_levered: DataFrame with levered strategy data
            base_trades: List of base strategy trades
            final_trades: List of levered strategy trades

        Returns:
            DataFrame with daily aggregated metrics including:
            - Price data (open, high, low, close)
            - Position data (base and levered)
            - P&L and cost data
            - Return and drawdown calculations

        Daily Aggregation Process:
        1. Initialize daily DataFrame with proper timezone
        2. Calculate daily OHLC price data
        3. Aggregate position metrics
        4. Process trade data and P&L
        5. Calculate returns and drawdowns
        """
        try:
            # Get timezone from input data - no conversion needed
            data_tz = df_base.index.tz
            if data_tz is None:
                raise ValueError("Input DataFrame must have timezone-aware index")

            # Create daily index with same timezone
            dates = sorted(set(df_base.index.date))
            daily_index = self._create_timezone_aware_index(df_base, dates)

            # Initialize daily DataFrame with timezone-aware index
            daily_data = pd.DataFrame(index=daily_index)

            # Initialize equity tracking
            daily_data['base_equity'] = self.config.initial_equity
            daily_data['equity'] = self.config.initial_equity
            base_equity = self.config.initial_equity
            final_equity = self.config.initial_equity

            # Process each trading day
            for date in dates:
                # Create date masks
                day_mask = df_base.index.date == date
                day_mask_levered = df_levered.index.date == date

                # Get day's data
                base_day_data = df_base[day_mask]
                levered_day_data = df_levered[day_mask_levered]

                # Get timestamp for current day (timezone-aware)
                daily_timestamp = pd.Timestamp(date).tz_localize(None).tz_localize(data_tz)

                # 1. Price Data
                self._aggregate_price_data(
                    daily_data=daily_data,
                    day_data=base_day_data,
                    timestamp=daily_timestamp
                )

                # 2. Position Metrics
                self._aggregate_position_metrics(
                    daily_data=daily_data,
                    base_data=base_day_data,
                    levered_data=levered_day_data,
                    timestamp=daily_timestamp
                )

                # 3. Process Base Strategy Trades
                base_day_trades = [t for t in base_trades if t.exit_time.date() == date]
                if base_day_trades:
                    daily_pnl_base = sum(t.pnl for t in base_day_trades)
                    daily_costs_base = sum(t.costs for t in base_day_trades)
                    win_count = sum(1 for t in base_day_trades if t.pnl > t.costs)

                    metrics = {
                        'base_pnl': daily_pnl_base,
                        'base_costs': daily_costs_base,
                        'base_trades': len(base_day_trades),
                        'base_win_rate': win_count / len(base_day_trades)
                    }

                    for key, value in metrics.items():
                        daily_data.loc[daily_timestamp, key] = value

                    base_equity += daily_pnl_base - daily_costs_base
                else:
                    # Initialize with zeros if no trades
                    zero_metrics = ['base_pnl', 'base_costs', 'base_trades', 'base_win_rate']
                    for metric in zero_metrics:
                        daily_data.loc[daily_timestamp, metric] = 0

                daily_data.loc[daily_timestamp, 'base_equity'] = base_equity

                # 4. Process Levered Strategy Trades
                final_day_trades = [t for t in final_trades if t.exit_time.date() == date]
                if final_day_trades:
                    daily_pnl_final = sum(t.pnl for t in final_day_trades)
                    daily_costs_final = sum(t.costs for t in final_day_trades)
                    win_count = sum(1 for t in final_day_trades if t.pnl > t.costs)

                    metrics = {
                        'pnl': daily_pnl_final,
                        'costs': daily_costs_final,
                        'trades': len(final_day_trades),
                        'win_rate': win_count / len(final_day_trades)
                    }

                    for key, value in metrics.items():
                        daily_data.loc[daily_timestamp, key] = value

                    final_equity += daily_pnl_final - daily_costs_final
                else:
                    # Initialize with zeros if no trades
                    zero_metrics = ['pnl', 'costs', 'trades', 'win_rate']
                    for metric in zero_metrics:
                        daily_data.loc[daily_timestamp, metric] = 0

                daily_data.loc[daily_timestamp, 'equity'] = final_equity

            # 5. Calculate Returns and Drawdowns
            # Returns calculations
            daily_data['base_returns'] = daily_data['base_equity'].pct_change().fillna(0)
            daily_data['returns'] = daily_data['equity'].pct_change().fillna(0)

            # High water mark and drawdown calculations
            daily_data['base_high_water_mark'] = daily_data['base_equity'].cummax()
            daily_data['high_water_mark'] = daily_data['equity'].cummax()
            daily_data['base_drawdown'] = (daily_data['base_equity'] / daily_data['base_high_water_mark']) - 1
            daily_data['drawdown'] = (daily_data['equity'] / daily_data['high_water_mark']) - 1

            # Log aggregation summary
            self._log_aggregation_summary(daily_data)

            return daily_data

        except Exception as e:
            self.logger.error(f"Daily aggregation failed: {str(e)}")
            raise

    def _aggregate_price_data(self, daily_data: pd.DataFrame, day_data: pd.DataFrame,
                              timestamp: pd.Timestamp) -> None:
        """Aggregate OHLC price data for a single day."""
        try:
            daily_data.loc[timestamp, 'close'] = day_data['Close'].iloc[-1]
            daily_data.loc[timestamp, 'open'] = day_data['Open'].iloc[0]
            daily_data.loc[timestamp, 'high'] = day_data['High'].max()
            daily_data.loc[timestamp, 'low'] = day_data['Low'].min()
        except Exception as e:
            self.logger.error(f"Price data aggregation failed: {str(e)}")
            raise

    def _aggregate_position_metrics(self, daily_data: pd.DataFrame, base_data: pd.DataFrame,
                                    levered_data: pd.DataFrame, timestamp: pd.Timestamp) -> None:
        """Aggregate position metrics for a single day."""
        try:
            # Base strategy position metrics
            base_positions = base_data['base_position_size']
            metrics = {
                'base_position': base_positions.iloc[-1],
                'max_base_position': abs(base_positions).max(),
                'avg_base_position': abs(base_positions).mean()
            }

            for key, value in metrics.items():
                daily_data.loc[timestamp, key] = value

            # Levered strategy position metrics
            levered_positions = levered_data['position_size']
            metrics = {
                'position': levered_positions.iloc[-1],
                'max_position': abs(levered_positions).max(),
                'avg_position': abs(levered_positions).mean()
            }

            for key, value in metrics.items():
                daily_data.loc[timestamp, key] = value
        except Exception as e:
            self.logger.error(f"Position metrics aggregation failed: {str(e)}")
            raise

    def _log_aggregation_summary(self, daily_data: pd.DataFrame) -> None:
        """Log summary of daily aggregation results."""
        try:
            self.logger.info("\nDaily Aggregation Summary:")
            self.logger.info(f"Period: {daily_data.index[0].date()} to {daily_data.index[-1].date()}")
            self.logger.info(f"Total trading days: {len(daily_data)}")

            # Return statistics
            total_return = (daily_data['equity'].iloc[-1] / daily_data['equity'].iloc[0] - 1) * 100
            base_return = (daily_data['base_equity'].iloc[-1] / daily_data['base_equity'].iloc[0] - 1) * 100

            self.logger.info("\nPerformance Summary:")
            self.logger.info(f"Base strategy return: {base_return:.2f}%")
            self.logger.info(f"Levered strategy return: {total_return:.2f}%")

            # Drawdown statistics
            max_dd = daily_data['drawdown'].min() * 100
            max_base_dd = daily_data['base_drawdown'].min() * 100

            self.logger.info("\nRisk Metrics:")
            self.logger.info(f"Max base drawdown: {max_base_dd:.2f}%")
            self.logger.info(f"Max levered drawdown: {max_dd:.2f}%")

            # Trade statistics
            total_trades = daily_data['trades'].sum()
            total_base_trades = daily_data['base_trades'].sum()

            self.logger.info("\nTrade Statistics:")
            self.logger.info(f"Total base trades: {total_base_trades:.0f}")
            self.logger.info(f"Total levered trades: {total_trades:.0f}")

        except Exception as e:
            self.logger.error(f"Failed to log aggregation summary: {str(e)}")


class RiskMetrics:
    """Calculates and manages risk metrics for trading strategies."""

    def __init__(self, config: Config):
        self.logger = LoggingConfig.get_logger(__name__)

        # Initialize with config attribute access
        self.windows = {
            'short_term': 21,
            'medium_term': 63,
            'long_term': 252
        }
        self.min_periods = 5
        self.required_windows = ['short', 'medium']
        self.risk_free_rate = 0.02

        self._validate_config()
        self._reset_caches()

    def _validate_required_metrics(self, metrics: pd.DataFrame) -> None:
        """
        Validate that all required metrics are present and valid.

        Args:
            metrics: DataFrame of calculated metrics

        Raises:
            ValueError: If required metrics are missing
        """
        for window in self.required_windows:
            required_metrics = [
                f'vol_{window}',
                f'return_{window}',
                f'sharpe_{window}'
            ]

            missing = [m for m in required_metrics if m not in metrics.columns]
            if missing:
                raise ValueError(f"Missing required metrics: {missing}")

            # Check for invalid values
            for metric in required_metrics:
                if metrics[metric].isnull().all():
                    raise ValueError(f"No valid values for required metric: {metric}")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate window sizes
        for window_name, size in self.windows.items():
            if size <= 0:
                raise ValueError(f"Window size must be positive: {window_name}={size}")
            if size < self.min_periods:
                raise ValueError(
                    f"Window size ({size}) cannot be smaller than min_periods ({self.min_periods})"
                )

        # Validate required windows exist
        missing_windows = set(self.required_windows) - set(name.split('_')[0]
                                                           for name in self.windows.keys())
        if missing_windows:
            raise ValueError(f"Required windows not found in configuration: {missing_windows}")

        # Validate risk-free rate
        if not 0 <= self.risk_free_rate <= 1:
            raise ValueError(f"Risk-free rate must be between 0 and 1: {self.risk_free_rate}")

    def _reset_caches(self) -> None:
        """Reset calculation caches."""
        self._cache = {
            'rolling': None,
            'summary': None,
            'intermediate': {}  # For storing intermediate calculations
        }

    def calculate_rolling_metrics(self, returns_obj: Union[pd.Series, BaseReturns, LeveredReturns]) -> MetricResult:
        """
        Calculate rolling metrics from either a Series or returns object with enhanced validation.

        Args:
            returns_obj: Either a returns Series or a BaseReturns/LeveredReturns object

        Returns:
            MetricResult containing rolling metrics DataFrame and calculation metadata
        """
        try:
            self.logger.info("Starting rolling metrics calculation")
            warning_messages = []

            # Get returns series based on input type with validation
            if isinstance(returns_obj, (BaseReturns, LeveredReturns)):
                returns = returns_obj.returns
            elif isinstance(returns_obj, pd.Series):
                returns = returns_obj
            else:
                raise ValueError(f"Unsupported returns type: {type(returns_obj)}")

            # Validate data
            if returns.empty:
                raise ValueError("Empty returns series provided")

            # Log initial state
            initial_days = len(returns)
            self.logger.info(f"Initial returns data: {initial_days} days")
            self.logger.info(f"Returns index is unique: {returns.index.is_unique}")

            # Handle duplicate dates with logging
            if not returns.index.is_unique:
                duplicate_count = returns.index.duplicated().sum()
                self.logger.info(f"Found {duplicate_count} duplicate dates in returns")
                duplicated_idx = returns.index[returns.index.duplicated(keep=False)]
                if len(duplicated_idx) > 0:
                    self.logger.info("Example duplicate dates and values:")
                    for idx in duplicated_idx[:5]:
                        self.logger.info(f"Date: {idx}")
                        self.logger.info(f"Values: {returns[returns.index == idx].values}")

            # Clean duplicates
            returns = returns[~returns.index.duplicated(keep='first')]
            if len(returns) != initial_days:
                self.logger.info(f"After removing duplicates: {len(returns)} days")

            # Handle NaN values
            nan_mask = returns.isna()
            nan_count = nan_mask.sum()
            if nan_count > 0:
                self.logger.info(f"Found {nan_count} NaN values")

            # Filter to valid data points
            returns = returns[~returns.isna()]
            available_days = len(returns)

            self.logger.info(f"After all filtering: {available_days} days")
            if initial_days != available_days:
                self.logger.info(f"Total filtered out: {initial_days - available_days} days")

            self.logger.info(f"Processing {available_days} days of returns data")

            # Initialize metrics DataFrame
            metrics = pd.DataFrame(index=returns.index)

            # Calculate windows and log
            windows = self._get_windows(available_days)
            self.logger.info(f"Using windows: {list(windows.keys())} with {available_days} available days")

            # Calculate metrics for each window
            for window_name, window_size in windows.items():
                if window_size > available_days:
                    warning_messages.append(f"Window size {window_size} larger than available data {available_days}")
                    continue

                min_periods = min(max(self.min_periods, window_size // 4), window_size)

                # Volatility calculation with proper annualization
                vol = returns.rolling(
                    window=window_size,
                    min_periods=min_periods
                ).std() * np.sqrt(252)
                metrics[f'vol_{window_name}'] = vol

                # Returns calculation with compound effect consideration
                roll_returns = returns.rolling(
                    window=window_size,
                    min_periods=min_periods
                ).apply(lambda x: np.prod(1 + x) ** (252 / window_size) - 1, raw=True)

                metrics[f'return_{window_name}'] = roll_returns

                # Sharpe ratio with proper handling of zero volatility
                excess_returns = roll_returns - self.risk_free_rate
                metrics[f'sharpe_{window_name}'] = (
                        excess_returns / vol.replace(0, np.nan)
                ).fillna(0)

            # Calculate proper equity curve and drawdown
            equity = (1 + returns).cumprod()
            metrics['drawdown'] = equity / equity.cummax() - 1
            metrics['high_water_mark'] = equity.cummax()

            return MetricResult(
                value=metrics,
                calculation_time=pd.Timestamp.now(),
                metric_type=MetricType.ROLLING,
                input_rows=available_days,
                warnings=warning_messages
            )

        except Exception as e:
            self.logger.error(f"Rolling metrics calculation failed: {str(e)}")
            raise

    def calculate_summary_metrics(self, returns_obj: Union[pd.Series, BaseReturns, LeveredReturns]) -> Dict[str, float]:
        """
        Calculate summary metrics with proper compound returns handling.

        Args:
            returns_obj: Either a returns Series or a BaseReturns/LeveredReturns object

        Returns:
            Dictionary of summary metrics including total return, Sharpe ratio, etc.
        """
        try:
            if self._cache['summary'] is not None:
                return self._cache['summary']

            # Extract returns series based on input type
            if isinstance(returns_obj, (BaseReturns, LeveredReturns)):
                returns = returns_obj.returns
            elif isinstance(returns_obj, pd.Series):
                returns = returns_obj
            else:
                raise TypeError(f"Unsupported returns data type: {type(returns_obj)}")

            # Calculate proper compound equity curve
            equity = (1 + returns).cumprod()
            trading_days = len(returns)
            years = trading_days / 252

            # Calculate metrics using compound returns
            total_return = equity.iloc[-1] - 1
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            volatility = returns.std() * np.sqrt(252)

            excess_return = annualized_return - self.risk_free_rate
            sharpe = excess_return / volatility if volatility > 0 else 0

            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

            summary = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'sortino_ratio': excess_return / downside_vol if downside_vol > 0 else 0,
                'max_drawdown': (equity / equity.cummax() - 1).min(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurt(),
                'var_95': returns.quantile(0.05),
                'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
                'best_return': returns.max(),
                'worst_return': returns.min(),
                'avg_return': returns.mean(),
                'avg_pos_return': returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0,
                'avg_neg_return': returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0,
                'pos_return_ratio': (returns > 0).mean(),
                'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum())
                    if returns[returns < 0].sum() != 0 else float('inf'),
                'trading_days': trading_days,
                'years': years
            }

            self._cache['summary'] = summary
            return summary

        except Exception as e:
            self.logger.error(f"Summary metrics calculation failed: {str(e)}")
            raise

    def calculate_metrics(self, returns_data: Union[pd.Series, BaseReturns, LeveredReturns],
                          metric_type: MetricType = MetricType.ROLLING,
                          caller: str = "") -> Union[MetricResult, Dict[str, float]]:
        """Calculate either rolling or summary metrics."""
        try:
            self.logger.info(f"\nMetrics calculation requested by: {caller}")
            self.logger.info(f"Metric type: {metric_type}")

            # Extract returns series based on input type
            if isinstance(returns_data, (BaseReturns, LeveredReturns)):
                returns = returns_data.returns
            elif isinstance(returns_data, pd.Series):
                returns = returns_data
            else:
                raise TypeError(f"Unsupported returns data type: {type(returns_data)}")

            self.logger.info(f"Returns length: {len(returns)}")
            self.logger.info(f"Returns date range: {returns.index.min()} to {returns.index.max()}")

            if metric_type == MetricType.ROLLING:
                return self.calculate_rolling_metrics(returns)
            else:
                return self.calculate_summary_metrics(returns)

        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {str(e)}")
            raise

    def _calculate_window_metrics(self, metrics: pd.DataFrame, returns: pd.Series,
                                  window_name: str, window_size: int,
                                  min_periods: int) -> pd.DataFrame:
        """
        Calculate all metrics for a specific window.

        Args:
            metrics: DataFrame to store results
            returns: Return series to analyze
            window_name: Name of the window period
            window_size: Size of the rolling window
            min_periods: Minimum periods required

        Returns:
            Updated metrics DataFrame
        """
        # Calculate volatility
        vol = self._calculate_volatility(returns, window_size, min_periods)
        metrics[f'vol_{window_name}'] = vol

        # Calculate compound returns
        ret = self._calculate_rolling_returns(returns, window_size, min_periods)
        metrics[f'return_{window_name}'] = ret

        # Calculate excess returns
        excess_ret = ret - (self.risk_free_rate / 252)  # Daily excess return
        metrics[f'excess_return_{window_name}'] = excess_ret

        # Calculate Sharpe ratio
        metrics[f'sharpe_{window_name}'] = self._calculate_rolling_sharpe(
            excess_ret, vol
        )

        # Calculate Sortino ratio
        metrics[f'sortino_{window_name}'] = self._calculate_rolling_sortino(
            returns, excess_ret, window_size
        )

        return metrics

    def _calculate_volatility(self, returns: pd.Series, window: int,
                              min_periods: int) -> pd.Series:
        """
        Calculate rolling annualized volatility.

        Args:
            returns: Return series
            window: Rolling window size
            min_periods: Minimum periods required

        Returns:
            Series of annualized volatility values
        """
        cache_key = f'vol_{window}'
        if cache_key in self._cache['intermediate']:
            return self._cache['intermediate'][cache_key]

        vol = returns.rolling(
            window=window,
            min_periods=min_periods
        ).std() * np.sqrt(252)

        self._cache['intermediate'][cache_key] = vol
        return vol

    def _calculate_rolling_returns(self, returns: pd.Series, window: int,
                                   min_periods: int) -> pd.Series:
        """
        Calculate rolling compound returns.

        Args:
            returns: Return series
            window: Rolling window size
            min_periods: Minimum periods required

        Returns:
            Series of rolling returns
        """
        cache_key = f'returns_{window}'
        if cache_key in self._cache['intermediate']:
            return self._cache['intermediate'][cache_key]

        roll_returns = (1 + returns).rolling(
            window=window,
            min_periods=min_periods
        ).apply(lambda x: np.prod(1 + x) ** (252 / window) - 1, raw=True)

        self._cache['intermediate'][cache_key] = roll_returns
        return roll_returns

    def _calculate_rolling_sharpe(self, excess_returns: pd.Series,
                                  volatility: pd.Series) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.

        Args:
            excess_returns: Excess return series
            volatility: Volatility series

        Returns:
            Series of Sharpe ratios
        """
        return np.where(volatility > 0, excess_returns / volatility, 0)

    def _calculate_rolling_sortino(self, returns: pd.Series, excess_returns: pd.Series,
                                   window: int) -> pd.Series:
        """
        Calculate rolling Sortino ratio.

        Args:
            returns: Return series
            excess_returns: Excess return series
            window: Rolling window size

        Returns:
            Series of Sortino ratios
        """
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0

        downside_vol = downside_returns.rolling(
            window=window,
            min_periods=self.min_periods  # Use class min_periods
        ).std() * np.sqrt(252)

        return np.where(downside_vol > 0, excess_returns / downside_vol, 0)

    def _calculate_drawdown(self, equity: pd.Series) -> pd.Series:
        """
        Calculate drawdown series.

        Args:
            equity: Equity curve series

        Returns:
            Series of drawdown values
        """
        high_water_mark = equity.cummax()
        return equity / high_water_mark - 1

    def _validate_input_data(self, daily_performance: pd.DataFrame) -> None:
        """Validate input data for risk metrics calculation."""
        if not isinstance(daily_performance, pd.DataFrame):
            raise ValueError("daily_performance must be a DataFrame")

        required_columns = {'returns', 'equity'}
        missing_columns = required_columns - set(daily_performance.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if daily_performance['returns'].isnull().any():
            raise ValueError("Found null values in returns series")

        if len(daily_performance) == 0:
            raise ValueError("Empty daily performance data")

    def _get_windows(self, available_days: Optional[int] = None) -> Dict[str, int]:
        """Get dictionary of usable window names and sizes."""
        # Start with base windows
        base_windows = {
            'short': self.windows['short_term'],
            'medium': self.windows['medium_term'],
        }

        # Only add long window if we have enough data
        if available_days and available_days >= self.windows['long_term']:
            base_windows['long'] = self.windows['long_term']

        if available_days:
            # Keep required windows regardless of availability
            windows = {
                name: size for name, size in base_windows.items()
                if name in self.required_windows or available_days >= size
            }
            self.logger.info(f"Using windows: {list(windows.keys())} "
                             f"with {available_days} available days")
            return windows

        return base_windows

    def clear_cache(self) -> None:
        """Clear all calculation caches."""
        self._reset_caches()


class RiskManager(ABC):
    """Abstract base class for risk management."""

    def __init__(self, config: Config, risk_metrics: RiskMetrics):
        self.logger = LoggingConfig.get_logger(__name__)
        self.config = config
        self.risk_metrics = risk_metrics
        self.risk_limits = RiskLimits(**config.risk_limits)
        self.contract_spec = self._get_contract_spec()

        # Get contract specification for the configured symbol
        self.contract_spec = ContractSpecification(
            symbol=config.symbol,
            tick_size=config.contract_specs[config.symbol]['tick_size'],
            multiplier=config.contract_specs[config.symbol]['multiplier'],
            margin=config.contract_specs[config.symbol]['margin'],
            market_open=self._parse_time(config.contract_specs[config.symbol]['market_open']),
            market_close=self._parse_time(config.contract_specs[config.symbol]['market_close']),
            last_entry=self._parse_time(config.contract_specs[config.symbol]['last_entry'])
        )

        # Initialize risk tracking
        self.current_drawdown = 0.0
        self.peak_equity = config.initial_equity
        self.daily_pnl = 0.0

    def _get_contract_spec(self) -> ContractSpecification:
        """Get contract specification from config."""
        spec_data = self.config.contract_specs[self.config.symbol]
        return ContractSpecification(
            symbol=self.config.symbol,
            tick_size=spec_data['tick_size'],
            multiplier=spec_data['multiplier'],
            margin=spec_data['margin'],
            market_open=self._parse_time(spec_data['market_open']),
            market_close=self._parse_time(spec_data['market_close']),
            last_entry=self._parse_time(spec_data['last_entry'])
        )

    def prepare_base_strategy_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame with base position sizes for evaluating strategy."""
        try:
            df = df.copy()

            # Calculate fixed base position size - no risk limits applied
            base_size = self.calculate_base_position_size(self.config.initial_equity)

            # Set fixed position sizes from signals
            df['base_position_size'] = df['signal'] * base_size

            # Add signal verification
            signal_times = df[df['signal'] != 0].index
            self.logger.info(f"\nVerifying signals:")
            self.logger.info(f"First signal time: {signal_times[0]}")
            self.logger.info(f"Last signal time: {signal_times[-1]}")
            self.logger.info(f"Sample of signals:")
            for time in signal_times[:5]:
                self.logger.info(f"  {time}: {df.loc[time, 'signal']}")

            # Initialize tracking columns
            df['position'] = 0
            df['position_size'] = df['base_position_size']
            df['current_equity'] = self.config.initial_equity  # Fixed for base strategy
            df['base_equity'] = self.config.initial_equity  # Fixed for base strategy

            # Log statistics about signals and positions
            self.logger.info("\nStep 3a Statistics:")
            self.logger.info("-" * 30)
            self.logger.info(f"Initial equity: ${self.config.initial_equity:,.2f}")
            self.logger.info(f"Base position size: {base_size:.2f} contracts")

            # Signal statistics
            total_signals = (df['signal'] != 0).sum()
            long_signals = (df['signal'] > 0).sum()
            short_signals = (df['signal'] < 0).sum()
            self.logger.info(f"\nSignal Distribution:")
            self.logger.info(f"Total signals: {total_signals}")
            self.logger.info(f"Long signals: {long_signals}")
            self.logger.info(f"Short signals: {short_signals}")

            # Position statistics
            self.logger.info(f"\nPosition Statistics:")
            self.logger.info(f"Fixed long position: {base_size:.2f} contracts")
            self.logger.info(f"Fixed short position: {-base_size:.2f} contracts")

            return df

        except Exception as e:
            self.logger.error(f"Base position preparation failed: {str(e)}")
            raise

    def calculate_base_position_size(self, initial_equity: float) -> float:
        """
        Calculate base position size using initial equity and contract margin.
        For base strategy, this is a fixed size based solely on initial equity.
        """
        try:
            # Simple calculation based on initial equity and margin only
            base_size = initial_equity / self.contract_spec.margin

            self.logger.info(f"\nBase Position Size Calculation:")
            self.logger.info(f"Initial equity: ${initial_equity:,.2f}")
            self.logger.info(f"Contract margin: ${self.contract_spec.margin:,.2f}")
            self.logger.info(f"Base size: {base_size:.2f} contracts")

            return base_size

        except Exception as e:
            self.logger.error(f"Base position size calculation failed: {str(e)}")
            return 1.0  # Conservative default

    def _parse_time(self, time_str: Union[str, time]) -> time:
        """Parse time string to time object if needed."""
        if isinstance(time_str, time):
            return time_str
        return datetime.strptime(time_str, '%H:%M').time()

    def initialize_risk_tracking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initialize risk tracking columns in DataFrame."""
        df = df.copy()

        risk_columns = {
            'volatility': 0.0,  # Rolling volatility
            'risk_scaling': 1.0,  # Risk-based position scaling
            'risk_adjusted_size': 0.0,  # Risk-adjusted position size
            'exposure_pct': 0.0,  # Position exposure as % of equity
            'margin_used_pct': 0.0  # Margin utilization percentage
        }

        for col, default in risk_columns.items():
            if col not in df.columns:
                df[col] = default

        return df

    @abstractmethod
    def calculate_position_sizes(self, daily_performance: pd.DataFrame,
                                 risk_metrics: pd.DataFrame) -> pd.Series:
        """Calculate position size multipliers based on risk metrics."""
        pass

    def apply_position_sizing(self, df: pd.DataFrame, position_sizes: pd.Series) -> pd.DataFrame:
        """
        Apply position sizing with strict index alignment and validation.

        Args:
            df: DataFrame with base positions
            position_sizes: Series of position size multipliers

        Returns:
            DataFrame with applied position sizing
        """
        try:
            df = df.copy()

            # Validate indexes
            if not df.index.equals(position_sizes.index):
                self.logger.warning("Position sizes index mismatch - attempting realignment")
                if len(position_sizes) != len(df):
                    self.logger.warning(f"Length mismatch - positions: {len(position_sizes)}, df: {len(df)}")

                # Ensure position_sizes aligns with df
                position_sizes = position_sizes.reindex(df.index, method='ffill')

                # Validate after realignment
                if position_sizes.isna().any():
                    self.logger.warning("NaN values after realignment - filling with 1.0")
                    position_sizes = position_sizes.fillna(1.0)

            # Initialize tracking
            current_equity = self.config.initial_equity
            df['current_equity'] = current_equity
            df['leverage'] = position_sizes

            # Process each day
            for date in pd.Series(df.index.date).unique():
                day_mask = df.index.date == date

                # Calculate position limits
                margin_factor = 0.9  # Safety margin
                max_contracts = (current_equity * margin_factor) / self.contract_spec.margin

                # Get day's data
                day_positions = df.loc[day_mask, 'base_position_size']
                day_sizing = position_sizes[day_mask]

                # Calculate and apply position sizes
                df.loc[day_mask, 'position_size'] = (
                        day_positions *
                        day_sizing *
                        (current_equity / self.config.initial_equity)
                ).clip(-max_contracts, max_contracts)

                # Calculate exposure
                df.loc[day_mask, 'exposure_pct'] = (
                        abs(df.loc[day_mask, 'position_size']) *
                        self.contract_spec.margin /
                        current_equity * 100
                )

                # Update equity if we have P&L
                if 'realized_pnl' in df.columns:
                    day_pnl = df.loc[day_mask, 'realized_pnl'].sum()
                    current_equity += day_pnl

            # Final validation
            if df['position_size'].isna().any():
                self.logger.error("NaN values in final position sizes")
                df['position_size'] = df['position_size'].fillna(0.0)

            # Log statistics
            self.logger.info("\nPosition Sizing Application Statistics:")
            self.logger.info(f"Average position size: {abs(df['position_size']).mean():.2f}")
            self.logger.info(f"Max position size: {abs(df['position_size']).max():.2f}")
            self.logger.info(f"Average exposure: {df['exposure_pct'].mean():.2f}%")
            self.logger.info(f"Max exposure: {df['exposure_pct'].max():.2f}%")

            return df

        except Exception as e:
            self.logger.error(f"Position sizing application failed: {str(e)}", exc_info=True)
            return df

    def check_risk_limits(self, df: pd.DataFrame) -> List[str]:
        """Check risk limits based on base positions only."""
        violations = []

        # Check against initial equity since this is base strategy
        position_exposure = (
                abs(df['base_position_size']) *
                self.contract_spec.margin /
                self.config.initial_equity
        )

        # Position size limit check
        if (position_exposure > self.risk_limits.position_limit_pct).any():
            violations.append(
                f"Position size exceeds {self.risk_limits.position_limit_pct:.1%} "
                f"of equity limit"
            )

        # Concentration limit check
        if (position_exposure > self.risk_limits.concentration_limit).any():
            violations.append(
                f"Position concentration exceeds {self.risk_limits.concentration_limit:.1%} limit"
            )

        # Daily loss limit check (as percentage of initial equity)
        daily_pnl = df.groupby(df.index.date)['pnl'].sum()
        max_daily_loss_pct = abs(daily_pnl.min() / self.config.initial_equity)
        if max_daily_loss_pct > self.risk_limits.max_daily_loss:
            violations.append(
                f"Daily loss of {max_daily_loss_pct:.1%} exceeds "
                f"{self.risk_limits.max_daily_loss:.1%} limit"
            )

        return violations

    def update_risk_metrics(self, equity: float, timestamp: pd.Timestamp) -> None:
        """Update risk tracking metrics."""
        # Update peak equity and drawdown
        self.peak_equity = max(self.peak_equity, equity)
        self.current_drawdown = (equity - self.peak_equity) / self.peak_equity

        # Reset daily P&L at market open
        if timestamp.time() == self.config.market_open:
            self.daily_pnl = 0.0

    @abstractmethod
    def get_risk_summary(self) -> dict:
        """Get current risk metrics summary."""
        pass

    def _validate_position_limits(self, position_size: float, current_equity: float) -> float:
        """Validate and adjust position size based on limits."""
        try:
            # Calculate maximum position based on margin requirements
            margin_max = (current_equity * self.risk_limits.position_limit_pct) / self.contract_spec.margin

            # Calculate maximum position based on concentration limit
            concentration_max = (current_equity * self.risk_limits.concentration_limit) / \
                                (self.contract_spec.multiplier * self.contract_spec.margin)

            # Take the minimum of all limits
            max_position = min(margin_max, concentration_max, self.risk_limits.max_position_size)

            # Apply limits
            return np.clip(position_size, -max_position, max_position)

        except Exception as e:
            self.logger.error(f"Position limit validation failed: {str(e)}")
            return 0.0

    def _calculate_exposure_metrics(self, position_size: float, price: float,
                                    current_equity: float) -> Dict[str, float]:
        """Calculate exposure and risk metrics for a position."""
        try:
            notional_exposure = abs(position_size * price * self.contract_spec.multiplier)
            margin_used = abs(position_size * self.contract_spec.margin)

            return {
                'notional_exposure': notional_exposure,
                'exposure_pct': (notional_exposure / current_equity) * 100,
                'margin_used': margin_used,
                'margin_used_pct': (margin_used / current_equity) * 100
            }

        except Exception as e:
            self.logger.error(f"Exposure calculation failed: {str(e)}")
            return {}

    def _apply_position_limits(self, position_sizes: pd.Series, equity: pd.Series) -> pd.Series:
        """Apply position limits based on risk parameters."""
        # Calculate maximum position size based on equity percentage
        equity_based_limit = equity * self.risk_limits.position_limit_pct

        # Apply both absolute and relative limits
        return position_sizes.clip(
            upper=np.minimum(
                self.risk_limits.max_position_size,
                equity_based_limit
            )
        )


class RiskManagerFactory:
    """Factory for creating risk manager instances."""

    @staticmethod
    def create(config: Config, risk_metrics: RiskMetrics) -> RiskManager:
        """
        Create a risk manager instance based on configuration.

        Args:
            config: Complete configuration object
            risk_metrics: Initialized RiskMetrics instance

        Returns:
            Configured risk manager instance

        Notes:
            - Uses config.risk_params['risk_manager_type'] to determine manager type
            - Available types: 'volatility', 'sharpe', 'adaptive', 'combined'
            - Defaults to VolatilityTargetRiskManager if type is unknown
        """
        risk_config = config.get_risk_manager_config()
        manager_type = risk_config['type'].lower()

        logger = LoggingConfig.get_logger(__name__)
        logger.info(f"Creating risk manager of type: {manager_type}")

        if manager_type == 'volatility':
            return VolatilityTargetRiskManager(config, risk_metrics)
        elif manager_type == 'sharpe':
            return SharpeRatioRiskManager(config, risk_metrics)
        elif manager_type == 'adaptive':
            return AdaptiveRiskManager(config, risk_metrics)
        elif manager_type == 'combined':
            # Create all managers for combination
            vol_manager = VolatilityTargetRiskManager(config, risk_metrics)
            sharpe_manager = SharpeRatioRiskManager(config, risk_metrics)
            adaptive_manager = AdaptiveRiskManager(config, risk_metrics)

            # Get weights from config or use default equal weights
            weights = risk_config.get('combined_weights', [0.4, 0.3, 0.3])

            return CombinedRiskManager(
                config=config,
                risk_metrics=risk_metrics,
                managers=[vol_manager, sharpe_manager, adaptive_manager],
                weights=weights
            )
        else:
            logger.warning(f"Unknown risk manager type: {manager_type}. Using volatility targeting.")
            return VolatilityTargetRiskManager(config, risk_metrics)

class VolatilityTargetRiskManager(RiskManager):
    """Concrete implementation of RiskManager using volatility targeting."""

    def __init__(self, config: Config, risk_metrics: RiskMetrics):
        super().__init__(config, risk_metrics)

        # Add debug logging for config values
        self.logger.debug(f"Raw volatility params from config: {config.volatility_params}")

        # Extract and validate volatility parameters from config
        vol_params = {
            'target_volatility': float(config.volatility_params.get('target_volatility', 0.15)),
            'estimation_window': int(config.volatility_params.get('estimation_window', 63)),
            'min_scaling': float(config.volatility_params.get('min_scaling', 0.5)),
            'max_scaling': float(config.volatility_params.get('max_scaling', 2.0)),
            'adaptation_rate': float(config.volatility_params.get('adaptation_rate', 0.1)),
            'vol_target_range': config.volatility_params.get('vol_target_range', (0.10, 0.20))
        }

        self.logger.info(f"Processed volatility parameters: {vol_params}")

        # Initialize with validated parameters
        self.vol_params = VolatilityParams(**vol_params)
        self.current_vol = None
        self.current_scaling = 1.0

        self.logger.info(f"Initialized VolatilityTargetRiskManager with parameters: {vol_params}")

    def _calculate_adaptation(self, risk_metrics: pd.DataFrame) -> float:
        """Calculate volatility adaptation factor."""
        if 'rolling_vol' in risk_metrics.columns:
            current_vol = risk_metrics['rolling_vol'].iloc[-1]
            target_vol = self.vol_params.target_volatility
            adaptation = np.exp(-self.vol_params.adaptation_rate *
                                (current_vol / target_vol - 1))
            return adaptation
        return 1.0

    def calculate_position_sizes(self, df: pd.DataFrame, risk_metrics: pd.DataFrame) -> pd.Series:
        """Calculate position size multipliers using volatility targeting."""
        try:
            # First calculate realized volatility if not available
            if 'vol_medium' not in risk_metrics.columns:
                self.logger.error("vol_medium not found in risk metrics")
                return pd.Series(1.0, index=df.index)

            # Get volatility and ensure it's positive
            daily_vol = risk_metrics['vol_medium']
            daily_target = self.vol_params.target_volatility

            self.logger.info(f"\nVolatility Analysis:")
            self.logger.info(f"Raw vol stats - mean: {daily_vol.mean():.2%}, "
                             f"min: {daily_vol.min():.2%}, max: {daily_vol.max():.2%}")

            # Replace zero/negative values with target vol
            daily_vol = daily_vol.replace(0, daily_target)
            daily_vol = daily_vol.mask(daily_vol < 0, daily_target)

            # Calculate scaling - increase position size when vol is low, decrease when high
            daily_scaling = pd.Series(
                daily_target / daily_vol,
                index=risk_metrics.index
            )

            # Create position sizes Series aligned with df index
            position_sizes = pd.Series(index=df.index, dtype=float)

            # Forward fill daily scaling to intraday points
            for day in pd.Series(df.index.date).unique():
                day_mask = df.index.date == day
                day_scaling = daily_scaling[daily_scaling.index.date == day]

                if not day_scaling.empty:
                    position_sizes.loc[day_mask] = day_scaling.iloc[0]
                else:
                    position_sizes.loc[day_mask] = 1.0

            # Apply scaling limits
            position_sizes = position_sizes.clip(
                lower=self.vol_params.min_scaling,
                upper=self.vol_params.max_scaling
            )

            # Add debugging info
            self.logger.info(f"\nPosition Sizing Details:")
            self.logger.info(f"Target volatility: {daily_target:.2%}")
            self.logger.info(f"Average daily vol: {daily_vol.mean():.2%}")
            self.logger.info(f"Average daily scaling: {daily_scaling.mean():.2f}")
            self.logger.info(f"Scaling stats - mean: {position_sizes.mean():.2f}, "
                             f"min: {position_sizes.min():.2f}, max: {position_sizes.max():.2f}")

            return position_sizes

        except Exception as e:
            self.logger.error(f"Position size calculation failed: {str(e)}")
            self.logger.error(f"Error details: ", exc_info=True)
            return pd.Series(1.0, index=df.index)

    def _calculate_volatility_scaling(self, risk_metrics: pd.DataFrame) -> pd.Series:
        """Calculate volatility-based position scaling with safety checks."""
        try:
            # Check if rolling_vol exists
            if 'rolling_vol' not in risk_metrics.columns:
                self.logger.warning("Rolling volatility not found in risk metrics, using default scaling of 1.0")
                return pd.Series(1.0, index=risk_metrics.index)

            current_vol = risk_metrics['rolling_vol']

            # Handle zero or NaN volatility
            current_vol = current_vol.replace(0, np.nan)
            current_vol = current_vol.fillna(self.vol_params.target_volatility)

            target_vol = self.vol_params.target_volatility

            # Ensure volatility stays within target range
            min_vol, max_vol = self.vol_params.vol_target_range
            target_vol = np.clip(target_vol, min_vol, max_vol)

            # Calculate scaling with safety checks
            scaling = np.where(
                current_vol > 0,
                target_vol / current_vol,
                1.0
            )

            # Clip scaling to prevent extreme values
            scaling = np.clip(
                scaling,
                self.vol_params.min_scaling,
                self.vol_params.max_scaling
            )

            return pd.Series(scaling, index=risk_metrics.index)

        except Exception as e:
            self.logger.error(f"Volatility scaling calculation failed: {str(e)}")
            # Return safe default scaling of 1.0 in case of errors
            return pd.Series(1.0, index=risk_metrics.index)

    def _calculate_drawdown_adjustment(self, risk_metrics: pd.DataFrame) -> pd.Series:
        """Calculate drawdown-based position adjustment."""
        try:
            # Use drawdown from risk metrics if available
            if 'drawdown' in risk_metrics.columns:
                drawdown = risk_metrics['drawdown']
            else:
                self.logger.warning("Drawdown not found in risk metrics, using default adjustment of 1.0")
                return pd.Series(1.0, index=risk_metrics.index)

            # Calculate adjustment factor (example: linear scaling based on drawdown)
            max_drawdown = self.risk_limits.max_drawdown
            adjustment = 1.0 + (drawdown / max_drawdown)  # Linear reduction based on drawdown

            # Ensure adjustment stays within reasonable bounds
            adjustment = adjustment.clip(0.5, 1.0)  # Never reduce position by more than 50%

            return adjustment

        except Exception as e:
            self.logger.error(f"Drawdown adjustment calculation failed: {str(e)}")
            return pd.Series(1.0, index=risk_metrics.index)

    def get_risk_summary(self) -> dict:
        """Get current risk metrics summary."""
        return {
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity,
            'daily_pnl': self.daily_pnl,
            'current_volatility': self.current_vol if self.current_vol is not None else 0.0,
            'current_scaling': self.current_scaling,
            'target_volatility': self.vol_params.target_volatility,
            'vol_target_range': self.vol_params.vol_target_range,
            'adaptation_rate': self.vol_params.adaptation_rate,
            'risk_limits': {
                'max_position_size': self.risk_limits.max_position_size,
                'max_daily_loss': self.risk_limits.max_daily_loss,
                'max_drawdown': self.risk_limits.max_drawdown,
                'position_limit_pct': self.risk_limits.position_limit_pct,
                'concentration_limit': self.risk_limits.concentration_limit
            }
        }


class SharpeRatioRiskManager(RiskManager):
    """Risk management based on Sharpe ratio and volatility targeting."""

    def __init__(self, config: Config, risk_metrics: RiskMetrics):
        super().__init__(config, risk_metrics)  # This will initialize contract_spec

        # Initialize Sharpe parameters
        sharpe_params = {
            'target_sharpe': float(config.sharpe_params.get('target_sharpe', 1.0)),
            'min_scaling': float(config.sharpe_params.get('min_scaling', 0.5)),
            'max_scaling': float(config.sharpe_params.get('max_scaling', 2.0)),
            'target_volatility': float(config.sharpe_params.get('target_volatility', 0.15)),
            'min_trades': int(config.sharpe_params.get('min_trades', 5)),
            'risk_free_rate': float(config.sharpe_params.get('risk_free_rate', 0.02)),
            'adaptation_rate': float(config.sharpe_params.get('adaptation_rate', 0.1)),
            'target_range': config.sharpe_params.get('target_range', (0.5, 2.0)),
            'window_type': str(config.sharpe_params.get('window_type', 'medium'))
        }
        self.params = SharpeParams(**sharpe_params)
        self.current_scaling = 1.0

    def calculate_position_sizes(self, df: pd.DataFrame, base_returns: BaseReturns) -> pd.Series:
        """Calculate position size multipliers with proper index alignment."""
        try:
            # Calculate metrics if not already done
            if base_returns.metrics is None:
                base_returns.calculate_metrics(self.risk_metrics)

            metrics_df = (base_returns.metrics.value
                          if isinstance(base_returns.metrics, MetricResult)
                          else base_returns.metrics)

            if metrics_df is None or metrics_df.empty:
                self.logger.warning("No metrics available for position sizing")
                return pd.Series(1.0, index=df.index)

            # Get required metrics
            sharpe_col = f'sharpe_{self.params.window_type}'
            vol_col = f'vol_{self.params.window_type}'

            if not all(col in metrics_df.columns for col in [sharpe_col, vol_col]):
                self.logger.warning(f"Missing required columns for sizing calculation")
                return pd.Series(1.0, index=df.index)

            # Calculate daily scaling
            daily_scaling = pd.Series(1.0, index=metrics_df.index)
            valid_mask = ~(metrics_df[sharpe_col].isna() | metrics_df[vol_col].isna())

            if valid_mask.any():
                # Get valid data
                sharpe = metrics_df.loc[valid_mask, sharpe_col].clip(-10, 10)
                vol = metrics_df.loc[valid_mask, vol_col].clip(0.0001)

                # Calculate components
                sharpe_scaling = (sharpe / self.params.target_sharpe).clip(
                    self.params.min_scaling,
                    self.params.max_scaling
                )
                vol_scaling = (self.params.target_volatility / vol).clip(
                    self.params.min_scaling,
                    self.params.max_scaling
                )

                # Apply adaptive scaling
                scaling = sharpe_scaling * vol_scaling
                scaling = scaling.ewm(alpha=self.params.adaptation_rate, adjust=False).mean()
                daily_scaling.loc[valid_mask] = scaling

            # Forward fill and enforce limits
            daily_scaling = daily_scaling.clip(
                self.params.min_scaling,
                self.params.max_scaling
            ).fillna(1.0)

            # Create position sizes aligned with df index
            position_sizes = pd.Series(index=df.index, dtype=float)

            # Map daily scaling to intraday points
            for date in daily_scaling.index:
                # Get all intraday points for this date
                day_mask = df.index.date == date.date()
                if day_mask.any():
                    position_sizes.loc[day_mask] = daily_scaling[date]

            # Fill any remaining points with 1.0
            position_sizes = position_sizes.fillna(1.0)

            # Log sizing statistics
            self.logger.info("\nPosition Sizing Statistics:")
            self.logger.info(f"Daily scaling range: [{daily_scaling.min():.2f}, {daily_scaling.max():.2f}]")
            self.logger.info(f"Average daily scaling: {daily_scaling.mean():.2f}")
            self.logger.info(f"Position sizes shape: {position_sizes.shape}")
            self.logger.info(f"NaN values in position sizes: {position_sizes.isna().sum()}")

            return position_sizes

        except Exception as e:
            self.logger.error(f"Position size calculation failed: {str(e)}", exc_info=True)
            return pd.Series(1.0, index=df.index)

    def _calculate_sharpe_scaling(self, sharpe_ratio: pd.Series) -> pd.Series:
        """Calculate scaling based on Sharpe ratio with safety checks."""
        try:
            # Replace invalid values with conservative defaults
            sharpe_ratio = sharpe_ratio.replace([np.inf, -np.inf], self.params.target_sharpe)
            sharpe_ratio = sharpe_ratio.fillna(self.params.target_sharpe)

            # Calculate basic scaling factor
            scaling = sharpe_ratio / self.params.target_sharpe

            # Apply adaptation rate for smoother transitions
            adapted_scaling = self.current_scaling + self.params.adaptation_rate * (scaling - self.current_scaling)

            # Store current scaling for next iteration
            self.current_scaling = adapted_scaling.iloc[-1] if len(adapted_scaling) > 0 else 1.0

            return adapted_scaling

        except Exception as e:
            self.logger.error(f"Sharpe scaling calculation failed: {str(e)}")
            return pd.Series(1.0, index=sharpe_ratio.index)

    def _calculate_adaptation(self, risk_metrics: pd.DataFrame) -> float:
        """Calculate Sharpe ratio adaptation factor."""
        if 'sharpe_medium' in risk_metrics.columns:
            current_sharpe = risk_metrics['sharpe_medium'].iloc[-1]
            target_sharpe = self.params.target_sharpe
            adaptation = np.exp(-self.params.adaptation_rate *
                              (current_sharpe / target_sharpe - 1))
            return adaptation
        return 1.0

    def get_risk_summary(self) -> dict:
        """Get current risk metrics summary."""
        return {
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity,
            'daily_pnl': self.daily_pnl,
            'current_sharpe': self.current_sharpe if self.current_sharpe is not None else 0.0,
            'current_scaling': self.current_scaling,
            'target_sharpe': self.params.target_sharpe,
            'target_range': self.params.target_range,
            'adaptation_rate': self.params.adaptation_rate,
            'min_trades': self.params.min_trades,
            'risk_limits': {
                'max_position_size': self.risk_limits.max_position_size,
                'max_daily_loss': self.risk_limits.max_daily_loss,
                'max_drawdown': self.risk_limits.max_drawdown,
                'position_limit_pct': self.risk_limits.position_limit_pct,
                'concentration_limit': self.risk_limits.concentration_limit
            }
        }


class AdaptiveRiskManager(RiskManager):
    """Risk management that adapts to market regimes."""

    def __init__(self, config: Config, risk_metrics: RiskMetrics):
        super().__init__(config, risk_metrics)

        # Extract and validate adaptive parameters from config
        adaptive_params = {
            'base_volatility': config.adaptive_params.get('base_volatility', 0.15),
            'regime_window': config.adaptive_params.get('regime_window', 252),
            'adaptation_rate': config.adaptive_params.get('adaptation_rate', 0.1),
            'min_scaling': config.adaptive_params.get('min_scaling', 0.5),
            'max_scaling': config.adaptive_params.get('max_scaling', 2.0),
            'vol_target_range': config.adaptive_params.get('vol_target_range', (0.10, 0.20)),
            'regime_thresholds': config.adaptive_params.get('regime_thresholds', (0.8, 1.2))
        }

        self.params = AdaptiveParams(**adaptive_params)
        self.logger.info(f"Initialized AdaptiveRiskManager with parameters: {adaptive_params}")
        self._current_regime = 'normal'

    def calculate_position_sizes(self, df: pd.DataFrame, risk_metrics: pd.DataFrame) -> pd.Series:
        """Calculate position size multipliers using regime-based adaptation."""
        try:
            if 'vol_medium' not in risk_metrics.columns:
                self.logger.error("vol_medium not found in risk metrics")
                return pd.Series(1.0, index=df.index)

            # Calculate daily volatility ratio
            daily_vol = risk_metrics['vol_medium']
            vol_ratio = daily_vol / self.params.base_volatility

            # Initialize daily scaling
            daily_scaling = pd.Series(1.0, index=risk_metrics.index)

            # Apply regime-based scaling if available
            if 'regime' in risk_metrics.columns:
                high_vol_mask = risk_metrics['regime'] == 'high_vol'
                low_vol_mask = risk_metrics['regime'] == 'low_vol'

                daily_scaling[high_vol_mask] = 1.0 / vol_ratio[high_vol_mask]
                daily_scaling[low_vol_mask] = 1.5 / vol_ratio[low_vol_mask]
                daily_scaling[~(high_vol_mask | low_vol_mask)] = 1.25 / vol_ratio[~(high_vol_mask | low_vol_mask)]

            # Forward fill to intraday by day
            intraday_scaling = pd.Series(index=df.index)
            for day in df.index.date.unique():
                day_mask = df.index.date == day
                day_scale = daily_scaling[daily_scaling.index.date == day]
                if not day_scale.empty:
                    intraday_scaling[day_mask] = day_scale.iloc[0]
                else:
                    intraday_scaling[day_mask] = 1.0

            # Apply adaptation and limits
            position_sizes = (1.0 + self.params.adaptation_rate * (intraday_scaling - 1.0)).clip(
                lower=self.params.min_scaling,
                upper=self.params.max_scaling
            )

            self.logger.info(f"Adaptive position sizing: mean={position_sizes.mean():.2f}, "
                             f"min={position_sizes.min():.2f}, max={position_sizes.max():.2f}")

            return position_sizes

        except Exception as e:
            self.logger.error(f"Position size calculation failed: {str(e)}")
            return pd.Series(1.0, index=df.index)

class CombinedRiskManager(RiskManager):
    """Combines multiple risk management approaches."""

    def __init__(self, config: Config, risk_metrics: RiskMetrics,
                 managers: List[RiskManager], weights: Optional[List[float]] = None):
        super().__init__(config, risk_metrics)
        self.managers = managers
        self.weights = weights or [1.0 / len(managers)] * len(managers)

        # Validate weights
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Weights must sum to 1.0")

        # Add adaptation rate for weight adjustment
        self.adaptation_rate = config.get('combined_params', {}).get('adaptation_rate', 0.1)
        self.performance_window = config.get('combined_params', {}).get('performance_window', 63)

        self.logger.info(f"Initialized CombinedRiskManager with {len(managers)} managers")
        self.logger.info(f"Initial weights: {self.weights}")

    def calculate_position_sizes(self, df: pd.DataFrame, risk_metrics: pd.DataFrame) -> pd.Series:
        """Calculate position size multipliers by combining multiple managers."""
        try:
            daily_scalings = []

            # Get scaling from each manager
            for manager, weight in zip(self.managers, self.weights):
                manager_scaling = manager.calculate_position_sizes(df, risk_metrics)
                daily_scalings.append(manager_scaling * weight)

            # Combine scalings
            if not daily_scalings:
                return pd.Series(1.0, index=df.index)

            position_sizes = sum(daily_scalings)

            # Apply final limits
            position_sizes = position_sizes.clip(
                lower=self.risk_limits.min_scalar,
                upper=self.risk_limits.max_scalar
            )

            self.logger.info(f"Combined position sizing: mean={position_sizes.mean():.2f}, "
                             f"min={position_sizes.min():.2f}, max={position_sizes.max():.2f}")

            return position_sizes

        except Exception as e:
            self.logger.error(f"Position size calculation failed: {str(e)}")
            return pd.Series(1.0, index=df.index)

    def _calculate_manager_performance(self, manager: RiskManager,
                                       daily_performance: pd.DataFrame,
                                       risk_metrics: pd.DataFrame) -> float:
        """Calculate performance score for a manager."""
        summary = manager.get_risk_summary()

        # Calculate performance metrics
        return_score = daily_performance['returns'].tail(self.performance_window).mean()
        vol_score = risk_metrics['rolling_vol'].tail(self.performance_window).mean()
        drawdown_score = summary.get('current_drawdown', 0)

        # Combine scores (can be customized based on preferences)
        performance = (return_score / max(vol_score, 0.0001)) * (1 + drawdown_score)
        return max(performance, 0)  # Ensure non-negative performance

    def _update_weights(self, performances: List[float]) -> None:
        """Update manager weights based on performance."""
        if not performances:
            return

        # Calculate new weights based on relative performance
        total_perf = sum(performances)
        if total_perf > 0:
            new_weights = [p / total_perf for p in performances]

            # Apply gradual adaptation
            self.weights = [
                w + self.adaptation_rate * (nw - w)
                for w, nw in zip(self.weights, new_weights)
            ]

            # Normalize weights to ensure they sum to 1
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]

            self.logger.info(f"Updated weights: {self.weights}")


class IntradayMomentumStrategy(Strategy):
    """Intraday momentum strategy implementation."""

    def __init__(self, config: Config, params: StrategyParameters, invert_signals: bool = False):
        """Initialize intraday momentum strategy.

        Args:
            config: System configuration
            params: Strategy-specific parameters
            invert_signals: Whether to invert strategy signals
        """
        # Call parent class constructor first
        super().__init__(name="IntradayMomentum", config=config)

        self.params = params
        self.invert_signals = invert_signals
        self._validate_parameters()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on noise area boundaries."""
        try:
            df = df.copy()
            df = self.calculate_noise_area(df)

            # Define valid trading times
            valid_trading_times = (
                (df.index.minute.isin(self.params.entry_times)) &
                (df.index.time >= self.contract_spec.market_open) &
                (df.index.time <= self.contract_spec.last_entry)
            )

            # Initialize signals
            df['signal'] = 0

            # Generate base signals
            long_signals = valid_trading_times & (df['Close'] > df['upper_bound'])
            short_signals = valid_trading_times & (df['Close'] < df['lower_bound'])

            # Apply signals based on invert_signals flag
            if self.invert_signals:
                df.loc[long_signals, 'signal'] = -1
                df.loc[short_signals, 'signal'] = 1
            else:
                df.loc[long_signals, 'signal'] = 1
                df.loc[short_signals, 'signal'] = -1

            # Calculate trailing stops
            df = self._calculate_stops(df)
            self._log_signal_statistics(df)

            return df

        except Exception as e:
            self.logger.error("Signal generation failed", exc_info=True)
            raise

    def check_exit_conditions(self, bar: pd.Series, position: int,
                          entry_price: float, current_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check if position should be exited."""
        try:
            # Market close check
            if current_time.time() >= self.contract_spec.market_close:
                return True, ExitReason.MARKET_CLOSE.value

            if position == 0:
                return False, ""

            # Check stop hit based on position direction
            if pd.isna(bar['stop']):
                return False, ""

            if position > 0:  # Long position
                stop_hit = bar['Close'] <= bar['stop']
                if stop_hit:
                    return True, ExitReason.BOUNDARY_STOP.value
            else:  # Short position
                stop_hit = bar['Close'] >= bar['stop']
                if stop_hit:
                    return True, ExitReason.BOUNDARY_STOP.value

            return False, ""

        except Exception as e:
            self.logger.error(f"Error checking exits: {str(e)}")
            return False, ""  # Conservative approach - don't force exit on error

    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""
        try:
            if self.params.lookback_days <= 0:
                raise ValueError("Lookback days must be positive")

            if self.params.volatility_multiplier <= 0:
                raise ValueError("Volatility multiplier must be positive")

            if not self.params.entry_times:
                raise ValueError("No entry times specified")

            # Validate trading hours consistency
            if not (self.contract_spec.market_open <=
                    self.contract_spec.last_entry <=
                    self.contract_spec.market_close):
                raise ValueError("Invalid trading hours sequence")

            self.logger.debug("Parameter validation successful")

        except Exception as e:
            self.logger.error(f"Parameter validation failed: {str(e)}")
            raise

    def calculate_noise_area(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate noise area boundaries as defined in the Beat the Market paper.

        The noise area represents the region where prices typically oscillate under
        normal market conditions. The boundaries account for overnight gaps by using
        max/min of Open and PrevClose as reference prices.
        """
        try:
            df = df.copy()

            # Create market hours mask
            market_hours = (
                    (df.index.time >= self.contract_spec.market_open) &
                    (df.index.time <= self.contract_spec.market_close)
            )

            # Calculate absolute move from open during market hours only
            df['pct_move_from_open'] = np.nan
            df.loc[market_hours, 'pct_move_from_open'] = (
                    abs(df.loc[market_hours, 'Close'] - df.loc[market_hours, 'day_open']) /
                    df.loc[market_hours, 'day_open']
            )

            # Calculate average move for each minute using rolling lookback
            df['avg_move'] = df.groupby('minute_of_day')['pct_move_from_open'].transform(
                lambda x: x.rolling(
                    window=self.params.lookback_days,
                    min_periods=max(5, self.params.lookback_days // 2)
                ).mean()
            )

            # Calculate reference prices for bounds
            df['ref_price_high'] = df[['day_open', 'prev_close']].max(axis=1)
            df['ref_price_low'] = df[['day_open', 'prev_close']].min(axis=1)

            # Calculate boundaries during market hours only
            df['upper_bound'] = np.nan
            df['lower_bound'] = np.nan

            mult = self.params.volatility_multiplier
            df.loc[market_hours, 'upper_bound'] = (
                    df.loc[market_hours, 'ref_price_high'] *
                    (1 + mult * df.loc[market_hours, 'avg_move'])
            )
            df.loc[market_hours, 'lower_bound'] = (
                    df.loc[market_hours, 'ref_price_low'] *
                    (1 - mult * df.loc[market_hours, 'avg_move'])
            )

            return df

        except Exception as e:
            self.logger.error(f"Noise area calculation failed: {str(e)}")
            raise

    def _calculate_stops(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate stops following paper methodology."""
        try:
            df = df.copy()
            market_hours = (
                    (df.index.time >= self.contract_spec.market_open) &
                    (df.index.time <= self.contract_spec.market_close)
            )

            # Initialize stop columns
            df['stop'] = np.nan

            # Set VWAP as stop level
            df.loc[market_hours, 'stop'] = df.loc[market_hours, 'vwap']

            return df

        except Exception as e:
            self.logger.error("Stop calculation failed", exc_info=True)
            raise

    def _log_signal_statistics(self, df: pd.DataFrame) -> None:
        """Log statistics about generated signals."""
        try:
            # Calculate valid trading times
            valid_trading_times = (
                    (df.index.minute.isin(self.params.entry_times)) &
                    (df.index.time >= self.contract_spec.market_open) &
                    (df.index.time <= self.contract_spec.last_entry)
            )

            # Count signals by type
            long_signals = (df['signal'] == 1).sum()
            short_signals = (df['signal'] == -1).sum()
            total_valid_bars = valid_trading_times.sum()

            self.logger.info("\nSignal Statistics:")
            self.logger.info(f"Long signals: {long_signals}")
            self.logger.info(f"Short signals: {short_signals}")
            self.logger.info(f"Valid trading bars: {total_valid_bars}")

            if total_valid_bars > 0:
                signal_rate = ((long_signals + short_signals) / total_valid_bars * 100)
                self.logger.info(f"Signal generation rate: {signal_rate:.2f}%")

            # Daily signal distribution
            daily_data = pd.DataFrame({
                'date': df.index.date,
                'signals': df['signal'].abs()
            })

            daily_signals = daily_data.groupby('date')['signals'].apply(
                lambda x: (x != 0).sum()
            )

            if not daily_signals.empty:
                self.logger.info("\nDaily Signal Distribution:")
                self.logger.info(f"  Days with signals: {(daily_signals > 0).sum()}")
                self.logger.info(f"  Average signals per day: {daily_signals.mean():.2f}")
                self.logger.info(f"  Max signals in one day: {daily_signals.max():.0f}")
                self.logger.info(f"  Min signals in one day: {daily_signals.min():.0f}")

                # Signal timing distribution
                signal_times = df[df['signal'] != 0].index.strftime('%H:%M')
                time_counts = signal_times.value_counts()

                if not time_counts.empty:
                    self.logger.info("\nTop Signal Times:")
                    for time, count in time_counts.head().items():
                        self.logger.info(f"  {time}: {count} signals")

            # Boundary statistics
            self.logger.info("\nBoundary Statistics:")
            self.logger.info(
                f"Average noise band width: {((df['upper_bound'] - df['lower_bound']) / df['ref_price_high']).mean():.4%}")
            self.logger.info(
                f"Average distance to upper bound: {((df['upper_bound'] - df['Close']) / df['Close']).mean():.4%}")
            self.logger.info(
                f"Average distance to lower bound: {((df['Close'] - df['lower_bound']) / df['Close']).mean():.4%}")

        except Exception as e:
            self.logger.warning(f"Failed to log signal statistics: {str(e)}")


class TradeManager:
    """
    Handles trade execution, tracking, and P&L calculations.
    Responsible for executing trades based on strategy signals and tracking performance.
    """

    def __init__(self, config: Config):
        self.logger = LoggingConfig.get_logger(__name__)
        self.config = config
        self.transaction_costs = TransactionCosts(**config.transaction_costs)

        # Debug settings with more granular control
        self.DEBUG_ALL_TRADES = False  # Set to True only when debugging trade flow
        self.DEBUG_FULL_DETAIL = False  # Set to True for full DataFrame info
        self.DEBUG_MIN_PNL = 1000.0  # Only log trades with PnL above this threshold
        self.DEBUG_LOG_LEVEL = logging.INFO  # Default to INFO level

        # Add daily equity tracking
        self.daily_equity = {}
        self.current_date = None
        self.current_equity = config.initial_equity

    def _create_timezone_aware_index(self, df: pd.DataFrame, dates: List[datetime.date]) -> pd.DatetimeIndex:
        """
        Create timezone-aware index preserving DataFrame's timezone.

        Args:
            df: DataFrame with timezone-aware index
            dates: List of dates to create index for

        Returns:
            DatetimeIndex with preserved timezone

        Raises:
            ValueError: If input DataFrame lacks timezone information
        """
        if df.index.tz is None:
            raise ValueError("Input DataFrame must have timezone-aware index")

        return pd.DatetimeIndex([
            pd.Timestamp(date).tz_localize(None).tz_localize(df.index.tz)
            for date in dates
        ])

    def execute_strategy(
            self,
            strategy: Strategy,
            df: pd.DataFrame,
            apply_sizing: bool = False
    ) -> Tuple[List[Trade], pd.DataFrame, BaseReturns]:
        """
        Execute a trading strategy and track performance on an intraday DataFrame,
        skipping holidays/non-trading days rather than forcing zero returns.

        **Summary of Changes**:
          1. We define a set of valid trading dates from `daily_index` (the unique
             trading days). Any date not in `daily_index` is treated as a holiday (no data).
          2. When `prev_date != current_date`, we only do the "day boundary" code if
             `prev_date` is in our valid trading days. That way, if `prev_date` is
             e.g. 7/04 (a holiday) and not in `daily_index`, we skip generating
             a daily return for that date. We effectively jump from 7/03 → 7/05.

        Args:
            strategy (Strategy): Strategy instance to execute (generates signals).
            df (pd.DataFrame): Intraday market data, possibly tz-aware for intraday use.
            apply_sizing (bool): Whether to apply position sizing or use base positions.

        Returns:
            (trades, df, base_returns):
                trades: List[Trade] objects from this run
                df: Updated DataFrame with tracking columns
                base_returns: daily returns after skipping non-trading days
        """
        try:
            # 1) Initialize intraday columns for trade tracking
            df = self.initialize_trade_tracking(df)

            trades: List[Trade] = []

            # Position variables
            position = 0
            position_size = 0
            base_size = 0
            entry_price = None
            entry_time = None

            # Track daily PnL
            daily_pnl = 0.0
            prev_date = None

            # Global stats dictionary
            stats = self._initialize_stats()

            # 2) Initialize daily returns objects (naive daily_index, daily_equity, daily_returns)
            daily_index, daily_equity, daily_returns = self._initialize_returns_tracking(df)

            # Turn daily_index’s `.date` into a set for quick membership checks
            valid_trading_dates = set(daily_index.date)

            # Track equity for intraday updates
            current_equity = self.config.initial_equity

            self.logger.info("Starting strategy execution")

            # 3) Process each intraday bar
            for i in range(1, len(df)):
                current_bar = df.iloc[i]
                current_time = df.index[i]
                current_date = current_time.date()  # Python date object

                # (a) If the date changed vs. previous bar
                if prev_date is not None and current_date != prev_date:
                    # If we had an open position, force-close it at the end of the previous day
                    if position != 0:
                        position, pnl_update, equity_update = self._handle_day_close(
                            df=df,
                            position=position,
                            prev_date=prev_date,
                            entry_time=entry_time,
                            entry_price=entry_price,
                            current_bar=current_bar,
                            strategy=strategy,
                            apply_sizing=apply_sizing,
                            trades=trades,
                            stats=stats,
                            daily_returns=daily_returns,
                            daily_pnl=daily_pnl,
                            current_equity=current_equity
                        )
                        daily_pnl += pnl_update
                        current_equity = equity_update

                    # IMPORTANT: only generate a daily return if prev_date is in valid_trading_dates
                    if prev_date in valid_trading_dates:
                        # day_idx for the day we’re finalizing
                        day_idx = daily_index[daily_index.date == prev_date]
                        if not day_idx.empty:
                            day_idx = day_idx[0]

                            # next_idx for the new day, if the new day is also in valid_trading_dates
                            if current_date in valid_trading_dates:
                                next_idx = daily_index[daily_index.date == current_date][0]
                            else:
                                # If current_date not in valid_trading_dates, we skip
                                # setting next_idx. We won't store daily_equity for current_date.
                                next_idx = None

                            # Compute that day’s returns
                            day_return, new_equity = self._calculate_daily_returns(
                                daily_index=daily_index,
                                pnl=daily_pnl,
                                prev_equity=daily_equity[day_idx]
                            )
                            daily_returns[day_idx] = day_return

                            # If we do have a valid next day, store the new equity
                            if next_idx is not None:
                                daily_equity[next_idx] = new_equity

                    # Reset daily tracking for the new date
                    daily_pnl = 0.0
                    position = 0
                    position_size = 0
                    base_size = 0
                    entry_price = None

                prev_date = current_date

                # Update intraday equity
                df.iloc[i, df.columns.get_loc('current_equity')] = current_equity

                # (b) Skip bars outside market hours
                if not current_bar['trading_hour']:
                    continue

                # (c) Enter position if no position and signal triggered
                if current_bar['signal'] != 0 and position == 0:
                    position, entry_price, entry_time = self._handle_entry(
                        current_bar=current_bar,
                        current_time=current_time,
                        stats=stats
                    )
                    if position != 0:
                        base_size = abs(current_bar['base_position_size'])
                        sizing_col = 'position_size' if apply_sizing else 'base_position_size'
                        position_size = abs(current_bar[sizing_col])

                # (d) If we have a position, check exit conditions
                elif position != 0:
                    position, pnl_update, equity_update = self._handle_exit(
                        current_bar=current_bar,
                        current_time=current_time,
                        position=position,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        strategy=strategy,
                        apply_sizing=apply_sizing,
                        trades=trades,
                        stats=stats,
                        daily_pnl=daily_pnl,
                        current_equity=current_equity,
                        base_size=base_size,
                        position_size=position_size
                    )
                    daily_pnl += pnl_update
                    current_equity = equity_update

                # Update intraday DataFrame columns
                df.iloc[i, df.columns.get_loc('position')] = position
                if position != 0:
                    df.iloc[i, df.columns.get_loc('entry_price')] = entry_price

            # 4) Handle final day if needed
            if prev_date is not None and daily_pnl != 0:
                # Only if prev_date in valid_trading_dates
                if prev_date in valid_trading_dates:
                    day_idx = daily_index[daily_index.date == prev_date][0]
                    day_return, _ = self._calculate_daily_returns(
                        daily_index=daily_index,
                        pnl=daily_pnl,
                        prev_equity=daily_equity[day_idx]
                    )
                    daily_returns[day_idx] = day_return

            # Summarize trades
            self._log_execution_summary(stats, current_equity)

            # 5) Filter daily returns so we only keep actual trading days
            filtered_returns = self._filter_returns_to_trading_days(daily_returns, df)
            if filtered_returns.index.tz is None:
                # Reattach tz if your BaseReturns requires it
                filtered_returns = filtered_returns.tz_localize(df.index.tz)

            # 6) Build BaseReturns from final daily returns
            base_returns_obj = BaseReturns(returns=filtered_returns)

            return trades, df, base_returns_obj

        except Exception as e:
            self.logger.error(f"Strategy execution failed: {str(e)}")
            raise

    def _calculate_daily_returns(self, daily_index: pd.DatetimeIndex,
                                 pnl: float, prev_equity: float) -> Tuple[float, float]:
        """
        Calculate daily returns and equity based on PnL.

        Args:
            daily_index: Index for the day's data
            pnl: Day's profit/loss
            prev_equity: Previous day's ending equity

        Returns:
            Tuple of (daily return, new equity)
        """
        if pnl == 0:
            return 0.0, prev_equity

        daily_return = pnl / prev_equity
        new_equity = prev_equity + pnl
        return daily_return, new_equity

    def _calculate_trade_pnl(self, exit_price: float, entry_price: float,
                             direction: int, quantity: float,
                             contract_spec: ContractSpecification) -> float:
        """Calculate trade P&L with improved validation."""
        try:
            # Validate inputs with detailed error messages
            if exit_price <= 0:
                raise ValueError(f"Invalid exit price: {exit_price}")
            if entry_price <= 0:
                raise ValueError(f"Invalid entry price: {entry_price}")
            if abs(direction) != 1:
                raise ValueError(f"Invalid direction: {direction} (must be 1 or -1)")
            if quantity <= 0:
                raise ValueError(f"Invalid quantity: {quantity}")
            if contract_spec.multiplier <= 0:
                raise ValueError(f"Invalid contract multiplier: {contract_spec.multiplier}")

            # Calculate P&L
            price_diff = exit_price - entry_price
            pnl = price_diff * direction * quantity * contract_spec.multiplier

            self._log_trade(
                "P&L calculation",
                data={
                    "exit_price": exit_price,
                    "entry_price": entry_price,
                    "direction": direction,
                    "quantity": quantity,
                    "price_diff": price_diff,
                    "pnl": pnl
                }
            )
            return pnl

        except Exception as e:
            self.logger.error(f"P&L calculation failed: {str(e)}")
            return 0.0  # Return 0 on error as a safe default

    def _calculate_trade_costs(self, trade: Trade) -> float:
        """Calculate transaction costs for a trade."""
        try:
            # Calculate commission per side
            commission = (self.transaction_costs.commission_rate *
                          trade.quantity * trade.contract_spec.multiplier)
            commission = max(commission, self.transaction_costs.min_commission)

            # Calculate slippage per side
            avg_price = (trade.entry_price + trade.exit_price) / 2
            slippage = (self.transaction_costs.slippage_rate * avg_price *
                        trade.quantity * trade.contract_spec.multiplier)

            # Total costs for entry and exit
            total_costs = (commission + slippage) * 2 + self.transaction_costs.fixed_costs

            self._log_trade("Cost calculation",
                            data={
                                "commission": commission,
                                "slippage": slippage,
                                "total": total_costs
                            })
            return total_costs

        except Exception as e:
            self.logger.error(f"Cost calculation failed: {str(e)}")
            return 0.0

    def initialize_trade_tracking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Initialize DataFrame with columns for tracking trades and performance.

        Args:
            df: Input DataFrame to initialize

        Returns:
            DataFrame with added tracking columns

        Columns added:
        - position: Current position (1 for long, -1 for short, 0 for flat)
        - entry_price: Entry price for current position
        - exit_price: Exit price for current position
        - trade_costs: Transaction costs for current trade
        - pnl: Profit/Loss for current trade
        - current_equity: Current portfolio equity
        - trade_status: Current trade status ('', 'OPEN', 'CLOSED')
        - exit_reason: Reason for trade exit
        - trade_direction: Direction of current trade
        """
        try:
            df = df.copy()

            tracking_columns = {
                'position': 0,
                'entry_price': 0.0,
                'exit_price': 0.0,
                'trade_costs': 0.0,
                'pnl': 0.0,
                'current_equity': self.config.initial_equity,
                'trade_status': '',
                'exit_reason': '',
                'trade_direction': 0,
            }

            # Add tracking columns with defaults
            for col, default in tracking_columns.items():
                if col not in df.columns:
                    df[col] = default

            # Verify column addition
            missing_cols = set(tracking_columns.keys()) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Failed to add columns: {missing_cols}")

            self._log_trade("Initialized trade tracking",
                            data={"columns": list(tracking_columns.keys())},
                            level=logging.INFO)

            return df

        except Exception as e:
            self.logger.error(f"Failed to initialize trade tracking: {str(e)}")
            raise

    def _handle_day_close(self, df: pd.DataFrame, position: int,
                          prev_date: datetime.date, entry_time: pd.Timestamp,
                          entry_price: float, current_bar: pd.Series,
                          strategy: Strategy, apply_sizing: bool, trades: List[Trade],
                          stats: Dict, daily_returns: pd.Series, daily_pnl: float,
                          current_equity: float) -> Tuple[int, float, float]:
        """Handle market close for a trading day."""
        try:
            # Store daily equity info in self.daily_equity
            if hasattr(self, 'daily_equity'):
                self.daily_equity[prev_date] = {
                    'start': current_equity - daily_pnl,
                    'end': current_equity
                }
                # Calculate and store daily return
                start_equity = current_equity - daily_pnl
                if start_equity > 0:
                    daily_return = daily_pnl / start_equity
                    if abs(daily_return) > 1.0:  # More than 100% in a day
                        self.logger.warning(
                            f"Large daily return detected: {daily_return:.2%} "
                            f"on {prev_date}. PnL: ${daily_pnl:,.2f}, "
                            f"Start equity: ${start_equity:,.2f}"
                        )
                    daily_returns[prev_date] = daily_return

            # Get previous day's market close time
            prev_close_time = pd.Timestamp.combine(
                prev_date,
                strategy.contract_spec.market_close
            ).tz_localize(df.index.tz)

            # Get close price at market close
            close_mask = df.index == prev_close_time
            if close_mask.any():
                exit_price = df[close_mask]['Close'].iloc[0]
                exit_bar = df[close_mask].iloc[0]
            else:
                # Get last price before market close
                last_bar = df[df.index <= prev_close_time].iloc[-1]
                exit_price = last_bar['Close']
                exit_bar = last_bar

            # Get position sizes
            base_size = abs(exit_bar['base_position_size'])
            final_size = abs(exit_bar['position_size' if apply_sizing else 'base_position_size'])

            pnl_update = 0.0
            equity_update = current_equity

            if base_size > 0 and final_size > 0:  # Only process if we have valid sizes
                # Calculate P&L
                trade_pnl = self._calculate_trade_pnl(
                    exit_price=exit_price,
                    entry_price=entry_price,
                    direction=position,
                    quantity=final_size,
                    contract_spec=strategy.contract_spec
                )

                # Create and log trade
                trade = self._create_trade(
                    entry_time=entry_time,
                    exit_time=prev_close_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=final_size,
                    direction=position,
                    pnl=trade_pnl,
                    contract_spec=strategy.contract_spec,
                    exit_reason=ExitReason.MARKET_CLOSE.value,
                    strategy_name=strategy.name,
                    symbol=strategy.contract_spec.symbol,
                    base_size=base_size,
                    final_size=final_size
                )

                trades.append(trade)
                self._update_stats(stats, trade, prev_close_time - entry_time)

                pnl_update = trade_pnl
                equity_update = current_equity + trade_pnl

            return 0, pnl_update, equity_update  # Always return position as 0 at day close

        except Exception as e:
            self.logger.error(f"Day close handling failed: {str(e)}")
            return 0, 0.0, current_equity  # Return safe values on error

    def _handle_entry(self, current_bar: pd.Series, current_time: pd.Timestamp,
                      stats: Dict) -> Tuple[int, float, pd.Timestamp]:
        """Handle new position entry."""
        try:
            # Validate position sizes
            base_size = abs(current_bar['base_position_size'])
            if base_size == 0:
                self.logger.warning("Zero base position size at entry")
                return 0, 0.0, current_time

            position = current_bar['signal']
            entry_price = current_bar['Close']
            entry_time = current_time

            stats['trades_initiated'] += 1
            if position > 0:
                stats['long_trades'] += 1
            else:
                stats['short_trades'] += 1

            self._log_trade("New position",
                            data={
                                "direction": "Long" if position > 0 else "Short",
                                "price": entry_price,
                                "time": entry_time,
                                "base_size": base_size
                            })

            return position, entry_price, entry_time

        except Exception as e:
            self.logger.error(f"Entry handling failed: {str(e)}")
            return 0, 0.0, current_time

    def _handle_position_update(self, current_bar: pd.Series, current_time: pd.Timestamp,
                                position: int, entry_time: Optional[pd.Timestamp] = None,
                                entry_price: Optional[float] = None) -> Tuple[int, float, float]:
        """Combined method for handling both entries and exits."""
        try:
            # Handle new entry if no position
            if position == 0 and current_bar['signal'] != 0:
                return self._handle_entry(current_bar, current_time)

            # Handle exit if in position
            elif position != 0:
                return self._handle_exit(current_bar, current_time, position,
                                         entry_time, entry_price)

            return position, 0.0, 0.0

        except Exception as e:
            self.logger.error(f"Position update failed: {str(e)}")
            return position, 0.0, 0.0
        
    def _handle_exit(self, current_bar: pd.Series, current_time: pd.Timestamp,
                     position: int, entry_time: pd.Timestamp, entry_price: float,
                     strategy: Strategy, apply_sizing: bool, trades: List[Trade],
                     stats: Dict, daily_pnl: float, current_equity: float,
                     base_size: float, position_size: float) -> Tuple[int, float, float]:
        """Handle position exit with explicit position sizing."""
        try:
            should_exit, exit_reason = strategy.check_exit_conditions(
                current_bar, position, entry_price, current_time
            )

            if should_exit and position_size > 0:  # Only exit if we have a valid position size
                exit_price = current_bar['Close']

                trade_pnl = self._calculate_trade_pnl(
                    exit_price=exit_price,
                    entry_price=entry_price,
                    direction=position,
                    quantity=position_size,
                    contract_spec=strategy.contract_spec
                )

                trade = self._create_trade(
                    entry_time=entry_time,
                    exit_time=current_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=position_size,
                    direction=position,
                    pnl=trade_pnl,
                    contract_spec=strategy.contract_spec,
                    exit_reason=exit_reason,
                    strategy_name=strategy.name,
                    symbol=strategy.contract_spec.symbol,
                    base_size=base_size,
                    final_size=position_size
                )

                trades.append(trade)
                self._update_stats(stats, trade, current_time - entry_time)
                daily_pnl += trade_pnl
                current_equity += trade_pnl
                position = 0

            return position, daily_pnl, current_equity

        except Exception as e:
            self.logger.error(f"Exit handling failed: {str(e)}")
            raise

    def _create_trade(self, **kwargs) -> Trade:
        """Create a Trade object with proper validation."""
        try:
            # Ensure required fields are present
            required_fields = {
                'entry_time', 'exit_time', 'entry_price', 'exit_price',
                'quantity', 'direction', 'pnl', 'contract_spec', 'exit_reason',
                'strategy_name', 'symbol', 'base_size', 'final_size'
            }

            missing_fields = required_fields - set(kwargs.keys())
            if missing_fields:
                raise ValueError(f"Missing required fields for trade creation: {missing_fields}")

            # Create temp trade for cost calculation
            temp_trade = Trade(**kwargs, costs=0.0,
                               base_return=0.0, final_return=0.0)

            # Calculate actual costs
            costs = self._calculate_trade_costs(temp_trade)

            # Create final trade with costs
            trade_args = {**kwargs}
            trade_args['costs'] = costs
            trade_args['base_return'] = kwargs['pnl'] / self.config.initial_equity
            trade_args['final_return'] = (kwargs['pnl'] - costs) / self.config.initial_equity

            trade = Trade(**trade_args)

            self._log_trade("Trade created", trade=trade)
            return trade

        except Exception as e:
            self.logger.error(f"Trade creation failed: {str(e)}")
            raise

    def process_trades(self, trades: List[Trade], df: pd.DataFrame) -> Tuple[List[Trade], pd.DataFrame]:
        """Process trades and update performance tracking."""
        try:
            processed_trades = []
            for trade in trades:
                processed_trade = self._create_trade(**self._trade_to_dict(trade))
                processed_trades.append(processed_trade)
                df = self._update_trade_tracking(processed_trade, df)

            return processed_trades, df

        except Exception as e:
            self.logger.error(f"Trade processing failed: {str(e)}")
            raise

    def get_trade_statistics(self, trades: List[Trade]) -> List[Dict]:
        """
        Generate detailed statistics for each trade.

        Args:
            trades: List of Trade objects to analyze

        Returns:
            List of dictionaries containing statistics for each trade
        """
        try:
            trade_metrics = []

            for trade in trades:
                metrics = {
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'duration_mins': (trade.exit_time - trade.entry_time).total_seconds() / 60,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'costs': trade.costs,
                    'net_pnl': trade.pnl - trade.costs,
                    'return': trade.final_return,
                    'exit_reason': trade.exit_reason,
                    'base_size': trade.base_size,
                    'final_size': trade.final_size,
                    'is_profitable': trade.pnl > trade.costs,
                    'leverage': trade.final_size / trade.base_size if trade.base_size != 0 else 1.0,
                    'mae': None,  # Maximum Adverse Excursion - can be added if we track intra-trade prices
                    'mfe': None  # Maximum Favorable Excursion - can be added if we track intra-trade prices
                }

                trade_metrics.append(metrics)

            self._log_trade_statistics_summary(trade_metrics)
            return trade_metrics

        except Exception as e:
            self.logger.error(f"Failed to generate trade statistics: {str(e)}")
            return []

    def _log_returns_debug(self, daily_returns: pd.Series, daily_equity: pd.Series) -> None:
        """Log detailed returns information for debugging."""
        self.logger.debug("\nDaily Returns Statistics:")
        self.logger.debug(f"Total trading days: {len(daily_returns)}")
        self.logger.debug(f"Days with non-zero returns: {(daily_returns != 0).sum()}")

        # Safe timezone check
        tz = getattr(daily_returns.index, 'tz', None)
        self.logger.debug(f"Returns index timezone: {tz}")

        # Daily return stats
        self.logger.debug(f"\nDaily Returns Values:")
        for date, ret in daily_returns.items():
            self.logger.debug(f"{date}: {ret:.6%}")

        self.logger.debug(f"\nSum of daily returns: {daily_returns.sum():.4f}")
        self.logger.debug(f"Compound return: {(1 + daily_returns).prod() - 1:.4f}")

        # Equity tracking
        self.logger.debug(f"\nEquity Progression:")
        for date, equity in daily_equity.items():
            self.logger.debug(f"{date}: ${equity:.2f}")

        self.logger.debug(f"\nStarting equity: {daily_equity.iloc[0]:.2f}")
        self.logger.debug(f"Final equity: {daily_equity.iloc[-1]:.2f}")

        # Add detailed index information
        self.logger.debug("\nIndex Information:")
        self.logger.debug(f"Returns index type: {type(daily_returns.index)}")
        self.logger.debug(f"First date: {daily_returns.index[0]}")
        self.logger.debug(f"Last date: {daily_returns.index[-1]}")
        self.logger.debug(f"Index freq: {getattr(daily_returns.index, 'freq', None)}")

    def _log_trade_statistics_summary(self, trade_metrics: List[Dict]) -> None:
        """
        Log summary statistics from trade metrics.

        Args:
            trade_metrics: List of trade metric dictionaries
        """
        try:
            if not trade_metrics:
                self.logger.info("No trades to analyze")
                return

            # Calculate aggregate statistics
            total_trades = len(trade_metrics)
            profitable_trades = sum(1 for t in trade_metrics if t['is_profitable'])
            win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0

            total_pnl = sum(t['pnl'] for t in trade_metrics)
            total_costs = sum(t['costs'] for t in trade_metrics)
            net_pnl = total_pnl - total_costs

            avg_duration = sum(t['duration_mins'] for t in trade_metrics) / total_trades

            # Log summary
            self.logger.info("\nTrade Statistics Summary:")
            self.logger.info(f"Total trades: {total_trades}")
            self.logger.info(f"Profitable trades: {profitable_trades} ({win_rate:.1f}%)")
            self.logger.info(f"Total P&L: ${total_pnl:,.2f}")
            self.logger.info(f"Total costs: ${total_costs:,.2f}")
            self.logger.info(f"Net P&L: ${net_pnl:,.2f}")
            self.logger.info(f"Average trade duration: {avg_duration:.1f} minutes")

            # Direction breakdown
            long_trades = sum(1 for t in trade_metrics if t['direction'] > 0)
            short_trades = sum(1 for t in trade_metrics if t['direction'] < 0)
            self.logger.info(f"\nDirection breakdown:")
            self.logger.info(f"Long trades: {long_trades}")
            self.logger.info(f"Short trades: {short_trades}")

            # Exit reason breakdown
            exit_reasons = {}
            for t in trade_metrics:
                exit_reasons[t['exit_reason']] = exit_reasons.get(t['exit_reason'], 0) + 1

            self.logger.info("\nExit reason breakdown:")
            for reason, count in exit_reasons.items():
                self.logger.info(f"{reason}: {count} trades ({(count / total_trades) * 100:.1f}%)")

        except Exception as e:
            self.logger.error(f"Failed to log trade statistics summary: {str(e)}")

    def _initialize_returns_tracking(self, df: pd.DataFrame) -> Tuple[pd.DatetimeIndex, pd.Series, pd.Series]:
        """
        Create daily tracking objects (index, equity, returns) for use in strategy execution.

        This method:
            1. Drops any existing tz from df.index (if present) to ensure a naive DatetimeIndex.
            2. Collects the sorted unique date portion of df.index.
            3. Creates:
                - daily_index: a naive DatetimeIndex of those dates
                - daily_equity: a Series of constant initial_equity (same shape as daily_index)
                - daily_returns: a Series of zeros (same shape as daily_index)

        Note: Because we're doing daily aggregates, we make everything naive to avoid
        conflicts with tz-aware vs. tz-naive indexes. Intra-day code can still
        remain tz-aware in DataManager, but for daily grouping we strip out tz here.

        Args:
            df (pd.DataFrame): Market data (with or without tz) from DataManager.

        Returns:
            (pd.DatetimeIndex, pd.Series, pd.Series):
                daily_index, daily_equity, daily_returns
        """
        try:
            # 1) Force df.index to be naive
            df_naive = df.copy()
            df_naive.index = self._force_naive_index(df_naive.index)

            # 2) Extract unique daily 'dates' from the naive index
            unique_dates = sorted(set(df_naive.index.date))

            # 3) Build a naive DatetimeIndex from those dates
            daily_index = pd.to_datetime(unique_dates)  # This yields a naive DatetimeIndex

            # Create Series for equity & returns, indexed by daily_index
            daily_equity = pd.Series(self.config.initial_equity, index=daily_index)
            daily_returns = pd.Series(0.0, index=daily_index)

            self.logger.debug(f"Initialized daily tracking with {len(daily_index)} unique days.")
            return daily_index, daily_equity, daily_returns

        except Exception as e:
            self.logger.error(f"Failed to initialize returns tracking: {str(e)}")
            raise

    def _filter_returns_to_trading_days(self, daily_returns: pd.Series, df: pd.DataFrame) -> pd.Series:
        """
        Filter the naive-indexed daily_returns Series so that it only includes
        days which appear as trading days in df['is_trading_period'].

        Steps:
            1. Force df.index to naive for grouping.
            2. groupby the normalized (midnight) index, checking if 'is_trading_period' is True any time that day.
            3. Convert that list of “active” days to a naive DatetimeIndex.
            4. Force daily_returns.index to naive.
            5. Return daily_returns restricted to those trading days.

        Args:
            daily_returns (pd.Series): The daily returns, with a naive DatetimeIndex from _initialize_returns_tracking.
            df (pd.DataFrame): The intraday DataFrame, with 'is_trading_period' column.

        Returns:
            pd.Series: daily_returns restricted to the set of trading days.
        """
        try:
            self.logger.debug("---- In _filter_returns_to_trading_days ----")

            # 1) Force df.index to naive for consistent grouping
            df_naive = df.copy()
            df_naive.index = self._force_naive_index(df_naive.index)

            # 2) Group by the "normalized" date portion.
            #    .normalize() on a naive DatetimeIndex => same day w/ 00:00:00.
            trading_mask = df_naive['is_trading_period'].groupby(df_naive.index.normalize()).any()

            # 3) Extract only the days that are True for is_trading_period
            active_days = trading_mask[trading_mask].index  # This is an Index of naive midnight Timestamps

            # Make sure it's a DatetimeIndex (naive)
            active_days = self._force_naive_index(active_days)

            self.logger.debug(f"Active trading days: {len(active_days)} found.")

            # 4) Force daily_returns index to naive as well
            daily_returns_naive = daily_returns.copy()
            daily_returns_naive.index = self._force_naive_index(daily_returns_naive.index)

            # 5) Filter daily_returns by only those active trading days
            filtered = daily_returns_naive[daily_returns_naive.index.isin(active_days)]

            self.logger.debug(
                f"Filtered daily_returns from {len(daily_returns_naive)} to {len(filtered)} entries."
            )

            return filtered

        except Exception as e:
            # Log detailed info
            self.logger.error(f"Failed to filter returns: {str(e)}")
            self.logger.error(f"daily_returns index info: {daily_returns.index}")
            self.logger.error(f"df index info: {df.index}")
            raise

    def _initialize_stats(self) -> Dict:
        """Initialize statistics tracking dictionary."""
        return {
            'trades_initiated': 0,
            'trades_completed': 0,
            'long_trades': 0,
            'short_trades': 0,
            'market_close_exits': 0,
            'stop_exits': 0,
            'profit_trades': 0,
            'loss_trades': 0,
            'total_pnl': 0.0,
            'max_trade_pnl': float('-inf'),
            'min_trade_pnl': float('inf'),
            'longest_trade_mins': 0,
            'shortest_trade_mins': float('inf')
        }

    def _update_stats(self, stats: Dict, trade: Trade, duration: pd.Timedelta) -> None:
        """
        Update trading statistics with completed trade information.

        Args:
            stats: Dictionary containing trading statistics
            trade: Completed trade object
            duration: Trade duration as Timedelta
        """
        try:
            # Update trade counters
            stats['trades_completed'] += 1

            # Update P&L statistics
            stats['total_pnl'] += trade.pnl
            stats['max_trade_pnl'] = max(stats['max_trade_pnl'], trade.pnl)
            stats['min_trade_pnl'] = min(stats['min_trade_pnl'], trade.pnl)

            # Update profitability stats
            if trade.pnl > trade.costs:
                stats['profit_trades'] += 1
            else:
                stats['loss_trades'] += 1

            # Update duration statistics (in minutes)
            duration_mins = duration.total_seconds() / 60
            stats['longest_trade_mins'] = max(stats['longest_trade_mins'], duration_mins)
            stats['shortest_trade_mins'] = min(stats['shortest_trade_mins'], duration_mins)

            # Update exit reason statistics
            if trade.exit_reason == ExitReason.MARKET_CLOSE.value:
                stats['market_close_exits'] += 1
            elif trade.exit_reason in {ExitReason.VWAP_STOP.value, ExitReason.BOUNDARY_STOP.value}:
                stats['stop_exits'] += 1

            self._log_trade(
                "Updated statistics",
                data={
                    'trade_pnl': trade.pnl,
                    'duration_mins': duration_mins,
                    'exit_reason': trade.exit_reason
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to update statistics: {str(e)}")
            raise

    def _update_trade_tracking(self, trade: Trade, df: pd.DataFrame) -> pd.DataFrame:
        """Update DataFrame with trade results."""
        try:
            # Update at exit time
            exit_mask = (df.index == trade.exit_time)
            if exit_mask.any():
                df.loc[exit_mask, 'trade_costs'] = trade.costs
                df.loc[exit_mask, 'pnl'] = trade.pnl
                df.loc[exit_mask, 'trade_status'] = 'CLOSED'
                df.loc[exit_mask, 'exit_reason'] = trade.exit_reason
                df.loc[exit_mask, 'position'] = 0

            # Update future equity
            future_mask = (df.index > trade.exit_time)
            if future_mask.any():
                df.loc[future_mask, 'current_equity'] += (trade.pnl - trade.costs)

            return df

        except Exception as e:
            self.logger.error(f"Trade tracking update failed: {str(e)}")
            raise

    def _force_naive_index(self, idx: pd.Index) -> pd.DatetimeIndex:
        """
        Convert any tz-aware DatetimeIndex (or array-like of date/time values) into a
        naive DatetimeIndex (no timezone).

        1. If `idx` is already a DatetimeIndex with no timezone, return as-is.
        2. If it is a tz-aware DatetimeIndex, remove the timezone via `.tz_localize(None)`.
        3. If `idx` is not a DatetimeIndex at all, attempt to convert it with `pd.to_datetime(..., errors='coerce')`
           and then remove the timezone.

        Args:
            idx (pd.Index): The index to convert or ensure is naive.

        Returns:
            pd.DatetimeIndex: A tz-naive (no timezone) DatetimeIndex.
        """
        if isinstance(idx, pd.DatetimeIndex):
            # If it already has a timezone, remove it
            if idx.tz is not None:
                return idx.tz_localize(None)
            else:
                # Already a naive DatetimeIndex
                return idx
        else:
            # If not even a DatetimeIndex, convert first
            idx_converted = pd.to_datetime(idx, errors='coerce')
            # Now ensure it's naive
            return idx_converted.tz_localize(None)

    def _log_execution_summary(self, stats: Dict, current_equity: float) -> None:
        """Log summary of strategy execution results with clear formatting."""
        summary_lines = [
            "\nExecution Summary:",
            "=" * 50,
            "\nTrading Activity:",
            f"Total trades: {stats['trades_completed']:,}",
            f"Long trades: {stats['long_trades']:,}",
            f"Short trades: {stats['short_trades']:,}",

            "\nProfitability:",
            f"Win rate: {(stats['profit_trades'] / stats['trades_completed'] * 100):.1f}%",
            f"Total P&L: ${stats['total_pnl']:,.2f}",
            f"Best trade: ${stats['max_trade_pnl']:,.2f}",
            f"Worst trade: ${stats['min_trade_pnl']:,.2f}",

            "\nDuration Statistics:",
            f"Average duration: {stats.get('avg_duration_mins', 0):.1f} minutes",
            f"Longest trade: {stats['longest_trade_mins']:.1f} minutes",
            f"Shortest trade: {stats['shortest_trade_mins']:.1f} minutes",

            "\nPortfolio Performance:",
            f"Initial equity: ${self.config.initial_equity:,.2f}",
            f"Final equity: ${current_equity:,.2f}",
            f"Total return: {((current_equity / self.config.initial_equity - 1) * 100):.1f}%",
            "=" * 50
        ]

        self.logger.info("\n".join(summary_lines))

    def _log_trade(self, msg: str, data: Dict = None,
                   trade: Trade = None, df: pd.DataFrame = None,
                   level: int = None) -> None:
        """Unified trade logging with timestamps and daily equity tracking."""
        if not self.DEBUG_ALL_TRADES:
            if trade and abs(trade.pnl) < self.DEBUG_MIN_PNL:
                return

        level = level or self.DEBUG_LOG_LEVEL

        # Build trade message
        trade_msg = [f"TRADE - {msg}"]

        # Add essential trade info if available
        if trade:
            essential_info = {
                'date': trade.exit_time.strftime('%Y-%m-%d'),
                'entry_time': trade.entry_time.strftime('%H:%M:%S'),
                'exit_time': trade.exit_time.strftime('%H:%M:%S'),
                'direction': trade.direction,
                'pnl': f"${trade.pnl:,.2f}",
                'exit_reason': trade.exit_reason,
                'entry_price': f"${trade.entry_price:,.2f}",
                'exit_price': f"${trade.exit_price:,.2f}",
                'quantity': trade.quantity
            }
            trade_msg.extend(f"  {k}: {v}" for k, v in essential_info.items())

            # Add equity tracking if we have it
            if hasattr(self, 'daily_equity'):
                trade_date = trade.exit_time.date()
                if trade_date in self.daily_equity:
                    day_equity = self.daily_equity[trade_date]
                    daily_return = (day_equity['end'] - day_equity['start']) / day_equity['start']
                    trade_msg.extend([
                        f"  start_equity: ${day_equity['start']:,.2f}",
                        f"  end_equity: ${day_equity['end']:,.2f}",
                        f"  daily_return: {daily_return:.4%}"
                    ])

        # Log basic trade info
        self.logger.log(level, "\n".join(trade_msg))

        # Log detailed info to debug log only if enabled
        if self.DEBUG_LOG_LEVEL <= logging.DEBUG:
            debug_msg = []

            if data:
                debug_msg.append("\nData:")
                debug_msg.extend(f"  {k}: {v}" for k, v in data.items())

            if trade:
                debug_msg.append("\nDetailed Trade Info:")
                detailed_info = {
                    'base_size': trade.base_size,
                    'final_size': trade.final_size,
                    'costs': f"${trade.costs:,.2f}",
                    'net_pnl': f"${(trade.pnl - trade.costs):,.2f}",
                    'contract_spec': trade.contract_spec.symbol,
                    'duration': f"{(trade.exit_time - trade.entry_time).total_seconds() / 60:.1f} minutes"
                }
                debug_msg.extend(f"  {k}: {v}" for k, v in detailed_info.items())

            if df is not None and self.DEBUG_FULL_DETAIL:
                debug_msg.append("\nDataFrame Info:")
                debug_msg.append(f"  Shape: {df.shape}")
                debug_msg.append(f"  Index range: {df.index[0]} to {df.index[-1]}")

            if debug_msg:
                self.logger.debug("\n".join(debug_msg))

    @staticmethod
    def _trade_to_dict(trade: Trade) -> Dict:
        """Convert Trade to dictionary for creation/logging."""
        return {
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'direction': trade.direction,
            'pnl': trade.pnl,
            'exit_reason': trade.exit_reason,
            'base_size': trade.quantity,
            'final_size': trade.quantity,
            'strategy_name': trade.strategy_name,
            'symbol': trade.contract_spec.symbol,
            'contract_spec': trade.contract_spec
        }


class TradingSystem:
    """Main trading system orchestrating all components."""

    def __init__(self, config: Config):
        """Initialize trading system with minimal logging."""
        self.logger = LoggingConfig.get_logger(__name__)
        self.config = config

        # Initialize core components without logging
        self.data_manager = DataManager(config)
        self.risk_metrics = RiskMetrics(config)
        self.trade_manager = TradeManager(config)
        self.risk_manager = RiskManagerFactory.create(
            config=config,
            risk_metrics=self.risk_metrics
        )

    def run(self, strategy: Strategy, symbol: str,
            days_to_analyze: Optional[int] = None,
            lookback_buffer: Optional[int] = None) -> TradingResults:
        """Execute trading strategy with proper component separation.

        Execution flow:
        1. Prepare market data
        2. Generate trading signals
        3. Execute base strategy to get base returns
        4. Calculate risk metrics and position sizing
        5. Execute levered strategy with position sizing
        6. Process results and create final performance metrics
        """
        try:
            # Step 1: Prepare market data
            self.logger.info("\nStep 1: Preparing market data...")
            df = self.data_manager.prepare_data_for_analysis(
                symbol=symbol,
                days_to_analyze=days_to_analyze,
                lookback_buffer=lookback_buffer
            )

            # Step 2: Generate trading signals
            self.logger.info("\nStep 2: Generating trading signals...")
            df = strategy.generate_signals(df)

            # Step 3: Execute base strategy
            self.logger.info("\nStep 3: Executing base strategy...")
            df_base = self.risk_manager.prepare_base_strategy_positions(df.copy())
            base_trades, df_base, base_returns = self.trade_manager.execute_strategy(
                strategy=strategy,
                df=df_base,
                apply_sizing=False
            )

            # Process base trades
            base_trades, df_base = self.trade_manager.process_trades(base_trades, df_base)

            # Step 4: Calculate risk metrics and position sizing
            self.logger.info("\nStep 4: Calculating risk metrics and position sizing...")
            if base_returns.metrics is None:
                base_returns.calculate_metrics(self.risk_metrics)
            position_sizes = self.risk_manager.calculate_position_sizes(df, base_returns)

            # Step 5: Execute levered strategy
            self.logger.info("\nStep 5: Executing levered strategy...")
            df_levered = df.copy()
            df_levered = self.risk_manager.prepare_base_strategy_positions(df_levered)
            df_levered = self.risk_manager.apply_position_sizing(df_levered, position_sizes)

            levered_trades, df_levered, _ = self.trade_manager.execute_strategy(
                strategy=strategy,
                df=df_levered,
                apply_sizing=True
            )

            # Process levered trades
            levered_trades, df_levered = self.trade_manager.process_trades(levered_trades, df_levered)

            # Step 6: Create final results
            self.logger.info("\nStep 6: Creating final results...")
            daily_position_sizes = position_sizes.reindex(base_returns.returns.index)
            levered_returns = LeveredReturns(
                position_sizes=daily_position_sizes,
                base_returns=base_returns
            )

            results = TradingResults(
                symbol=symbol,
                strategy_name=strategy.name,
                base_trades=base_trades,
                final_trades=levered_trades,
                trade_metrics=self.trade_manager.get_trade_statistics(levered_trades),
                daily_performance=strategy.aggregate_to_daily(
                    df_base=df_base,
                    df_levered=df_levered,
                    base_trades=base_trades,
                    final_trades=levered_trades
                ),
                execution_data=df_levered[df_levered['is_trading_period']].copy(),
                config=self.config,
                contract_spec=strategy.contract_spec,
                timestamp=pd.Timestamp.now(tz=df.index.tz),
                base_returns=base_returns,
                levered_returns=levered_returns
            )

            return results

        except Exception as e:
            self.logger.error(f"Strategy execution failed: {str(e)}")
            raise

    def _log_execution_summary(self, results: TradingResults) -> None:
        """Log comprehensive execution summary."""
        self.logger.info("\nExecution Summary:")
        self.logger.info("=" * 50)
        self.logger.info(f"Trading period: {results.daily_performance.index[0].date()} to "
                         f"{results.daily_performance.index[-1].date()}")

        # Base strategy results
        self.logger.info("\nBase Strategy:")
        self.logger.info(f"Total trades: {len(results.base_trades)}")
        base_pnl = sum(t.pnl for t in results.base_trades)
        base_costs = sum(t.costs for t in results.base_trades)
        self.logger.info(f"Total P&L: ${base_pnl:,.2f}")
        self.logger.info(f"Total costs: ${base_costs:,.2f}")
        self.logger.info(f"Net P&L: ${(base_pnl - base_costs):,.2f}")

        # Levered strategy results
        self.logger.info("\nLevered Strategy:")
        self.logger.info(f"Total trades: {len(results.final_trades)}")
        final_pnl = sum(t.pnl for t in results.final_trades)
        final_costs = sum(t.costs for t in results.final_trades)
        self.logger.info(f"Total P&L: ${final_pnl:,.2f}")
        self.logger.info(f"Total costs: ${final_costs:,.2f}")
        self.logger.info(f"Net P&L: ${(final_pnl - final_costs):,.2f}")

        self.logger.info("=" * 50)

    def _update_daily_performance(self,
                                  daily_performance: pd.DataFrame,
                                  base_returns: BaseReturns,
                                  levered_returns: LeveredReturns,
                                  base_trades: List[Trade],
                                  final_trades: List[Trade]) -> pd.DataFrame:
        """
        Update daily performance data using return objects.

        Args:
            daily_performance: Original daily performance DataFrame
            base_returns: BaseReturns object
            levered_returns: LeveredReturns object
            base_trades: List of base trades
            final_trades: List of final trades

        Returns:
            Updated daily performance DataFrame
        """
        # Calculate base and levered returns statistics
        df = daily_performance.copy()

        # Update equity curves
        base_equity = (1 + base_returns.returns).cumprod() * self.config.initial_equity
        levered_equity = (1 + levered_returns.returns).cumprod() * self.config.initial_equity

        # Add derived data
        df['base_equity'] = base_equity
        df['equity'] = levered_equity

        # Add trades data
        trade_dates = pd.Series(df.index.date).unique()
        for date in trade_dates:
            day_mask = df.index.date == date

            # Process base trades
            base_day_trades = [t for t in base_trades if t.exit_time.date() == date]
            if base_day_trades:
                df.loc[day_mask, 'base_trades'] = len(base_day_trades)
                df.loc[day_mask, 'base_pnl'] = sum(t.pnl for t in base_day_trades)
                df.loc[day_mask, 'base_costs'] = sum(t.costs for t in base_day_trades)

            # Process final trades
            final_day_trades = [t for t in final_trades if t.exit_time.date() == date]
            if final_day_trades:
                df.loc[day_mask, 'trades'] = len(final_day_trades)
                df.loc[day_mask, 'pnl'] = sum(t.pnl for t in final_day_trades)
                df.loc[day_mask, 'costs'] = sum(t.costs for t in final_day_trades)

        # Add metrics if available
        if base_returns.metrics is not None:
            metrics_df = base_returns.metrics.value if isinstance(base_returns.metrics,
                                                                  MetricResult) else base_returns.metrics
            df = pd.concat([df, metrics_df.add_prefix('base_')], axis=1)

        if levered_returns.metrics is not None:
            metrics_df = levered_returns.metrics.value if isinstance(levered_returns.metrics,
                                                                     MetricResult) else levered_returns.metrics
            df = pd.concat([df, metrics_df.add_prefix('levered_')], axis=1)

        return df


class ReportingSystem:
    """Centralized reporting system that uses BaseReturns and LeveredReturns dataclasses."""

    def __init__(self, config: Config):
        self.logger = LoggingConfig.get_logger(__name__)
        self.config = config
        # Use attribute access instead of .get()
        self.base_output_dir = Path(self.config.output_paths[self.config.symbol])
        self.timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_output_dir / f"run_{self.timestamp}"
        self._init_directories()

    def _init_directories(self) -> None:
        """Initialize output directories for reports."""
        try:
            self.base_output_dir.mkdir(parents=True, exist_ok=True)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created run directory: {self.run_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize directories: {str(e)}")
            raise

    def generate_reports(self, trading_results: TradingResults) -> None:
        """Generate comprehensive trading reports using returns dataclasses."""
        try:
            self.logger.info("Starting report generation")

            # Validate returns objects exist
            if not trading_results.base_returns:
                raise ValueError("No base returns data available")

            # Log performance
            self._log_performance(trading_results)

            # Generate performance reports
            perf_path = self._get_output_path("performance")
            self.generate_performance_report(trading_results, perf_path)

            # Generate trade analysis
            trade_path = self._get_output_path("trades.csv")
            self._generate_trade_analysis(trading_results, trade_path)

            # Save complete results
            results_path = self._get_output_path("results.pkl")
            self._save_results(trading_results, results_path)

            self.logger.info(f"\nReports generated in directory: {self.run_dir}")

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise

    def generate_performance_report(self, results: TradingResults, base_path: Path) -> None:
        """Generate HTML performance reports for base and levered returns."""
        try:
            # Generate base strategy report
            if results.base_returns is not None:
                if not isinstance(results.base_returns, BaseReturns):
                    raise TypeError("base_returns must be a BaseReturns instance")

                base_returns = results.base_returns.returns
                if not base_returns.empty:
                    base_path_str = str(base_path) + "_base.html"
                    qs.reports.html(
                        returns=base_returns,
                        output=base_path_str,
                        title=f"{results.strategy_name} Base Performance Report"
                    )
                    self.logger.info(f"Base performance report generated at {base_path_str}")

                    # Save base metrics
                    if results.base_returns.metrics is not None:
                        self._save_metrics(results.base_returns.metrics, base_path_str.replace('.html', '_metrics.csv'))

            # Generate levered strategy report
            if results.levered_returns is not None:
                if not isinstance(results.levered_returns, LeveredReturns):
                    raise TypeError("levered_returns must be a LeveredReturns instance")

                levered_returns = results.levered_returns.returns
                if not levered_returns.empty:
                    levered_path_str = str(base_path) + "_levered.html"
                    qs.reports.html(
                        returns=levered_returns,
                        output=levered_path_str,
                        title=f"{results.strategy_name} Levered Performance Report"
                    )
                    self.logger.info(f"Levered performance report generated at {levered_path_str}")

                    # Save levered metrics
                    if results.levered_returns.metrics is not None:
                        self._save_metrics(results.levered_returns.metrics,
                                           levered_path_str.replace('.html', '_metrics.csv'))

        except Exception as e:
            self.logger.error(f"Performance report generation failed: {str(e)}")
            raise

    def _generate_trade_analysis(self, results: TradingResults, output_path: Path) -> None:
        """Generate comprehensive trade analysis."""
        try:
            trades_df = self._create_trade_analysis_df(results.final_trades or results.base_trades)

            if trades_df.empty:
                self.logger.info("\nNo trades to analyze")
                return

            # Save analysis to CSV
            trades_df.to_csv(output_path)
            self.logger.info(f"Trade analysis saved to {output_path}")

            # Log trade statistics
            self._log_trade_statistics(results)

        except Exception as e:
            self.logger.error(f"Trade analysis generation failed: {str(e)}")
            raise

    def _create_trade_analysis_df(self, trades: List[Trade]) -> pd.DataFrame:
        """Create comprehensive trade analysis DataFrame."""
        if not trades:
            return pd.DataFrame()

        try:
            analysis = []
            for trade in trades:
                analysis.append({
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'duration_mins': (trade.exit_time - trade.entry_time).total_seconds() / 60,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'costs': trade.costs,
                    'net_pnl': trade.pnl - trade.costs,
                    'exit_reason': trade.exit_reason,
                    'base_size': trade.base_size,
                    'final_size': trade.final_size,
                    'base_return': trade.base_return,
                    'final_return': trade.final_return
                })

            return pd.DataFrame(analysis)

        except Exception as e:
            self.logger.error(f"Trade analysis DataFrame creation failed: {str(e)}")
            return pd.DataFrame()

    def _save_metrics(self, metrics: Union[MetricResult, pd.DataFrame], path: str) -> None:
        """Save metrics to CSV with proper handling of MetricResult objects."""
        try:
            if isinstance(metrics, MetricResult):
                metrics_df = metrics.value
            elif isinstance(metrics, pd.DataFrame):
                metrics_df = metrics
            else:
                raise TypeError(f"Unsupported metrics type: {type(metrics)}")

            metrics_df.to_csv(path)
            self.logger.info(f"Metrics saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {str(e)}")
            raise

    def _log_performance(self, results: TradingResults) -> None:
        """Log comprehensive performance metrics from returns dataclasses."""
        try:
            self.logger.info("\n" + "=" * 50)
            self.logger.info(f"TRADING SUMMARY: {results.strategy_name}")
            self.logger.info("=" * 50)

            # Base strategy performance
            if results.base_returns:
                self._log_base_performance(results.base_returns)

            # Levered strategy performance
            if results.levered_returns:
                self._log_levered_performance(results.levered_returns)

        except Exception as e:
            self.logger.error(f"Performance logging failed: {str(e)}")

    def _log_base_performance(self, base_returns: BaseReturns) -> None:
        """Log base strategy performance metrics."""
        self.logger.info("\nBASE STRATEGY PERFORMANCE")
        self.logger.info("-" * 30)

        self.logger.debug("\nBaseReturns Statistics:")
        self.logger.debug(f"Number of returns: {len(base_returns.returns)}")
        self.logger.debug(f"Sum of returns: {base_returns.returns.sum():.4f}")
        self.logger.debug(f"Compound return: {(1 + base_returns.returns).prod() - 1:.4f}")

        if base_returns.summary_metrics:
            metrics = base_returns.summary_metrics
            self._log_metrics_summary(metrics)
        else:
            self.logger.warning("No base strategy metrics available")

    def _log_levered_performance(self, levered_returns: LeveredReturns) -> None:
        """Log levered strategy performance metrics."""
        self.logger.info("\nLEVERED STRATEGY PERFORMANCE")
        self.logger.info("-" * 30)

        if levered_returns.summary_metrics:
            metrics = levered_returns.summary_metrics
            self._log_metrics_summary(metrics)

            # Add position sizing statistics
            self._log_position_sizing(levered_returns.position_sizes)
        else:
            self.logger.warning("No levered strategy metrics available")

    def _log_metrics_summary(self, metrics: Dict[str, float]) -> None:
        """Log summary metrics in a standardized format."""
        key_metrics = {
            'total_return': ('Total Return', '{:.2%}'),
            'annualized_return': ('Annualized Return', '{:.2%}'),
            'volatility': ('Volatility', '{:.2%}'),
            'sharpe_ratio': ('Sharpe Ratio', '{:.2f}'),
            'sortino_ratio': ('Sortino Ratio', '{:.2f}'),
            'max_drawdown': ('Max Drawdown', '{:.2%}'),
            'win_rate': ('Win Rate', '{:.2%}'),
            'profit_factor': ('Profit Factor', '{:.2f}')
        }

        for key, (label, fmt) in key_metrics.items():
            if key in metrics:
                self.logger.info(f"{label}: {fmt.format(metrics[key])}")

    def _log_position_sizing(self, position_sizes: pd.Series) -> None:
        """Log position sizing statistics."""
        if position_sizes.empty:
            return

        self.logger.info("\nPosition Sizing Statistics:")
        self.logger.info(f"Average Size: {position_sizes.mean():.2f}x")
        self.logger.info(f"Maximum Size: {position_sizes.max():.2f}x")
        self.logger.info(f"Minimum Size: {position_sizes.min():.2f}x")

    def _log_trade_statistics(self, results: TradingResults) -> None:
        """Log comprehensive trade statistics."""
        try:
            trades = results.final_trades if results.final_trades else results.base_trades

            if not trades:
                self.logger.info("\nNo trades to analyze")
                return

            self.logger.info("\nTrade Statistics:")
            self.logger.info(f"Total trades: {len(trades)}")

            # Profitability metrics
            profitable_trades = [t for t in trades if t.pnl > t.costs]
            win_rate = len(profitable_trades) / len(trades)
            avg_profit = np.mean([t.pnl for t in profitable_trades]) if profitable_trades else 0
            avg_loss = np.mean([t.pnl for t in trades if t.pnl <= t.costs]) if len(trades) > len(
                profitable_trades) else 0

            self.logger.info(f"Win rate: {win_rate:.2%}")
            self.logger.info(f"Average profit: ${avg_profit:,.2f}")
            self.logger.info(f"Average loss: ${avg_loss:,.2f}")

            # Duration metrics
            durations = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades]
            self.logger.info("\nDuration Statistics:")
            self.logger.info(f"Average duration: {np.mean(durations):.1f} minutes")
            self.logger.info(f"Longest trade: {np.max(durations):.1f} minutes")
            self.logger.info(f"Shortest trade: {np.min(durations):.1f} minutes")

            # Exit analysis
            self.logger.info("\nExit Reasons:")
            exit_reasons = [t.exit_reason for t in trades]
            reason_counts = {}
            for reason in exit_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

            for reason, count in reason_counts.items():
                self.logger.info(f"  {reason}: {count} ({count / len(trades):.1%})")

        except Exception as e:
            self.logger.error(f"Trade statistics logging failed: {str(e)}")

    def _get_output_path(self, filename: str) -> Path:
        """Get path for output file in run directory."""
        return self.run_dir / filename

    def _save_results(self, results: TradingResults, save_path: Path) -> None:
        """Save complete trading results."""
        try:
            save_data = {
                'symbol': results.symbol,
                'strategy': results.strategy_name,
                'base_returns': results.base_returns,
                'levered_returns': results.levered_returns,
                'daily_performance': results.daily_performance,
                'timestamp': results.timestamp
            }

            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            self.logger.info(f"Results saved to {save_path}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise

def setup_logging() -> None:
    """
    Initialize the logging system.
    This should be called once at application startup.
    """
    LoggingConfig().setup()


def create_strategy(config: Config, symbol: str, invert_signals: bool = False) -> IntradayMomentumStrategy:
    """Create an instance of IntradayMomentumStrategy with proper parameters."""
    params = StrategyParameters(
        lookback_days=config.strategy_params['lookback_days'],
        volatility_multiplier=config.strategy_params['volatility_multiplier'],
        min_holding_period=pd.Timedelta(minutes=1),
        entry_times=[0, 30]
    )

    # Remove the contract_spec creation since it's handled in the Strategy base class
    return IntradayMomentumStrategy(
        config=config,
        params=params,
        invert_signals=invert_signals
    )

# Helper function to validate trading hours
def _validate_trading_hours(market_open: time, market_close: time, last_entry: time) -> None:
    """Validate trading hours are in correct sequence."""
    if market_open >= market_close:
        raise ValueError("Market open must be before market close")
    if last_entry <= market_open or last_entry >= market_close:
        raise ValueError("Last entry must be between market open and close")

def _initialize_configuration(params: Dict, logger: logging.Logger) -> Optional[Config]:
    """Initialize system configuration."""
    try:
        config = Config(config_file='config.yaml')

        # Validate symbol
        if params['symbol'] not in config.input_files:
            logger.error(
                f"Symbol '{params['symbol']}' not found in config. "
                f"Available symbols: {list(config.input_files.keys())}"
            )
            return None

        # Log paths
        logger.info(f"Input data path: {config.input_files[params['symbol']]}")
        logger.info(f"Output path: {config.output_paths[params['symbol']]}")

        return config

    except FileNotFoundError:
        logger.error("Configuration file 'config.yaml' not found")
        return None
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}", exc_info=True)
        return None



def _initialize_components(
        config: Config,
        params: Dict,
        logger: logging.Logger
) -> Optional[Tuple[DataManager, TradingSystem, Strategy]]:
    """Initialize system components."""
    try:
        # Initialize components
        data_manager = DataManager(config)
        trading_system = TradingSystem(config, data_manager)

        # Create strategy instance
        strategy = create_strategy(config, params['symbol'])

        logger.info("Successfully initialized system components")
        return data_manager, trading_system, strategy

    except Exception as e:
        logger.error(f"Failed to initialize system components: {str(e)}", exc_info=True)
        return None


def _save_results(
        results: pd.DataFrame,
        metrics: Dict,
        logger: logging.Logger,
        filename: str = 'last_run_results.pkl'
) -> None:
    """Save results to pickle file with error handling."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump({
                'results': results,
                'metrics': metrics,
                'timestamp': datetime.now()
            }, f)
        logger.info(f"Results saved to '{filename}'")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}", exc_info=True)


def main() -> Optional[TradingResults]:
    """Main execution function for the trading system."""
    logger = LoggingConfig.get_logger(__name__)

    try:
        # Load configuration without logging steps
        try:
            config = Config('config.yaml')
        except Exception as e:
            logger.error(f"Configuration loading failed: {str(e)}")
            return None

        # Initialize trading system
        try:
            trading_system = TradingSystem(config)
        except Exception as e:
            logger.error(f"Trading system initialization failed: {str(e)}")
            return None

        # Create strategy
        try:
            strategy = create_strategy(config, config.symbol)
        except Exception as e:
            logger.error(f"Strategy creation failed: {str(e)}")
            return None

        # Execute strategy - all step logging happens in run()
        try:
            trading_results = trading_system.run(
                strategy=strategy,
                symbol=config.symbol,
                days_to_analyze=config.days_to_analyze,
                lookback_buffer=config.lookback_buffer
            )

            if trading_results is None:
                logger.error("Strategy execution failed to produce results")
                return None

        except Exception as e:
            logger.error(f"Strategy execution failed: {str(e)}")
            return None

        # Generate reports
        try:
            reporting_system = ReportingSystem(config)
            reporting_system.generate_reports(trading_results)
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")

        return trading_results

    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    # Set up logging configuration first thing
    setup_logging()
    logger = LoggingConfig.get_logger(__name__)

    # Execute main function
    results = main()

    # Set exit code based on execution success
    if results is None:
        sys.exit(1)
    else:
        sys.exit(0)