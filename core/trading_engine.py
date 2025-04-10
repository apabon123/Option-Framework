"""
Trading Engine Module

This module provides a comprehensive framework for backtesting and executing trading
strategies with position management, margin calculation, and performance reporting.
"""

import os
import logging
import traceback
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import io
import sys
from tabulate import tabulate
import re
import math
import time
import platform

# Handle imports properly whether run as script or imported as module
if __name__ == "__main__":
    # When run as a script, use absolute imports
    from core.data_manager import DataManager
    from core.position import Position, OptionPosition
    from core.portfolio import Portfolio
    from core.margin import MarginCalculator, OptionMarginCalculator, SPANMarginCalculator
    from core.hedging import HedgingManager
    from core.reporting import ReportingSystem
    from core.risk_manager import RiskManager
    from core.margin_management import PortfolioRebalancer
    from core.position_sizer import PositionSizer
    from core.risk_scaler import RiskScaler
else:
    # When imported as a module, use relative imports
    from .data_manager import DataManager
    from .position import Position, OptionPosition
    from .portfolio import Portfolio
    from .margin import MarginCalculator, OptionMarginCalculator, SPANMarginCalculator
    from .hedging import HedgingManager
    from .reporting import ReportingSystem
    from .risk_manager import RiskManager
    from .margin_management import PortfolioRebalancer
    from .position_sizer import PositionSizer
    from .risk_scaler import RiskScaler


class LoggingManager:
    """
    Manages logging configuration and operations for trading applications.

    This class centralizes all logging functionality including setup,
    filtering, and redirection of output streams.
    """

    def __init__(self):
        """Initialize the LoggingManager"""
        self.logger = None
        self.log_file = None
        self.original_stdout = None
        self.original_stderr = None
        self.component_log_levels = {}

    def log_section_header(self, title: str, date: Optional[datetime] = None) -> None:
        """
        Log a section header with optional date.

        Args:
            title: Section title
            date: Optional date to include in header
        """
        if self.logger:
            header = "=" * 50
            self.logger.info(header)
            if date:
                self.logger.info(f"{title} [{date.strftime('%Y-%m-%d')}]:")
            else:
                self.logger.info(f"{title}:")

    def log_section_footer(self) -> None:
        """Log a section footer."""
        if self.logger:
            self.logger.info("=" * 50)

    def log_subsection_header(self, title: str) -> None:
        """
        Log a subsection header.

        Args:
            title: Subsection title
        """
        if self.logger:
            self.logger.info("-" * 50)
            self.logger.info(f"{title}:")

    def log_trade_manager(self, message: str) -> None:
        """
        Log a TradeManager message for trade execution tracking.

        Args:
            message: The trade manager message to log
        """
        if self.logger:
            self.logger.info(f"[TradeManager] {message}")

    def log_risk_scaling(self, message: str, indent: bool = False) -> None:
        """
        Log risk scaling metrics and decisions.

        Args:
            message: The risk scaling message to log
            indent: If True, indents the message for nested logging
        """
        if self.logger:
            prefix = "  " if indent else ""
            self.logger.info(f"{prefix}[Risk Scaling] {message}")

    def log_hedge_update(self, message: str) -> None:
        """
        Log hedge position updates.

        Args:
            message: The hedge update message to log
        """
        if self.logger:
            self.logger.info(f"[Hedge Update] {message}")

    def log_position_update(self, message: str) -> None:
        """
        Log position updates.

        Args:
            message: The position update message to log
        """
        if self.logger:
            self.logger.info(f"[Position Update] {message}")

    def log_daily_return(self, message: str, indent: bool = False) -> None:
        """
        Log daily return metrics.

        Args:
            message: The daily return message to log
            indent: If True, indents the message for nested logging
        """
        if self.logger:
            prefix = "  " if indent else ""
            self.logger.info(f"{prefix}[Daily Return] {message}")

    def setup_logging(self, config_dict: Dict[str, Any], verbose_console: bool = True,
                      debug_mode: bool = False, clean_format: bool = True) -> logging.Logger:
        """
        Set up logging configuration.

        Args:
            config_dict: Configuration dictionary
            verbose_console: Whether to output verbose logs to console
            debug_mode: Enable debug level logging
            clean_format: Use clean logging format without timestamp/level

        Returns:
            logging.Logger: Configured logger
        """
        # Get log level from config
        log_level_str = config_dict.get('logging', {}).get('level', 'INFO')
        log_levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        # Use DEBUG if debug_mode is True, otherwise use the configured level
        log_level = logging.DEBUG if debug_mode else log_levels.get(
            log_level_str, logging.INFO)

        # Set up component-specific log levels
        self.component_log_levels = {}

        # First check for component-specific levels in the new 'component_levels' structure
        if 'logging' in config_dict and 'component_levels' in config_dict['logging']:
            for component, level in config_dict['logging']['component_levels'].items():
                self.component_log_levels[component] = level
                
        # Check for component-specific logging settings in the older 'components' structure
        elif 'logging' in config_dict and 'components' in config_dict['logging']:
            # Check for margin logging settings
            if 'margin' in config_dict['logging']['components']:
                level = config_dict['logging']['components']['margin'].get(
                    'level', 'standard')
                self.component_log_levels['margin'] = str(
                    level) if level is not None else 'standard'

            # Check for portfolio logging settings
            if 'portfolio' in config_dict['logging']['components']:
                level = config_dict['logging']['components']['portfolio'].get(
                    'level', 'standard')
                self.component_log_levels['portfolio'] = str(
                    level) if level is not None else 'standard'
        else:
            # Fallback to the old structure for backward compatibility
            # Check for margin logging settings
            if 'margin' in config_dict and 'logging' in config_dict['margin']:
                level = config_dict['margin']['logging'].get(
                    'level', 'standard')
                self.component_log_levels['margin'] = str(
                    level) if level is not None else 'standard'

            # Check for portfolio logging settings
            if 'portfolio' in config_dict and 'logging' in config_dict['portfolio']:
                level = config_dict['portfolio']['logging'].get(
                    'level', 'standard')
                self.component_log_levels['portfolio'] = str(
                    level) if level is not None else 'standard'

        # Print the component log levels for debugging
        print(f"Component log levels after setup: {self.component_log_levels}")

        # Create logger
        logger = logging.getLogger('trading_engine')
        logger.setLevel(log_level)

        # Clear any existing handlers
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        # Determine log file path
        log_filename = self.build_log_filename(config_dict)

        # Get output directory from config and ensure it exists
        output_dir = config_dict.get('paths', {}).get('output_dir', '.')
        os.makedirs(output_dir, exist_ok=True)

        # Create full log file path
        log_file = os.path.join(output_dir, log_filename)
        self.log_file = log_file

        # Create file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)  # Use the same level for file

        if clean_format:
            # Use a clean format without timestamps and log levels
            file_formatter = logging.Formatter('%(message)s')
        else:
            # Use a standard format with timestamps and log levels
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s')

        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

        # Create console handler for verbose output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Create custom formatter for console output
        class ColoredFormatter(logging.Formatter):
            """Custom formatter with colored output for console."""

            COLORS = {
                'WARNING': '\033[93m',  # Yellow
                'INFO': '\033[92m',     # Green
                'DEBUG': '\033[94m',    # Blue
                'CRITICAL': '\033[91m',  # Red
                'ERROR': '\033[91m',    # Red
                'ENDC': '\033[0m',      # End color
                'BOLD': '\033[1m',      # Bold
                'UNDERLINE': '\033[4m'  # Underline
            }

            def format(self, record):
                # Get the plain message
                message = super().format(record)

                # Apply color formatting based on message content
                if record.levelno >= logging.WARNING:
                    # Warning/Error messages in yellow/red
                    level_color = self.COLORS.get(record.levelname, '')
                    message = f"{level_color}{message}{self.COLORS['ENDC']}"
                elif '[STATUS]' in message:
                    # Status messages in bold green
                    message = f"{self.COLORS['BOLD']}{self.COLORS['INFO']}{message}{self.COLORS['ENDC']}"
                elif record.levelno == logging.INFO:
                    # Regular info messages in green
                    message = f"{self.COLORS['INFO']}{message}{self.COLORS['ENDC']}"
                else:
                    # Debug messages in blue
                    message = f"{self.COLORS['DEBUG']}{message}{self.COLORS['ENDC']}"

                return message

        # Use clean format for console to match file output
        if clean_format:
            console_formatter = ColoredFormatter('%(message)s')
        else:
            console_formatter = ColoredFormatter('%(asctime)s - %(message)s')

        console_handler.setFormatter(console_formatter)

        # Only add console handler if verbose_console is True
        if verbose_console:
            logger.addHandler(console_handler)

        # Also redirect stderr to logger for error handling
        sys.stderr = io.StringIO()

        # Log basic information about the application
        logger.info(f"Logger initialized at {log_file}")

        # Store logger for later use
        self.logger = logger

        # Set the manager attribute on the logger so MarginCalculator can use it
        setattr(logger, 'manager', self)

        return logger

    def build_log_filename(self, config_dict: Dict[str, Any]) -> str:
        """
        Build a descriptive log filename based on configuration.

        Args:
            config_dict: Configuration dictionary

        Returns:
            str: Formatted log filename
        """
        # Get strategy name
        strategy_name = config_dict.get(
            'strategy', {}).get('name', 'UnknownStrategy')

        # Get current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Check if this is a backtest or live trading
        mode = config_dict.get('run_mode', 'backtest')

        # Format filename
        filename = f"{strategy_name}_{mode}_{timestamp}.log"

        return filename

    def get_log_file_path(self) -> Optional[str]:
        """
        Get the path to the current log file.

        Returns:
            str: Path to log file, or None if not set
        """
        return self.log_file

    def cleanup(self):
        """Clean up logging resources and restore stdout/stderr."""
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr

    def get_component_log_level(self, component_name: str) -> str:
        """
        Get the log level for a specific component.

        Args:
            component_name: Name of the component (e.g., 'margin', 'portfolio')

        Returns:
            str: Log level for the component
        """
        if component_name in self.component_log_levels:
            # Make sure we're returning a string
            level = self.component_log_levels[component_name]
            if isinstance(level, str):
                return level
            # Try to convert to string if it's not already
            try:
                return str(level)
            except:
                return "standard"  # Default if conversion fails

        # Default to standard level
        return "standard"

    def should_log(self, *args, **kwargs):
        """
        Simplified should_log method that always returns True to avoid comparison errors.
        """
        # This is a temporary workaround - always allow logging
        return True

    def disable(self):
        """
        Disable logging.
        This method is added for compatibility with code expecting a disable method.
        """
        if self.logger:
            self.logger.disabled = True

    def _clear_cache(self):
        """
        Clear any internal caches.
        This method is added for compatibility with code expecting a _clear_cache method.
        """
        # Currently no cache to clear, but included for compatibility
        pass


class Strategy:
    """
    Base strategy class for implementing trading logic.

    This is an abstract base class that defines the interface for strategies.
    Concrete strategy implementations should inherit from this class and
    override the required methods.
    """

    def __init__(self, name: str, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the strategy.

        Args:
            name: Strategy name
            config: Strategy configuration dictionary
            logger: Logger instance
        """
        self.name = name
        self.config = config
        self.logger = logger or logging.getLogger('trading_engine')

        # Performance tracking
        self.performance_history = []

    def generate_signals(self, current_date: datetime, daily_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate trading signals for the current date.

        This is the main entry point for strategy implementations.

        Args:
            current_date: Current trading date
            daily_data: Data for the current trading day

        Returns:
            list: List of trading signal dictionaries
        """
        # This is an abstract method that should be overridden by concrete strategies
        self.logger.warning(
            "Using base Strategy.generate_signals - this should be overridden")
        return []

    def check_exit_conditions(self, position: Position, market_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a position should be exited based on market conditions.

        Args:
            position: Position to check
            market_data: Current market data for the position's symbol

        Returns:
            tuple: (should_exit, reason) - Boolean indicating whether to exit, and the reason
        """
        # This is an abstract method that should be overridden by concrete strategies
        return False, "No exit condition met"

    def update_metrics(self, portfolio_metrics: Dict[str, Any]) -> None:
        """
        Update strategy-specific metrics based on portfolio performance.

        Args:
            portfolio_metrics: Portfolio performance metrics
        """
        # Store performance metrics
        self.performance_history.append(portfolio_metrics)


class TradingEngine:
    """
    Core trading engine for backtesting and simulation.

    This class orchestrates the components of a trading system:
    - Data Management: Loading and preprocessing price data
    - Portfolio Management: Tracking positions and calculating performance
    - Strategy: Generating signals and trade ideas
    - Execution: Simulating order fills and tracking trades
    - Reporting: Generating performance reports and visualizations
    """

    def __init__(self, config: Dict[str, Any], strategy=None, logger=None):
        """
        Initialize the Trading Engine.

        Args:
            config: Configuration dictionary
            strategy: Optional strategy instance (will be created from config if not provided)
            logger: Optional logger instance
        """
        # Store config and other initialization parameters
        self.config = config
        self._strategy_instance = strategy

        # Set up logging first
        self.logging_manager = LoggingManager()
        self.logger = logger or self.logging_manager.setup_logging(
            config,
            verbose_console=config.get('logging', {}).get(
                'verbose_console', True),
            debug_mode=config.get('logging', {}).get('debug_mode', False),
            clean_format=config.get('logging', {}).get('clean_format', True)
        )

        # Introduce ourselves to the logs
        self.logger.info("=" * 80)
        self.logger.info("TRADING ENGINE INITIALIZATION")
        self.logger.info("=" * 80)

        # Print the library versions in use
        self._log_environment_info()

        # Initialize data containers
        self.daily_data = None
        self.current_date = None
        self.days_processed = 0  # Track number of days processed

        # Initialize tracking metrics
        self.metrics_history = []

        # Extract main configuration parameters
        self.initial_capital = self.config.get(
            'portfolio', {}).get('initial_capital', 100000)
        self.max_leverage = self.config.get(
            'portfolio', {}).get('max_leverage', 1.0)

        # Initialize current date and other date-related fields
        self.start_date = self._parse_date(
            self.config.get('dates', {}).get('start_date'))
        self.end_date = self._parse_date(
            self.config.get('dates', {}).get('end_date'))
        self.current_date = None

        # Initialize containers for data
        self.portfolio = None  # Will be initialized in _init_components
        self.risk_manager = None  # Will be initialized in _init_components
        self.strategy = None  # Will be initialized in _init_components
        self.data_manager = None  # Will be initialized in _init_components
        self.margin_calculator = None  # Will be initialized in _init_components
        self.margin_manager = None  # Will be initialized in _init_components
        self.reporting_system = None  # Will be initialized in _init_components
        self.hedging_manager = None  # Will be set up if hedging is enabled

        # Tracking variables
        self.performance_metrics = {
            'daily_returns': [],
            'equity_curve': [],
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'risk_metrics': {}
        }

        # Tracking dictionaries for daily activities
        self.today_added_positions = {}  # Track positions added today by symbol
        self.today_signals_by_symbol = {}  # Track signals generated today by symbol

        # Store starting time for performance tracking
        self.start_time = time.time()

        # Store original date format from config for consistency
        self.date_format = self.config.get('date_format', '%Y-%m-%d')
        
        # Cache for underlying prices
        self.cached_underlying_prices = {}

        # Initialize components (portfolio, risk manager, etc.)
        self._init_components()

    def _init_components(self) -> None:
        """Initialize trading engine components."""
        self.logger.info("=" * 60)
        self.logger.info("TRADING ENGINE COMPONENT INITIALIZATION")
        self.logger.info("=" * 60)

        # Get configuration values from the portfolio section
        portfolio_config = self.config.get('portfolio', {})
        max_position_size = portfolio_config.get('max_position_size_pct', 0.05)

        # Create margin calculator first, as it's needed by portfolio
        # First check margin_management config, then fall back to margin config
        margin_config = self.config.get(
            'margin_management', self.config.get('margin', {}))

        # Check for included margin config
        if 'includes' in self.config and 'margin_config' in self.config['includes']:
            # If there's an included margin config, it takes precedence
            included_margin_path = self.config['includes']['margin_config']
            self.logger.info(
                f"Using included margin config: {included_margin_path}")

            # If the included file exists and has a margin_calculator_type, use that
            if os.path.exists(included_margin_path):
                try:
                    with open(included_margin_path, 'r') as f:
                        import yaml
                        included_margin_config = yaml.safe_load(f)
                        # Merge the included config with existing margin config
                        for key, value in included_margin_config.items():
                            if key not in margin_config:
                                margin_config[key] = value
                    self.logger.info(
                        f"Successfully loaded included margin config")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load included margin config: {e}")

        # Determine the margin calculator type to use
        margin_type = margin_config.get('margin_calculator_type', 'standard')

        # Log which margin calculator type we're using
        self.logger.info(f"INITIALIZING MARGIN CALCULATOR: {margin_type}")

        if margin_type.lower() == 'option':
            self.margin_calculator = OptionMarginCalculator(
                max_leverage=self.max_leverage,
                logger=self.logger
            )
            # Set logging manager separately
            self.margin_calculator.logging_manager = self.logging_manager
        elif margin_type.lower() == 'span':
            # Load SPAN parameters if provided in config
            span_params = margin_config.get('span_parameters', {})

            # Extract explicit SPAN parameters from config
            price_move_pct = span_params.get('price_move_pct', 0.05)
            vol_shift_pct = span_params.get('vol_shift_pct', 0.3)
            initial_margin_percentage = span_params.get(
                'initial_margin_percentage', 0.1)
            hedge_credit_rate = span_params.get('hedge_credit_rate', 0.8)
            min_scan_risk_percentage = span_params.get(
                'min_scan_risk_percentage', 0.25)
            max_margin_to_premium_ratio = span_params.get(
                'max_margin_to_premium_ratio', 20.0)

            self.logger.info(
                f"Creating SPAN margin calculator with parameters:")
            self.logger.info(
                f"  Initial margin %: {initial_margin_percentage:.2%}")
            self.logger.info(f"  Price move %: {price_move_pct:.2%}")
            self.logger.info(f"  Vol shift %: {vol_shift_pct:.2%}")
            self.logger.info(f"  Hedge credit: {hedge_credit_rate:.2%}")

            # Create with parameters from config
            self.margin_calculator = SPANMarginCalculator(
                max_leverage=self.max_leverage,
                initial_margin_percentage=initial_margin_percentage,
                price_move_pct=price_move_pct,
                vol_shift_pct=vol_shift_pct,
                hedge_credit_rate=hedge_credit_rate,
                min_scan_risk_percentage=min_scan_risk_percentage,
                max_margin_to_premium_ratio=max_margin_to_premium_ratio,
                logger=self.logger
            )
            # Set logging manager separately
            self.margin_calculator.logging_manager = self.logging_manager

            # Store reference to SPAN calculator for use in position sizing
            self._span_calculator = self.margin_calculator
        else:
            # Use standard margin calculator
            self.logger.warning(
                f"Using basic margin calculator - this is not recommended for options trading")
            self.margin_calculator = MarginCalculator(
                max_leverage=self.max_leverage,
                logger=self.logger
            )
            # Set logging manager separately
            self.margin_calculator.logging_manager = self.logging_manager

        # Create portfolio with the margin calculator
        self.logger.info("INITIALIZING PORTFOLIO")
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            max_position_size_pct=max_position_size,
            max_portfolio_delta=portfolio_config.get(
                'max_portfolio_delta', 0.2),
            logger=self.logger,
            margin_calculator=self.margin_calculator
        )

        # Check if hedging is enabled in the configuration
        hedging_enabled = self.config.get('hedging', {}).get('enabled', False)

        # Initialize the hedging manager if hedging is enabled - do this before position sizer
        if hedging_enabled:
            self.logger.info("INITIALIZING HEDGING MANAGER (ENABLED)")
            hedging_config = self.config.get('hedging', {})
            self.hedging_manager = HedgingManager(
                portfolio=self.portfolio,
                config=hedging_config,
                logger=self.logger
            )
            self.logger.info(
                f"  Hedge symbol: {self.hedging_manager.hedge_symbol}")
            self.logger.info(
                f"  Delta hedging: {'Enabled' if self.hedging_manager.enable_hedging else 'Disabled'}")
            self.logger.info(
                f"  Target delta ratio: {hedging_config.get('target_delta_ratio', 0.0)}")
        else:
            self.hedging_manager = None
            self.logger.info("HEDGING: DISABLED")

        # Initialize the risk scaler if risk scaling is enabled
        risk_scaling_config = self.config.get('risk_scaling', {})
        if risk_scaling_config.get('enabled', True):  # Default to enabled for backward compatibility
            self.logger.info("INITIALIZING RISK SCALER")
            self.risk_scaler = RiskScaler(
                config=self.config,
                logger=self.logger
            )
        else:
            self.risk_scaler = None
            self.logger.info("RISK SCALING: DISABLED")

        # Create position sizer (formerly risk manager)
        position_sizing_config = self.config.get('position_sizing', {})
        if position_sizing_config.get('enabled', True):  # Default to enabled for backward compatibility
            self.logger.info("INITIALIZING POSITION SIZER")
            self.position_sizer = PositionSizer(
                config=self.config,
                logger=self.logger
            )

            # Connect hedging manager to position sizer if both exist
            if self.hedging_manager and hasattr(self.position_sizer, 'hedging_manager'):
                self.position_sizer.hedging_manager = self.hedging_manager
                self.logger.info(
                    "[TradingEngine] Connected hedging manager to position sizer for integrated margin calculation")

            # Connect risk scaler to position sizer if both exist
            if self.risk_scaler and hasattr(self.position_sizer, 'risk_scaler'):
                self.position_sizer.risk_scaler = self.risk_scaler
                self.logger.info(
                    "[TradingEngine] Connected risk scaler to position sizer for risk-adjusted position sizing")
        else:
            self.position_sizer = None
            self.logger.info("POSITION SIZING: DISABLED - Using fixed position sizes")

        # For backward compatibility, alias position_sizer as risk_manager
        self.risk_manager = self.position_sizer

        # Create data manager
        self.logger.info("INITIALIZING DATA MANAGER")
        self.data_manager = DataManager(
            config=self.config,
            logger=self.logger
        )

        # Create strategy
        self.logger.info("INITIALIZING STRATEGY")
        self.strategy = self._create_strategy()

        # Create the reporting system
        self.logger.info("INITIALIZING REPORTING SYSTEM")
        self.reporting_system = ReportingSystem(
            config=self.config,
            portfolio=self.portfolio,
            logger=self.logger,
            trading_engine=self
        )

        # Initialize margin management AFTER hedging manager is created
        if 'margin_management' in self.config:
            self.logger.info("INITIALIZING MARGIN MANAGER")
            # Initialize the margin manager with the portfolio
            self.margin_manager = PortfolioRebalancer(
                self.portfolio,
                self.config.get('margin_management', {}),
                self.logger
            )

            # Connect the hedging manager to the margin manager if both exist
            if self.hedging_manager and hasattr(self.margin_manager, 'hedging_manager'):
                self.margin_manager.hedging_manager = self.hedging_manager
                self.logger.info(
                    "[TradingEngine] Connected hedging manager to margin manager for integrated margin calculation")
        else:
            self.margin_manager = None
            self.logger.info("MARGIN MANAGEMENT: DISABLED")

        # Double-check all connections for proper integration
        if self.hedging_manager:
            # Verify risk manager connection
            if hasattr(self.risk_manager, 'hedging_manager') and self.risk_manager.hedging_manager == self.hedging_manager:
                self.logger.info(
                    "✓ Hedging integration with risk manager: VERIFIED")
            else:
                self.logger.warning(
                    "✗ Hedging integration with risk manager: FAILED - Connection not established")

            # Verify margin manager connection if it exists
            if self.margin_manager:
                if hasattr(self.margin_manager, 'hedging_manager') and self.margin_manager.hedging_manager == self.hedging_manager:
                    self.logger.info(
                        "✓ Hedging integration with margin manager: VERIFIED")
                else:
                    self.logger.warning(
                        "✗ Hedging integration with margin manager: FAILED - Connection not established")

        # Ensure portfolio has correct margin calculator
        calculator_class_name = type(self.portfolio.margin_calculator).__name__
        calculator_display_name = calculator_class_name
        if calculator_class_name == "SPANMarginCalculator":
            calculator_display_name = "SPAN"
        elif calculator_class_name == "OptionMarginCalculator":
            calculator_display_name = "Option"
        elif calculator_class_name == "MarginCalculator":
            calculator_display_name = "Basic"

        # Log verification of the portfolio's margin calculator
        self.logger.info(
            f"✓ Portfolio margin calculator: {calculator_display_name}")

        # Log summary of all initialized components
        self.logger.info("=" * 60)
        self.logger.info("INITIALIZATION COMPLETE - COMPONENT SUMMARY")
        self.logger.info("-" * 60)
        self.logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        self.logger.info(
            f"Margin calculator: {type(self.margin_calculator).__name__}")
        self.logger.info(f"Strategy: {type(self.strategy).__name__}")
        self.logger.info(
            f"Margin management: {'Enabled' if self.margin_manager else 'Disabled'}")
        self.logger.info(
            f"Hedging: {'Enabled' if self.hedging_manager else 'Disabled'}")

        # Log date range for the backtest
        if self.start_date and self.end_date:
            date_format = "%Y-%m-%d"
            self.logger.info(
                f"Backtest period: {self.start_date.strftime(date_format)} to {self.end_date.strftime(date_format)}")

        self.logger.info("=" * 60)

    def _create_strategy(self) -> Strategy:
        """
        Create and configure a strategy based on config.

        Returns:
            Strategy: Configured strategy instance
        """
        # If a strategy instance was provided in the constructor, use that
        if self._strategy_instance is not None:
            return self._strategy_instance

        # Otherwise, fall back to creating a base Strategy (which will warn that it should be overridden)
        strategy_config = self.config.get('strategy', {})
        strategy_name = strategy_config.get('name', 'DefaultStrategy')
        self.logger.warning(
            f"No strategy instance provided - creating base Strategy ({strategy_name})")
        return Strategy(name=strategy_name, config=strategy_config, logger=self.logger)

    def load_data(self) -> bool:
        """Load and preprocess data for trading."""
        if self.data_manager is None:
            self.logger.error("Data manager not initialized")
            return False

        # Load data
        self.logger.info("Loading data...")
        try:
            print("Attempting to load data file...")
            data = self.data_manager.load_data()

            if data is None or len(data) == 0:
                self.logger.error("No data loaded")
                print("ERROR: No data loaded from file")
                return False

            # Preprocess data
            print(f"Data loaded with {len(data)} rows. Preprocessing...")
            self.logger.info("Preprocessing data...")
            self.data = self.data_manager.preprocess_data(data)

            self.logger.debug(
                f"Data loaded: {len(self.data)} rows, {self.data.columns.tolist()}")
            print(
                f"Preprocessing complete - {len(self.data)} rows ready for trading")
            return True
        except Exception as e:
            self.logger.error(f"Error in load_data: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            print(f"ERROR loading data: {e}")
            return False

    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the backtest over the specified date range.

        Returns:
            Dict[str, Any]: Dictionary containing backtest results
        """
        # Initialize result tracking
        results = {
            'initial_value': self.portfolio.initial_capital,
            'final_value': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'report_path': None
        }

        # First, make sure data is loaded
        self.logger.info("Loading data for backtest...")
        if not self.load_data():
            self.logger.error("Failed to load data for backtest")
            return results

        # Get the dates for the backtest
        start_date = self.config.get('dates', {}).get('start_date')
        end_date = self.config.get('dates', {}).get('end_date')

        self.logger.info(
            f"Configuration date range: {start_date} to {end_date}")

        # Convert string dates to datetime objects if necessary
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            self.logger.debug(
                f"Converted start_date string to datetime: {start_date}")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            self.logger.debug(
                f"Converted end_date string to datetime: {end_date}")

        # Generate trading dates in the specified range
        self.logger.info(
            "Retrieving available trading dates from data_manager...")
        trading_dates = self.data_manager.get_dates()
        self.logger.info(
            f"Data manager returned {len(trading_dates)} trading dates")

        if not trading_dates:
            self.logger.error("No trading dates available")
            return results

        # Debug: print first few dates to see what format they're in
        if trading_dates:
            self.logger.debug(f"First 5 trading dates: {trading_dates[:5]}")
            self.logger.debug(f"Date type: {type(trading_dates[0])}")

        # Filter by date range
        original_count = len(trading_dates)
        if start_date:
            # Check if we need to convert date format
            if isinstance(trading_dates[0], str):
                trading_dates = [datetime.strptime(
                    d, '%Y-%m-%d') for d in trading_dates]

            # Try to handle different date object types
            filtered_dates = []
            for d in trading_dates:
                # Convert to date object if it's a datetime
                if isinstance(d, datetime):
                    d_date = d.date()
                elif isinstance(d, pd._libs.tslibs.timestamps.Timestamp):
                    d_date = d.date()
                else:
                    d_date = d

                # Compare with start_date (convert start_date to date if it's datetime)
                start_date_for_compare = start_date.date() if isinstance(
                    start_date, datetime) else start_date
                if d_date >= start_date_for_compare:
                    filtered_dates.append(d)

            trading_dates = filtered_dates
            self.logger.info(
                f"After start date filter: {len(trading_dates)} dates remain")

        if end_date:
            # Handle different date object types for end date filtering
            filtered_dates = []
            for d in trading_dates:
                # Convert to date object if it's a datetime
                if isinstance(d, datetime):
                    d_date = d.date()
                elif isinstance(d, pd._libs.tslibs.timestamps.Timestamp):
                    d_date = d.date()
                else:
                    d_date = d

                # Compare with end_date (convert end_date to date if it's datetime)
                end_date_for_compare = end_date.date() if isinstance(
                    end_date, datetime) else end_date
                if d_date <= end_date_for_compare:
                    filtered_dates.append(d)

            trading_dates = filtered_dates
            self.logger.info(
                f"After end date filter: {len(trading_dates)} dates remain")

        trading_dates = sorted(trading_dates)

        if not trading_dates:
            self.logger.error("No trading dates in the specified range")
            self.logger.error(
                f"Original date range had {original_count} dates before filtering")
            return results

        # Print backtest details
        self.logger.info(
            f"Backtest range: {trading_dates[0]} to {trading_dates[-1]}")
        self.logger.info(f"Total trading days: {len(trading_dates)}")
        print(f"Backtest range: {trading_dates[0]} to {trading_dates[-1]}")
        print(f"Total trading days: {len(trading_dates)}")

        self.logger.info(
            f"Starting backtest with strategy: {self.strategy.name}")
        self.logger.info(
            f"Initial capital: ${self.portfolio.initial_capital:,.2f}")

        # Process each trading day
        total_dates = len(trading_dates)
        for i, current_date in enumerate(trading_dates):
            # Print progress every 5% or every day if total days <= 20
            if total_dates <= 20 or i % max(1, int(total_dates * 0.05)) == 0:
                progress_pct = (i / total_dates) * 100
                print(
                    f"Progress: {i}/{total_dates} days ({progress_pct:.1f}%) - Processing {current_date.strftime('%Y-%m-%d')}")

            # Process the current trading day
            self._process_trading_day(current_date)

        # Calculate backtest results
        final_value = self.portfolio.get_portfolio_value()
        initial_value = self.portfolio.initial_capital

        # Calculate total return
        if initial_value > 0:
            total_return = (final_value - initial_value) / initial_value
        else:
            total_return = 0

        # Calculate other metrics
        rolling_metrics = self.calculate_rolling_metrics()

        # Update results
        results['final_value'] = final_value
        results['total_return'] = total_return
        results['sharpe_ratio'] = rolling_metrics.get('sharpe_ratio', 0)
        results['max_drawdown'] = rolling_metrics.get('max_drawdown', 0)

        # Generate report
        if hasattr(self, 'reporting_system') and self.reporting_system:
            print("Generating HTML report...")
            self.logger.info("===========================================")
            self.logger.info(
                f"Generating HTML report '{self.strategy.name}_backtest'...")

            try:
                report_path = self.reporting_system.generate_html_report(
                    self.portfolio,
                    self.strategy.name,
                    self.strategy.config
                )

                results['report_path'] = report_path

                self.logger.info(f"Report saved to {report_path}")
                self.logger.info("===========================================")
            except Exception as e:
                self.logger.error(f"Error generating report: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

        # Log final results
        self.logger.info("===========================================")
        self.logger.info(
            f"Backtest completed. Final value: ${final_value:,.2f}")
        self.logger.info(f"Total return: {total_return:.2%}")

        print(f"Backtest complete - Final value: ${final_value:,.2f}")

        # Enhance reporting output to include verification files
        if hasattr(self, 'reporting_system') and self.reporting_system:
            try:
                final_report_data = self.reporting_system.run_reports(
                    self.portfolio, self, self.current_date)

                # Add verification files to results
                if 'pre_trade_file' in final_report_data:
                    results['pre_trade_verification_file'] = final_report_data['pre_trade_file']
                if 'post_trade_file' in final_report_data:
                    results['post_trade_verification_file'] = final_report_data['post_trade_file']

                # Log the verification file paths
                if 'pre_trade_verification_file' in results:
                    self.logger.info(
                        f"Pre-trade verification file: {results['pre_trade_verification_file']}")
                if 'post_trade_verification_file' in results:
                    self.logger.info(
                        f"Post-trade verification file: {results['post_trade_verification_file']}")
            except Exception as e:
                self.logger.error(f"Error generating final reports: {e}")
                self.logger.debug(traceback.format_exc())

        return results

    def _process_trading_day(self, current_date: datetime) -> None:
        """
        Process a single trading day, including open position updates, 
        signal generation, and execution.
        
        Args:
            current_date: Date to process
        """
        # Store the current date for use in other methods
        self.current_date = current_date
        
        # Get daily data for this date
        daily_data = self.data_manager.get_data_for_date(current_date)
        
        if daily_data is None or len(daily_data) == 0:
            self.logger.warning(f"No data available for {current_date}, skipping day")
            return
            
        # Update positions with latest market data
        self._update_positions(current_date, daily_data)
        
        # Log pre-trade summary
        # Check for skip_summaries in a safer way
        skip_summaries = False
        if hasattr(self, 'mode_settings') and isinstance(self.mode_settings, dict):
            skip_summaries = self.mode_settings.get('skip_summaries', False)
        
        if not skip_summaries:
            self._log_pre_trade_summary(current_date, daily_data)
        
        # Perform position management (exit checks, etc.)
        self._manage_positions(current_date, daily_data)
        
        # Execute trading activity (signal generation and execution)
        self._execute_trading_activities(current_date, daily_data)
        
        # Record daily metrics in the portfolio
        self.portfolio.record_daily_metrics(current_date)
        
        # Log performance metrics
        self._log_rolling_risk_metrics()

    def _log_pre_trade_summary(self, current_date, daily_data, add_header=True):
        """
        Log a pre-trade summary of the current portfolio status.
        
        Args:
            current_date: The current trading date
            daily_data: Daily market data
            add_header: Whether to add a header to the summary
        """
        if not self.portfolio:
            self.logger.warning("No portfolio available for pre-trade summary")
            return

        # Format the header with the current date
        formatted_date = current_date
        if isinstance(current_date, datetime):
            formatted_date = current_date.strftime('%Y-%m-%d')
        elif not isinstance(current_date, str):
            formatted_date = str(current_date)
        
        if add_header:
            self.logging_manager.log_section_header(f"PRE-TRADE SUMMARY - {formatted_date}")
        
        # Get current portfolio value
        portfolio_value = self.portfolio.get_portfolio_value()
        self.logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
        
        # Calculate daily PnL if we have previous value
        if hasattr(self, 'previous_portfolio_value') and self.previous_portfolio_value > 0:
            daily_pnl = portfolio_value - self.previous_portfolio_value
            daily_pnl_percent = (daily_pnl / self.previous_portfolio_value) * 100
            
            # Initialize P&L components
            option_pnl = 0
            equity_pnl = 0
            
            # Calculate P&L breakdown by position type
            for symbol, position in self.portfolio.positions.items():
                if (hasattr(position, 'position_type') and position.position_type == 'option') or isinstance(position, OptionPosition):
                    # For options, calculate based on price change (similar to equity)
                    # For short options: (previous_day_price - current_price) * contracts * 100
                    # For long options: (current_price - previous_day_price) * contracts * 100
                    if hasattr(position, 'previous_day_price') and hasattr(position, 'current_price'):
                        # Use previous_day_price for daily P&L calculations
                        price_diff = position.previous_day_price - position.current_price if position.is_short else position.current_price - position.previous_day_price
                        # Use previous_day_contracts instead of current contracts for consistent P&L calculation
                        pos_option_pnl = price_diff * abs(position.previous_day_contracts) * 100
                        option_pnl += pos_option_pnl
                        self.logger.debug(f"Debug - Option P&L for {symbol}: previous_day_price={position.previous_day_price}, current_price={position.current_price}, price_diff={price_diff}, contracts={position.previous_day_contracts}, is_short={position.is_short}, P&L=${pos_option_pnl:.2f}")
                elif hasattr(position, 'contracts') and hasattr(position, 'previous_day_price') and hasattr(position, 'current_price'):
                    # For equities, calculate P&L based on price change
                    # Use previous_day_contracts instead of current contracts for consistent P&L calculation
                    if hasattr(position, 'is_short') and position.is_short:
                        # For short positions, price decrease is profit, price increase is loss
                        pos_equity_pnl = position.previous_day_contracts * (position.previous_day_price - position.current_price)
                    else:
                        # For long positions, price increase is profit, price decrease is loss
                        pos_equity_pnl = position.previous_day_contracts * (position.current_price - position.previous_day_price)
                    equity_pnl += pos_equity_pnl
                    self.logger.debug(f"Debug - Equity P&L for {symbol}: previous_day_price={position.previous_day_price}, current_price={position.current_price}, contracts={position.previous_day_contracts}, is_short={getattr(position, 'is_short', False)}, P&L=${pos_equity_pnl:.2f}")
            
            # Cash/other P&L is what's left after accounting for options and equities
            cash_pnl = daily_pnl - option_pnl - equity_pnl
            
            # Log the P&L breakdown
            self.logger.info(f"Daily P&L: ${daily_pnl:.2f} ({daily_pnl_percent:.2f}%)")
            self.logger.info(f"  • Option P&L: ${option_pnl:.2f}")
            self.logger.info(f"  • Equity P&L: ${equity_pnl:.2f}")
            self.logger.info(f"  • Cash/Other P&L: ${cash_pnl:.2f}")
        
        # Log open positions
        self._log_open_positions()
        
        # Log margin details - calculate margin just once and store results for reuse
        # FIX: Cache the portfolio metrics after calculation to avoid redundant calls
        portfolio_metrics = self.portfolio.get_portfolio_metrics()
        self._cached_portfolio_metrics = portfolio_metrics
        self._log_margin_details(detailed=True, reuse_existing_metrics=True)
        
        # Log portfolio Greek risk - reuse the cached metrics
        self._log_portfolio_greek_risk()

    def _log_open_positions(self):
        """
        Log open positions including options and equities.
        """
        # Create a table for option positions
        option_positions = []
        zero_contract_positions = []
        
        # Process option positions
        for symbol, position in self.portfolio.positions.items():
            # Skip positions with 0 contracts
            if position.contracts == 0:
                zero_contract_positions.append(symbol)
                continue
                
            if isinstance(position, OptionPosition):
                # Get option details
                option_type = position.option_type
                strike = f"${float(position.strike):.2f}"
                
                # Get expiry date - try multiple attributes since the class might have different naming
                expiry = "N/A"
                dte = 0
                
                # Try different attribute names for expiry
                if hasattr(position, 'expiry_date'):
                    expiry = position.expiry_date
                elif hasattr(position, 'expiration'):
                    expiry = position.expiration
                elif hasattr(position, 'expiry'):
                    expiry = position.expiry
                elif hasattr(position, 'option_data') and isinstance(position.option_data, dict):
                    # Try to get from option_data dictionary
                    expiry = position.option_data.get('Expiration', position.option_data.get('expiry', 'N/A'))
                
                # Calculate DTE if we have a valid expiry
                if expiry != "N/A":
                    if isinstance(expiry, str):
                        try:
                            expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
                            # Use the current date from the backtest
                            if hasattr(self, 'current_date') and self.current_date:
                                current_date = self.current_date
                            else:
                                current_date = datetime.now()
                            # Calculate absolute value of days difference
                            dte = abs((expiry_date - current_date).days)
                        except ValueError:
                            dte = 0
                    elif isinstance(expiry, datetime):
                        # Use the current date from the backtest
                        if hasattr(self, 'current_date') and self.current_date:
                            current_date = self.current_date
                        else:
                            current_date = datetime.now()
                        # Calculate absolute value of days difference
                        dte = abs((expiry - current_date).days)
                
                # Use days_to_expiry attribute if available and we couldn't calculate DTE
                if dte == 0 and hasattr(position, 'days_to_expiry'):
                    dte = abs(position.days_to_expiry)  # Make sure it's positive
                
                # Get position size and direction
                contracts = position.contracts
                direction = "-" if position.is_short else ""
                
                # Get price and value
                price = position.current_price
                value = position.current_price * abs(contracts) * 100
                
                # Get Greeks
                delta = position.current_delta * contracts
                gamma = position.current_gamma * contracts
                theta = position.current_theta * contracts
                vega = position.current_vega * contracts
                
                # Calculate total P&L (realized + unrealized)
                total_pnl = 0
                if hasattr(position, 'realized_pnl') and hasattr(position, 'unrealized_pnl'):
                    total_pnl = position.realized_pnl + position.unrealized_pnl
                
                # Format the row data for tabulate
                option_positions.append([
                    symbol,
                    option_type,
                    strike,
                    expiry,
                    dte,
                    f"{direction}{contracts}",
                    f"${price:.2f}",
                    f"${value:,.2f}",
                    f"{delta:.4f}",
                    f"{gamma:.4f}",
                    f"{theta:.2f}",
                    f"{vega:.2f}",
                    f"${total_pnl:.2f}"
                ])
        
        # Create a table for equity positions
        equity_positions = []
        
        # Process equity positions
        for symbol, position in self.portfolio.positions.items():
            # Skip positions with 0 contracts
            if position.contracts == 0:
                continue
                
            if not isinstance(position, OptionPosition):
                # Get basic position info
                shares = position.contracts
                # Add minus sign for short positions
                direction = "-" if position.is_short else ""
                price = position.current_price
                value = position.current_price * abs(shares)
                
                # Get delta (for stocks, delta is 1.0 per share if long, -1.0 if short)
                delta = position.current_delta
                
                # Calculate total P&L based on current price and previous day price
                total_pnl = 0
                
                if hasattr(position, 'previous_day_price') and hasattr(position, 'current_price') and hasattr(position, 'previous_day_contracts'):
                    # Calculate P&L based on price change, similar to the pre-trade summary calculation
                    if hasattr(position, 'is_short') and position.is_short:
                        # For short positions, price decrease is profit, price increase is loss
                        total_pnl = position.previous_day_contracts * (position.previous_day_price - position.current_price)
                    else:
                        # For long positions, price increase is profit, price decrease is loss
                        total_pnl = position.previous_day_contracts * (position.current_price - position.previous_day_price)
                elif hasattr(position, 'realized_pnl') and hasattr(position, 'unrealized_pnl'):
                    # If we don't have previous day data, fall back to unrealized + realized
                    total_pnl = position.realized_pnl + position.unrealized_pnl
                
                # Format the row
                equity_positions.append([
                    symbol, f"{direction}{shares}", f"${price:.2f}",
                    f"${value:.2f}", delta, f"${total_pnl:.2f}"
                ])
        
        # Display option positions table if we have any
        if option_positions:
            self.logger.info("Open Option Positions:")
            headers = ["Symbol", "Type", "Strike", "Expiry", "DTE", "Contracts", "Price", "Value",
                      "Delta", "Gamma", "Theta", "Vega", "Total P&L"]
            table = tabulate(option_positions, headers=headers, tablefmt="grid")
            for line in table.split('\n'):
                self.logger.info(line)
            
        # Display equity positions table if we have any
        if equity_positions:
            self.logger.info("Open Equity Positions:")
            headers = ["Symbol", "Shares", "Price", "Value", "Delta", "Total P&L"]
            table = tabulate(equity_positions, headers=headers, tablefmt="grid")
            for line in table.split('\n'):
                self.logger.info(line)
            
        # Log any positions with zero contracts that were skipped
        if zero_contract_positions:
            self.logger.debug(f"Skipped {len(zero_contract_positions)} positions with 0 contracts: {', '.join(zero_contract_positions)}")

    def _log_margin_details(self, detailed=True, reuse_existing_metrics=False):
        """
        Log margin details.

        Args:
            detailed: Whether to show detailed margin information
            reuse_existing_metrics: If True, reuse existing portfolio metrics instead of recalculating
        """
        if reuse_existing_metrics and hasattr(self, '_cached_portfolio_metrics') and self._cached_portfolio_metrics:
            portfolio_metrics = self._cached_portfolio_metrics
            self.logger.debug(
                "Reusing cached portfolio metrics for margin details")
        else:
            portfolio_metrics = self.portfolio.get_portfolio_metrics()
            # Cache the metrics for potential reuse
            self._cached_portfolio_metrics = portfolio_metrics

        # Get margin values
        total_margin = portfolio_metrics.get('total_margin', 0)
        available_margin = portfolio_metrics.get('available_margin', 0)
        current_leverage = portfolio_metrics.get('current_leverage', 0)
        portfolio_value = portfolio_metrics.get('portfolio_value', 0)

        # Log the basic margin info
        self.logger.info("Margin Details:")

        # Determine and log which margin calculator is being used
        margin_calculator_type = "Unknown"
        if hasattr(self.portfolio, 'margin_calculator') and self.portfolio.margin_calculator:
            # Get the actual class name
            calculator_class_name = type(
                self.portfolio.margin_calculator).__name__

            # Map the class name to a more user-friendly display name
            if calculator_class_name == "SPANMarginCalculator":
                margin_calculator_type = "SPAN"
            elif calculator_class_name == "OptionMarginCalculator":
                margin_calculator_type = "Option"
            elif calculator_class_name == "MarginCalculator":
                margin_calculator_type = "Basic"
            else:
                margin_calculator_type = calculator_class_name

            self.logger.info(
                f"  Margin Calculator Type: {margin_calculator_type}")
        else:
            self.logger.info(
                "  Margin Calculator: Not Set (Using position level calculation)")

        # Log configuration details for the calculator
        if hasattr(self.portfolio, 'margin_calculator') and self.portfolio.margin_calculator:
            calculator = self.portfolio.margin_calculator
            if hasattr(calculator, 'hedge_credit_rate'):
                try:
                    hedge_credit_rate = float(calculator.hedge_credit_rate)
                    self.logger.info(
                        f"  Hedge Credit Rate: {hedge_credit_rate:.2f} (SPAN margin offset)")
                except (TypeError, ValueError):
                    self.logger.info(
                        f"  Hedge Credit Rate: {calculator.hedge_credit_rate}")

            if hasattr(calculator, 'initial_margin_percentage'):
                try:
                    initial_margin_pct = float(
                        calculator.initial_margin_percentage)
                    self.logger.info(
                        f"  Initial Margin Percentage: {initial_margin_pct:.2%}")
                except (TypeError, ValueError):
                    self.logger.info(
                        f"  Initial Margin Percentage: {calculator.initial_margin_percentage}")

            if hasattr(calculator, 'max_leverage'):
                # Fix: Special handling for max_leverage
                if hasattr(calculator.max_leverage, 'name') and calculator.max_leverage.__class__.__name__ == 'Logger':
                    # This is actually a logger object
                    self.logger.info(f"  Max Leverage: 12.00x (default)")
                else:
                    # Try to convert to float
                    try:
                        max_leverage_value = float(calculator.max_leverage)
                        self.logger.info(
                            f"  Max Leverage: {max_leverage_value:.2f}x")
                    except (TypeError, ValueError):
                        # If it's not a number, just log it as is
                        self.logger.info(
                            f"  Max Leverage: {calculator.max_leverage}")

        # Log the margin details
        self.logger.info(f"  Total Margin Requirement: ${total_margin:,.2f}")
        self.logger.info(f"  Available Margin: ${available_margin:,.2f}")

        # Calculate margin utilization percentage
        if portfolio_value > 0:
            margin_utilization = (total_margin / portfolio_value) * 100
            self.logger.info(
                f"  Margin Utilization: {margin_utilization:.2f}%")

            # Add warning if margin utilization is high
            if margin_utilization > 50 and detailed:
                warning_level = "HIGH" if margin_utilization > 75 else "MODERATE"
                self.logger.warning(
                    f"  Margin Utilization Warning: {warning_level} - {margin_utilization:.2f}% of available capital is being used for margin")
        else:
            self.logger.info(
                f"  Margin Utilization: N/A (NLV is zero or negative)")

        # If detailed, show margin breakdown by position
        if detailed:
            # Get margin by position and hedging benefits from portfolio metrics
            margin_by_position = portfolio_metrics.get(
                'margin_by_position', {})
            hedging_benefits = portfolio_metrics.get('hedging_benefits', 0)

            # Log the total hedging benefits
            if hedging_benefits > 0:
                self.logger.info(
                    f"  Hedging Benefits: ${hedging_benefits:,.2f}")
                self.logger.info(
                    f"  Standalone Sum of Margins: ${(total_margin + hedging_benefits):,.2f}")
                hedge_reduction_pct = (hedging_benefits / (total_margin + hedging_benefits)
                                       * 100) if (total_margin + hedging_benefits) > 0 else 0
                self.logger.info(
                    f"  Hedge Reduction: {hedge_reduction_pct:.2f}% of margin")

            # Show margin breakdown by position
            if margin_by_position:
                self.logger.info("  Margin Breakdown by Position:")

                # Sort positions by margin (highest first)
                sorted_positions = sorted(
                    margin_by_position.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Log top positions by margin
                for i, (symbol, margin) in enumerate(sorted_positions):
                    position = self.portfolio.positions.get(symbol)
                    position_type = "Option" if hasattr(
                        position, 'option_type') else "Stock"
                    is_short = getattr(position, 'is_short', False)
                    direction = "Short" if is_short else "Long"

                    # Format margin percentage of total
                    margin_pct = (margin / total_margin *
                                  100) if total_margin > 0 else 0

                    # Get position details
                    contracts = getattr(position, 'contracts', 0)

                    self.logger.info(
                        f"    {i+1}. {symbol} ({direction} {position_type}): ${margin:,.2f} ({margin_pct:.1f}%), {contracts} contracts")

                    # Limit to top 10 positions if there are many
                    if i >= 9 and len(sorted_positions) > 12:
                        remaining = len(sorted_positions) - 10
                        self.logger.info(
                            f"    ... and {remaining} more positions")
                        break

            self.logger.info(
                "--------------------------------------------------")

    def _log_post_trade_summary(self, current_date, add_header=True):
        """
        Log a summary of the portfolio status after trading.
        
        Args:
            current_date: The current trading date
            add_header: Whether to add a header to the summary
        """
        if not self.portfolio:
            self.logger.warning("No portfolio available for post-trade summary")
            return
        
        # Format the header with the current date
        formatted_date = current_date
        if isinstance(current_date, datetime):
            formatted_date = current_date.strftime('%Y-%m-%d')
        elif not isinstance(current_date, str):
            formatted_date = str(current_date)
        
        if add_header:
            self.logging_manager.log_section_header(f"POST-TRADE SUMMARY - {formatted_date}")
        
        # Get current portfolio value
        portfolio_value = self.portfolio.get_portfolio_value()
        self.logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
        
        # Calculate daily PnL if we have previous value
        if hasattr(self, 'previous_portfolio_value') and self.previous_portfolio_value > 0:
            daily_pnl = portfolio_value - self.previous_portfolio_value
            daily_pnl_percent = (daily_pnl / self.previous_portfolio_value) * 100
            
            # Initialize P&L components
            option_pnl = 0
            equity_pnl = 0
            
            # Calculate P&L breakdown by position type
            for symbol, position in self.portfolio.positions.items():
                if (hasattr(position, 'position_type') and position.position_type == 'option') or isinstance(position, OptionPosition):
                    if hasattr(position, 'previous_day_price') and hasattr(position, 'current_price'):
                        price_diff = position.previous_day_price - position.current_price if position.is_short else position.current_price - position.previous_day_price
                        pos_option_pnl = price_diff * abs(position.previous_day_contracts) * 100
                        option_pnl += pos_option_pnl
                elif hasattr(position, 'contracts') and hasattr(position, 'previous_day_price') and hasattr(position, 'current_price'):
                    # For equities, calculate P&L based on price change
                    if hasattr(position, 'is_short') and position.is_short:
                        # For short positions, price decrease is profit, price increase is loss
                        pos_equity_pnl = position.previous_day_contracts * (position.previous_day_price - position.current_price)
                    else:
                        # For long positions, price increase is profit, price decrease is loss
                        pos_equity_pnl = position.previous_day_contracts * (position.current_price - position.previous_day_price)
                    equity_pnl += pos_equity_pnl
            
            # Cash/other P&L is what's left after accounting for options and equities
            cash_pnl = daily_pnl - option_pnl - equity_pnl
            
            # Log the P&L breakdown
            self.logger.info(f"Daily P&L: ${daily_pnl:.2f} ({daily_pnl_percent:.2f}%)")
            self.logger.info(f"  • Option P&L: ${option_pnl:.2f}")
            self.logger.info(f"  • Equity P&L: ${equity_pnl:.2f}")
            self.logger.info(f"  • Cash/Other P&L: ${cash_pnl:.2f}")
            
            # If we have realized P&L for today, add it
            if hasattr(self.portfolio, 'today_realized_pnl') and self.portfolio.today_realized_pnl != 0:
                self.logger.info(f"  • Today's Realized P&L: ${self.portfolio.today_realized_pnl:.2f}")
        
        # Log open positions
        self._log_open_positions()
        
        # Log margin details - FIX: Use cached metrics if available to avoid recalculation
        if hasattr(self, '_cached_portfolio_metrics') and self._cached_portfolio_metrics:
            # Reuse existing metrics to avoid recalculating margin
            self._log_margin_details(detailed=False, reuse_existing_metrics=True)
        else:
            # Cache portfolio metrics for potential reuse by other methods
            portfolio_metrics = self.portfolio.get_portfolio_metrics()
            self._cached_portfolio_metrics = portfolio_metrics
            self._log_margin_details(detailed=False, reuse_existing_metrics=True)
        
        # Log portfolio Greek risk - reuse cached metrics
        self._log_portfolio_greek_risk()

    def _calculate_metrics_window(self, window_size):
        """
        Calculate risk metrics for a specified window of past performance data.

        Args:
            window_size: Number of days to include in the window

        Returns:
            dict: Dictionary containing the calculated metrics
        """
        # Initialize default values
        results = {
            'sharpe_ratio': 0.0,
            'annual_volatility': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0
        }

        # If we don't have any metrics or window_size is invalid, return defaults
        if not hasattr(self.portfolio, 'daily_metrics') or not self.portfolio.daily_metrics or window_size <= 0:
            return results

        # Get the daily metrics into a list
        daily_metrics = list(self.portfolio.daily_metrics.values())

        # Ensure we don't try to use more data than we have
        window_size = min(window_size, len(daily_metrics))

        # Get the most recent window_size metrics
        window_metrics = daily_metrics[-window_size:]

        # Extract daily returns - if returns are not available, use value changes
        if 'returns' in window_metrics[0]:
            returns = [m.get('returns', 0) for m in window_metrics]
        else:
            # Calculate returns from portfolio values
            values = [m.get('portfolio_value', 0) for m in window_metrics]
            returns = []

            # Calculate daily returns
            for i in range(1, len(values)):
                if values[i-1] > 0:
                    daily_return = (values[i] - values[i-1]) / values[i-1]
                    returns.append(daily_return)
                else:
                    returns.append(0)

            # Add a zero return for the first day
            returns.insert(0, 0)

        # Calculate metrics
        if len(returns) > 1:
            # Calculate volatility (standard deviation of returns)
            daily_volatility = self._calculate_std_dev(
                returns) if len(returns) > 1 else 0

            # Annualize volatility (approximate by multiplying by sqrt(252))
            annual_volatility = daily_volatility * \
                (252 ** 0.5) if daily_volatility else 0
            results['annual_volatility'] = annual_volatility

            # Calculate Sharpe ratio
            avg_return = sum(returns) / len(returns)
            annualized_return = (1 + avg_return) ** 252 - 1
            risk_free_rate = 0.02  # Assumed 2% risk-free rate

            # Avoid division by zero
            if annual_volatility > 0:
                sharpe_ratio = (annualized_return -
                                risk_free_rate) / annual_volatility
                results['sharpe_ratio'] = sharpe_ratio

            # Calculate max drawdown
            values = [m.get('portfolio_value', 0) for m in window_metrics]
            max_drawdown = self._calculate_max_drawdown(values)
            results['max_drawdown'] = max_drawdown

            # Calculate total return over the period
            if values[0] > 0:
                total_return = (values[-1] - values[0]) / values[0]
                results['total_return'] = total_return

        return results

    def _calculate_std_dev(self, values):
        """
        Calculate standard deviation of a list of values.

        Args:
            values: List of numeric values

        Returns:
            float: Standard deviation
        """
        if len(values) <= 1:
            return 0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def _calculate_max_drawdown(self, values):
        """
        Calculate maximum drawdown from a list of values.

        Args:
            values: List of numeric values representing portfolio values

        Returns:
            float: Maximum drawdown as a percentage (0 to 1)
        """
        if not values or len(values) <= 1:
            return 0

        # Calculate running maximum and drawdown
        max_value = values[0]
        max_drawdown = 0

        for value in values:
            if value > max_value:
                max_value = value

            # Calculate drawdown if we have a peak
            if max_value > 0:
                drawdown = (max_value - value) / max_value
                max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _log_rolling_risk_metrics(self):
        """
        Calculate and log rolling risk metrics for different time windows.
        This includes Sharpe ratios and volatilities over short, medium, and long timeframes.
        """
        try:
            # Get risk config parameters
            risk_config = self.config.get('risk', {})
            short_window = risk_config.get('short_window', 21)
            medium_window = risk_config.get('medium_window', 63)
            long_window = risk_config.get('long_window', 252)

            # Check if we have any returns data
            if not hasattr(self.portfolio, 'daily_returns') or not self.portfolio.daily_returns:
                self.logger.warning(
                    "No daily returns data for rolling metrics calculation")
                return

            # Log the number of daily returns for debugging
            num_returns = len(self.portfolio.daily_returns)
            self.logger.debug(
                f"Calculating rolling metrics with {num_returns} daily returns")

            # Extract daily returns series
            returns = []
            dates = []
            for day_return in self.portfolio.daily_returns:
                returns.append(day_return.get('return', 0))
                dates.append(day_return.get('date'))
                self.logger.debug(
                    f"Return entry: date={day_return.get('date')}, return={day_return.get('return', 0)}")

            # Need at least 2 returns for meaningful metrics
            if len(returns) < 2:
                self.logger.info(
                    "--------------------------------------------------")
                self.logger.info("Rolling Risk Metrics:")
                self.logger.info(
                    "  Not enough data yet (need at least 2 days of returns)")
                return

            # Convert to pandas Series for rolling calculations
            try:
                import pandas as pd
                import numpy as np

                # Create Series (handle both datetime and string dates)
                if all(isinstance(d, str) for d in dates):
                    returns_series = pd.Series(
                        returns, index=pd.to_datetime(dates))
                else:
                    returns_series = pd.Series(returns, index=dates)

                self.logger.debug(
                    f"Created returns series with shape: {returns_series.shape}")

                # Header for rolling metrics
                self.logger.info(
                    "--------------------------------------------------")
                self.logger.info("Rolling Risk Metrics:")

                # Use expanding window with minimum periods of 2 if we have limited data
                min_periods = min(2, len(returns))

                # Calculate metrics using expanding window since we have limited data
                mean_return = returns_series.expanding(
                    min_periods=min_periods).mean() * 252
                std_return = returns_series.expanding(
                    min_periods=min_periods).std() * np.sqrt(252)

                # Calculate Sharpe ratio (annualized)
                sharpe = mean_return.iloc[-1] / \
                    std_return.iloc[-1] if std_return.iloc[-1] > 0 else 0
                volatility = std_return.iloc[-1] * 100  # Convert to percentage

                # Display expanding window metrics
                self.logger.info(
                    f"  Expanding Window (n={len(returns_series)})")
                self.logger.info(
                    f"    Sharpe: {sharpe:.2f}, Volatility: {volatility:.2f}%")

                # Only show short/medium/long windows when we have more data
                if len(returns_series) >= 5:
                    # Short window - default to expanding if not enough data
                    self.logger.info(
                        f"  Short Window (target={short_window} days)")
                    if len(returns_series) < short_window:
                        # Still using expanding window metrics
                        self.logger.info(
                            f"    Sharpe: {sharpe:.2f}, Volatility: {volatility:.2f}% (expanding window)")
                    else:
                        # Full rolling window
                        short_mean = returns_series.rolling(
                            short_window, min_periods=5).mean() * 252
                        short_std = returns_series.rolling(
                            short_window, min_periods=5).std() * np.sqrt(252)
                        short_sharpe = short_mean.iloc[-1] / \
                            short_std.iloc[-1] if short_std.iloc[-1] > 0 else 0
                        # Convert to percentage
                        short_vol = short_std.iloc[-1] * 100
                        self.logger.info(
                            f"    Sharpe: {short_sharpe:.2f}, Volatility: {short_vol:.2f}% (rolling window)")

                    # Medium window
                    self.logger.info(
                        f"  Medium Window (target={medium_window} days)")
                    if len(returns_series) < medium_window:
                        # Still using expanding window metrics
                        self.logger.info(
                            f"    Sharpe: {sharpe:.2f}, Volatility: {volatility:.2f}% (expanding window)")
                    else:
                        # Full rolling window
                        med_mean = returns_series.rolling(
                            medium_window, min_periods=5).mean() * 252
                        med_std = returns_series.rolling(
                            medium_window, min_periods=5).std() * np.sqrt(252)
                        med_sharpe = med_mean.iloc[-1] / \
                            med_std.iloc[-1] if med_std.iloc[-1] > 0 else 0
                        med_vol = med_std.iloc[-1] * \
                            100  # Convert to percentage
                        self.logger.info(
                            f"    Sharpe: {med_sharpe:.2f}, Volatility: {med_vol:.2f}% (rolling window)")

                    # Long window
                    self.logger.info(
                        f"  Long Window (target={long_window} days)")
                    if len(returns_series) < long_window:
                        # Still using expanding window metrics
                        self.logger.info(
                            f"    Sharpe: {sharpe:.2f}, Volatility: {volatility:.2f}% (expanding window)")
                    else:
                        # Full rolling window
                        long_mean = returns_series.rolling(
                            long_window, min_periods=5).mean() * 252
                        long_std = returns_series.rolling(
                            long_window, min_periods=5).std() * np.sqrt(252)
                        long_sharpe = long_mean.iloc[-1] / \
                            long_std.iloc[-1] if long_std.iloc[-1] > 0 else 0
                        long_vol = long_std.iloc[-1] * \
                            100  # Convert to percentage
                        self.logger.info(
                            f"    Sharpe: {long_sharpe:.2f}, Volatility: {long_vol:.2f}% (rolling window)")

            except Exception as e:
                import traceback
                self.logger.warning(f"Error calculating rolling metrics: {e}")
                self.logger.debug(traceback.format_exc())
        except Exception as e:
            import traceback
            self.logger.warning(f"Exception in _log_rolling_risk_metrics: {e}")
            self.logger.debug(traceback.format_exc())

    def _update_positions(self, current_date: datetime, daily_data: pd.DataFrame) -> None:
        """
        Update positions with the latest market data and greeks.

        Args:
            current_date: Current trading date
            daily_data: Data for the current trading day
        """
        self.logger.info(
            f"Updating positions with market data for {current_date.strftime('%Y-%m-%d')}")

        # Mark the start of a new day - Clear the cached metrics
        if hasattr(self, '_cached_portfolio_metrics'):
            self._cached_portfolio_metrics = None

        # Skip if no positions to manage
        if not self.portfolio.positions:
            self.logger.info("No positions to update")
            return True

        # Store pre-update values for comparison
        self.pre_update_values = {}
        for symbol, position in self.portfolio.positions.items():
            self.pre_update_values[symbol] = position.current_price * position.contracts * (
                100 if isinstance(position, OptionPosition) else 1)

        # Get the latest market data for the current date
        market_data = daily_data

        try:
            # First update positions using the portfolio's update_market_data method
            # This handles the basic position updates, but SUPPRESS the POST-TRADE summary until later
            if hasattr(self.portfolio, 'update_market_data') and callable(self.portfolio.update_market_data):
                # Create a dictionary of market data by symbol for quick lookup
                market_data_by_symbol = {}

                # For option positions
                if hasattr(daily_data, 'columns') and 'OptionSymbol' in daily_data.columns:
                    for _, row in daily_data.iterrows():
                        if 'OptionSymbol' in row:
                            symbol = row['OptionSymbol']
                            market_data_by_symbol[symbol] = row

                # Update the portfolio with the market data but don't generate POST-TRADE summary yet
                # Pass None for the current_date to suppress the POST-TRADE summary
                self.portfolio.update_market_data(market_data_by_symbol, None)
                self.logger.debug(
                    "Updated positions using portfolio.update_market_data")

            # Then use our enhanced method to ensure all Greeks and prices are properly set
            # This is especially important for the hedge position and option Greeks
            self._update_positions_market_data(daily_data)

            # Log the updated positions
            self.logger.info("Positions updated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
            return False

    def _update_positions_market_data(self, daily_data):
        """
        Updates all positions with the latest market data for greeks and prices

        Args:
            daily_data: Dictionary containing daily market data
        """
        self.logger.debug("Updating positions with latest market data")

        # Determine the underlying symbol from hedging configuration or option positions
        underlying_symbol = None

        # First try to get it from hedging configuration
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            underlying_symbol = self.hedging_manager.hedge_symbol

        # If still None, try to extract from option positions
        if underlying_symbol is None and self.portfolio and hasattr(self.portfolio, 'positions'):
            for symbol, position in self.portfolio.positions.items():
                if hasattr(position, 'instrument_data') and position.instrument_data:
                    if hasattr(position.instrument_data, 'get'):
                        # Try to get UnderlyingSymbol from instrument_data
                        underlying_symbol = position.instrument_data.get(
                            'UnderlyingSymbol')
                        if underlying_symbol:
                            break

        # Default to 'SPY' if we still don't have an underlying
        if underlying_symbol is None:
            underlying_symbol = 'SPY'  # Default based on the positions in logs

        # Find the latest underlying price from daily_data
        latest_underlying_price = None

        # First check if we have a direct underlying price in the data
        if hasattr(daily_data, 'iterrows'):
            for _, row in daily_data.iterrows():
                # Try to find the underlying symbol in various columns
                symbol_match = False
                for col in ['Symbol', 'UnderlyingSymbol']:
                    if col in row and row[col] == underlying_symbol:
                        symbol_match = True
                        break

                if symbol_match:
                    # Try to get price from different columns
                    for price_col in ['UnderlyingPrice', 'Last', 'Close', 'MidPrice']:
                        if price_col in row and row[price_col] > 0:
                            latest_underlying_price = row[price_col]
                            self.logger.debug(
                                f"Found underlying price for {underlying_symbol}: ${latest_underlying_price:.2f}")
                            break

                    if latest_underlying_price is not None:
                        break

        # If we still don't have a price, try other methods
        if latest_underlying_price is None and hasattr(daily_data, 'get'):
            if 'closing_price' in daily_data and underlying_symbol in daily_data.get('closing_price', {}):
                latest_underlying_price = daily_data['closing_price'][underlying_symbol]
                self.logger.debug(
                    f"Using closing price for {underlying_symbol}: ${latest_underlying_price:.2f}")

        # Log if we couldn't find a price
        if latest_underlying_price is None:
            self.logger.warning(
                f"Could not find price for underlying {underlying_symbol} in market data")

        # Update each position's market data
        for symbol, position in self.portfolio.positions.items():
            try:
                # For equity positions (including the underlying)
                if not isinstance(position, OptionPosition):
                    # Look for price specifically for this equity symbol
                    equity_price = None

                    # First try to find in daily_data DataFrame
                    if hasattr(daily_data, 'iterrows'):
                        for _, row in daily_data.iterrows():
                            if 'Symbol' in row and row['Symbol'] == symbol:
                                # Try a variety of price columns
                                for price_col in ['Last', 'Close', 'MidPrice', 'UnderlyingPrice']:
                                    if price_col in row and row[price_col] > 0:
                                        equity_price = row[price_col]
                                        break

                                if equity_price is not None:
                                    break

                    # If no specific price found, use the underlying price for the hedge symbol
                    if equity_price is None and symbol == underlying_symbol and latest_underlying_price is not None:
                        equity_price = latest_underlying_price

                    # If we have a valid price, update the position
                    if equity_price is not None and equity_price > 0:
                        # For equity positions, delta is the number of shares (positive for long, negative for short)
                        delta = position.contracts if not position.is_short else -position.contracts
                        position.current_price = equity_price
                        position.current_delta = delta
                        self.logger.debug(
                            f"Updated equity position {symbol}: price=${equity_price:.2f}, delta={delta}")
                    else:
                        self.logger.warning(
                            f"No price found for equity position {symbol}, price not updated")

                # For option positions
                elif isinstance(position, OptionPosition):
                    # Find the option in the chain data
                    option_data = None

                    # Try to locate the option in the market data
                    if hasattr(daily_data, 'iterrows'):
                        for _, row in daily_data.iterrows():
                            if 'OptionSymbol' in row and row['OptionSymbol'] == symbol:
                                option_data = row
                                break

                    # Update the position if we found data
                    if option_data is not None:
                        # Get price from option data - try different columns
                        price = None
                        for price_col in ['MidPrice', 'Last', 'Bid', 'Ask']:
                            if price_col in option_data and option_data[price_col] > 0:
                                price = option_data[price_col]
                                break

                        # Only update if we have a valid price
                        if price is not None and price > 0:
                            # Extract Greeks from the data
                            delta = option_data['Delta'] if 'Delta' in option_data else None
                            gamma = option_data['Gamma'] if 'Gamma' in option_data else None
                            theta = option_data['Theta'] if 'Theta' in option_data else None
                            vega = option_data['Vega'] if 'Vega' in option_data else None
                            iv = option_data['IV'] if 'IV' in option_data else None

                            # Make sure underlying price is set
                            if 'UnderlyingPrice' in option_data and option_data['UnderlyingPrice'] > 0:
                                position.underlying_price = option_data['UnderlyingPrice']
                                self.logger.debug(
                                    f"Set underlying price for {symbol}: ${position.underlying_price:.2f}")
                            elif latest_underlying_price is not None and latest_underlying_price > 0:
                                position.underlying_price = latest_underlying_price
                                self.logger.debug(
                                    f"Set underlying price for {symbol} from latest price: ${position.underlying_price:.2f}")
                            else:
                                # Try to get underlying price from a stock position with matching symbol
                                for stock_symbol, stock_pos in self.portfolio.positions.items():
                                    if stock_symbol == underlying_symbol and hasattr(stock_pos, 'current_price'):
                                        position.underlying_price = stock_pos.current_price
                                        self.logger.debug(
                                            f"Set underlying price for {symbol} from stock position: ${position.underlying_price:.2f}")
                                        break

                            # Update position with new data
                            position.update_market_data(
                                price=price,
                                delta=delta,
                                gamma=gamma,
                                theta=theta,
                                vega=vega,
                                implied_volatility=iv
                            )

                            # If this is a short position, make sure delta is negative
                            if position.is_short and delta is not None and delta > 0:
                                position.current_delta = -delta

                            self.logger.debug(
                                f"Updated option position {symbol}: price=${price:.2f}, delta={position.current_delta:.4f}")
                    else:
                        self.logger.warning(
                            f"No market data found for option position {symbol}, position not updated")

            except Exception as e:
                self.logger.warning(f"Error updating position {symbol}: {e}")

        # After all positions are updated, make sure the hedge position has a correct delta
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            hedge_position = self.hedging_manager.get_hedge_position()
            if hedge_position:
                # Ensure the hedge position delta reflects the actual number of contracts
                hedge_position.current_delta = hedge_position.contracts if not hedge_position.is_short else - \
                    hedge_position.contracts
                self.logger.debug(
                    f"Updated hedge position with delta: {hedge_position.current_delta}")

        self.logger.debug("Finished updating positions with market data")

    def _manage_positions(self, current_date: datetime, daily_data: pd.DataFrame) -> None:
        """
        Manage existing positions, checking exit conditions.

        Args:
            current_date: Current trading date
            daily_data: Data for the current trading day
        """
        # Skip if no positions to manage
        if not self.portfolio.positions:
            return

        # Create a dictionary of market data by symbol for quick lookup
        market_data_by_symbol = {}

        # For option positions
        if 'OptionSymbol' in daily_data.columns:
            # Group data by option symbol
            for _, row in daily_data.iterrows():
                symbol = row['OptionSymbol']
                market_data_by_symbol[symbol] = row.to_dict()

        # Check exit conditions for each position
        positions_to_close = []

        for symbol, position in list(self.portfolio.positions.items()):
            # Get market data for this position
            if symbol in market_data_by_symbol:
                market_data = market_data_by_symbol[symbol]

                # Check if strategy wants to exit
                should_exit, reason = self.strategy.check_exit_conditions(
                    position, market_data)

                if should_exit:
                    positions_to_close.append((symbol, reason))

        # Close positions that met exit conditions
        for symbol, reason in positions_to_close:
            if symbol in self.portfolio.positions:
                # Get current price from market data
                price = None
                if symbol in market_data_by_symbol:
                    price = market_data_by_symbol[symbol].get('MidPrice')

                # Close position
                position = self.portfolio.positions[symbol]
                quantity = position.contracts

                pnl = self.portfolio.remove_position(
                    symbol=symbol,
                    price=price,
                    reason=reason,
                    skip_margin_calc=True  # Skip redundant margin calculation
                )

                # Categorize the reason for closing
                close_category = self._categorize_close_reason(reason)

                # Log detailed info about the closed position
                self.logger.info(
                    f"[TradeManager] Closed position {symbol} - Reason: {close_category} - Detail: {reason} - P&L: ${pnl:,.2f}")
            else:
                self.logger.warning(
                    f"Cannot close position {symbol} - not found in portfolio")

    def _execute_trading_activities(self, current_date: datetime, daily_data: pd.DataFrame, skip_summaries=False) -> None:
        """
        Execute trading activities with structured flow, including hedging.

        This method implements the complete trading lifecycle with the following phases:
        1. PRETRADE ANALYSIS - Evaluate current market and portfolio state
        2. PORTFOLIO REBALANCING - Ensure portfolio is within margin limits
        3. STRATEGY EVALUATION - Generate trading signals based on market conditions
        4. HEDGING ANALYSIS - Calculate hedging requirements for new trades (optional)
        5. MARGIN CALCULATION - Calculate margin requirements for trades and hedges
        6. POSITION SIZING - Determine appropriate position sizes based on risk parameters
        7. TRADE EXECUTION - Execute primary trades and associated hedges
        8. POSITION MANAGEMENT - Add positions to portfolio with proper tracking
        9. POSTTRADE ANALYSIS - Evaluate the updated portfolio state and metrics

        Args:
            current_date: Current processing date 
            daily_data: Market data for the current date
            skip_summaries: Skip redundant summary logging when called from _process_trading_day
        """
        # --------------------------------------------------------
        # PHASE 1: PRETRADE ANALYSIS
        # Evaluate current portfolio state before making trading decisions
        # --------------------------------------------------------
        start_time = time.time()

        # Update positions first to ensure we have current market data
        self._update_positions_market_data(daily_data)

        if not skip_summaries:
            self.logger.info("PHASE 1: PRETRADE ANALYSIS")
            self.logger.info("-" * 50)

            # Perform pre-trade activities
            self._log_pre_trade_summary(current_date, daily_data)

        # --------------------------------------------------------
        # PHASE 2: PORTFOLIO REBALANCING
        # Ensure portfolio is within margin limits before adding new positions
        # --------------------------------------------------------
        # Check margin management needs if margin manager exists
        if self.margin_manager:
            self.logger.info("PHASE 2: PORTFOLIO REBALANCING")
            self.logger.info("-" * 50)

            # Convert to market data by symbol for margin analysis
            market_data_by_symbol = self.data_manager.get_market_data_by_symbol(
                daily_data)

            # Analyze current margin status
            margin_status = self.margin_manager.analyze_margin_status(
                current_date)

            # Log margin analysis
            if margin_status.get('needs_rebalancing', False):
                self.logger.warning("Margin rebalancing required")
                self.logger.warning(
                    f"  Current margin utilization: {margin_status['margin_utilization']:.2%}")
                self.logger.warning(
                    f"  Target margin utilization: {self.margin_manager.target_margin_threshold:.2%}")
                
                # Here we would implement systematic position reduction
                # based on the margin_manager's recommendations
                if hasattr(self.margin_manager, 'rebalance_positions'):
                    self.margin_manager.rebalance_positions(current_date, daily_data)
            else:
                self.logger.info("Portfolio margin within acceptable limits, no rebalancing needed")

        # --------------------------------------------------------
        # PHASE 3: STRATEGY EVALUATION
        # Identify potential new trading opportunities based on market conditions
        # --------------------------------------------------------
        self.logger.info("PHASE 3: STRATEGY EVALUATION")
        self.logger.info("-" * 50)
        
        # Generate trading signals
        signals = self.strategy.generate_signals(current_date, daily_data)
        
        if signals:
            self.logger.info(f"Generated {len(signals)} trading signals")
            for i, signal in enumerate(signals):
                self.logger.debug(f"  Signal {i+1}: {signal}")
        else:
            self.logger.info("No trading signals generated for today")
        
        # --------------------------------------------------------
        # PHASE 4-7: SIGNAL PROCESSING AND EXECUTION
        # For each signal: Calculate hedging requirements, determine margin,
        # size the position appropriately, and execute trades
        # --------------------------------------------------------
        if signals:
            self.logger.info("PHASES 4-7: SIGNAL PROCESSING AND EXECUTION")
            self.logger.info("-" * 50)
            self._execute_signals(signals, daily_data, current_date)
        
        # --------------------------------------------------------
        # PHASE 8: POSITION MANAGEMENT - EXIT CONDITION CHECKS
        # Check existing positions against exit criteria
        # --------------------------------------------------------
        if self.portfolio.positions:
            self.logger.info("PHASE 8: POSITION MANAGEMENT - EXIT CHECKS")
            self.logger.info("-" * 50)
            self._check_exit_conditions(daily_data, current_date)

        # --------------------------------------------------------
        # PHASE 8B: POSITION MANAGEMENT - PORTFOLIO HEDGING ADJUSTMENTS
        # Adjust hedge positions based on overall portfolio exposure
        # --------------------------------------------------------
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            self.logger.info("PHASE 8B: POSITION MANAGEMENT - HEDGING ADJUSTMENTS")
            self.logger.info("-" * 50)
            
            self.logger.info("[Hedging] Analyzing portfolio exposure")
            delta_summary = self.hedging_manager.calculate_hedge_requirements(
                daily_data, current_date)

            # Log the portfolio exposure
            if delta_summary:
                self.logger.info(
                    f"[Hedging] Portfolio delta: {delta_summary.get('portfolio_delta', 0):.4f}")
                self.logger.info(
                    f"[Hedging] Portfolio delta ($): ${delta_summary.get('portfolio_dollar_delta', 0):.2f}")
                self.logger.info(
                    f"[Hedging] Target delta ratio: {delta_summary.get('target_delta_ratio', 0):.2f}")
                self.logger.info(
                    f"[Hedging] Required delta adjustment: {delta_summary.get('required_delta_adjustment', 0):.4f}")

                # Apply hedging based on config
                self.hedging_manager.apply_hedging(
                    current_date=current_date,
                    market_data=daily_data
                )

            # Check if hedging was applied
            current_hedge = self.hedging_manager.get_hedge_position()
            if current_hedge:
                self.logger.info(
                    f"[Hedging] Current hedge: {current_hedge.contracts} shares of {current_hedge.symbol}")
                self.logger.info(
                    f"[Hedging] Hedge delta: {current_hedge.current_delta:.4f}")

        # --------------------------------------------------------
        # PHASE 9: POSTTRADE ANALYSIS
        # Evaluate the portfolio after trading to ensure all metrics are within limits
        # --------------------------------------------------------
        if not skip_summaries:
            self.logging_manager.log_section_header("PHASE 9: POSTTRADE ANALYSIS")
            self.logger.info("-" * 50)

            # Log post-trade portfolio summary
            self._log_post_trade_summary(current_date)

        # Calculate and log execution time
        end_time = time.time()
        execution_time = end_time - start_time
        self.logger.info(
            f"Trading day execution completed in {execution_time:.2f} seconds")

    def _check_exit_conditions(self, daily_data: pd.DataFrame, current_date: datetime) -> None:
        """
        Check exit conditions for all positions and close if needed.

        Args:
            daily_data: Data for the current trading day
            current_date: Current trading date
        """
        positions_to_close = []
        market_data_by_symbol = {}

        # Create a dictionary of market data by symbol for quick lookup
        if hasattr(daily_data, 'columns') and 'OptionSymbol' in daily_data.columns:
            for _, row in daily_data.iterrows():
                if 'OptionSymbol' in row:
                    symbol = row['OptionSymbol']
                    market_data_by_symbol[symbol] = row

        # Check exit conditions for each position
        for symbol, position in list(self.portfolio.positions.items()):
            # Get market data for this position
            if symbol in market_data_by_symbol:
                market_data = market_data_by_symbol[symbol]

                # Check if strategy wants to exit
                should_exit, reason = self.strategy.check_exit_conditions(
                    position, market_data)

                if should_exit:
                    positions_to_close.append((symbol, reason))

        # Close positions that met exit conditions
        for symbol, reason in positions_to_close:
            if symbol in self.portfolio.positions:
                # Get current price from market data
                price = None
                if symbol in market_data_by_symbol:
                    price = market_data_by_symbol[symbol].get('MidPrice')

                # Close position
                position = self.portfolio.positions[symbol]
                quantity = position.contracts

                pnl = self.portfolio.remove_position(
                    symbol=symbol,
                    price=price,
                    reason=reason,
                    skip_margin_calc=True  # Skip redundant margin calculation
                )

                # Categorize the reason for closing
                close_category = self._categorize_close_reason(reason)

                # Log detailed info about the closed position
                self.logger.info(
                    f"[TradeManager] Closed position {symbol} - Reason: {close_category} - Detail: {reason} - P&L: ${pnl:,.2f}")
            else:
                self.logger.warning(
                    f"Cannot close position {symbol} - not found in portfolio")

    def _execute_signals(self, signals: List[Dict[str, Any]], daily_data: pd.DataFrame, current_date: datetime) -> None:
        """
        Execute trading signals with integrated margin and hedging calculation.

        Args:
            signals: List of trading signals from strategy
            daily_data: DataFrame of market data
            current_date: Current simulation date
        """
        self.logger.info("[TradeManager] Executing trading signals")

        # Prepare hedge market data - extract and cache underlying prices
        # This ensures consistency across all subsequent hedging operations
        self._prepare_hedge_market_data(daily_data)

        # Convert daily data to market data by symbol for margin impact checks
        market_data_by_symbol = self.data_manager.get_market_data_by_symbol(
            daily_data)
            
        # Inject cached underlying prices into market data for consistency
        for symbol, price in self.cached_underlying_prices.items():
            if symbol in market_data_by_symbol:
                market_data_by_symbol[symbol]['UnderlyingPrice'] = price
            else:
                # Create a minimal entry if the symbol doesn't exist
                market_data_by_symbol[symbol] = {
                    'Symbol': symbol,
                    'UnderlyingSymbol': symbol,
                    'UnderlyingPrice': price,
                    'Close': price,
                    'MidPrice': price
                }

        # Process each signal
        executed_signals = []

        for signal in signals:
            # Normalize action to lowercase
            action = signal.get('action', '').lower()
            symbol = signal.get('symbol')

            # Map buy/sell actions to open with appropriate is_short flag
            if action == 'buy':
                # Convert 'buy' to 'open' with is_short=False
                action = 'open'
                signal['is_short'] = False
                self.logger.debug(
                    f"Mapped 'buy' action to 'open' with is_short=False")

            elif action == 'sell':
                # Convert 'sell' to 'open' with is_short=True
                action = 'open'
                signal['is_short'] = True
                self.logger.debug(
                    f"Mapped 'sell' action to 'open' with is_short=True")

            if action == 'open':
                # Handle opening a new position
                is_short = signal.get('is_short', False)
                quantity = signal.get('quantity', 0)
                price = signal.get('price')
                position_type = signal.get(
                    'position_type', 'option')  # Default to option
                instrument_data = signal.get('instrument_data', {})

                # Get market data for the symbol
                market_data = market_data_by_symbol.get(symbol, {})

                # Ensure we have instrument data - use market data if needed
                if not instrument_data and market_data:
                    instrument_data = market_data

                # Extract execution data if present
                execution_data = {'date': current_date}
                if 'execution_data' in signal:
                    execution_data.update(signal['execution_data'])

                # Debug info
                self.logger.debug(
                    f"Opening position: {symbol} {quantity} {'short' if is_short else 'long'} {position_type}")

                # STEP 1: HEDGING CALCULATION - Determine hedge position before margin check
                hedge_position = None
                if position_type.lower() == 'option' and self.hedging_manager and hasattr(self.hedging_manager, 'create_theoretical_hedge_position'):
                    # Create temporary option position for hedging calculation
                    self.logger.info(
                        f"[PHASE 4: HEDGING] Calculating hedge position for {symbol}")

                    # Format option_data for hedging manager
                    option_data = {
                        'symbol': symbol,
                        'OptionSymbol': symbol,
                        'price': price,
                        'Delta': instrument_data.get('Delta', 0),
                        'Gamma': instrument_data.get('Gamma', 0),
                        'Theta': instrument_data.get('Theta', 0),
                        'Vega': instrument_data.get('Vega', 0),
                        'Strike': instrument_data.get('Strike', 0),
                        'Expiration': instrument_data.get('Expiration', None),
                        'UnderlyingSymbol': instrument_data.get('UnderlyingSymbol', 'SPY'),
                        'UnderlyingPrice': instrument_data.get('UnderlyingPrice', 0),
                        'Type': 'put' if 'P' in symbol else 'call'
                    }

                    # Create a temporary position
                    from core.position import OptionPosition
                    temp_position = OptionPosition(
                        symbol=symbol,
                        option_data={
                            'Type': option_data['Type'],
                            'Strike': option_data['Strike'],
                            'Expiration': option_data['Expiration'],
                            'UnderlyingSymbol': option_data['UnderlyingSymbol'],
                            'UnderlyingPrice': option_data['UnderlyingPrice']
                        },
                        contracts=quantity,
                        entry_price=price,
                        current_price=price,
                        is_short=is_short,
                        logger=self.logger
                    )

                    # Set greeks on the position
                    temp_position.current_delta = option_data['Delta']
                    temp_position.current_gamma = option_data['Gamma']
                    temp_position.current_theta = option_data['Theta']
                    temp_position.current_vega = option_data['Vega']

                    # Get hedge position
                    hedge_position = self.hedging_manager.create_theoretical_hedge_position(
                        temp_position)

                    if hedge_position:
                        self.logger.info(
                            f"[PHASE 4: HEDGING] Hedge required: {hedge_position.contracts} shares of {hedge_position.symbol}")
                        self.logger.info(
                            f"  Hedge delta: {hedge_position.current_delta}")

                # PHASE 5-6: MARGIN CALCULATION AND POSITION SIZING
                if symbol and position_type.lower() == 'option':
                    self.logger.info(f"[PHASE 5-6: MARGIN & SIZING] Calculating margin and position size for {symbol}")
                    
                    # If we have a hedge position, pass it to position sizing
                    # by adding it to the instrument data
                    if hedge_position:
                        instrument_data['hedge_position'] = {
                            'symbol': hedge_position.symbol,
                            'contracts': hedge_position.contracts,
                            'delta': hedge_position.current_delta,
                            'price': hedge_position.current_price,
                            'is_short': hedge_position.is_short
                        }
                        
                        # Also pass the margin we already calculated to avoid redundant calculation
                        if hasattr(self, 'portfolio') and self.portfolio and hasattr(self.portfolio, 'margin_calculator'):
                            # Calculate the position margin with the hedge
                            self.logger.info(f"[Portfolio] Using SPAN margin calculation")
                            self.logger.info(f"[Position Sizing] Using hedging manager for margin calculation with full hedging benefits")
                            
                            # Instead of creating a new option position and hedge position here,
                            # use the hedging manager to create a properly initialized theoretical hedge position
                            # This ensures all necessary attributes are set correctly
                            from core.position import OptionPosition
                            
                            # Create a temporary option position first
                            temp_option_position = OptionPosition(
                                symbol=symbol,
                                contracts=1,  # Calculate per contract
                                entry_price=price,
                                is_short=is_short,
                                option_data=instrument_data,
                                logger=self.logger
                            )
                            
                            # Ensure the option position has required attributes for margin calculation
                            # Add these attributes to properly calculate delta risk
                            if 'UnderlyingPrice' in instrument_data and instrument_data['UnderlyingPrice'] > 0:
                                temp_option_position.underlying_price = instrument_data['UnderlyingPrice']
                            elif 'underlying_price' in instrument_data and instrument_data['underlying_price'] > 0:
                                temp_option_position.underlying_price = instrument_data['underlying_price']
                            else:
                                # Extract underlying symbol for looking up price if needed
                                underlying_symbol = None
                                if 'UnderlyingSymbol' in instrument_data:
                                    underlying_symbol = instrument_data['UnderlyingSymbol']
                                elif symbol.startswith(('SPY', 'QQQ', 'IWM', 'DIA')):
                                    underlying_symbol = symbol[:3]
                                
                                # Try to get price from market data
                                if underlying_symbol and underlying_symbol in market_data_by_symbol:
                                    temp_option_position.underlying_price = market_data_by_symbol[underlying_symbol].get('Price', 0)
                                    self.logger.info(f"Using market data underlying price for {underlying_symbol}: ${temp_option_position.underlying_price:.2f}")
                            
                            # Set Greeks for proper risk calculation
                            if 'Delta' in instrument_data:
                                temp_option_position.current_delta = instrument_data['Delta']
                            elif 'delta' in instrument_data:
                                temp_option_position.current_delta = instrument_data['delta']
                                
                            if 'Gamma' in instrument_data:
                                temp_option_position.current_gamma = instrument_data['Gamma']
                            elif 'gamma' in instrument_data:
                                temp_option_position.current_gamma = instrument_data['gamma']
                                
                            if 'Vega' in instrument_data:
                                temp_option_position.current_vega = instrument_data['Vega']
                            elif 'vega' in instrument_data:
                                temp_option_position.current_vega = instrument_data['vega']
                                
                            if 'Theta' in instrument_data:
                                temp_option_position.current_theta = instrument_data['Theta']
                            elif 'theta' in instrument_data:
                                temp_option_position.current_theta = instrument_data['theta']
                            
                            if hasattr(self, 'hedging_manager') and self.hedging_manager:
                                # Use the hedging manager to create a properly initialized theoretical hedge position
                                temp_hedge_position = self.hedging_manager.create_theoretical_hedge_position(temp_option_position)
                                
                                if temp_hedge_position:
                                    # Start SPAN margin calculation with the properly initialized positions
                                    self.logger.info(f"[Margin] Portfolio calculation for 2 positions")
                                    self.logger.info(f"[Margin] Starting SPAN margin calculation for {symbol}")
                                    
                                    # Prepare for margin calculation
                                    positions = {
                                        symbol: temp_option_position,
                                        temp_hedge_position.symbol: temp_hedge_position
                                    }
                                    
                                    # Calculate margin using portfolio's calculator
                                    margin_result = self.portfolio.margin_calculator.calculate_portfolio_margin(positions)
                                    margin_with_hedge = margin_result.get('total_margin', 0)
                                    hedge_benefit = margin_result.get('hedging_benefits', 0)
                                    
                                    # Add margin details to instrument data for position sizing
                                    instrument_data['margin_per_contract'] = margin_with_hedge
                                    instrument_data['hedge_benefit'] = hedge_benefit
                                    
                                    self.logger.info(f"[Position Sizing] Hedged margin calculation:")
                                    self.logger.info(f"  Total margin for 1 contract: ${margin_with_hedge:.2f}")
                                    self.logger.info(f"  Hedging benefits: ${hedge_benefit:.2f}")
                                    self.logger.info(f"  Margin per contract (with hedging): ${margin_with_hedge:.2f}")
                                else:
                                    self.logger.warning(f"[Position Sizing] Could not create theoretical hedge position, falling back to simplified margin")
                                    # If we can't create a theoretical hedge, fall back to basic calculation
                                    margin_with_hedge = price * 100 * self.portfolio.margin_calculator.max_leverage
                                    instrument_data['margin_per_contract'] = margin_with_hedge
                                    instrument_data['hedge_benefit'] = 0
                            else:
                                # No hedging manager available, do a basic calculation without hedging
                                self.logger.warning(f"[Position Sizing] No hedging manager available, calculating margin without hedge")
                                positions = {symbol: temp_option_position}
                                margin_result = self.portfolio.margin_calculator.calculate_portfolio_margin(positions)
                                margin_with_hedge = margin_result.get('total_margin', 0)
                                instrument_data['margin_per_contract'] = margin_with_hedge
                                instrument_data['hedge_benefit'] = 0

                    # Perform position sizing with hedging consideration
                    sizing_result = self._position_sizing(
                        symbol, quantity, price, is_short, instrument_data, daily_data)

                    if sizing_result and 'position_size' in sizing_result:
                        quantity = sizing_result['position_size']
                        self.logger.info(
                            f"[PHASE 6: SIZING] Position sizing for {symbol}: Original={signal.get('quantity', 0)}, Calculated={quantity}")

                    # If quantity is 0 or None after position sizing, skip this position
                    if not quantity:
                        self.logger.warning(
                            f"[PHASE 6: SIZING] Position sizing returned zero quantity for {symbol}, skipping")
                        continue

                # PHASE 7: TRADE EXECUTION - Execute the primary trade
                if instrument_data:
                    self.logger.info(f"[PHASE 7: EXECUTION] Executing trade for {symbol}")
                    
                    # Add position - Pass skip_margin_calc=True to avoid redundant margin calculation
                    position = self.portfolio.add_position(
                        symbol=symbol,
                        instrument_data=instrument_data,
                        quantity=quantity,
                        price=price,
                        position_type=position_type,  # Use detected position_type
                        is_short=is_short,
                        execution_data=execution_data,
                        # Pass the reason from the signal
                        reason=signal.get('reason'),
                        skip_margin_calc=True  # Skip redundant margin calculation
                    )

                    # Enhanced logging
                    if position:
                        position_value = price * quantity * \
                            100 if position_type == 'option' else price * quantity
                        self.logger.info(
                            f"[PHASE 7: EXECUTION] Added position: {quantity} {'short' if is_short else 'long'} {position_type} {symbol} @ {price}")
                        self.logger.info(
                            f"  Position value: ${position_value:,.2f}")

                        # Store in today's added positions
                        self.today_added_positions[symbol] = {
                            'contracts': quantity,
                            'price': price,
                            'value': position_value,
                            'data': instrument_data,
                            'position_type': position_type,  # Include position type in tracking
                            'is_short': is_short  # Store position direction for sign adjustment
                        }

                        # PHASE 7B: TRADE EXECUTION - Add hedge position if needed
                        if hedge_position and self.hedging_manager:
                            self.logger.info(f"[PHASE 7B: HEDGE EXECUTION] Executing hedge trade for {symbol}")
                            
                            # Scale hedge position based on actual quantity
                            if quantity != signal.get('quantity', 0) and signal.get('quantity', 0) > 0:
                                scale_factor = quantity / \
                                    signal.get('quantity', 0)
                                hedge_position.contracts = int(
                                    hedge_position.contracts * scale_factor)

                            # Add the hedge position to the portfolio
                            self.logger.info(
                                f"[PHASE 7B: HEDGE EXECUTION] Adding hedge position: {hedge_position.contracts} shares of {hedge_position.symbol}")

                            hedge_price = market_data_by_symbol.get(hedge_position.symbol, {}).get(
                                'MidPrice', hedge_position.current_price)

                            hedge_instrument_data = {
                                'Symbol': hedge_position.symbol,
                                'Type': 'stock',
                                'Delta': 1.0 if not hedge_position.is_short else -1.0,
                                'UnderlyingSymbol': hedge_position.symbol,
                                'UnderlyingPrice': hedge_price
                            }

                            # Add hedge position to portfolio - Pass skip_margin_calc=True
                            self.portfolio.add_position(
                                symbol=hedge_position.symbol,
                                instrument_data=hedge_instrument_data,
                                quantity=hedge_position.contracts,
                                price=hedge_price,
                                position_type='stock',
                                is_short=hedge_position.is_short,
                                execution_data=execution_data,
                                reason=f"Hedge for {symbol}",
                                skip_margin_calc=True  # Skip redundant margin calculation
                            )
                else:
                    self.logger.warning(
                        f"[PHASE 7: EXECUTION] Cannot open position {symbol}: No instrument data available")

            # For close positions
            elif action == 'close':
                # Get the required data for closing a position
                quantity = signal.get('quantity')  # None means close all
                price = signal.get('price')
                reason = signal.get('reason', 'Signal')

                # Log the signal
                self.logger.info(
                    f"[PHASE 8: POSITION MANAGEMENT] Closing {quantity if quantity else 'all'} {symbol}: {reason}")

                # Build execution data
                execution_data = {'date': current_date}
                if 'execution_data' in signal:
                    execution_data.update(signal['execution_data'])

                # Check if we have the position
                if symbol in self.portfolio.positions:
                    # Find linked hedge position before closing
                    hedge_position = None
                    if self.hedging_manager:
                        # Get the position to be closed
                        position_to_close = self.portfolio.positions[symbol]

                        # Check if this is an option position
                        if hasattr(position_to_close, 'position_type') and position_to_close.position_type == 'option':
                            # Try to find corresponding hedge
                            underlying_symbol = position_to_close.underlying_symbol if hasattr(
                                position_to_close, 'underlying_symbol') else 'SPY'

                            # If we have a position in the underlying, it might be a hedge
                            if underlying_symbol in self.portfolio.positions:
                                potential_hedge = self.portfolio.positions[underlying_symbol]

                                # Check if this is a likely hedge (opposite delta sign)
                                if (position_to_close.current_delta * potential_hedge.current_delta < 0):
                                    hedge_position = potential_hedge
                                    self.logger.info(
                                        f"[PHASE 8: POSITION MANAGEMENT] Found corresponding hedge position for {symbol}: {underlying_symbol}")

                    # Close the main position
                    self.logger.info(f"[PHASE 8: POSITION MANAGEMENT] Closing primary position: {symbol}")
                    pnl = self.portfolio.remove_position(
                        symbol=symbol,
                        quantity=quantity,
                        price=price,
                        execution_data=execution_data,
                        reason=reason,
                        skip_margin_calc=True  # Skip redundant margin calculation
                    )

                    # Categorize the reason for closing
                    close_category = self._categorize_close_reason(reason)

                    # Enhanced logging for PnL with reason
                    self.logger.info(
                        f"[PHASE 8: POSITION MANAGEMENT] Closed position - Reason: {close_category} - Detail: {reason} - P&L: ${pnl:,.2f}")

                    # STEP 3: CLOSE THE HEDGE POSITION IF FOUND
                    if hedge_position and hedge_position.symbol in self.portfolio.positions:
                        # Calculate how much of the hedge to close
                        hedge_close_qty = None  # Default to close all
                        if quantity is not None:
                            # Calculate proportion of position closed
                            original_qty = self.portfolio.positions[symbol].contracts
                            proportion = quantity / original_qty if original_qty > 0 else 1.0
                            hedge_close_qty = int(
                                hedge_position.contracts * proportion)

                        # Close hedge position
                        hedge_price = market_data_by_symbol.get(hedge_position.symbol, {}).get(
                            'MidPrice', hedge_position.current_price)
                        hedge_pnl = self.portfolio.remove_position(
                            symbol=hedge_position.symbol,
                            quantity=hedge_close_qty,
                            price=hedge_price,
                            execution_data=execution_data,
                            reason=f"Closing hedge for {symbol}",
                            skip_margin_calc=True  # Skip redundant margin calculation
                        )

                        self.logger.info(
                            f"[TradeManager] Closed hedge position {hedge_position.symbol} - P&L: ${hedge_pnl:,.2f}")
                else:
                    self.logger.warning(
                        f"Cannot close position {symbol} - not found in portfolio")

            else:
                self.logger.warning(f"Unknown action in signal: {action}")

        # Ensure all positions have their attributes set correctly for margin calculation
        self._ensure_position_attributes_for_margin()
        
        # Use the last calculated margin from the margin calculator if available
        # This avoids redundant recalculation after positions were just added
        if hasattr(self.portfolio, 'margin_calculator') and hasattr(self.portfolio.margin_calculator, '_last_calculated_margin'):
            margin_req = self.portfolio.margin_calculator._last_calculated_margin
            self.logger.info(f"Updated margin requirement after all signals: ${margin_req:,.2f}")
        else:
            # Only calculate if we don't have a recent value
            margin_req = self.calculate_margin_requirement(reuse_last_calculation=False)
            self.logger.info(f"Updated margin requirement after all signals: ${margin_req:,.2f}")

        # Calculate portfolio metrics after executing signals
        portfolio_metrics = self.portfolio.get_portfolio_metrics()
        self.logger.debug(
            f"Portfolio after signals: ${portfolio_metrics['portfolio_value']:,.2f}, {len(self.portfolio.positions)} positions")
        self.logger.debug(
            f"  Cash balance: ${portfolio_metrics['cash_balance']:,.2f}")
        self.logger.debug(
            f"  Delta: {portfolio_metrics['delta']:.2f} (${portfolio_metrics['dollar_delta']:,.2f})")

    def _categorize_close_reason(self, reason: str) -> str:
        """
        Categorize the reason for closing a position into standard categories.

        Args:
            reason: The original reason provided for closing the position

        Returns:
            str: Categorized reason (DTE, STOP, TARGET, or the original reason)
        """
        reason_lower = reason.lower()

        # Check for days to expiry/expiration
        if any(term in reason_lower for term in ['days to expiry', 'dte', 'expiration', 'expiry']):
            return 'DTE'

        # Check for stop loss
        elif any(term in reason_lower for term in ['stop loss', 'stop-loss', 'stop', 'max loss']):
            return 'STOP'

        # Check for profit target
        elif any(term in reason_lower for term in ['profit target', 'target', 'profit goal', 'take profit']):
            return 'TARGET'

        # Return original reason if not matching any category
        return reason

    def calculate_rolling_metrics(self) -> Dict[str, float]:
        """
        Calculate and store rolling performance metrics.

        Returns:
            dict: Dictionary of performance metrics
        """
        # Check if we have enough history
        if not hasattr(self.portfolio, 'daily_returns') or len(self.portfolio.daily_returns) < 5:
            return {}

        # Calculate base metrics
        perf_metrics = self.portfolio.get_performance_metrics()

        # Extract Sharpe ratio
        current_sharpe = perf_metrics.get('sharpe_ratio', 0)

        # Store in history
        if len(self.metrics_history) >= 252:  # One year of trading days
            self.metrics_history.pop(0)  # Remove oldest entry

        self.metrics_history.append({
            'date': self.portfolio.daily_returns[-1]['date'],
            'sharpe_ratio': current_sharpe,
            'volatility': perf_metrics.get('volatility', 0),
            'return': perf_metrics.get('return', 0)
        })

        # Calculate mean and standard deviation of Sharpe ratio
        sharpe_values = [m.get('sharpe_ratio', 0)
                         for m in self.metrics_history if m.get('sharpe_ratio', 0) != 0]

        if len(sharpe_values) >= 5:
            mean_sharpe = np.mean(sharpe_values)
            std_sharpe = np.std(sharpe_values)

            # Calculate z-score
            z_score = (current_sharpe - mean_sharpe) / \
                std_sharpe if std_sharpe > 0 else 0

            # Calculate scaling factor based on z-score
            # Higher z-score means better performance, so scale up
            scaling_factor = min(max(0.5 + (z_score * 0.1), 0.1), 1.0)

            # Add derived metrics
            perf_metrics.update({
                'mean_sharpe': mean_sharpe,
                'std_sharpe': std_sharpe,
                'z_score': z_score,
                'scaling_factor': scaling_factor,
                'nan_count': sum(1 for m in self.metrics_history if m.get('sharpe_ratio', 0) == 0)
            })

        return perf_metrics

    def _log_portfolio_greek_risk(self):
        """
        Log portfolio Greek risk information.
        """
        portfolio_metrics = self.portfolio.get_portfolio_metrics()
        
        # Calculate option delta directly from positions to ensure accuracy
        option_delta = 0
        option_dollar_delta = 0
        
        for symbol, position in self.portfolio.positions.items():
            if isinstance(position, OptionPosition):
                # Get delta with correct sign (already adjusted for short/long in the position)
                raw_delta = position.current_delta * position.contracts
                
                if hasattr(position, 'underlying_price') and position.underlying_price > 0:
                    dollar_delta = raw_delta * 100 * position.underlying_price
                else:
                    # Use a default underlying price if not available
                    underlying_price = 100
                    for opt_pos in self.portfolio.positions.values():
                        if hasattr(opt_pos, 'underlying_price') and opt_pos.underlying_price > 0:
                            underlying_price = opt_pos.underlying_price
                            break
                    dollar_delta = raw_delta * 100 * underlying_price
                
                option_delta += raw_delta
                option_dollar_delta += dollar_delta
        
        # Get the hedge delta from the portfolio
        hedge_delta = portfolio_metrics.get('hedge_delta', 0)
        hedge_dollar_delta = portfolio_metrics.get('hedge_dollar_delta', 0)
        
        # Calculate total delta
        total_delta = option_delta + hedge_delta
        total_dollar_delta = option_dollar_delta + hedge_dollar_delta
        
        # Get other Greeks from portfolio metrics
        gamma = portfolio_metrics.get('gamma', 0)
        theta = portfolio_metrics.get('theta', 0)
        vega = portfolio_metrics.get('vega', 0)
        
        # Get dollar-based Greeks
        dollar_gamma = portfolio_metrics.get('dollar_gamma', gamma * 100)
        dollar_theta = portfolio_metrics.get('dollar_theta', theta)
        dollar_vega = portfolio_metrics.get('dollar_vega', vega)

        # Log portfolio Greek risk
        self.logger.info("--------------------------------------------------")
        self.logger.info("Portfolio Greek Risk:")
        self.logger.info("--------------------------------------------------")
        self.logger.info(f"  Option Delta: {option_delta:.3f} (${option_dollar_delta:.2f})")
        self.logger.info(f"  Hedge Delta: {hedge_delta:.3f} (${hedge_dollar_delta:.2f})")
        self.logger.info(f"  Total Delta: {total_delta:.3f} (${total_dollar_delta:.2f})")
        
        # Determine if the portfolio is majority short options
        short_option_count = 0
        long_option_count = 0
        for symbol, position in self.portfolio.positions.items():
            if hasattr(position, 'position_type') and position.position_type == 'option' and hasattr(position, 'is_short'):
                if position.is_short:
                    short_option_count += 1
                else:
                    long_option_count += 1
        
        # For portfolios that are predominantly short options, flip the sign of Greeks for display
        # Short options benefit from theta decay and are hurt by gamma and vega increases
        if short_option_count > long_option_count and short_option_count > 0:
            # Flip signs for risk representation from a short option perspective
            display_gamma = -gamma
            display_dollar_gamma = -dollar_gamma
            display_theta = -theta
            display_dollar_theta = -dollar_theta
            display_vega = -vega
            display_dollar_vega = -dollar_vega
            
            # Log with adjusted signs for a short-biased portfolio
            self.logger.info(f"  Gamma: {display_gamma:.6f} (${display_dollar_gamma:.2f} per 1% move)")
            self.logger.info(f"  Theta: ${display_dollar_theta:.2f} per day")
            self.logger.info(f"  Vega: ${display_dollar_vega:.2f} per 1% IV")
        else:
            # Use standard signs for long-biased portfolio
            self.logger.info(f"  Gamma: {gamma:.6f} (${dollar_gamma:.2f} per 1% move)")
            self.logger.info(f"  Theta: ${dollar_theta:.2f} per day")
            self.logger.info(f"  Vega: ${dollar_vega:.2f} per 1% IV")

        # Calculate portfolio value components
        nlv = portfolio_metrics.get('net_liquidation_value', 0)
        cash_balance = self.portfolio.cash_balance
        market_value = 0
        equity_value = 0
        long_options_value = 0
        total_liabilities = 0
        short_options_value = 0
        short_equity_value = 0
        
        # Calculate values from positions
        for symbol, position in self.portfolio.positions.items():
            if isinstance(position, OptionPosition):
                position_value = position.current_price * position.contracts * 100
                if position.is_short:
                    # For short options, this is a liability
                    short_options_value += position_value
                else:
                    # For long options, this is an asset
                    long_options_value += position_value
            else:  # Stock position
                position_value = abs(position.current_price * position.contracts)
                if hasattr(position, 'is_short') and position.is_short:
                    # Short stock is a liability
                    short_equity_value += position_value
                else:
                    # Long stock is an asset
                    equity_value += position_value
                
        market_value = equity_value + long_options_value
        total_liabilities = short_options_value + short_equity_value
        
        # Log portfolio value breakdown
        self.logger.info("Portfolio Value Breakdown:")
        self.logger.info("--------------------------------------------------")
        self.logger.info(f"  Net Liquidation Value (NLV): ${nlv:,.2f}")
        self.logger.info(f"    Cash Balance: ${cash_balance:,.2f}")
        self.logger.info(f"    Market Value of Securities: ${market_value:,.2f}")
        self.logger.info(f"      — Equities: ${equity_value:,.2f}")
        self.logger.info(f"      — Long Options: ${long_options_value:,.2f}")
        self.logger.info(f"    Total Liabilities: ${total_liabilities:,.2f}")
        self.logger.info(f"      — Short Options: ${short_options_value:,.2f}")
        self.logger.info(f"      — Short Equities: ${short_equity_value:,.2f}")

    def _position_sizing(self, symbol: str, quantity: int, price: float, is_short: bool,
                         instrument_data: Dict[str, Any], daily_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply position sizing rules to determine the appropriate position size.

        This method uses the position sizer to calculate the optimal position size,
        taking into account margin requirements, portfolio constraints, and risk scaling.

        Args:
            symbol: Instrument symbol
            quantity: Original requested quantity
            price: Current price
            is_short: Whether this is a short position
            instrument_data: Instrument data dictionary
            daily_data: Daily market data

        Returns:
            dict: Dictionary with position sizing information
        """
        # Skip position sizing if disabled or for zero quantity
        if quantity <= 0 or self.position_sizer is None:
            return {'position_size': quantity}

        # Get portfolio value directly - this ensures we get the current value
        portfolio_value = self.portfolio.get_portfolio_value()
        
        # Get current portfolio metrics
        portfolio_metrics = self.portfolio.get_performance_metrics()
        
        # Ensure the metrics have the critical values properly set
        # This is the key fix to ensure proper position sizing
        if portfolio_metrics.get('net_liquidation_value', 0) <= 0:
            portfolio_metrics['net_liquidation_value'] = portfolio_value
            
        # Double-check other required metrics
        if 'available_margin' not in portfolio_metrics or portfolio_metrics['available_margin'] <= 0:
            total_margin = portfolio_metrics.get('total_margin', 0)
            portfolio_metrics['available_margin'] = max(portfolio_value - total_margin, 0)

        # Pass the portfolio object for access to margin calculator
        portfolio_metrics['portfolio'] = self.portfolio

        # Get current date for portfolio history lookup - handle different index types
        current_date = None
        if len(daily_data) > 0:
            # The index might be a datetime or could be numeric (like numpy.int64)
            idx_value = daily_data.index[0]
            if hasattr(idx_value, 'date'):
                # This is a datetime-like object
                current_date = idx_value.date()
            elif isinstance(idx_value, (int, np.integer, float, np.floating)):
                # This is a numeric index - could be a timestamp or just a sequence
                # Convert to date if it looks like a timestamp
                if isinstance(idx_value, (int, np.integer)) and idx_value > 10000:
                    try:
                        # Try to convert assuming it's a timestamp
                        current_date = pd.Timestamp(idx_value).date()
                    except:
                        # Just use the numeric value
                        current_date = idx_value
                else:
                    current_date = idx_value
            else:
                # Use the string representation or whatever it is
                current_date = idx_value

        # Calculate portfolio returns for risk scaling
        returns = self.portfolio.get_returns_series()
        
        # Apply risk scaling if enabled
        risk_scaling = 1.0  # Default neutral scaling
        if self.risk_scaler and self.risk_scaler.enabled:
            risk_scaling = self.risk_scaler.calculate_risk_scaling(returns)
        
        # Use position sizer to calculate final position size
        if hasattr(self.position_sizer, 'calculate_position_size'):
            try:
                position_size = self.position_sizer.calculate_position_size(
                    instrument_data, portfolio_metrics, risk_scaling
                )
            except Exception as e:
                self.logger.error(f"Error in position sizing calculation: {str(e)}")
                self.logger.debug(f"Position sizing error details:", exc_info=True)
                position_size = quantity  # Fall back to original quantity
            
            # Make sure we have a valid integer
            if not isinstance(position_size, (int, float)) or pd.isna(position_size):
                position_size = quantity
            else:
                position_size = int(position_size)  # Ensure integer

            self.logger.debug(
                f"Position sizing for {symbol}: Original={quantity}, Calculated={position_size}")

            # Return the calculated position size
            return {'position_size': position_size}
        else:
            # If position sizer doesn't have the method, return original quantity
            self.logger.debug(
                f"Position sizer doesn't have calculate_position_size method, using original: {quantity}")
            return {'position_size': quantity}

    def _check_position_margin_impact(self, symbol: str, contracts: int, price: float, current_market_data: Dict):
        """
        Check if adding a position would have acceptable margin impact.

        This method evaluates the margin impact of adding a new position,
        considering both the position itself and its potential hedge.

        Args:
            symbol: Instrument symbol
            contracts: Number of contracts
            price: Current price
            current_market_data: Market data dictionary

        Returns:
            bool: Whether position can be added (True if acceptable margin impact)
        """
        # Skip for zero contracts
        if contracts <= 0:
            return True

        # If no margin manager, we can't properly check margin impact
        if self.margin_manager is None:
            # Get risk manager's last margin calculation if available
            margin_per_contract = 0
            if hasattr(self.risk_manager, '_last_margin_per_contract') and self.risk_manager._last_margin_per_contract > 0:
                margin_per_contract = self.risk_manager._last_margin_per_contract
                self.logger.warning(
                    f"[MarginCheck] No margin manager available, using risk manager margin per contract: ${margin_per_contract:.2f}")

                # Calculate margin impact using a simplified approach
                position_margin = margin_per_contract * contracts
                portfolio_value = self.portfolio.get_portfolio_value()
                current_margin = self.portfolio.calculate_margin_requirement()
                total_new_margin = current_margin + position_margin
                margin_utilization = total_new_margin / \
                    portfolio_value if portfolio_value > 0 else 1.0

                # Use high margin threshold from config or default to 85%
                high_margin_threshold = self.config.get(
                    'margin_management', {}).get('high_margin_threshold', 0.85)
                can_add = margin_utilization < high_margin_threshold

                self.logger.warning(
                    f"[MarginCheck] Simplified margin check (no margin manager):")
                self.logger.warning(
                    f"  Position margin: ${position_margin:.2f}")
                self.logger.warning(
                    f"  New total margin: ${total_new_margin:.2f}")
                self.logger.warning(
                    f"  Margin utilization: {margin_utilization:.2%}")
                self.logger.warning(
                    f"  High threshold: {high_margin_threshold:.2%}")
                self.logger.warning(f"  Can add position: {can_add}")

                return can_add
        else:
            # No margin information available, allow with caution
            self.logger.warning(
                f"[MarginCheck] No margin manager or risk manager margin data available")
            self.logger.warning(
                f"[MarginCheck] ALLOWING POSITION WITH CAUTION - RECOMMEND ENABLING MARGIN MANAGEMENT")
            return True

        # Get data for the instrument if available
        instrument_data = {}
        if symbol in current_market_data:
            instrument_data = current_market_data[symbol]

        # Get delta for hedging calculation
        delta = instrument_data.get('Delta', 0)

        # For hedging, we need to know the underlying
        underlying_symbol = instrument_data.get('UnderlyingSymbol', None)
        if not underlying_symbol and 'P' in symbol:
            # Try to extract underlying from option symbol
            parts = symbol.split('P')
            if len(parts) > 0:
                potential_underlying = ''.join(
                    c for c in parts[0] if c.isalpha())
                if potential_underlying:
                    underlying_symbol = potential_underlying

        # Get the underlying price if available
        underlying_price = 0
        if underlying_symbol and underlying_symbol in current_market_data:
            underlying_price = current_market_data[underlying_symbol].get(
                'MidPrice', 0)
        elif 'UnderlyingPrice' in instrument_data:
            underlying_price = instrument_data.get('UnderlyingPrice', 0)

        # Calculate position delta for hedging
        position_delta = delta * contracts

        # Create a position data dictionary for margin manager
        position_data = {
            'symbol': symbol,
            'contracts': contracts,
            'price': price,
            'underlying_price': underlying_price,
            'instrument_data': instrument_data,
            'is_short': True  # Assuming short for options selling
        }

        # Get the margin_per_contract if available from the risk manager
        if hasattr(self.risk_manager, '_last_margin_per_contract') and self.risk_manager._last_margin_per_contract > 0:
            margin_per_contract = self.risk_manager._last_margin_per_contract
            position_data['margin_per_contract'] = margin_per_contract
            self.logger.info(
                f"[MarginCheck] Using margin per contract from risk manager: ${margin_per_contract:.2f}")

        # Determine if position can be added with margin impact, including hedge
        can_add, margin_details = self.margin_manager.can_add_position_with_hedge(
            position_data,
            hedge_delta=-position_delta  # Hedge in opposite direction of position delta
        )

        # Log the margin impact details
        if can_add:
            self.logger.info(
                f"[MarginCheck] Position margin impact is acceptable:")
        else:
            self.logger.warning(
                f"[MarginCheck] Position margin impact is too high:")

        self.logger.info(
            f"  Current margin utilization: {margin_details['current_utilization']:.2%}")
        self.logger.info(
            f"  New margin utilization: {margin_details['new_utilization']:.2%}")
        self.logger.info(
            f"  Additional margin required: ${margin_details['additional_margin']:.2f}")

        return can_add

    def _process_day(self, date_obj) -> Dict[str, Any]:
        """
        Process a single trading day.

        Args:
            date_obj: Date to process

        Returns:
            dict: Results for the day
        """
        # Set current date
        self.current_date = date_obj

        # Reset tracking dictionaries for daily activities
        self.today_added_positions = {}
        self.today_signals_by_symbol = {}

        # Format date for display
        date_str = date_obj.strftime(self.date_format)

        # Log start of day processing
        self.logger.info("=" * 80)
        self.logger.info(f"PROCESSING TRADING DAY: {date_str}")
        self.logger.info("=" * 80)

        # Get daily data for this date
        try:
            self.daily_data = self.data_manager.get_data_for_date(date_obj)
            if self.daily_data is None or len(self.daily_data) == 0:
                self.logger.warning(
                    f"No data available for {date_str}, skipping day")
                return {'status': 'skipped', 'reason': 'no_data'}

            # Convert the daily data to market_data_by_symbol for easier access
            market_data_by_symbol = self.data_manager.get_market_data_by_symbol(
                self.daily_data)

            # Log number of symbols with data
            self.logger.info(
                f"Data available for {len(market_data_by_symbol)} symbols on {date_str}")
        except Exception as e:
            self.logger.error(f"Error getting data for {date_str}: {str(e)}")
            self.logger.exception("Stack trace:")
            return {'status': 'error', 'reason': 'data_error', 'error': str(e)}

        # Update existing positions with current market data
        try:
            self._update_positions_with_market_data(market_data_by_symbol)
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
            self.logger.exception("Stack trace:")
            return {'status': 'error', 'reason': 'position_update_error', 'error': str(e)}

        # Execute trading activities
        try:
            # Execute all trading activities for the day
            self._execute_trading_activities(date_obj, market_data_by_symbol)
        except Exception as e:
            self.logger.error(f"Error executing trading day: {str(e)}")
            self.logger.exception("Stack trace:")
            return {'status': 'error', 'reason': 'execution_error', 'error': str(e)}

        # Update portfolio metrics at end of day
        try:
            portfolio_metrics = self.portfolio.get_portfolio_metrics()
            self.logger.info("POST-TRADE SUMMARY:")
            self.logger.info(
                f"Portfolio value: ${portfolio_metrics['portfolio_value']:,.2f}")

            # Calculate and log P&L
            if hasattr(self, 'prev_portfolio_value'):
                daily_pnl = portfolio_metrics['portfolio_value'] - \
                    self.prev_portfolio_value
                daily_return_pct = daily_pnl / \
                    self.prev_portfolio_value if self.prev_portfolio_value > 0 else 0
                self.logger.info(
                    f"Daily P&L: ${daily_pnl:,.2f} ({daily_return_pct:.2%})")

            # Store current value for next day's comparison
            self.prev_portfolio_value = portfolio_metrics['portfolio_value']

            # Get and log margin metrics
            margin_details = portfolio_metrics.get('margin_details', {})
            margin_calculator_type = margin_details.get(
                'calculator_type', 'Basic')
            total_margin = margin_details.get('total_margin', 0)
            max_leverage = margin_details.get('max_leverage', 1.0)

            self.logger.info("MARGIN DETAILS:")
            self.logger.info(f"  Calculator: {margin_calculator_type}")
            self.logger.info(f"  Max leverage: {max_leverage:.2f}x")
            self.logger.info(
                f"  Total margin requirement: ${total_margin:,.2f}")

            if total_margin > 0:
                margin_utilization = total_margin / \
                    portfolio_metrics['portfolio_value']
                self.logger.info(
                    f"  Margin utilization: {margin_utilization:.2%}")

            # Track metrics for this day
            day_metrics = {
                'date': date_str,
                'portfolio_value': portfolio_metrics['portfolio_value'],
                'margin_requirement': total_margin,
                'open_positions': len(self.portfolio.positions),
                'delta_exposure': portfolio_metrics.get('portfolio_delta', 0),
                'delta_as_pct': portfolio_metrics.get('delta_percentage', 0)
            }

            # Append to metrics history
            self.metrics_history.append(day_metrics)

            # Increment days processed counter
            self.days_processed += 1

            # Return success with day metrics
            return {'status': 'success', 'metrics': day_metrics}

        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            self.logger.exception("Stack trace:")
            return {'status': 'error', 'reason': 'metrics_error', 'error': str(e)}

    def _parse_date(self, date_str):
        """Parse date string to datetime object."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                self.logger.error(f"Could not parse date: {date_str}")
                return None

    def _log_environment_info(self):
        """Log information about the execution environment."""
        import platform
        import sys
        import pandas as pd
        import numpy as np

        self.logger.info("EXECUTION ENVIRONMENT:")
        self.logger.info(f"  Python version: {sys.version.split()[0]}")
        self.logger.info(f"  OS: {platform.system()} {platform.version()}")
        self.logger.info(f"  Pandas version: {pd.__version__}")
        self.logger.info(f"  NumPy version: {np.__version__}")

        self.logger.info("-" * 80)

    def _prepare_hedge_market_data(self, daily_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract and cache underlying prices from market data to ensure consistency across all hedging processes.
        This method should be called before any hedging calculations to ensure the same prices are used everywhere.

        Args:
            daily_data: DataFrame of market data for the current day

        Returns:
            Dict[str, float]: Dictionary of underlying symbols and their prices
        """
        self.logger.info("PHASE 3.5: PREPARING HEDGE MARKET DATA")
        self.logger.info("-" * 50)
        
        # Clear previous cache
        self.cached_underlying_prices = {}
        
        # Identify all relevant underlying symbols
        underlying_symbols = set()
        
        # Add the primary hedge symbol if we have a hedging manager
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            primary_symbol = self.hedging_manager.hedge_symbol
            underlying_symbols.add(primary_symbol)
            self.logger.info(f"Primary hedge symbol: {primary_symbol}")
            
        # Also add symbols from existing positions
        if self.portfolio and hasattr(self.portfolio, 'positions'):
            for symbol, position in self.portfolio.positions.items():
                if hasattr(position, 'instrument_data') and position.instrument_data:
                    if hasattr(position.instrument_data, 'get'):
                        underlying = position.instrument_data.get('UnderlyingSymbol')
                        if underlying:
                            underlying_symbols.add(underlying)
        
        # Extract prices for all identified symbols
        for underlying_symbol in underlying_symbols:
            # First check if we have a direct underlying price in the data
            if hasattr(daily_data, 'iterrows'):
                for _, row in daily_data.iterrows():
                    # Try to find the underlying symbol in various columns
                    symbol_match = False
                    for col in ['Symbol', 'UnderlyingSymbol']:
                        if col in row and row[col] == underlying_symbol:
                            symbol_match = True
                            break

                    if symbol_match:
                        # Try to get price from different columns
                        for price_col in ['UnderlyingPrice', 'Last', 'Close', 'MidPrice']:
                            if price_col in row and row[price_col] > 0:
                                self.cached_underlying_prices[underlying_symbol] = row[price_col]
                                # Break out of price column loop - we have a price
                                break

                            # If we found a price, break out of the row iteration
                            if underlying_symbol in self.cached_underlying_prices:
                                break
            
            # If we still don't have a price, try other methods
            if underlying_symbol not in self.cached_underlying_prices and hasattr(daily_data, 'get'):
                if 'closing_price' in daily_data and underlying_symbol in daily_data.get('closing_price', {}):
                    self.cached_underlying_prices[underlying_symbol] = daily_data['closing_price'][underlying_symbol]
        
            # Check if we found a price through any method
            if underlying_symbol not in self.cached_underlying_prices:
                self.logger.warning(f"Could not find price for {underlying_symbol} in market data")
                
                # Use a realistic default based on the symbol
                default_prices = {
                    'SPY': 475.0,
                    'QQQ': 430.0,
                    'IWM': 200.0,
                    'DIA': 380.0
                }
                default_price = default_prices.get(underlying_symbol, 450.0)
                self.cached_underlying_prices[underlying_symbol] = default_price
                self.logger.warning(f"Using default price for {underlying_symbol}: ${default_price:.2f}")
            else:
                # Only log the price once after we've found it
                self.logger.info(f"Found underlying price for {underlying_symbol}: ${self.cached_underlying_prices[underlying_symbol]:.2f}")
        
        # Share cached prices with the hedging manager
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            if hasattr(self.hedging_manager, 'set_cached_prices'):
                self.hedging_manager.set_cached_prices(self.cached_underlying_prices)
            else:
                # If the method doesn't exist (legacy code), add it as an attribute
                self.hedging_manager.cached_prices = self.cached_underlying_prices
        
        self.logger.info(f"Cached prices set for {len(self.cached_underlying_prices)} symbols")
        self.logger.info(f"Prepared market data for {len(self.cached_underlying_prices)} underlying symbols")
        return self.cached_underlying_prices

    def calculate_margin_requirement(self, reuse_last_calculation=True):
        """
        Calculate the total margin requirement for the portfolio.
        This also ensures all positions have current prices and greeks updated.
        
        Args:
            reuse_last_calculation: Whether to reuse the last calculated margin value if available
        
        Returns:
            float: Total margin requirement
        """
        # Check if we can reuse the last calculated margin
        if reuse_last_calculation and hasattr(self.portfolio.margin_calculator, '_last_calculated_margin'):
            return self.portfolio.margin_calculator._last_calculated_margin
            
        # First, make sure all positions have updated underlying prices
        self._ensure_position_attributes_for_margin()
        
        # Then calculate margin as before
        result = self.portfolio.calculate_margin_requirement()
        
        # Store for future reuse
        if hasattr(self.portfolio, 'margin_calculator'):
            self.portfolio.margin_calculator._last_calculated_margin = result
            
        return result
        
    def _ensure_position_attributes_for_margin(self):
        """
        Ensures all positions have the necessary attributes for margin calculation,
        particularly making sure underlying prices are properly set for all options.
        """
        if not hasattr(self, 'portfolio') or not self.portfolio or not hasattr(self.portfolio, 'positions'):
            return
            
        # First, build a dictionary of underlying prices from stock positions
        underlying_prices = {}
        for symbol, position in self.portfolio.positions.items():
            # If it's a stock/underlying position, add its price to our lookup
            if hasattr(position, 'position_type') and position.position_type == 'stock':
                if hasattr(position, 'current_price') and position.current_price > 0:
                    underlying_prices[symbol] = position.current_price
                    self.logger.debug(f"Using price for {symbol} from stock position: ${position.current_price:.2f}")
        
        # Also add any cached underlying prices
        if hasattr(self, 'cached_underlying_prices'):
            for symbol, price in self.cached_underlying_prices.items():
                if price > 0:
                    underlying_prices[symbol] = price
                    self.logger.debug(f"Using cached underlying price for {symbol}: ${price:.2f}")
                    
        # Now update options with appropriate underlying prices
        for symbol, position in self.portfolio.positions.items():
            # Check if this is an option position that needs an underlying price
            if (hasattr(position, 'position_type') and position.position_type == 'option' and
                (not hasattr(position, 'underlying_price') or position.underlying_price <= 0)):
                
                # Determine the underlying symbol
                underlying_symbol = None
                if hasattr(position, 'underlying') and position.underlying:
                    underlying_symbol = position.underlying
                elif hasattr(position, 'underlying_symbol') and position.underlying_symbol:
                    underlying_symbol = position.underlying_symbol
                elif hasattr(position, 'instrument_data') and position.instrument_data:
                    underlying_symbol = position.instrument_data.get('UnderlyingSymbol')
                # Extract from option symbol if needed
                elif len(symbol) >= 3 and symbol[:3] in ['SPY', 'QQQ', 'IWM', 'DIA']:
                    underlying_symbol = symbol[:3]
                
                # Set the underlying price if we have it
                if underlying_symbol and underlying_symbol in underlying_prices:
                    position.underlying_price = underlying_prices[underlying_symbol]
                    self.logger.info(f"Updated underlying price for {symbol}: ${position.underlying_price:.2f}")
                elif 'SPY' in underlying_prices:  # Default to SPY if nothing else found
                    position.underlying_price = underlying_prices['SPY']
                    self.logger.warning(f"Using SPY price as default for {symbol}: ${position.underlying_price:.2f}")
                
        # Ensure all option positions have required greeks
        for symbol, position in self.portfolio.positions.items():
            if hasattr(position, 'position_type') and position.position_type == 'option':
                # Ensure delta is set - critical for margin calculation
                if not hasattr(position, 'current_delta') or position.current_delta == 0:
                    # Try to set from instrument data
                    if hasattr(position, 'instrument_data') and position.instrument_data:
                        delta = position.instrument_data.get('Delta', position.instrument_data.get('delta', 0))
                        if delta != 0:
                            position.current_delta = delta
                            self.logger.info(f"Set delta for {symbol} from instrument data: {delta:.4f}")

    def _execute_backtest_cycle(self, trading_day, current_date):
        """
        Execute a single backtest cycle for the given trading day.
        
        Args:
            trading_day: Market data for the trading day
            current_date: Current date of simulation
            
        Returns:
            dict: Results for the trading day
        """
        # Start daily cycle
        self.logger.info(f"[Day {current_date.strftime('%Y-%m-%d')}] Starting trading cycle")
        
        # PHASE 1: INITIALIZE
        try:
            # Clear today's tracking dictionaries
            self.today_added_positions = {}
            self.today_closed_positions = {}
            
            # Update the current date
            self.current_date = current_date
            
            # Get realized P&L for the day
            if hasattr(self, 'portfolio') and self.portfolio:
                self.realized_daily_pnl = 0  # Reset daily P&L
            
            # PHASE 2: POSITION MANAGEMENT
            # First check exits for existing positions
            self._check_exit_conditions(trading_day, current_date)
            
            # Update the portfolio with latest prices
            if hasattr(self, 'portfolio') and self.portfolio:
                # Create a market data dictionary by symbol
                market_data_by_symbol = self.data_manager.get_market_data_by_symbol(trading_day) if hasattr(self, 'data_manager') else {}
                
                # Update positions with market data
                self.portfolio.update_market_data(market_data_by_symbol, current_date)
            
            # Update positions with latest prices and greeks
            self._update_positions_market_data(trading_day)
            
            # PHASE 3: STRATEGY SIGNALS
            # Get signals from the strategy
            signals = self.strategy.generate_signals(
                trading_day, self.portfolio, current_date)
            
            # Process strategy signals
            if signals:
                self._execute_signals(signals, trading_day, current_date)
                
            # Make sure all position attributes are up to date for margin calculation
            self._ensure_position_attributes_for_margin()
            
            # PHASE 9: PORTFOLIO UPDATE
            # Calculate EOD metrics and margin requirements
            # Make sure all positions have current prices and Greeks
            if hasattr(self, 'portfolio') and self.portfolio:
                try:
                    # Calculate margin requirements
                    margin_req = self.calculate_margin_requirement()
                    self.logger.info(f"EOD Margin requirement: ${margin_req:,.2f}")
                    
                    # Get portfolio metrics
                    portfolio_metrics = self.portfolio.get_portfolio_metrics()
                    
                    # Log key metrics
                    self.logger.info(f"EOD Portfolio value: ${portfolio_metrics['portfolio_value']:,.2f}")
                    self.logger.info(f"EOD Cash balance: ${portfolio_metrics['cash']:,.2f}")
                    self.logger.info(f"EOD Net liquidation value: ${portfolio_metrics['net_liquidation_value']:,.2f}")
                    
                    # Log position breakdown
                    position_summary = portfolio_metrics.get('positions_summary', {})
                    if position_summary:
                        self.logger.info("EOD Positions summary:")
                        for symbol, details in position_summary.items():
                            self.logger.info(f"  {symbol}: {details['quantity']} @ ${details['current_price']:.2f} = ${details['value']:,.2f}")
                    
                    # Store metrics
                    self.daily_metrics[current_date] = portfolio_metrics
                    
                except Exception as e:
                    self.logger.error(f"Error calculating portfolio metrics: {e}")
                    if self.debug:
                        import traceback
                        self.logger.error(traceback.format_exc())
            
            # Return stats for the day
            day_result = {
                'date': current_date,
                'portfolio_value': self.portfolio.get_portfolio_value() if hasattr(self, 'portfolio') and self.portfolio else 0,
                'cash': self.portfolio.cash if hasattr(self, 'portfolio') and self.portfolio else 0,
                'margin_requirement': margin_req if 'margin_req' in locals() else 0,
                'positions': len(self.portfolio.positions) if hasattr(self, 'portfolio') and self.portfolio and hasattr(self.portfolio, 'positions') else 0,
                'realized_pnl': self.realized_daily_pnl if hasattr(self, 'realized_daily_pnl') else 0
            }
            
            return day_result
        
        except Exception as e:
            self.logger.error(f"Error in trading cycle for {current_date}: {e}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            
            # Return empty result on error
            return {
                'date': current_date,
                'error': str(e)
            }


if __name__ == "__main__":
    # Sample usage if run directly
    import json

    # Load configuration
    try:
        with open("config/config.yaml", "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Use fallback config
        config = {
            "portfolio": {
                "initial_capital": 100000,
                "max_position_size_pct": 0.05
            },
            "strategy": {
                "name": "SampleStrategy",
                "type": "SampleStrategy"
            },
            "data": {
                "sources": [
                    {"type": "csv", "path": "data/sample.csv"}
                ]
            }
        }

    # Create and run trading engine
    engine = TradingEngine(config)
    engine.load_data()
    results = engine.run_backtest()

    print(f"Backtest results: {results}")
