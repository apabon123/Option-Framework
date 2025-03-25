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
        log_level = logging.DEBUG if debug_mode else log_levels.get(log_level_str, logging.INFO)
        
        # Set up component-specific log levels
        self.component_log_levels = {}
        
        # Check for component-specific logging settings in the new structure
        if 'logging' in config_dict and 'components' in config_dict['logging']:
            # Check for margin logging settings
            if 'margin' in config_dict['logging']['components']:
                level = config_dict['logging']['components']['margin'].get('level', 'standard')
                self.component_log_levels['margin'] = str(level) if level is not None else 'standard'
                
            # Check for portfolio logging settings
            if 'portfolio' in config_dict['logging']['components']:
                level = config_dict['logging']['components']['portfolio'].get('level', 'standard')
                self.component_log_levels['portfolio'] = str(level) if level is not None else 'standard'
        else:
            # Fallback to the old structure for backward compatibility
            # Check for margin logging settings
            if 'margin' in config_dict and 'logging' in config_dict['margin']:
                level = config_dict['margin']['logging'].get('level', 'standard')
                self.component_log_levels['margin'] = str(level) if level is not None else 'standard'
                
            # Check for portfolio logging settings
            if 'portfolio' in config_dict and 'logging' in config_dict['portfolio']:
                level = config_dict['portfolio']['logging'].get('level', 'standard')
                self.component_log_levels['portfolio'] = str(level) if level is not None else 'standard'
        
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
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
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
                'CRITICAL': '\033[91m', # Red
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
        strategy_name = config_dict.get('strategy', {}).get('name', 'UnknownStrategy')
        
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
        self.logger.warning("Using base Strategy.generate_signals - this should be overridden")
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
    
    def __init__(self, config: Dict[str, Any], strategy: Strategy = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the trading engine.
        
        Args:
            config: Configuration dictionary
            strategy: Trading strategy instance (optional)
            logger: Logger instance (optional)
        """
        # Store config
        self.config = config
        
        # Set up logging
        self.logger = logger or logging.getLogger('trading_engine')
        self.logging_manager = None
        # No need to call _setup_logging here as we're getting the logger as a parameter
        
        # Store the provided strategy instance
        self._strategy_instance = strategy
        
        # Initialize components to None (will be set in init_components)
        self.data_manager = None
        self.position_manager = None
        self.portfolio = None
        self.risk_manager = None
        self.margin_calculator = None
        self.margin_manager = None
        self.reporting_system = None
        self.strategy = None  # Will be initialized in _init_components
        
        # Initialize state variables
        self.data = None
        self.trade_history = []
        self.performance_history = []
        
        # Setup initial default params
        self.dates_trading = []
        self.start_date = None
        self.end_date = None
        self.current_date = None
        self.days_processed = 0  # Track number of days processed
        
        # Initialize tracking metrics
        self.metrics_history = []
        
        # Import SimpleLoggingManager for consistency across the application
        from simple_logging import SimpleLoggingManager
        
        # Initialize the logging manager and logger
        self.logging_manager = SimpleLoggingManager()
        self.logger = self.logging_manager.setup_logging(
            self.config,
            verbose_console=True,
            debug_mode=self.config.get('debug_mode', False)
        )
        
        # Extract main configuration parameters
        self.initial_capital = self.config.get('portfolio', {}).get('initial_capital', 100000)
        self.max_leverage = self.config.get('portfolio', {}).get('max_leverage', 1.0)
        
        # Initialize current date and other date-related fields
        self.start_date = self._parse_date(self.config.get('dates', {}).get('start_date'))
        self.end_date = self._parse_date(self.config.get('dates', {}).get('end_date'))
        self.current_date = None
        
        # Initialize containers for data
        self.strategy = None  # Will be initialized in _init_components
        self.margin_calculator = None  # Will be initialized in _init_components
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
        
        # Store starting time for performance tracking
        self.start_time = time.time()
        
        # Store original date format from config for consistency
        self.date_format = self.config.get('date_format', '%Y-%m-%d')
        
        # Initialize components (portfolio, risk manager, etc.)
        self._init_components()
        
    def _init_components(self) -> None:
        """Initialize trading engine components."""
        # Get configuration values from the portfolio section
        portfolio_config = self.config.get('portfolio', {})
        max_position_size = portfolio_config.get('max_position_size_pct', 0.05)
        
        # Create portfolio first
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            logger=self.logger
        )
        self.logger.info(f"Portfolio initialized with ${self.initial_capital:,.2f} capital")
        
        # Create risk manager
        self.risk_manager = RiskManager(
            config=self.config,
            logger=self.logger
        )
        self.logger.info(f"Risk manager initialized with max position size: {max_position_size:.2%}")
        
        # Create strategy
        self.strategy = self._create_strategy()
        
        # Create data manager
        self.data_manager = DataManager(
            config=self.config,
            logger=self.logger
        )
        
        # Create margin calculator
        # First check margin_management config, then fall back to margin config
        margin_config = self.config.get('margin_management', self.config.get('margin', {}))
        margin_type = margin_config.get('margin_calculator_type', 'standard')
        
        # Log which margin calculator type we're using
        self.logger.info(f"Using margin calculator type: {margin_type}")
        
        if margin_type == 'option':
            self.logger.info("[INIT] Using OptionMarginCalculator")
            self.margin_calculator = OptionMarginCalculator(logger=self.logger)
            # Set logging manager separately
            self.margin_calculator.logging_manager = self.logging_manager
        elif margin_type == 'span':
            self.logger.info("[INIT] Using SPANMarginCalculator")
            # Load SPAN parameters if provided in config
            span_params = {}
            if 'span_parameters' in self.config.get('margin', {}):
                span_params = self.config.get('margin', {}).get('span_parameters', {})
                
            self.margin_calculator = SPANMarginCalculator(
                logger=self.logger,
                params=span_params
            )
            # Set logging manager separately
            self.margin_calculator.logging_manager = self.logging_manager
        else:
            self.logger.info("[INIT] Using standard MarginCalculator")
            self.margin_calculator = MarginCalculator(logger=self.logger)  # Pass the logger instance
            # Set logging manager separately
            self.margin_calculator.logging_manager = self.logging_manager
        
        # Connect margin calculator to portfolio (now portfolio is guaranteed to exist)
        self.portfolio.set_margin_calculator(self.margin_calculator)
        
        # Import and use the real margin manager from margin_management module
        from core.margin_management import PortfolioRebalancer
        
        # Create and configure the real margin manager
        self.logger.info("[INIT] Creating PortfolioRebalancer for margin management")
        self.margin_manager = PortfolioRebalancer(
            portfolio=self.portfolio,
            config=self.config,
            logger=self.logger
        )
        
        # Create reporting system
        self.reporting_system = ReportingSystem(
            config=self.config,
            logger=self.logger
        )
        
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
        self.logger.warning(f"No strategy instance provided - creating base Strategy ({strategy_name})")
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
            
            self.logger.info(f"Data loaded: {len(self.data)} rows, {self.data.columns.tolist()}")
            print(f"Preprocessing complete - {len(self.data)} rows ready for trading")
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
        
        # Get the dates for the backtest
        start_date = self.config.get('dates', {}).get('start_date')
        end_date = self.config.get('dates', {}).get('end_date')
        
        self.logger.info(f"Configuration date range: {start_date} to {end_date}")
        
        # Convert string dates to datetime objects if necessary
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            self.logger.debug(f"Converted start_date string to datetime: {start_date}")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            self.logger.debug(f"Converted end_date string to datetime: {end_date}")
            
        # Generate trading dates in the specified range
        self.logger.info("Retrieving available trading dates from data_manager...")
        trading_dates = self.data_manager.get_dates()
        self.logger.info(f"Data manager returned {len(trading_dates)} trading dates")
        
        if not trading_dates:
            self.logger.error("No trading dates available")
            return results
            
        # Debug: print first few dates to see what format they're in
        if trading_dates:
            self.logger.info(f"First 5 trading dates: {trading_dates[:5]}")
            self.logger.info(f"Date type: {type(trading_dates[0])}")
        
        # Filter by date range
        original_count = len(trading_dates)
        if start_date:
            # Check if we need to convert date format
            if isinstance(trading_dates[0], str):
                trading_dates = [datetime.strptime(d, '%Y-%m-%d') for d in trading_dates]
                
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
                start_date_for_compare = start_date.date() if isinstance(start_date, datetime) else start_date
                if d_date >= start_date_for_compare:
                    filtered_dates.append(d)
                    
            trading_dates = filtered_dates
            self.logger.info(f"After start date filter: {len(trading_dates)} dates remain")
            
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
                end_date_for_compare = end_date.date() if isinstance(end_date, datetime) else end_date
                if d_date <= end_date_for_compare:
                    filtered_dates.append(d)
                    
            trading_dates = filtered_dates
            self.logger.info(f"After end date filter: {len(trading_dates)} dates remain")
            
        trading_dates = sorted(trading_dates)
        
        if not trading_dates:
            self.logger.error("No trading dates in the specified range")
            self.logger.error(f"Original date range had {original_count} dates before filtering")
            return results
            
        # Print backtest details
        self.logger.info(f"Backtest range: {trading_dates[0]} to {trading_dates[-1]}")
        self.logger.info(f"Total trading days: {len(trading_dates)}")
        print(f"Backtest range: {trading_dates[0]} to {trading_dates[-1]}")
        print(f"Total trading days: {len(trading_dates)}")
        
        self.logger.info(f"Starting backtest with strategy: {self.strategy.name}")
        self.logger.info(f"Initial capital: ${self.portfolio.initial_capital:,.2f}")
        
        # Process each trading day
        total_dates = len(trading_dates)
        for i, current_date in enumerate(trading_dates):
            # Print progress every 5% or every day if total days <= 20
            if total_dates <= 20 or i % max(1, int(total_dates * 0.05)) == 0:
                progress_pct = (i / total_dates) * 100
                print(f"Progress: {i}/{total_dates} days ({progress_pct:.1f}%) - Processing {current_date.strftime('%Y-%m-%d')}")
            
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
            self.logger.info(f"Generating HTML report '{self.strategy.name}_backtest'...")
            
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
        self.logger.info(f"Backtest completed. Final value: ${final_value:,.2f}")
        self.logger.info(f"Total return: {total_return:.2%}")
        
        print(f"Backtest complete - Final value: ${final_value:,.2f}")
        
        # Enhance reporting output to include verification files
        if hasattr(self, 'reporting_system') and self.reporting_system:
            try:
                final_report_data = self.reporting_system.run_reports(self.portfolio, self, self.current_date)
                
                # Add verification files to results
                if 'pre_trade_file' in final_report_data:
                    results['pre_trade_verification_file'] = final_report_data['pre_trade_file']
                if 'post_trade_file' in final_report_data:
                    results['post_trade_verification_file'] = final_report_data['post_trade_file']
                    
                # Log the verification file paths
                if 'pre_trade_verification_file' in results:
                    self.logger.info(f"Pre-trade verification file: {results['pre_trade_verification_file']}")
                if 'post_trade_verification_file' in results:
                    self.logger.info(f"Post-trade verification file: {results['post_trade_verification_file']}")
            except Exception as e:
                self.logger.error(f"Error generating final reports: {e}")
                self.logger.debug(traceback.format_exc())
            
        return results

    def _process_trading_day(self, current_date: datetime) -> None:
        """
        Process a single trading day.
        
        Args:
            current_date: Current trading date
        """
        # Set the current date for use in other methods
        self.current_date = current_date
        
        # Get data for this trading day
        daily_data = self.data_manager.get_data_for_date(current_date)
        if daily_data is None or daily_data.empty:
            self.logger.warning(f"No data found for {current_date}")
            return
        
        try:
            # Update positions with market data
            self._update_positions(current_date, daily_data)
            
            # Log pre-trade summary
            self._log_pre_trade_summary(current_date, daily_data)
            
            # Execute trading activities
            self._execute_trading_activities(current_date, daily_data)
            
            # Log post-trade summary
            self._log_post_trade_summary(current_date)
            
            # Record metrics (do this before marking end of day to match post-trade summary values)
            self.portfolio.record_daily_metrics(current_date)
            
            # Mark the end of the trading day for all positions
            for symbol, position in self.portfolio.positions.items():
                if hasattr(position, 'mark_end_of_day'):
                    position.mark_end_of_day()
            
            # Add to progress counter
            self.days_processed += 1
            
        except Exception as e:
            self.logger.error(f"Error processing trading day {current_date}: {e}")
            raise

    def _log_pre_trade_summary(self, current_date, daily_data):
        """
        Log the pre-trade summary of the portfolio status.
        This includes portfolio value, open positions, and P&L.
        """
        self.logger.info("==================================================")
        self.logger.info(f"PRE-TRADE SUMMARY - {current_date.strftime('%Y-%m-%d')}:")
        
        # Calculate and log portfolio value
        portfolio_value = self.portfolio.get_portfolio_value()
        self.logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
        
        # Calculate and log P&L if we have a previous portfolio value
        if hasattr(self, 'previous_portfolio_value') and self.previous_portfolio_value is not None:
            daily_pnl = portfolio_value - self.previous_portfolio_value
            daily_pnl_percent = (daily_pnl / self.previous_portfolio_value) * 100 if self.previous_portfolio_value > 0 else 0
            
            # Initialize P&L components
            option_pnl = 0
            equity_pnl = 0
            
            # Calculate P&L breakdown by position type
            for symbol, position in self.portfolio.positions.items():
                self.logger.debug(f"Debug - Position {symbol}: type={getattr(position, 'position_type', 'unknown')}, current_price={getattr(position, 'current_price', 'N/A')}, prev_price={getattr(position, 'prev_price', 'N/A')}, previous_day_price={getattr(position, 'previous_day_price', 'N/A')}, is_short={getattr(position, 'is_short', 'N/A')}")
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
                    pos_equity_pnl = position.previous_day_contracts * (position.current_price - position.previous_day_price)
                    equity_pnl += pos_equity_pnl
                    self.logger.debug(f"Debug - Equity P&L for {symbol}: previous_day_price={position.previous_day_price}, current_price={position.current_price}, contracts={position.previous_day_contracts}, P&L=${pos_equity_pnl:.2f}")
            
            # Cash/other P&L is what's left after accounting for options and equities
            cash_pnl = daily_pnl - option_pnl - equity_pnl
            
            # Log the P&L breakdown
            self.logger.info(f"Daily P&L: ${daily_pnl:.2f} ({daily_pnl_percent:.2f}%)")
            self.logger.info(f"  • Option P&L: ${option_pnl:.2f}")
            self.logger.info(f"  • Equity P&L: ${equity_pnl:.2f}")
            self.logger.info(f"  • Cash/Other P&L: ${cash_pnl:.2f}")
        
        # Log open positions
        self._log_open_positions()
        
        # Log margin details - This is the first time we calculate margin in this cycle
        self._log_margin_details(detailed=True, reuse_existing_metrics=False)
        
        # Log portfolio Greek risk
        self._log_portfolio_greek_risk()
        
        self.logger.info("==================================================")

    def _log_open_positions(self):
        """
        Log open positions in a formatted table.
        """
        # Create a table for option positions
        option_positions = []
        
        # Track positions with zero contracts for debugging
        zero_contract_positions = []
        
        # Process option positions
        for symbol, position in self.portfolio.positions.items():
            # Skip positions with 0 contracts
            if position.contracts == 0:
                zero_contract_positions.append(symbol)
                continue
                
            if isinstance(position, OptionPosition):
                self.logger.debug(f"Formatting option position for display: {symbol}")
                
                # Get basic position info
                contracts = position.contracts
                direction = "-" if position.is_short else ""  # Add minus sign for short positions
                price = position.current_price
                value = position.current_price * abs(contracts) * 100
                
                # Get option details
                option_type = "Option"
                if hasattr(position, 'option_type') and position.option_type:
                    option_type = position.option_type.capitalize()
                
                # Try multiple ways to get expiry and strike
                expiry = "N/A"
                strike = "N/A"
                expiry_date = None  # For DTE calculation
                
                # First try the direct attributes - debug info first
                self.logger.debug(f"Checking position attributes for {symbol}:")
                
                # Get strike price using the get_strike() method
                strike_value = position.get_strike()
                if strike_value:
                    strike = f"${strike_value:.2f}"
                    
                # Get expiry date
                if hasattr(position, 'expiry_date') and position.expiry_date:
                    expiry_date = position.expiry_date
                    expiry = expiry_date.strftime('%Y-%m-%d')
                elif hasattr(position, 'expiration') and position.expiration:
                    expiry_date = position.expiration
                    expiry = expiry_date.strftime('%Y-%m-%d')
                
                # Calculate days to expiry
                dte = "N/A"
                if expiry_date:
                    if isinstance(expiry_date, str):
                        try:
                            expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d')
                        except ValueError:
                            # Could not parse date, keep as N/A
                            pass
                    
                    if isinstance(expiry_date, datetime):
                        current_date = self.current_date
                        if isinstance(current_date, str):
                            current_date = datetime.strptime(current_date, '%Y-%m-%d')
                        
                        # Calculate business days between dates
                        dte = 0
                        while current_date < expiry_date:
                            # Skip weekends
                            if current_date.weekday() < 5:  # 0-4 are Monday to Friday
                                dte += 1
                            current_date += timedelta(days=1)
                
                # Get Greeks with display formatting
                greeks = position.get_greeks(for_display=True)
                delta = greeks['delta']
                gamma = greeks['gamma']
                theta = greeks['theta']
                vega = greeks['vega']
                
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
                direction = "-" if position.is_short else ""  # Add minus sign for short positions
                price = position.current_price
                value = position.current_price * abs(shares)
                
                # Get delta (for stocks, delta is 1.0 per share if long, -1.0 if short)
                delta = position.current_delta
                
                # Calculate total P&L (realized + unrealized)
                total_pnl = 0
                if hasattr(position, 'realized_pnl') and hasattr(position, 'unrealized_pnl'):
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
            self.logger.debug("Reusing cached portfolio metrics for margin details")
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
            calculator_class_name = type(self.portfolio.margin_calculator).__name__
            
            # Map the class name to a more user-friendly display name
            if calculator_class_name == "SPANMarginCalculator":
                margin_calculator_type = "SPAN"
            elif calculator_class_name == "OptionMarginCalculator":
                margin_calculator_type = "Option"
            elif calculator_class_name == "MarginCalculator":
                margin_calculator_type = "Basic"
            else:
                margin_calculator_type = calculator_class_name
                
            self.logger.info(f"  Margin Calculator Type: {margin_calculator_type}")
        else:
            self.logger.info("  Margin Calculator: Not Set (Using position level calculation)")
        
        # Log configuration details for the calculator
        if hasattr(self.portfolio, 'margin_calculator') and self.portfolio.margin_calculator:
            calculator = self.portfolio.margin_calculator
            if hasattr(calculator, 'hedge_credit_rate'):
                try:
                    hedge_credit_rate = float(calculator.hedge_credit_rate)
                    self.logger.info(f"  Hedge Credit Rate: {hedge_credit_rate:.2f} (SPAN margin offset)")
                except (TypeError, ValueError):
                    self.logger.info(f"  Hedge Credit Rate: {calculator.hedge_credit_rate}")
                    
            if hasattr(calculator, 'initial_margin_percentage'):
                try:
                    initial_margin_pct = float(calculator.initial_margin_percentage)
                    self.logger.info(f"  Initial Margin Percentage: {initial_margin_pct:.2%}")
                except (TypeError, ValueError):
                    self.logger.info(f"  Initial Margin Percentage: {calculator.initial_margin_percentage}")
                    
            if hasattr(calculator, 'max_leverage'):
                # Fix: Special handling for max_leverage
                if hasattr(calculator.max_leverage, 'name') and calculator.max_leverage.__class__.__name__ == 'Logger':
                    # This is actually a logger object
                    self.logger.info(f"  Max Leverage: 12.00x (default)")
                else:
                    # Try to convert to float
                    try:
                        max_leverage_value = float(calculator.max_leverage)
                        self.logger.info(f"  Max Leverage: {max_leverage_value:.2f}x")
                    except (TypeError, ValueError):
                        # If it's not a number, just log it as is
                        self.logger.info(f"  Max Leverage: {calculator.max_leverage}")
        
        # Log the margin details
        self.logger.info(f"  Total Margin Requirement: ${total_margin:,.2f}")
        self.logger.info(f"  Available Margin: ${available_margin:,.2f}")
        
        # Calculate margin utilization percentage
        if portfolio_value > 0:
            margin_utilization = (total_margin / portfolio_value) * 100
            self.logger.info(f"  Margin Utilization: {margin_utilization:.2f}%")
            
            # Add warning if margin utilization is high
            if margin_utilization > 50 and detailed:
                warning_level = "HIGH" if margin_utilization > 75 else "MODERATE"
                self.logger.warning(f"  Margin Utilization Warning: {warning_level} - {margin_utilization:.2f}% of available capital is being used for margin")
        else:
            self.logger.info(f"  Margin Utilization: N/A (NLV is zero or negative)")
        
        # If detailed, show margin breakdown by position
        if detailed:
            # Get margin by position and hedging benefits from portfolio metrics
            margin_by_position = portfolio_metrics.get('margin_by_position', {})
            hedging_benefits = portfolio_metrics.get('hedging_benefits', 0)
            
            # Log the total hedging benefits
            if hedging_benefits > 0:
                self.logger.info(f"  Hedging Benefits: ${hedging_benefits:,.2f}")
                self.logger.info(f"  Standalone Sum of Margins: ${(total_margin + hedging_benefits):,.2f}")
                hedge_reduction_pct = (hedging_benefits / (total_margin + hedging_benefits) * 100) if (total_margin + hedging_benefits) > 0 else 0
                self.logger.info(f"  Hedge Reduction: {hedge_reduction_pct:.2f}% of margin")
            
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
                    position_type = "Option" if hasattr(position, 'option_type') else "Stock"
                    is_short = getattr(position, 'is_short', False)
                    direction = "Short" if is_short else "Long"
                    
                    # Format margin percentage of total
                    margin_pct = (margin / total_margin * 100) if total_margin > 0 else 0
                    
                    # Get position details
                    contracts = getattr(position, 'contracts', 0)
                    
                    self.logger.info(f"    {i+1}. {symbol} ({direction} {position_type}): ${margin:,.2f} ({margin_pct:.1f}%), {contracts} contracts")
                    
                    # Limit to top 10 positions if there are many
                    if i >= 9 and len(sorted_positions) > 12:
                        remaining = len(sorted_positions) - 10
                        self.logger.info(f"    ... and {remaining} more positions")
                        break
            
            self.logger.info("--------------------------------------------------")

    def _log_post_trade_summary(self, current_date):
        """
        Log the post-trade summary of the portfolio status.
        This includes portfolio value, open positions, and P&L.
        """
        # Log section header
        self.logger.info("==================================================")
        self.logger.info(f"POST-TRADE SUMMARY - {current_date.strftime('%Y-%m-%d')}:")
        
        # Get and update portfolio value
        portfolio_value = self.portfolio.get_portfolio_value()
        
        # Calculate daily P&L if we have a previous portfolio value
        if hasattr(self, 'previous_portfolio_value') and self.previous_portfolio_value is not None:
            daily_pnl = portfolio_value - self.previous_portfolio_value
            daily_pnl_percent = (daily_pnl / self.previous_portfolio_value) * 100 if self.previous_portfolio_value > 0 else 0
            
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
                    pos_equity_pnl = position.previous_day_contracts * (position.current_price - position.previous_day_price)
                    equity_pnl += pos_equity_pnl
            
            # Cash/other P&L is what's left after accounting for options and equities
            cash_pnl = daily_pnl - option_pnl - equity_pnl
            
            # Log portfolio value and P&L breakdown
            self.logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
            self.logger.info(f"Daily P&L: ${daily_pnl:.2f} ({daily_pnl_percent:.2f}%)")
            self.logger.info(f"  • Option P&L: ${option_pnl:.2f}")
            self.logger.info(f"  • Equity P&L: ${equity_pnl:.2f}")
            self.logger.info(f"  • Cash/Other P&L: ${cash_pnl:.2f}")
            
            # Also log realized P&L today if we have any
            if hasattr(self, 'today_realized_pnl') and self.today_realized_pnl:
                total_realized = sum(self.today_realized_pnl.values())
                if total_realized != 0:
                    self.logger.info(f"Today's Realized P&L: ${total_realized:.2f} (from closed positions)")
        else:
            self.logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
        
        # Log open positions
        self._log_open_positions()
        
        # Log margin details - Reuse metrics to avoid duplicate calculations
        self._log_margin_details(detailed=True, reuse_existing_metrics=True)
        
        # Log portfolio Greek risk
        self._log_portfolio_greek_risk()
        
        # Store today's portfolio value for tomorrow's P&L calculation
        self.previous_portfolio_value = portfolio_value
        
        # Reset today's realized P&L tracker
        self.today_realized_pnl = {}
        
        # Log the daily metrics if we are tracking them
        if hasattr(self.portfolio, 'daily_metrics') and self.portfolio.daily_metrics:
            self.logger.info("-" * 50)
            
            # Display rolling risk metrics
            self.logger.info("Rolling Risk Metrics:")
            
            # Get total days in metrics
            total_days = len(self.portfolio.daily_metrics)
            
            # Expanding window (always all data)
            expanding_metrics = self._calculate_metrics_window(total_days)
            self.logger.info(f"  Expanding Window (n={total_days})")
            self.logger.info(f"    Sharpe: {expanding_metrics['sharpe_ratio']:.2f}, Volatility: {expanding_metrics['annual_volatility']:.2%}")
            
            # Log short, medium, and long window metrics if we have enough data
            if total_days > 5:  # At least 5 days of data
                # Define target window sizes but use expanding if not enough data
                short_window = min(21, total_days)  # Target: 21 days (1 month)
                medium_window = min(63, total_days)  # Target: 63 days (3 months)
                long_window = min(252, total_days)  # Target: 252 days (1 year)
                
                # Only show the shorter windows if different from expanding
                if short_window < total_days:
                    short_metrics = self._calculate_metrics_window(short_window)
                    self.logger.info(f"  Short Window (target=21 days)")
                    self.logger.info(f"    Sharpe: {short_metrics['sharpe_ratio']:.2f}, Volatility: {short_metrics['annual_volatility']:.2%}")
                else:
                    self.logger.info(f"  Short Window (target=21 days)")
                    self.logger.info(f"    Sharpe: {expanding_metrics['sharpe_ratio']:.2f}, Volatility: {expanding_metrics['annual_volatility']:.2%} (expanding window)")
                
                if medium_window < total_days and medium_window != short_window:
                    medium_metrics = self._calculate_metrics_window(medium_window)
                    self.logger.info(f"  Medium Window (target=63 days)")
                    self.logger.info(f"    Sharpe: {medium_metrics['sharpe_ratio']:.2f}, Volatility: {medium_metrics['annual_volatility']:.2%}")
                else:
                    self.logger.info(f"  Medium Window (target=63 days)")
                    
                    if medium_window == short_window:
                        self.logger.info(f"    Sharpe: {expanding_metrics['sharpe_ratio']:.2f}, Volatility: {expanding_metrics['annual_volatility']:.2%} (same as short window)")
                    else:
                        self.logger.info(f"    Sharpe: {expanding_metrics['sharpe_ratio']:.2f}, Volatility: {expanding_metrics['annual_volatility']:.2%} (expanding window)")
                
                if long_window < total_days and long_window != medium_window:
                    long_metrics = self._calculate_metrics_window(long_window)
                    self.logger.info(f"  Long Window (target=252 days)")
                    self.logger.info(f"    Sharpe: {long_metrics['sharpe_ratio']:.2f}, Volatility: {long_metrics['annual_volatility']:.2%}")
                else:
                    self.logger.info(f"  Long Window (target=252 days)")
                    
                    if long_window == medium_window:
                        self.logger.info(f"    Sharpe: {expanding_metrics['sharpe_ratio']:.2f}, Volatility: {expanding_metrics['annual_volatility']:.2%} (same as medium window)")
                    else:
                        self.logger.info(f"    Sharpe: {expanding_metrics['sharpe_ratio']:.2f}, Volatility: {expanding_metrics['annual_volatility']:.2%} (expanding window)")
        
        # Log the end of the summary
        self.logger.info("=="*25)

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
            daily_volatility = self._calculate_std_dev(returns) if len(returns) > 1 else 0
            
            # Annualize volatility (approximate by multiplying by sqrt(252))
            annual_volatility = daily_volatility * (252 ** 0.5) if daily_volatility else 0
            results['annual_volatility'] = annual_volatility
            
            # Calculate Sharpe ratio
            avg_return = sum(returns) / len(returns)
            annualized_return = (1 + avg_return) ** 252 - 1
            risk_free_rate = 0.02  # Assumed 2% risk-free rate
            
            # Avoid division by zero
            if annual_volatility > 0:
                sharpe_ratio = (annualized_return - risk_free_rate) / annual_volatility
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
                self.logger.warning("No daily returns data for rolling metrics calculation")
                return
            
            # Log the number of daily returns for debugging
            num_returns = len(self.portfolio.daily_returns)
            self.logger.debug(f"Calculating rolling metrics with {num_returns} daily returns")
            
            # Extract daily returns series
            returns = []
            dates = []
            for day_return in self.portfolio.daily_returns:
                returns.append(day_return.get('return', 0))
                dates.append(day_return.get('date'))
                self.logger.debug(f"Return entry: date={day_return.get('date')}, return={day_return.get('return', 0)}")
            
            # Need at least 2 returns for meaningful metrics
            if len(returns) < 2:
                self.logger.info("--------------------------------------------------")
                self.logger.info("Rolling Risk Metrics:")
                self.logger.info("  Not enough data yet (need at least 2 days of returns)")
                return
            
            # Convert to pandas Series for rolling calculations
            try:
                import pandas as pd
                import numpy as np
                
                # Create Series (handle both datetime and string dates)
                if all(isinstance(d, str) for d in dates):
                    returns_series = pd.Series(returns, index=pd.to_datetime(dates))
                else:
                    returns_series = pd.Series(returns, index=dates)
                
                self.logger.debug(f"Created returns series with shape: {returns_series.shape}")
                
                # Header for rolling metrics
                self.logger.info("--------------------------------------------------")
                self.logger.info("Rolling Risk Metrics:")
                
                # Use expanding window with minimum periods of 2 if we have limited data
                min_periods = min(2, len(returns))
                
                # Calculate metrics using expanding window since we have limited data
                mean_return = returns_series.expanding(min_periods=min_periods).mean() * 252
                std_return = returns_series.expanding(min_periods=min_periods).std() * np.sqrt(252)
                
                # Calculate Sharpe ratio (annualized)
                sharpe = mean_return.iloc[-1] / std_return.iloc[-1] if std_return.iloc[-1] > 0 else 0
                volatility = std_return.iloc[-1] * 100  # Convert to percentage
                
                # Display expanding window metrics
                self.logger.info(f"  Expanding Window (n={len(returns_series)})")
                self.logger.info(f"    Sharpe: {sharpe:.2f}, Volatility: {volatility:.2f}%")
                
                # Only show short/medium/long windows when we have more data
                if len(returns_series) >= 5:
                    # Short window - default to expanding if not enough data
                    self.logger.info(f"  Short Window (target={short_window} days)")
                    if len(returns_series) < short_window:
                        # Still using expanding window metrics
                        self.logger.info(f"    Sharpe: {sharpe:.2f}, Volatility: {volatility:.2f}% (expanding window)")
                    else:
                        # Full rolling window
                        short_mean = returns_series.rolling(short_window, min_periods=5).mean() * 252
                        short_std = returns_series.rolling(short_window, min_periods=5).std() * np.sqrt(252)
                        short_sharpe = short_mean.iloc[-1] / short_std.iloc[-1] if short_std.iloc[-1] > 0 else 0
                        short_vol = short_std.iloc[-1] * 100  # Convert to percentage
                        self.logger.info(f"    Sharpe: {short_sharpe:.2f}, Volatility: {short_vol:.2f}% (rolling window)")
                    
                    # Medium window  
                    self.logger.info(f"  Medium Window (target={medium_window} days)")
                    if len(returns_series) < medium_window:
                        # Still using expanding window metrics
                        self.logger.info(f"    Sharpe: {sharpe:.2f}, Volatility: {volatility:.2f}% (expanding window)")
                    else:
                        # Full rolling window
                        med_mean = returns_series.rolling(medium_window, min_periods=5).mean() * 252
                        med_std = returns_series.rolling(medium_window, min_periods=5).std() * np.sqrt(252)
                        med_sharpe = med_mean.iloc[-1] / med_std.iloc[-1] if med_std.iloc[-1] > 0 else 0
                        med_vol = med_std.iloc[-1] * 100  # Convert to percentage
                        self.logger.info(f"    Sharpe: {med_sharpe:.2f}, Volatility: {med_vol:.2f}% (rolling window)")
                    
                    # Long window
                    self.logger.info(f"  Long Window (target={long_window} days)")
                    if len(returns_series) < long_window:
                        # Still using expanding window metrics
                        self.logger.info(f"    Sharpe: {sharpe:.2f}, Volatility: {volatility:.2f}% (expanding window)")
                    else:
                        # Full rolling window
                        long_mean = returns_series.rolling(long_window, min_periods=5).mean() * 252
                        long_std = returns_series.rolling(long_window, min_periods=5).std() * np.sqrt(252)
                        long_sharpe = long_mean.iloc[-1] / long_std.iloc[-1] if long_std.iloc[-1] > 0 else 0
                        long_vol = long_std.iloc[-1] * 100  # Convert to percentage
                        self.logger.info(f"    Sharpe: {long_sharpe:.2f}, Volatility: {long_vol:.2f}% (rolling window)")
                    
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
        self.logger.info(f"Updating positions with market data for {current_date.strftime('%Y-%m-%d')}")
        
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
            self.pre_update_values[symbol] = position.current_price * position.contracts * (100 if isinstance(position, OptionPosition) else 1)
        
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
                self.logger.debug("Updated positions using portfolio.update_market_data")
            
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
                        underlying_symbol = position.instrument_data.get('UnderlyingSymbol')
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
                            self.logger.debug(f"Found underlying price for {underlying_symbol}: ${latest_underlying_price:.2f}")
                            break
                    
                    if latest_underlying_price is not None:
                        break
        
        # If we still don't have a price, try other methods
        if latest_underlying_price is None and hasattr(daily_data, 'get'):
            if 'closing_price' in daily_data and underlying_symbol in daily_data.get('closing_price', {}):
                latest_underlying_price = daily_data['closing_price'][underlying_symbol]
                self.logger.debug(f"Using closing price for {underlying_symbol}: ${latest_underlying_price:.2f}")
        
        # Log if we couldn't find a price
        if latest_underlying_price is None:
            self.logger.warning(f"Could not find price for underlying {underlying_symbol} in market data")
        
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
                        self.logger.debug(f"Updated equity position {symbol}: price=${equity_price:.2f}, delta={delta}")
                    else:
                        self.logger.warning(f"No price found for equity position {symbol}, price not updated")
                
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
                                self.logger.debug(f"Set underlying price for {symbol}: ${position.underlying_price:.2f}")
                            elif latest_underlying_price is not None and latest_underlying_price > 0:
                                position.underlying_price = latest_underlying_price
                                self.logger.debug(f"Set underlying price for {symbol} from latest price: ${position.underlying_price:.2f}")
                            else:
                                # Try to get underlying price from a stock position with matching symbol
                                for stock_symbol, stock_pos in self.portfolio.positions.items():
                                    if stock_symbol == underlying_symbol and hasattr(stock_pos, 'current_price'):
                                        position.underlying_price = stock_pos.current_price
                                        self.logger.debug(f"Set underlying price for {symbol} from stock position: ${position.underlying_price:.2f}")
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
                            
                            self.logger.debug(f"Updated option position {symbol}: price=${price:.2f}, delta={position.current_delta:.4f}")
                    else:
                        self.logger.warning(f"No market data found for option position {symbol}, position not updated")
            
            except Exception as e:
                self.logger.warning(f"Error updating position {symbol}: {e}")
        
        # After all positions are updated, make sure the hedge position has a correct delta
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            hedge_position = self.hedging_manager.get_hedge_position()
            if hedge_position:
                # Ensure the hedge position delta reflects the actual number of contracts
                hedge_position.current_delta = hedge_position.contracts if not hedge_position.is_short else -hedge_position.contracts
                self.logger.debug(f"Updated hedge position with delta: {hedge_position.current_delta}")
        
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
                should_exit, reason = self.strategy.check_exit_conditions(position, market_data)
                
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
                    reason=reason
                )
                
                # Categorize the reason for closing
                close_category = self._categorize_close_reason(reason)
                
                # Log detailed info about the closed position
                self.logger.info(f"[TradeManager] Closed position {symbol} - Reason: {close_category} - Detail: {reason} - P&L: ${pnl:,.2f}")
            else:
                self.logger.warning(f"Cannot close position {symbol} - not found in portfolio")
                    
    def _execute_trading_activities(self, current_date: datetime, daily_data: pd.DataFrame) -> None:
        """
        Execute trading activities for the current trading day.
        
        This method orchestrates:
        - Margin management and rebalancing
        - Generating trading signals from the strategy
        - Evaluating and filtering signals
        - Executing valid signals
        - Adjusting hedge positions
        
        Args:
            current_date: The current trading date
            daily_data: Data for the current trading day
        """
        try:
            # Convert daily data to market data dictionary for easier access
            market_data_by_symbol = self.data_manager.get_market_data_by_symbol(daily_data)
            
            # --------------------------------------------------------
            # 1. MARGIN MANAGEMENT - Rebalance if needed
            # --------------------------------------------------------
            self.logger.info("MARGIN MANAGEMENT:")
            self.logger.info("-" * 50)
            
            # Use the data_manager method which already exists
            market_data_by_symbol = self.data_manager.get_market_data_by_symbol(daily_data)
            rebalance_result = self.margin_manager.rebalance_portfolio(current_date, market_data_by_symbol)
            
            if rebalance_result.get('rebalanced', False):
                self.logger.info("[TradingEngine] Portfolio rebalanced")
                # Log additional rebalance details if available
                if 'positions_closed' in rebalance_result:
                    self.logger.info(f"[TradingEngine] Positions closed: {rebalance_result['positions_closed']}")
                if 'margin_reduced' in rebalance_result:
                    self.logger.info(f"[TradingEngine] Margin reduced: ${rebalance_result['margin_reduced']:.2f}")
            else:
                self.logger.info("No rebalancing needed, margin utilization within limits")
            
            # --------------------------------------------------------
            # 2. SIGNAL GENERATION - Get signals from strategy
            # --------------------------------------------------------
            self.logger.info("STRATEGY SIGNALS:")
            self.logger.info("-" * 50)
            
            signals = self.strategy.generate_signals(
                current_date,
                daily_data
            )
            
            if signals:
                self.logger.info(f"[Strategy] Generated {len(signals)} signals for {current_date.strftime('%Y-%m-%d')}")
                
                # --------------------------------------------------------
                # 3. SIGNAL EXECUTION - Open/close positions based on signals
                # --------------------------------------------------------
                self.logger.info("SIGNAL EXECUTION:")
                self.logger.info("-" * 50)
                
                self._execute_signals(signals, daily_data, current_date)
            else:
                self.logger.info(f"[Strategy] No trading signals for {current_date.strftime('%Y-%m-%d')}")
            
            # --------------------------------------------------------
            # 4. EXIT CONDITION CHECK - Review existing positions 
            # --------------------------------------------------------
            self.logger.info("EXIT CONDITION CHECK:")
            self.logger.info("-" * 50)
            
            self._check_exit_conditions(daily_data, current_date)
            
            # --------------------------------------------------------
            # 5. HEDGING - Apply hedging if enabled
            # --------------------------------------------------------
            if self.hedging_manager:
                self.logger.info("PORTFOLIO HEDGING:")
                self.logger.info("-" * 50)
                
                # Apply hedging based on config
                self.hedging_manager.apply_hedging(
                    current_date=current_date,
                    market_data=daily_data
                )
        
        except Exception as e:
            self.logger.error(f"Error during trading activities: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
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
                should_exit, reason = self.strategy.check_exit_conditions(position, market_data)
                
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
                
                # Create execution_data with the provided current date
                execution_data = {'date': current_date}
                
                pnl = self.portfolio.remove_position(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    execution_data=execution_data,
                    reason=reason
                )
                
                # Categorize the reason for closing
                close_category = self._categorize_close_reason(reason)
                
                # Enhanced logging for PnL with reason - include both category and full reason
                self.logger.info(f"[TradeManager] Closed position - Reason: {close_category} - Detail: {reason} - P&L: ${pnl:,.2f}")
            else:
                self.logger.warning(f"Cannot close position {symbol} - not found in portfolio")
                    
    def _execute_signals(self, signals: List[Dict[str, Any]], daily_data: pd.DataFrame, current_date: datetime) -> None:
        """
        Execute trading signals.
        
        Args:
            signals: List of trading signals to execute
            daily_data: Data for the current trading day
            current_date: Current trading date
        """
        # Convert daily data to market data dictionary for easier access
        market_data_by_symbol = self.data_manager.get_market_data_by_symbol(daily_data)
        
        # Log that we're executing signals
        self.logger.info("[TradeManager] Executing trading signals")
        
        # Initialize today's added positions dictionary
        self.today_added_positions = {}
        
        # Process each signal
        for signal in signals:
            action = signal.get('action', '').lower()  # Normalize action to lowercase
            symbol = signal.get('symbol')
            
            # Map buy/sell actions to open with appropriate is_short flag
            if action == 'buy':
                # Convert 'buy' to 'open' with is_short=False
                action = 'open'
                signal['is_short'] = False
                self.logger.debug(f"Mapped 'buy' action to 'open' with is_short=False")
            
            elif action == 'sell':
                # Convert 'sell' to 'open' with is_short=True
                action = 'open'
                signal['is_short'] = True
                self.logger.debug(f"Mapped 'sell' action to 'open' with is_short=True")
            
            if action == 'open':
                # Handle opening a new position
                is_short = signal.get('is_short', False)
                quantity = signal.get('quantity', 0)
                price = signal.get('price')
                position_type = signal.get('position_type', 'option')  # Default to option
                instrument_data = signal.get('instrument_data')
                
                # Extract execution data if present
                execution_data = {'date': current_date}
                if 'execution_data' in signal:
                    execution_data.update(signal['execution_data'])
                
                # Debug info
                self.logger.debug(f"Opening position: {symbol} {quantity} {'short' if is_short else 'long'} {position_type}")
                
                # Position sizing check for options
                if symbol and position_type.lower() == 'option':
                    # Position sizing logic (unchanged)
                    sizing_result = self._position_sizing(symbol, quantity, price, is_short, instrument_data, daily_data)
                    
                    if sizing_result and 'position_size' in sizing_result:
                        quantity = sizing_result['position_size']
                        self.logger.debug(f"Position sizing for {symbol}: {quantity} contracts")
                
                # Check margin impact before executing the trade
                if position_type.lower() == 'option' and not self._check_position_margin_impact(symbol, quantity, price, market_data_by_symbol):
                    self.logger.warning(f"Skipping {symbol} due to margin constraints")
                    continue
                
                # Execute the open position
                if instrument_data:
                    # Add position
                    position = self.portfolio.add_position(
                        symbol=symbol,
                        instrument_data=instrument_data,
                        quantity=quantity,
                        price=price,
                        position_type=position_type,  # Use detected position_type
                        is_short=is_short,
                        execution_data=execution_data,
                        reason=signal.get('reason')  # Pass the reason from the signal
                    )
                    
                    # Enhanced logging
                    if position:
                        position_value = price * quantity * 100 if position_type == 'option' else price * quantity
                        self.logger.info(f"[TradeManager] Added {quantity} {'contracts' if position_type == 'option' else 'shares'} of {symbol}")
                        self.logger.info(f"  Position value: ${position_value:,.2f}")
                        
                        # Store in today's added positions
                        self.today_added_positions[symbol] = {
                            'contracts': quantity,
                            'price': price,
                            'value': position_value,
                            'data': instrument_data,
                            'position_type': position_type,  # Include position type in tracking
                            'is_short': is_short  # Store position direction for sign adjustment
                        }
                else:
                    self.logger.warning(f"Cannot open position {symbol}: No instrument data available")
            
            # For close positions
            elif action == 'close':
                # Get the required data for closing a position
                quantity = signal.get('quantity')  # None means close all
                price = signal.get('price')
                reason = signal.get('reason', 'Signal')
                
                # Log the signal
                self.logger.info(f"[TradeManager] Closing {quantity if quantity else 'all'} {symbol}: {reason}")
                
                # Build execution data
                execution_data = {'date': current_date}
                if 'execution_data' in signal:
                    execution_data.update(signal['execution_data'])
                    
                # Check if we have the position
                if symbol in self.portfolio.positions:
                    # Close position
                    pnl = self.portfolio.remove_position(
                        symbol=symbol,
                        quantity=quantity,
                        price=price,
                        execution_data=execution_data,
                        reason=reason
                    )
                    
                    # Categorize the reason for closing
                    close_category = self._categorize_close_reason(reason)
                    
                    # Enhanced logging for PnL with reason - include both category and full reason
                    self.logger.info(f"[TradeManager] Closed position - Reason: {close_category} - Detail: {reason} - P&L: ${pnl:,.2f}")
                else:
                    self.logger.warning(f"Cannot close position {symbol} - not found in portfolio")
                    
            else:
                self.logger.warning(f"Unknown action in signal: {action}")
                
        # Calculate portfolio metrics after executing signals
        portfolio_metrics = self.portfolio.get_portfolio_metrics()
        self.logger.debug(f"Portfolio after signals: ${portfolio_metrics['portfolio_value']:,.2f}, {len(self.portfolio.positions)} positions")
        self.logger.debug(f"  Cash balance: ${portfolio_metrics['cash_balance']:,.2f}")
        self.logger.debug(f"  Delta: {portfolio_metrics['delta']:.2f} (${portfolio_metrics['dollar_delta']:,.2f})")
    
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
        sharpe_values = [m.get('sharpe_ratio', 0) for m in self.metrics_history if m.get('sharpe_ratio', 0) != 0]
        
        if len(sharpe_values) >= 5:
            mean_sharpe = np.mean(sharpe_values)
            std_sharpe = np.std(sharpe_values)
            
            # Calculate z-score
            z_score = (current_sharpe - mean_sharpe) / std_sharpe if std_sharpe > 0 else 0
            
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
        
        # Get option delta directly from the portfolio method
        option_delta = self.portfolio.get_option_delta()
        hedge_delta = self.portfolio.get_hedge_delta()
        total_delta = option_delta + hedge_delta
        
        # Calculate dollar values
        underlying_price = None
        if hasattr(self, 'market_data') and self.market_data is not None:
            # Try to extract SPY price or other main index
            for symbol in ['SPY', 'SPX', '^SPX']:
                if symbol in self.market_data:
                    data = self.market_data[symbol]
                    if hasattr(data, 'get'):
                        underlying_price = data.get('UnderlyingPrice', data.get('Close', None))
                    elif hasattr(data, 'UnderlyingPrice'):
                        underlying_price = data.UnderlyingPrice
                    
                    if underlying_price is not None:
                        break
        
        # Fallback to a default price if we couldn't get one
        if underlying_price is None or underlying_price <= 0:
            underlying_price = 100
        
        # Calculate dollar values
        option_dollar_delta = option_delta * 100 * underlying_price
        hedge_dollar_delta = hedge_delta * 100 * underlying_price
        total_dollar_delta = total_delta * 100 * underlying_price
        
        # Get other Greeks
        gamma = portfolio_metrics.get('gamma', 0)
        theta = portfolio_metrics.get('theta', 0)
        vega = portfolio_metrics.get('vega', 0)
        
        # Calculate dollar Greeks with correct scaling
        # Gamma is per 1% move in the underlying, which is equivalent to a point move * underlying/100
        dollar_gamma = gamma * 100 * (underlying_price/100)**2
        
        # Theta and vega are already properly scaled in dollar_theta and dollar_vega from portfolio_metrics
        # Remove the redundant multiplication by 100 here since it's already applied in position_inventory.py
        dollar_theta = portfolio_metrics.get('dollar_theta', theta)  # Use pre-calculated dollar_theta
        dollar_vega = portfolio_metrics.get('dollar_vega', vega)  # Use pre-calculated dollar_vega
        
        # Log portfolio Greek risk
        self.logger.info("--------------------------------------------------")
        self.logger.info("Portfolio Greek Risk:")
        self.logger.info("--------------------------------------------------")
        self.logger.info(f"  Option Delta: {option_delta:.3f} (${option_dollar_delta:.2f})")
        self.logger.info(f"  Hedge Delta: {hedge_delta:.3f} (${hedge_dollar_delta:.2f})")
        self.logger.info(f"  Total Delta: {total_delta:.3f} (${total_dollar_delta:.2f})")
        self.logger.info(f"  Gamma: {gamma:.6f} (${dollar_gamma:.2f} per 1% move)")
        self.logger.info(f"  Theta: ${dollar_theta:.2f} per day")
        self.logger.info(f"  Vega: ${dollar_vega:.2f} per 1% IV")
        
        # Calculate portfolio value components
        nlv = portfolio_metrics['portfolio_value']
        cash_balance = self.portfolio.cash_balance
        market_value = 0
        equity_value = 0
        long_options_value = 0
        total_liabilities = 0
        short_options_value = 0
        
        # Calculate values from positions
        for symbol, position in self.portfolio.positions.items():
            if isinstance(position, OptionPosition):
                position_value = position.current_price * position.contracts * 100
                if position.is_short:
                    short_options_value += position_value
                else:
                    long_options_value += position_value
            else:  # Stock position
                position_value = position.current_price * position.contracts
                equity_value += position_value
        
        market_value = equity_value + long_options_value
        total_liabilities = short_options_value
        
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
        
        # Log margin details is now handled by _log_margin_details method
        self.logger.info("--------------------------------------------------")

    def _position_sizing(self, symbol: str, quantity: int, price: float, is_short: bool, 
                         instrument_data: Dict[str, Any], daily_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            symbol: Symbol of the instrument
            quantity: Requested quantity
            price: Current price
            is_short: Whether this is a short position
            instrument_data: Instrument data dictionary
            daily_data: Daily market data
            
        Returns:
            Dict containing position sizing information including 'position_size'
        """
        if not hasattr(self, 'risk_manager') or not self.risk_manager:
            # If no risk manager, return the original quantity
            self.logger.debug(f"No risk manager available, using original position size: {quantity}")
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
        
        # Log portfolio metrics being used for position sizing
        self.logger.debug(f"Portfolio metrics for position sizing:")
        self.logger.debug(f"  Portfolio Value: ${portfolio_value:.2f}")
        self.logger.debug(f"  NLV: ${portfolio_metrics['net_liquidation_value']:.2f}")
        self.logger.debug(f"  Available Margin: ${portfolio_metrics['available_margin']:.2f}")
        self.logger.debug(f"  Total Margin: ${portfolio_metrics.get('total_margin', 0):.2f}")
        
        # Use risk manager to calculate position size
        if hasattr(self.risk_manager, 'calculate_position_size'):
            # Calculate position size using the risk manager
            position_size = self.risk_manager.calculate_position_size(
                instrument_data, 
                portfolio_metrics,
                risk_scaling=getattr(self, 'current_risk_scaling', 1.0)
            )
            
            self.logger.debug(f"Position sizing for {symbol}: Original={quantity}, Calculated={position_size}")
            
            # Return the calculated position size
            return {'position_size': position_size}
        else:
            # If risk manager doesn't have the method, return original quantity
            self.logger.debug(f"Risk manager doesn't have calculate_position_size method, using original: {quantity}")
            return {'position_size': quantity}

    def _check_position_margin_impact(self, symbol: str, contracts: int, price: float, current_market_data: Dict):
        """
        Check the margin impact of adding a new position and its associated hedge.
        
        Args:
            symbol: Option symbol
            contracts: Number of contracts
            price: Current price
            current_market_data: Market data dictionary
            
        Returns:
            bool: True if the position can be added, False otherwise
        """
        # Get the instrument data
        instrument_data = current_market_data.get(symbol, {})
        
        # Estimate position margin requirement
        # For simplicity, use a conservative estimate for short options
        underlying_price = instrument_data.get('UnderlyingPrice', 0)
        if underlying_price <= 0:
            self.logger.warning(f"Cannot estimate margin impact - no underlying price for {symbol}")
            return True  # Default to allowing the position if we can't calculate
            
        # Get delta to estimate hedge requirements
        delta = instrument_data.get('Delta', 0)
        
        # For short options, margin is typically 20% of underlying plus option premium
        is_short = self.config.get('strategy', {}).get('is_short', True)
        
        # Calculate estimated margin
        if is_short:
            option_price = instrument_data.get('MidPrice', price)
            option_value = option_price * contracts * 100  # Option contract value
            
            # Estimate margin requirement (simplified)
            margin_requirement = option_value + (underlying_price * contracts * 100 * 0.2)
        else:
            # For long options, margin is just the premium
            margin_requirement = price * contracts * 100
            
        # Calculate estimated hedge delta
        target_delta_ratio = self.config.get('hedging', {}).get('target_delta_ratio', 0.1)
        hedge_delta = 0
        
        if is_short:
            # For short options, we need to hedge in the opposite direction
            # Simplified hedge calculation
            position_delta = delta * contracts  # Delta is typically negative for puts, positive for calls
            hedge_delta = -position_delta  # Hedge in opposite direction
        
        # Get the margin_per_contract if available from the risk manager
        margin_per_contract = 0
        # Check if we have the margin_per_contract from the risk manager
        if hasattr(self.risk_manager, '_last_margin_per_contract') and self.risk_manager._last_margin_per_contract > 0:
            margin_per_contract = self.risk_manager._last_margin_per_contract
            self.logger.info(f"[TradeManager] Using margin per contract from risk manager: ${margin_per_contract:.2f}")
        
        # Check margin impact with the margin manager
        position_data = {
            'symbol': symbol,
            'contracts': contracts,
            'price': price,
            'margin_requirement': margin_requirement,
            'underlying_price': underlying_price,
            'instrument_data': instrument_data,
            'is_short': is_short
        }
        
        # Add margin_per_contract if we have it from the risk manager
        if margin_per_contract > 0:
            position_data['margin_per_contract'] = margin_per_contract
        
        can_add, details = self.margin_manager.can_add_position_with_hedge(position_data, hedge_delta)
        
        if not can_add:
            self.logger.warning(f"Cannot add position {symbol} - margin impact too high")
            self.logger.info(f"  Current margin utilization: {details['current_utilization']:.2%}")
            self.logger.info(f"  New margin utilization would be: {details['new_utilization']:.2%}")
            self.logger.info(f"  Threshold: {details['threshold']:.2%}")
            
        return can_add

    def _process_day(self, current_date: datetime, daily_data: pd.DataFrame) -> None:
        """
        Process a single trading day.
        
        Args:
            current_date: Current trading date
            daily_data: DataFrame of market data for the current day
        """
        # Convert daily data to market data dictionary for easier access
        market_data_by_symbol = self.data_manager.get_market_data_by_symbol(daily_data)
        
        # Update portfolio with current market data
        self.portfolio.update_positions(current_date, market_data_by_symbol)
        
        # Check if margin rebalancing is needed before making any new trades
        # This helps ensure we have margin capacity for new positions
        # Use the market_data_by_symbol that was already created above
        rebalance_result = self.margin_manager.rebalance_portfolio(current_date, market_data_by_symbol)
        
        if rebalance_result.get('rebalanced', False):
            self.logger.info("[TradingEngine] Portfolio rebalanced")
            # Log additional rebalance details if available
            if 'positions_closed' in rebalance_result:
                self.logger.info(f"[TradingEngine] Positions closed: {rebalance_result['positions_closed']}")
            if 'margin_reduced' in rebalance_result:
                self.logger.info(f"[TradingEngine] Margin reduced: ${rebalance_result['margin_reduced']:.2f}")
        
        # Extract filtered candidates based on strategy criteria
        candidates = self.strategy.filter_candidates(daily_data)
        
        # Check portfolio capacity for new positions
        can_add_position = self.portfolio.can_add_position()
        
        # Generate and execute signals
        signals = []
        if can_add_position and len(candidates) > 0:
            # Select the best candidate
            selected_symbol, selected_data = self.strategy.select_best_candidate(candidates)
            self.logger.info(f"Selected best candidate: {selected_symbol}")
            
            # Get current price
            current_price = selected_data.get('MidPrice', 0)
            
            # Validate basic parameters
            is_valid, reason = self.strategy.validate_trade_params(selected_data)
            
            if not is_valid:
                self.logger.warning(f"Trade validation failed: {reason}")
            elif current_price <= 0:
                self.logger.warning(f"Invalid price ({current_price}) for {selected_symbol}")
            else:
                # Calculate position size
                contracts = self.strategy.calculate_position_size(
                    selected_data,
                    self.portfolio.get_portfolio_value()
                )
                
                # Check margin impact before executing the trade
                if self._check_position_margin_impact(selected_symbol, contracts, current_price, market_data_by_symbol):
                    # Create a signal
                    signal = {
                        'symbol': selected_symbol,
                        'action': 'sell' if self.config.get('strategy', {}).get('is_short', True) else 'buy',
                        'quantity': contracts,
                        'price': current_price,
                        'reason': 'New position'
                    }
                    signals.append(signal)
                else:
                    self.logger.info(f"Skipping {selected_symbol} due to margin constraints")
        
        # Execute signals
        for signal in signals:
            self._execute_signal(signal, market_data_by_symbol.get(signal['symbol'], {}))
        
        # Monitor existing positions
        exit_signals = self.strategy.check_exit_conditions(self.portfolio, market_data_by_symbol)
        
        # Execute exit signals
        for exit_signal in exit_signals:
            self._execute_signal(exit_signal, market_data_by_symbol.get(exit_signal['symbol'], {}))
        
        # Apply hedging adjustments
        if self.config.get('hedging', {}).get('enabled', False):
            self.hedging_manager.apply_hedging(current_date, market_data_by_symbol)
        
        # Log portfolio status
        self._log_portfolio_status()
        
        # Mark end of day for all positions (for daily P&L tracking)
        self.portfolio.mark_end_of_day()

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