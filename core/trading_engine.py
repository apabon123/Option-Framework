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

# Handle imports properly whether run as script or imported as module
if __name__ == "__main__":
    # Add the parent directory to the path so we can run this file directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_manager import DataManager
    from core.position import Position, OptionPosition
    from core.portfolio import Portfolio
    from core.margin import MarginCalculator, OptionMarginCalculator, SPANMarginCalculator
    from core.hedging import HedgingManager
    from core.reporting import ReportingSystem
    from core.risk_manager import RiskManager
else:
    # When imported as a module, use relative imports
    from .data_manager import DataManager
    from .position import Position, OptionPosition
    from .portfolio import Portfolio
    from .margin import MarginCalculator, OptionMarginCalculator, SPANMarginCalculator
    from .hedging import HedgingManager
    from .reporting import ReportingSystem
    from .risk_manager import RiskManager


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
        Set up logging for the trading application.

        Args:
            config_dict: Configuration dictionary
            verbose_console: If True, all logs go to console. If False, only status updates go to console.
            debug_mode: If True, enables DEBUG level logging. If False, uses INFO level.
            clean_format: If True, uses a clean format without timestamps and log levels.

        Returns:
            logger: Configured logger instance
        """
        import sys

        # Store original stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Create output directory if it doesn't exist
        output_dir = config_dict.get('paths', {}).get('output_dir', 'logs')
        os.makedirs(output_dir, exist_ok=True)

        # Build log filename based on configuration
        log_filename = os.path.join(output_dir, self.build_log_filename(config_dict))
        self.log_file = log_filename

        # Create the logger
        logger = logging.getLogger('trading_engine')

        # Set the root logger level first
        logging.getLogger().setLevel(logging.WARNING)

        # Set this specific logger's level
        logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        logger.propagate = False  # Prevent propagation to parent loggers

        # Remove any existing handlers to avoid duplicates
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)

        # File handler - logs everything including DEBUG if debug_mode is True
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

        if clean_format:
            # Use a clean format without timestamps and log levels
            file_formatter = logging.Formatter('%(message)s')
        else:
            # Use standard format with timestamps and log levels
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler - now with color formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create a colored formatter for more visible console output
        class ColoredFormatter(logging.Formatter):
            """Custom formatter with colored output for console."""
            
            COLORS = {
                'WARNING': '\033[93m',  # Yellow
                'INFO': '\033[92m',     # Green
                'DEBUG': '\033[94m',    # Blue
                'CRITICAL': '\033[91m', # Red
                'ERROR': '\033[91m',    # Red
                'ENDC': '\033[0m',      # Reset
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
        logger.info(f"Logger initialized at {log_filename}")
        
        # Store logger for later use
        self.logger = logger
        
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
    Core trading engine for backtesting and executing trading strategies.
    
    This class orchestrates the entire trading process, including:
    - Loading and preprocessing data
    - Instantiating and configuring the strategy
    - Managing the portfolio
    - Calculating risk metrics
    - Processing daily trading activities
    - Tracking and reporting performance
    """
    
    def __init__(self, config_dict: Dict[str, Any], strategy: Optional[Strategy] = None):
        """
        Initialize the trading engine.
        
        Args:
            config_dict: Configuration dictionary
            strategy: Optional strategy instance (if None, will be created from config)
        """
        # Set up logging first
        self.logging_manager = LoggingManager()
        self.logger = self.logging_manager.setup_logging(
            config_dict,
            verbose_console=config_dict.get('verbose', False),
            debug_mode=config_dict.get('debug', False)
        )
        
        # Store configuration
        self.config = config_dict
        self.logger.debug(f"Configuration: {self.config}")
        
        # Initialize components
        self.data_manager = DataManager(self.config, self.logger)
        self.logger.debug(f"DataManager initialized with config: {self.config.get('paths', {})}")
        
        # Initialize other components later
        self.data = None  # Will hold the processed data
        self.portfolio = None
        self.hedging_manager = None
        self.margin_calculator = None
        self.reporting_system = None
        self.risk_manager = None
        
        # If strategy is provided, use it; otherwise create from config
        self.strategy = strategy
        if self.strategy is None:
            self.strategy = self._create_strategy()
            
        # Performance metrics
        self.metrics_history = []
        
        # Initialize components
        self._init_components()
        
    def _init_components(self) -> None:
        """Initialize trading engine components."""
        # Get configuration values from the portfolio section
        portfolio_config = self.config.get('portfolio', {})
        initial_capital = portfolio_config.get('initial_capital', 100000)
        max_position_size = portfolio_config.get('max_position_size_pct', 0.05)
        max_portfolio_delta = portfolio_config.get('max_portfolio_delta', 0.20)
        
        # Create portfolio
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            max_position_size_pct=max_position_size,
            max_portfolio_delta=max_portfolio_delta,
            logger=self.logger
        )
        
        # Create risk manager
        self.risk_manager = RiskManager(
            config=self.config,
            logger=self.logger
        )
        self.logger.info(f"Risk manager initialized with max position size: {max_position_size:.2%}")
        
        # Create margin calculator
        margin_config = self.config.get('margin', {})
        margin_type = margin_config.get('type', 'option')
        
        if margin_type == 'option':
            self.margin_calculator = OptionMarginCalculator(self.logger)
        elif margin_type == 'span':
            self.margin_calculator = SPANMarginCalculator(self.logger)
        else:
            self.margin_calculator = MarginCalculator(self.logger)
        
        # Create hedging manager if hedging is enabled
        hedging_config = self.config.get('hedging', {})
        hedging_enabled = hedging_config.get('enabled', False)
        
        if hedging_enabled:
            self.hedging_manager = HedgingManager(
                config=hedging_config,
                portfolio=self.portfolio,
                logger=self.logger
            )
            self.logger.info(f"Hedging enabled: mode={hedging_config.get('mode', 'delta_neutral')}")
        else:
            self.hedging_manager = None
            self.logger.info("Hedging disabled")
        
        # Create reporting system if reporting is enabled
        reporting_config = self.config.get('reporting', {})
        reporting_enabled = reporting_config.get('enabled', True)
        
        if reporting_enabled:
            self.reporting_system = ReportingSystem(
                config=reporting_config,
                portfolio=self.portfolio,
                logger=self.logger,
                trading_engine=self  # Pass a reference to the trading engine instance
            )
        else:
            self.reporting_system = None
        
        # Log completion
        self.logger.info("Trading engine components initialized")
        
    def _create_strategy(self) -> Strategy:
        """
        Create and configure a strategy based on config.
        
        Returns:
            Strategy: Configured strategy instance
        """
        strategy_config = self.config.get('strategy', {})
        strategy_name = strategy_config.get('name', 'DefaultStrategy')
        strategy_type = strategy_config.get('type', 'DefaultStrategy')
        
        # Placeholder - in a real implementation, would dynamically load the strategy class
        # For now, just return a base Strategy instance
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
        
        # Convert string dates to datetime objects if necessary
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        # Generate trading dates in the specified range
        trading_dates = self.data_manager.get_dates()
        if not trading_dates:
            self.logger.error("No trading dates available")
            return results
            
        # Filter by date range
        if start_date:
            trading_dates = [d for d in trading_dates if d >= start_date]
        if end_date:
            trading_dates = [d for d in trading_dates if d <= end_date]
            
        trading_dates = sorted(trading_dates)
        
        if not trading_dates:
            self.logger.error("No trading dates in the specified range")
            return results
            
        # Print backtest details
        print(f"Backtest range: {trading_dates[0].strftime('%Y-%m-%d')} to {trading_dates[-1].strftime('%Y-%m-%d')}")
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
            current_date: The trading date to process
        """
        # Store the current date for later use
        self.current_date = current_date
        
        # Get the daily data for this date
        daily_data = self.data_manager.get_data_for_date(current_date)
        
        if daily_data is None or daily_data.empty:
            self.logger.warning(f"No data for {current_date.strftime('%Y-%m-%d')}, skipping day")
            return
            
        # Log the current day being processed
        self.logger.info(f"\n\n")
        self.logger.info(f"==================================================")
        self.logger.info(f"TRADING DAY: {current_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"==================================================")
        
        # ----- PRE-TRADE PHASE -----
        # Update existing positions without changing portfolio exposure
        self._update_positions(current_date, daily_data)
        
        # Pre-trade summary and market analysis
        self._log_pre_trade_summary(current_date, daily_data)
        
        # ----- TRADING ACTIVITIES PHASE -----
        self.logger.info(f"\n")
        self.logger.info(f"==================================================")
        self.logger.info(f"TRADING ACTIVITIES [{current_date.strftime('%Y-%m-%d')}]:")
        self.logger.info(f"==================================================")
        
        # Execute trading activities (generates signals, evaluates, executes)
        self._execute_trading_activities(current_date, daily_data)
        
        # ----- POST-TRADE PHASE -----
        # Post-trade summary
        self._log_post_trade_summary(current_date)
        
        # Update portfolio statistics and record daily metrics
        try:
            # Try to call record_daily_metrics if it exists
            if hasattr(self.portfolio, 'record_daily_metrics'):
                self.portfolio.record_daily_metrics(current_date)
            else:
                # Otherwise just log the current portfolio value
                portfolio_value = self.portfolio.get_portfolio_value()
                self.logger.info(f"Portfolio value on {current_date.strftime('%Y-%m-%d')}: ${portfolio_value:,.2f}")
                
                # Simple implementation to record metrics if custom method doesn't exist
                if not hasattr(self.portfolio, 'daily_metrics'):
                    self.portfolio.daily_metrics = {}
                
                # Determine cash value safely
                try:
                    cash = self.portfolio.cash
                except AttributeError:
                    # If 'cash' doesn't exist, use initial_capital as a fallback
                    cash = self.portfolio.initial_capital
                    
                # Store the metrics
                self.portfolio.daily_metrics[current_date] = {
                    'date': current_date,
                    'portfolio_value': portfolio_value,
                    'cash': cash
                }
        except Exception as e:
            self.logger.warning(f"Error recording daily metrics: {e}")
            print(f"Warning: {e}")
            # Continue processing even if this fails

        # Log completion of day processing
        self.logger.info(f"\n")
        self.logger.info(f"==================================================")
        self.logger.info(f"END OF TRADING DAY: {current_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"==================================================")
        self.logger.info(f"\n")

    def _log_pre_trade_summary(self, current_date: datetime, daily_data: pd.DataFrame) -> None:
        """
        Log pre-trade portfolio summary after marking-to-market.
        
        Args:
            current_date: Current trading date
            daily_data: Daily market data
        """
        if not self.portfolio.positions:
            self.logger.info("PRE-TRADE Summary: No open positions")
            return
            
        portfolio_metrics = self.portfolio.get_portfolio_metrics()
        greeks = self.portfolio.get_portfolio_greeks()
        
        # Log "PRE-TRADE Summary" header with date
        self.logger.info("==================================================")
        self.logger.info(f"PRE-TRADE Summary [{current_date.strftime('%Y-%m-%d')}]:")
        self.logger.info("==================================================")
        
        # Log position updates (moved from _update_positions)
        if hasattr(self, 'pre_update_values'):
            for symbol, position in self.portfolio.positions.items():
                new_value = position.current_price * position.contracts * 100 if isinstance(position, OptionPosition) else position.current_price * position.contracts
                change = new_value - self.pre_update_values.get(symbol, 0)
                
                self.logger.info(f"Position: {symbol} - ${new_value:.2f} (change: ${change:.2f})")
        
        # Calculate and log daily return at the top of the summary
        # We'll look for the most recent daily return
        daily_pnl = 0
        daily_return_pct = 0
        
        if hasattr(self.portfolio, 'daily_returns') and self.portfolio.daily_returns:
            # Find the most recent return entry
            for return_entry in reversed(self.portfolio.daily_returns):
                if return_entry.get('date') and return_entry.get('date') < current_date:
                    daily_pnl = return_entry.get('pnl', 0)
                    daily_return_pct = return_entry.get('return', 0)
                    # If the most recent entry has a date earlier than today, use it
                    break
            
            # Log daily P&L info
            self.logger.info(f"Daily P&L: ${daily_pnl:,.2f} ({daily_return_pct:.2%})")
            
            # Also log option and hedge PnL components if available
            if len(self.portfolio.daily_returns) > 0:
                latest_return = None
                # Find the most recent return entry
                for entry in reversed(self.portfolio.daily_returns):
                    if entry.get('date') and entry.get('date') < current_date:
                        latest_return = entry
                        break
                
                if latest_return:
                    option_pnl = latest_return.get('unrealized_pnl_change', 0)
                    hedge_pnl = latest_return.get('realized_pnl', 0)  # Using realized_pnl as hedge_pnl
                    self.logger.info(f"  Option PnL: ${option_pnl:,.2f}")
                    self.logger.info(f"  Hedge PnL: ${hedge_pnl:,.2f}")
        
        # Portfolio position info
        self.logger.info(f"Open Trades: {len(self.portfolio.positions)}")
        
        # Calculate exposure using our enhanced method
        exposure_pct = self.portfolio.get_total_position_exposure()
        self.logger.info(f"Total Position Exposure: {exposure_pct:.1%} of NLV")
        
        portfolio_value = portfolio_metrics['portfolio_value']
        self.logger.info(f"Net Liq: ${portfolio_value:,.0f}")
        self.logger.info(f"  Cash Balance: ${portfolio_metrics['cash_balance']:,.0f}")
        
        # Calculate liability using our enhanced method
        total_liability = self.portfolio.get_total_liability()
        self.logger.info(f"  Total Liability: ${total_liability:,.0f}")
        
        # Margin info
        self.logger.info(f"Total Margin Requirement: ${portfolio_metrics['total_margin']:,.0f}")
        self.logger.info(f"Available Margin: ${portfolio_metrics['available_margin']:,.0f}")
        self.logger.info(f"Margin-Based Leverage: {portfolio_metrics['current_leverage']:.2f}")
        
        # Add detailed margin section
        self.logger.info("\nMargin Details:")
        margin_type = "None"
        if hasattr(self, 'margin_calculator'):
            margin_type = self.margin_calculator.__class__.__name__
        self.logger.info(f"  Margin Method: {margin_type}")
        
        # Add calculation details based on the margin type
        if hasattr(self, 'margin_calculator'):
            if isinstance(self.margin_calculator, OptionMarginCalculator):
                self.logger.info("  Calculation: Option margin based on OCC rules")
                self.logger.info(f"  OTM Margin Multiplier: {self.margin_calculator.otm_margin_multiplier:.2f}")
            elif isinstance(self.margin_calculator, SPANMarginCalculator):
                self.logger.info("  Calculation: SPAN risk-based margin")
                self.logger.info(f"  Initial Margin: {self.margin_calculator.initial_margin_percentage:.1%} of notional")
                self.logger.info(f"  Maintenance Margin: {self.margin_calculator.maintenance_margin_percentage:.1%} of notional")
                self.logger.info(f"  Volatility Multiplier: {self.margin_calculator.volatility_multiplier:.2f}")
                self.logger.info(f"  Hedge Credit Rate: {self.margin_calculator.hedge_credit_rate:.2f}")
            else:
                self.logger.info("  Calculation: Basic margin based on position value")
                self.logger.info(f"  Max Leverage: {self.margin_calculator.max_leverage:.2f}")
        
        # Portfolio Greeks section
        self.logger.info("\nPortfolio Greek Risk:")
        
        # Option delta
        self.logger.info(f"  Option Delta: {greeks['delta']:.3f} (${greeks['dollar_delta']:,.2f})")
        
        # Hedge delta if available
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            hedge_delta = self.hedging_manager.current_hedge_delta if hasattr(self.hedging_manager, 'current_hedge_delta') else 0
            hedge_dollar_delta = self.hedging_manager.current_dollar_delta if hasattr(self.hedging_manager, 'current_dollar_delta') else 0
            self.logger.info(f"  Hedge Delta: {hedge_delta:.3f} (${hedge_dollar_delta:,.2f})")
            
            # Total delta (options + hedge)
            total_delta = greeks['delta'] + hedge_delta
            total_dollar_delta = greeks['dollar_delta'] + hedge_dollar_delta
            self.logger.info(f"  Total Delta: {total_delta:.3f} (${total_dollar_delta:,.2f})")
        
        # Other Greeks
        self.logger.info(f"  Gamma: {greeks['gamma']:.6f} (${greeks['dollar_gamma']:,.2f} per 1% move)")
        self.logger.info(f"  Theta: ${greeks['dollar_theta']:,.2f} per day")
        self.logger.info(f"  Vega: ${greeks['dollar_vega']:,.2f} per 1% IV")
        
        # Open trades table with improved formatting
        self.logger.info("--------------------------------------------------")
        self.logger.info("\nOpen Trades Table:")
        self.logger.info("-" * 140)
        
        # Modified header with better alignment
        header = (
            f"{'Symbol':<16} "
            f"{'Contracts':>9} "
            f"{'Entry':>8} "
            f"{'Current':>8} "
            f"{'Value':>9} "
            f"{'NLV%':>6} "
            f"{'Underlying':>10} "  
            f"{'Delta':>8} "
            f"{'Gamma':>9} "
            f"{'Theta':>8} "
            f"{'Vega':>8} "
            f"{'Margin':>9} "
            f"{'DTE':>5}"
        )
        self.logger.info(header)
        self.logger.info("-" * 140)
        
        portfolio_value = portfolio_metrics['portfolio_value']
        for symbol, position in self.portfolio.positions.items():
            # Calculate position metrics
            is_option = isinstance(position, OptionPosition)
            
            # Use entry price if current price is 0
            price = position.current_price if position.current_price > 0 else position.avg_entry_price
            pos_value = price * position.contracts * (100 if is_option else 1)
            pos_pct = pos_value / portfolio_value if portfolio_value > 0 else 0
            
            # Get margin if available
            margin = position.get_margin_requirement() if hasattr(position, 'get_margin_requirement') else 0
            
            # Format the position info line (with better spacing)
            position_line = (
                f"{symbol:<16} "
                f"{position.contracts:>9d} "
                f"${position.avg_entry_price:>6.2f} "
                f"${price:>6.2f} "
                f"${pos_value:>7,.0f} "
                f"{pos_pct:>5.1%} "
                f"${position.underlying_price:>8.2f} "
                f"{position.current_delta:>8.3f} "
                f"{position.current_gamma:>8.6f} "
                f"${position.current_theta:>6.2f} "
                f"${position.current_vega:>6.2f} "
                f"${margin:>7,.0f} "
                f"{position.days_to_expiry if hasattr(position, 'days_to_expiry') else 0:>5d}"
            )
            self.logger.info(position_line)
        
        # Print table footer with total position value
        self.logger.info("-" * 140)
        
        # Calculate total position value using our enhanced method
        total_exposure = self.portfolio.get_total_position_exposure() * portfolio_value
        total_margin = self.portfolio.get_margin_requirement() if hasattr(self.portfolio, 'get_margin_requirement') else 0
        
        # Format the total line with better spacing
        total_line = (
            f"{'TOTAL':<16} "
            f"{'':<9} "
            f"{'':<8} "
            f"{'':<8} "
            f"${total_exposure:>7,.0f} "
            f"{exposure_pct:>5.1%} "
            f"{'':<10} "
            f"{'':<8} "
            f"{'':<9} "
            f"{'':<8} "
            f"{'':<8} "
            f"${total_margin:>7,.0f} "
            f"{'':<5}"
        )
        self.logger.info(total_line)
        self.logger.info("-" * 140)
        
        # Rolling metrics moved to post-trade summary only
        
        # Add the generation of the pre-trade verification file
        if hasattr(self, 'reporting_system') and self.reporting_system:
            try:
                pre_trade_file = self.reporting_system.generate_business_logic_verification_file(
                    self.portfolio, current_date, pre_trade=True)
                self.logger.debug(f"Pre-trade verification file generated: {pre_trade_file}")
            except Exception as e:
                self.logger.error(f"Error generating pre-trade verification file: {e}")
                self.logger.debug(traceback.format_exc())
        
        # Log completion of day processing
        self.logger.info(f"[TradeManager] Completed processing for {current_date.strftime('%Y-%m-%d')}")
    
    def _execute_trading_activities(self, current_date: datetime, daily_data: pd.DataFrame) -> None:
        """
        Execute all trading activities for the day including position management and signal execution.
        
        Args:
            current_date: Current trading date
            daily_data: Daily market data
        """
        # Now the Trading Activities section header is directly handled by _process_trading_day method
        
        # Check debug flags
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            self.logger.info(f"[DEBUG] CPD Hedging enabled: {self.hedging_manager.enable_hedging}")
            self.logger.info(f"[DEBUG] Hedge mode: {self.hedging_manager.hedge_mode}")
            if hasattr(self.hedging_manager, 'target_delta_ratio'):
                self.logger.info(f"[DEBUG] Target delta ratio: {self.hedging_manager.target_delta_ratio}")
            if hasattr(self.hedging_manager, 'target_portfolio_delta'):
                self.logger.info(f"[DEBUG] Target portfolio delta: {self.hedging_manager.target_portfolio_delta}")
        
        # Execute Risk Scaling
        self.logger.info("--------------------------------------------------")
        self.logger.info("1. Risk Scaling:")
        self.logger.info("--------------------------------------------------")
        # Calculate and store rolling performance metrics
        metrics = self.calculate_rolling_metrics()
        
        # Log risk scaling information if metrics are available
        if metrics and 'sharpe_ratio' in metrics:
            self.logger.info(f"[Risk Scaling] Current Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, Hist Mean: {metrics.get('mean_sharpe', 0):.2f}, Std: {metrics.get('std_sharpe', 0):.2f}, NaN Count: {metrics.get('nan_count', 0)}, z: {metrics.get('z_score', 0):.2f} => Scaling: {metrics.get('scaling_factor', 1.0):.2f}")
            scaling_factor = metrics.get('scaling_factor', 1.0)
            scaling_desc = "Aggressive" if scaling_factor > 0.9 else "Moderate" if scaling_factor > 0.7 else "Cautious" if scaling_factor > 0.4 else "Defensive"
            self.logger.info(f"  [Risk Scaling] {scaling_desc} scaling factor ({scaling_factor:.2f}) - {'aggressive' if scaling_factor > 0.9 else 'normal' if scaling_factor > 0.7 else 'cautious' if scaling_factor > 0.4 else 'minimal'} position sizing")
            
            # Add Portfolio Rebalancer Analysis
            portfolio_value = self.portfolio.get_portfolio_value()
            margin_buffer = self.config.get('margin_management', {}).get('margin_buffer_pct', 0.1)
            max_margin_allowed = portfolio_value * (1 - margin_buffer)
            current_margin = self.portfolio.get_total_margin_requirement()
            available_margin = max_margin_allowed - current_margin
            
            self.logger.info("[Portfolio Rebalancer Analysis]")
            self.logger.info(f"  Current NLV: ${portfolio_value:.0f}")
            self.logger.info(f"  Risk Scaling Factor: {scaling_factor:.2f}")
            self.logger.info(f"  Maximum Margin Allowed: ${max_margin_allowed:.0f} (with {margin_buffer*100:.0f}% buffer)")
            self.logger.info(f"  Current Margin: ${current_margin:.0f}")
            self.logger.info(f"  Available Margin: ${available_margin:.0f} ({(available_margin/portfolio_value*100):.2f}% of NLV)")
        else:
            self.logger.info("Not enough history for risk scaling metrics")
        
        # Execute Portfolio Rebalancing
        self.logger.info("--------------------------------------------------")
        self.logger.info("2. Portfolio Rebalancing:")
        self.logger.info("--------------------------------------------------")
        # Manage existing positions (check exit conditions)
        # This includes checking for profit targets, stop losses, etc.
        self._manage_positions(current_date, daily_data)
        
        # Execute Trading Signals
        self.logger.info("--------------------------------------------------")
        self.logger.info("3. New Trades:")
        self.logger.info("--------------------------------------------------")
        
        # Generate new signals from the strategy
        signals = self.strategy.generate_signals(current_date, daily_data)
        if signals:
            self.logger.info(f"Generated {len(signals)} trading signals")
            
            # Execute signals
            self._execute_signals(signals, daily_data, current_date)
            
            # Display New Trades Table if we added any new positions
            if hasattr(self, 'today_added_positions') and self.today_added_positions:
                self.logger.info("\nNew Trades Table:")
                self.logger.info("-" * 100)
                header = (
                    f"{'Symbol':<16} "
                    f"{'Contracts':>9} "
                    f"{'Price':>10} "
                    f"{'Value':>12} "
                    f"{'Delta':>10} "
                    f"{'Gamma':>10} "
                    f"{'Theta':>10} "
                    f"{'Vega':>10} "
                    f"{'DTE':>8}"
                )
                self.logger.info(header)
                self.logger.info("-" * 100)
                
                for symbol, trade_data in self.today_added_positions.items():
                    contracts = trade_data.get('contracts', 0)
                    price = trade_data.get('price', 0)
                    value = trade_data.get('value', 0)
                    
                    option_data = trade_data.get('data', {})
                    delta = 0
                    gamma = 0
                    theta = 0
                    vega = 0
                    dte = 0
                    
                    if hasattr(option_data, 'get'):
                        # Dictionary-like access
                        delta = option_data.get('Delta', 0)
                        if 'Gamma' in option_data:
                            gamma = option_data.get('Gamma', 0)
                        if 'Theta' in option_data:
                            theta = option_data.get('Theta', 0)
                        if 'Vega' in option_data:
                            vega = option_data.get('Vega', 0)
                        if 'DaysToExpiry' in option_data:
                            dte = option_data.get('DaysToExpiry', 0)
                            
                    elif hasattr(option_data, 'index'):
                        # Pandas Series
                        delta = option_data['Delta'] if 'Delta' in option_data.index else 0
                        if 'Gamma' in option_data.index:
                            gamma = option_data['Gamma'] if 'Gamma' in option_data.index else 0
                        if 'Theta' in option_data.index:
                            theta = option_data['Theta'] if 'Theta' in option_data.index else 0
                        if 'Vega' in option_data.index:
                            vega = option_data['Vega'] if 'Vega' in option_data.index else 0
                        if 'DaysToExpiry' in option_data.index:
                            dte = option_data['DaysToExpiry']
                    else:
                        # Handle case where option_data is a dictionary without 'get' method
                        if isinstance(option_data, dict):
                            delta = option_data.get('Delta', 0)
                            gamma = option_data.get('Gamma', 0)
                            theta = option_data.get('Theta', 0)
                            vega = option_data.get('Vega', 0)
                            if 'DaysToExpiry' in option_data:
                                dte = option_data['DaysToExpiry']
                            elif 'Expiration' in option_data:
                                # Calculate days to expiry from expiration date
                                expiry = option_data['Expiration']
                                date = current_date
                                dte = (expiry - date).days
                    
                    self.logger.info(f"{symbol:<16}{contracts:>10}${price:>8.2f}${value:>10.2f}{delta:>10.3f}{gamma:>10.6f}{theta:>9.2f}{vega:>9.2f}{dte:>8}")
                
                self.logger.info("-" * 100)
            
        else:
            self.logger.info("No new trade signals generated")
        
        # Manage hedging separately after signals have been executed
        if hasattr(self, 'hedging_manager') and self.hedging_manager and self.hedging_manager.enable_hedging:
            try:
                # Log pre-hedge metrics
                self.logger.info("--------------------------------------------------")
                self.logger.info("4. Hedge Management:")
                self.logger.info("--------------------------------------------------")
                
                # Execute hedging
                hedge_signals = self.hedging_manager.generate_hedge_signals(daily_data, current_date)
                
                if hedge_signals:
                    self.logger.info(f"Generated {len(hedge_signals)} hedging signals")
                    self._execute_signals(hedge_signals, daily_data, current_date)
                else:
                    self.logger.info("No hedge adjustments needed")
                
            except Exception as e:
                self.logger.error(f"Error in hedging: {e}")
        
        self.logger.info("--------------------------------------------------")
        
    def _log_post_trade_summary(self, current_date: datetime) -> None:
        """
        Log post-trade portfolio summary after all trading activities.
        
        Args:
            current_date: Current trading date
        """
        if not self.portfolio.positions:
            return
            
        portfolio_metrics = self.portfolio.get_portfolio_metrics()
        greeks = self.portfolio.get_portfolio_greeks()
        
        # Log "POST-TRADE Summary" header with date
        self.logger.info("==================================================")
        self.logger.info(f"POST-TRADE Summary [{current_date.strftime('%Y-%m-%d')}]:")
        
        # Calculate daily return
        if hasattr(self.portfolio, 'daily_returns') and self.portfolio.daily_returns:
            # Find the most recent return entry
            latest_return = None
            for entry in reversed(self.portfolio.daily_returns):
                if entry.get('date') and entry.get('date') <= current_date:
                    latest_return = entry
                    break
                    
            if latest_return:
                daily_pnl = latest_return.get('pnl', 0)
                daily_return_pct = latest_return.get('return', 0)
                self.logger.info(f"Daily P&L: ${daily_pnl:,.2f} ({daily_return_pct:.2%})")
                
                # Log option and hedge PnL
                option_pnl = latest_return.get('unrealized_pnl_change', 0)
                hedge_pnl = latest_return.get('realized_pnl', 0)
                self.logger.info(f"  Option PnL: ${option_pnl:,.2f}")
                self.logger.info(f"  Hedge PnL: ${hedge_pnl:,.2f}")
        
        # Portfolio position info
        self.logger.info(f"Open Trades: {len(self.portfolio.positions)}")
        
        # Calculate exposure using our enhanced method
        exposure_pct = self.portfolio.get_total_position_exposure()
        self.logger.info(f"Total Position Exposure: {exposure_pct:.1%} of NLV")
        
        portfolio_value = portfolio_metrics['portfolio_value']
        self.logger.info(f"Net Liq: ${portfolio_value:,.0f}")
        self.logger.info(f"  Cash Balance: ${portfolio_metrics['cash_balance']:,.0f}")
        
        # Calculate liability using our enhanced method
        total_liability = self.portfolio.get_total_liability()
        self.logger.info(f"  Total Liability: ${total_liability:,.0f}")
        
        # Add hedge info if available
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            # Get hedge PnL from portfolio if available, otherwise use 0
            hedge_position = self.portfolio.positions.get(self.hedging_manager.hedge_symbol)
            hedge_pnl = hedge_position.realized_pnl if hedge_position else 0
            self.logger.info(f"  Self Hedge (Hedge PnL): ${hedge_pnl:,.2f}")
        
        # Margin info
        self.logger.info(f"Total Margin Requirement: ${portfolio_metrics['total_margin']:,.0f}")
        self.logger.info(f"Available Margin: ${portfolio_metrics['available_margin']:,.0f}")
        self.logger.info(f"Margin-Based Leverage: {portfolio_metrics['current_leverage']:.2f}")
        
        # Add detailed margin section
        self.logger.info("\nMargin Details:")
        margin_type = "None"
        if hasattr(self, 'margin_calculator'):
            margin_type = self.margin_calculator.__class__.__name__
        self.logger.info(f"  Margin Method: {margin_type}")
        
        # Add calculation details based on the margin type
        if hasattr(self, 'margin_calculator'):
            if isinstance(self.margin_calculator, OptionMarginCalculator):
                self.logger.info("  Calculation: Option margin based on OCC rules")
                self.logger.info(f"  OTM Margin Multiplier: {self.margin_calculator.otm_margin_multiplier:.2f}")
            elif isinstance(self.margin_calculator, SPANMarginCalculator):
                self.logger.info("  Calculation: SPAN risk-based margin")
                self.logger.info(f"  Initial Margin: {self.margin_calculator.initial_margin_percentage:.1%} of notional")
                self.logger.info(f"  Maintenance Margin: {self.margin_calculator.maintenance_margin_percentage:.1%} of notional")
                self.logger.info(f"  Volatility Multiplier: {self.margin_calculator.volatility_multiplier:.2f}")
                self.logger.info(f"  Hedge Credit Rate: {self.margin_calculator.hedge_credit_rate:.2f}")
            else:
                self.logger.info("  Calculation: Basic margin based on position value")
                self.logger.info(f"  Max Leverage: {self.margin_calculator.max_leverage:.2f}")
        
        # Portfolio Greeks section
        self.logger.info("\nPortfolio Greek Risk:")
        
        # Option delta
        self.logger.info(f"  Option Delta: {greeks['delta']:.3f} (${greeks['dollar_delta']:,.2f})")
        
        # Hedge delta if available
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            hedge_delta = self.hedging_manager.current_hedge_delta if hasattr(self.hedging_manager, 'current_hedge_delta') else 0
            hedge_dollar_delta = self.hedging_manager.current_dollar_delta if hasattr(self.hedging_manager, 'current_dollar_delta') else 0
            self.logger.info(f"  Hedge Delta: {hedge_delta:.3f} (${hedge_dollar_delta:,.2f})")
            
            # Total delta (options + hedge)
            total_delta = greeks['delta'] + hedge_delta
            total_dollar_delta = greeks['dollar_delta'] + hedge_dollar_delta
            self.logger.info(f"  Total Delta: {total_delta:.3f} (${total_dollar_delta:,.2f})")
        
        # Other Greeks
        self.logger.info(f"  Gamma: {greeks['gamma']:.6f} (${greeks['dollar_gamma']:,.2f} per 1% move)")
        self.logger.info(f"  Theta: ${greeks['dollar_theta']:,.2f} per day")
        self.logger.info(f"  Vega: ${greeks['dollar_vega']:,.2f} per 1% IV")
        
        # Open trades table with improved formatting
        self.logger.info("--------------------------------------------------")
        self.logger.info("\nOpen Trades Table:")
        self.logger.info("-" * 140)
        
        # Modified header with better alignment
        header = (
            f"{'Symbol':<16} "
            f"{'Contracts':>9} "
            f"{'Entry':>8} "
            f"{'Current':>8} "
            f"{'Value':>9} "
            f"{'NLV%':>6} "
            f"{'Underlying':>10} "  
            f"{'Delta':>8} "
            f"{'Gamma':>9} "
            f"{'Theta':>8} "
            f"{'Vega':>8} "
            f"{'Margin':>9} "
            f"{'DTE':>5}"
        )
        self.logger.info(header)
        self.logger.info("-" * 140)
        
        portfolio_value = portfolio_metrics['portfolio_value']
        for symbol, position in self.portfolio.positions.items():
            # Calculate position metrics
            is_option = isinstance(position, OptionPosition)
            
            # Use entry price if current price is 0
            price = position.current_price if position.current_price > 0 else position.avg_entry_price
            pos_value = price * position.contracts * (100 if is_option else 1)
            pos_pct = pos_value / portfolio_value if portfolio_value > 0 else 0
            
            # Get margin if available
            margin = position.get_margin_requirement() if hasattr(position, 'get_margin_requirement') else 0
            
            # Format the position info line (with better spacing)
            position_line = (
                f"{symbol:<16} "
                f"{position.contracts:>9d} "
                f"${position.avg_entry_price:>6.2f} "
                f"${price:>6.2f} "
                f"${pos_value:>7,.0f} "
                f"{pos_pct:>5.1%} "
                f"${position.underlying_price:>8.2f} "
                f"{position.current_delta:>8.3f} "
                f"{position.current_gamma:>8.6f} "
                f"${position.current_theta:>6.2f} "
                f"${position.current_vega:>6.2f} "
                f"${margin:>7,.0f} "
                f"{position.days_to_expiry if hasattr(position, 'days_to_expiry') else 0:>5d}"
            )
            self.logger.info(position_line)
        
        # Print table footer with total position value
        self.logger.info("-" * 140)
        
        # Calculate total position value using our enhanced method
        total_exposure = self.portfolio.get_total_position_exposure() * portfolio_value
        total_margin = self.portfolio.get_margin_requirement() if hasattr(self.portfolio, 'get_margin_requirement') else 0
        
        # Format the total line with better spacing
        total_line = (
            f"{'TOTAL':<16} "
            f"{'':<9} "
            f"{'':<8} "
            f"{'':<8} "
            f"${total_exposure:>7,.0f} "
            f"{exposure_pct:>5.1%} "
            f"{'':<10} "
            f"{'':<8} "
            f"{'':<9} "
            f"{'':<8} "
            f"{'':<8} "
            f"${total_margin:>7,.0f} "
            f"{'':<5}"
        )
        self.logger.info(total_line)
        self.logger.info("-" * 140)
        
        # Rolling metrics output
        if hasattr(self.portfolio, 'get_rolling_metrics'):
            rolling_metrics = self.portfolio.get_rolling_metrics()
            if rolling_metrics:
                self.logger.info("\nRolling Metrics:")
                self.logger.info(f"  Expanding Window (all obs, min 5 required): Sharpe: {rolling_metrics.get('expanding_sharpe', 0):.2f}, Volatility: {rolling_metrics.get('expanding_volatility', 0):.2%}")
                self.logger.info(f"  Short Window (21 days, rolling): Sharpe: {rolling_metrics.get('short_sharpe', 0):.2f}, Volatility: {rolling_metrics.get('short_volatility', 0):.2%}")
                self.logger.info(f"  Medium Window (63 days, rolling): Sharpe: {rolling_metrics.get('medium_sharpe', 0):.2f}, Volatility: {rolling_metrics.get('medium_volatility', 0):.2%}")
                self.logger.info(f"  Long Window (252 days, rolling): Sharpe: {rolling_metrics.get('long_sharpe', 0):.2f}, Volatility: {rolling_metrics.get('long_volatility', 0):.2%}")
        
        self.logger.info("==================================================")
        
        # Add the generation of the post-trade verification file
        if hasattr(self, 'reporting_system') and self.reporting_system:
            try:
                post_trade_file = self.reporting_system.generate_business_logic_verification_file(
                    self.portfolio, current_date, pre_trade=False)
                self.logger.debug(f"Post-trade verification file generated: {post_trade_file}")
            except Exception as e:
                self.logger.error(f"Error generating post-trade verification file: {e}")
                self.logger.debug(traceback.format_exc())
        
        # Log completion of day processing
        self.logger.info(f"[TradeManager] Completed processing for {current_date.strftime('%Y-%m-%d')}")
    
    def _update_positions(self, current_date: datetime, daily_data: pd.DataFrame) -> None:
        """
        Update all positions with latest market data.
        
        Args:
            current_date: Current trading date
            daily_data: Data for the current trading day
        """
        # Skip if no positions to update
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
        
        # Store pre-update position values for logging (will be used later in _log_pre_trade_summary)
        self.pre_update_values = {}
        for symbol, position in self.portfolio.positions.items():
            self.pre_update_values[symbol] = position.current_price * position.contracts * 100 if isinstance(position, OptionPosition) else position.current_price * position.contracts
        
        # Update portfolio positions (passing silent=True to prevent logging POST-TRADE summary)
        self.portfolio.update_market_data(market_data_by_symbol, None)  # Don't pass current_date to avoid POST-TRADE summary

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
                
                # Log detailed info about the closed position
                self.logger.info(f"[TradeManager] Closed position {symbol}: {reason}")
                self.logger.info(f"  Contracts: {quantity}")
                self.logger.info(f"  P&L: ${pnl:,.2f}")
                
    def _execute_signals(self, signals: List[Dict[str, Any]], daily_data: pd.DataFrame, current_date: datetime) -> None:
        """
        Execute trading signals.
        
        Args:
            signals: List of trading signals to execute
            daily_data: Data for the current trading day
            current_date: Current trading date
        """
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
            
            # For open positions
            if action == 'open':
                # Get the required data for opening a position
                base_quantity = signal.get('quantity', 1)  # Base quantity from strategy
                price = signal.get('price')
                position_type = signal.get('type', 'option')
                is_short = signal.get('is_short', True)
                
                # Calculate portfolio metrics for position sizing
                portfolio_metrics = self.portfolio.get_portfolio_metrics()
                
                # Get instrument data for position sizing
                instrument_data = signal.get('instrument_data', {})
                
                # Use RiskManager to determine actual position size
                if hasattr(self, 'risk_manager') and instrument_data and position_type == 'option':
                    # Set up data for position sizing
                    sizing_data = instrument_data.copy()
                    if 'price' not in sizing_data:
                        sizing_data['price'] = price
                    if 'symbol' not in sizing_data and 'OptionSymbol' in sizing_data:
                        sizing_data['symbol'] = sizing_data['OptionSymbol']
                        
                    # Calculate the appropriate position size with default risk scaling
                    risk_scaling = 1.0  # Default risk scaling
                    if hasattr(self, 'current_risk_scaling'):
                        risk_scaling = self.current_risk_scaling
                        
                    scaled_quantity = self.risk_manager.calculate_position_size(
                        sizing_data,
                        portfolio_metrics,
                        risk_scaling
                    )
                    
                    # Use the scaled quantity if it's valid
                    if scaled_quantity > 0:
                        quantity = scaled_quantity
                        self.logger.info(f"[Position Sizing] Scaled quantity from {base_quantity} to {quantity} contracts")
                    else:
                        quantity = base_quantity
                        self.logger.warning(f"[Position Sizing] Failed to scale position, using base quantity: {base_quantity}")
                else:
                    quantity = base_quantity
                    if position_type == 'option':
                        self.logger.warning(f"[Position Sizing] No risk manager available or missing data, using base quantity: {base_quantity}")
                
                # Log the signal
                self.logger.info(f"[TradeManager] Opening {quantity} {'short' if is_short else 'long'} {position_type} {symbol}")
                
                # Build execution data
                execution_data = {'date': current_date}
                if 'execution_data' in signal:
                    execution_data.update(signal['execution_data'])
                    
                # Get instrument data
                if not instrument_data:
                    # Look up in daily data
                    for _, row in daily_data.iterrows():
                        if 'OptionSymbol' in row and row['OptionSymbol'] == symbol:
                            instrument_data = row.to_dict()
                            break
                
                # Execute the open position
                if instrument_data:
                    # Add position
                    position = self.portfolio.add_position(
                        symbol=symbol,
                        instrument_data=instrument_data,
                        quantity=quantity,
                        price=price,
                        position_type=position_type,
                        is_short=is_short,
                        execution_data=execution_data
                    )
                    
                    # Enhanced logging
                    if position:
                        position_value = price * quantity * 100 if position_type == 'option' else price * quantity
                        self.logger.info(f"[TradeManager] Added {quantity} contracts of {symbol}")
                        self.logger.info(f"  Position value: ${position_value:,.2f}")
                        
                        # Store in today's added positions
                        self.today_added_positions[symbol] = {
                            'contracts': quantity,
                            'price': price,
                            'value': position_value,
                            'data': instrument_data
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
                    
                    # Enhanced logging for PnL
                    self.logger.info(f"[TradeManager] Closed position - P&L: ${pnl:,.2f}")
                else:
                    self.logger.warning(f"Cannot close position {symbol} - not found in portfolio")
                    
            else:
                self.logger.warning(f"Unknown action in signal: {action}")
                
        # Calculate portfolio metrics after executing signals
        portfolio_metrics = self.portfolio.get_portfolio_metrics()
        self.logger.debug(f"Portfolio after signals: ${portfolio_metrics['portfolio_value']:,.2f}, {len(self.portfolio.positions)} positions")
        self.logger.debug(f"  Cash balance: ${portfolio_metrics['cash_balance']:,.2f}")
        self.logger.debug(f"  Delta: {portfolio_metrics['delta']:.2f} (${portfolio_metrics['dollar_delta']:,.2f})")
        
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