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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
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
else:
    # When imported as a module, use relative imports
    from .data_manager import DataManager
    from .position import Position, OptionPosition
    from .portfolio import Portfolio
    from .margin import MarginCalculator, OptionMarginCalculator, SPANMarginCalculator
    from .hedging import HedgingManager
    from .reporting import ReportingSystem


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
            verbose_console=config_dict.get('verbose_console', True),
            debug_mode=config_dict.get('debug_mode', False)
        )
        
        # Store configuration
        self.config = config_dict
        
        # Initialize other components later
        self.data_manager = None
        self.data = None  # Will hold the processed data
        self.portfolio = None
        self.hedging_manager = None
        self.margin_calculator = None
        self.reporting_system = None
        
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
        # Get configuration values
        initial_capital = self.config.get('initial_capital', 100000)
        max_position_size = self.config.get('max_position_size', 0.05)
        max_portfolio_delta = self.config.get('max_portfolio_delta', 0.20)
        
        # Create portfolio
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            max_position_size_pct=max_position_size,
            max_portfolio_delta=max_portfolio_delta,
            logger=self.logger
        )
        
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
                logger=self.logger
            )
        else:
            self.reporting_system = None
            
        # Create data manager
        data_config = self.config.get('data', {})
        data_sources = data_config.get('sources', [])
        
        if not data_sources:
            self.logger.warning("No data sources specified in configuration")
            
        self.data_manager = DataManager(
            config=data_config,
            logger=self.logger
        )
        
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
        
    def load_data(self) -> None:
        """Load and preprocess data for trading."""
        if self.data_manager is None:
            self.logger.error("Data manager not initialized")
            return
            
        # Load data
        self.logger.info("Loading data...")
        data = self.data_manager.load_data()
        
        if data is None or len(data) == 0:
            self.logger.error("No data loaded")
            return
            
        # Preprocess data
        self.logger.info("Preprocessing data...")
        self.data = self.data_manager.preprocess_data(data)
        
        self.logger.info(f"Data loaded: {len(self.data)} rows, {self.data.columns.tolist()}")
        
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run a backtest using the configured strategy.
        
        Returns:
            dict: Backtest results
        """
        # Check if data is loaded
        if self.data is None or len(self.data) == 0:
            self.logger.error("No data loaded for backtest")
            return {"error": "No data loaded"}
            
        # Check for required date column
        if 'DataDate' not in self.data.columns:
            self.logger.error("Data missing required 'DataDate' column")
            return {"error": "Missing date column"}
            
        self.logger.info(f"Starting backtest with strategy: {self.strategy.name}")
        self.logger.info(f"Initial capital: ${self.portfolio.initial_capital:,.2f}")
        
        try:
            # Get unique dates for processing
            self.data['DataDate'] = pd.to_datetime(self.data['DataDate'])
            unique_dates = self.data['DataDate'].dt.date.unique()
            unique_dates.sort()
            
            total_days = len(unique_dates)
            self.logger.info(f"Backtest period: {unique_dates[0]} to {unique_dates[-1]} ({total_days} trading days)")
            
            # Process each trading day
            processed_days = 0
            for date in unique_dates:
                # Convert date to datetime for filtering
                current_date = pd.to_datetime(date)
                
                # Log progress periodically
                processed_days += 1
                if processed_days % 1 == 0 or processed_days == total_days:
                    progress_pct = processed_days / total_days * 100
                    self.logger.info(f"Progress: {processed_days}/{total_days} days ({progress_pct:.1f}%) - Processing {current_date.strftime('%Y-%m-%d')} - DT.05")
                    
                # Process the current trading day
                self._process_trading_day(current_date)
                
            # Calculate final performance metrics
            performance_metrics = self.portfolio.get_performance_metrics()
            
            # Generate report if reporting is enabled
            report_path = None
            if self.reporting_system:
                try:
                    report_path = self.reporting_system.generate_html_report(
                        self.portfolio.equity_history,
                        performance_metrics,
                        self.portfolio.transactions,
                        f"{self.strategy.name}_backtest"
                    )
                except Exception as e:
                    self.logger.error(f"Error generating report: {e}")
                    self.logger.error(traceback.format_exc())
            
            # Compile results
            results = {
                "initial_capital": self.portfolio.initial_capital,
                "final_value": self.portfolio.get_portfolio_value(),
                "total_return": performance_metrics.get('return', 0),
                "sharpe_ratio": performance_metrics.get('sharpe_ratio', 0),
                "max_drawdown": performance_metrics.get('max_drawdown', 0),
                "volatility": performance_metrics.get('volatility', 0),
                "cagr": performance_metrics.get('cagr', 0),
                "transaction_count": len(self.portfolio.transactions),
                "report_path": report_path
            }
            
            self.logger.info(f"Backtest completed. Final value: ${results['final_value']:,.2f}")
            self.logger.info(f"Total return: {results['total_return']:.2%}")
            
            return results
            
        except Exception as e:
            # Handle any unexpected errors during backtest
            self.logger.error(f"Error during backtest: {e}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
            
    def _log_pre_trade_summary(self, current_date: datetime, daily_data: pd.DataFrame) -> None:
        """
        Log pre-trade portfolio summary after marking-to-market.
        
        Args:
            current_date: Current trading date
            daily_data: Daily market data
        """
        if not self.portfolio.positions:
            return
            
        portfolio_metrics = self.portfolio.get_portfolio_metrics()
        greeks = self.portfolio.get_portfolio_greeks()
        
        # Log "PRE-TRADE Summary" header with date
        self.logger.info("==================================================")
        self.logger.info(f"PRE-TRADE Summary [{current_date.strftime('%Y-%m-%d')}]:")
        
        # Calculate daily return
        if hasattr(self.portfolio, 'daily_returns') and self.portfolio.daily_returns:
            latest_return = self.portfolio.daily_returns[-1]
            daily_pnl = latest_return.get('pnl', 0)
            daily_return_pct = latest_return.get('return', 0)
            self.logger.info(f"Daily P&L: ${daily_pnl:.0f} ({daily_return_pct:.2%})")
            
            # Log option and hedge PnL
            option_pnl = latest_return.get('unrealized_pnl_change', 0)
            hedge_pnl = latest_return.get('realized_pnl', 0)  # Using realized_pnl as hedge_pnl
            self.logger.info(f"  Option PnL: ${option_pnl:.0f}")
            self.logger.info(f"  Hedge PnL: ${hedge_pnl:.0f}")
        
        # Portfolio position info
        self.logger.info(f"Open Trades: {len(self.portfolio.positions)}")
        
        # Calculate exposure
        exposure_pct = portfolio_metrics['position_value'] / portfolio_metrics['portfolio_value'] if portfolio_metrics['portfolio_value'] > 0 else 0
        self.logger.info(f"Total Position Exposure: {exposure_pct:.1%} of NLV")
        self.logger.info(f"Net Liq: ${portfolio_metrics['portfolio_value']:.0f}")
        self.logger.info(f"  Cash Balance: ${portfolio_metrics['cash_balance']:.0f}")
        self.logger.info(f"  Total Liability: ${-portfolio_metrics['position_value']:.0f}")
        
        # Add hedge info if available
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            hedge_pnl = latest_return.get('realized_pnl', 0)  # Using realized_pnl as hedge_pnl
            self.logger.info(f"  Self Hedge (Hedge PnL): ${hedge_pnl:.0f}")
        
        # Margin info
        self.logger.info(f"Total Margin Requirement: ${portfolio_metrics['total_margin']:.0f}")
        self.logger.info(f"Available Margin: ${portfolio_metrics['available_margin']:.0f}")
        self.logger.info(f"Margin-Based Leverage: {portfolio_metrics['current_leverage']:.2f}")
        
        # Portfolio Greeks section
        self.logger.info("\nPortfolio Greek Risk:")
        
        # Option delta
        self.logger.info(f"  Option Delta: {greeks['delta']:.3f} (${greeks['dollar_delta']:.2f})")
        
        # Hedge delta if available
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            hedge_delta = self.hedging_manager.current_hedge_delta if hasattr(self.hedging_manager, 'current_hedge_delta') else 0
            hedge_dollar_delta = self.hedging_manager.current_dollar_delta if hasattr(self.hedging_manager, 'current_dollar_delta') else 0
            self.logger.info(f"  Hedge Delta: {hedge_delta:.3f} (${hedge_dollar_delta:.2f})")
            
            # Total delta (options + hedge)
            total_delta = greeks['delta'] + hedge_delta
            total_dollar_delta = greeks['dollar_delta'] + hedge_dollar_delta
            self.logger.info(f"  Total Delta: {total_delta:.3f} (${total_dollar_delta:.2f})")
        
        # Other Greeks
        self.logger.info(f"  Gamma: {greeks['gamma']:.6f} (${greeks['dollar_gamma']:.2f} per 1% move)")
        self.logger.info(f"  Theta: ${greeks['dollar_theta']:.2f} per day")
        self.logger.info(f"  Vega: ${greeks['dollar_vega']:.2f} per 1% IV")
        
        # Open trades table
        self.logger.info("--------------------------------------------------")
        self.logger.info("\nOpen Trades Table:")
        self.logger.info("-" * 120)
        self.logger.info(f"{'Symbol':<20}{'Contracts':>10}{'Entry':>8}{'Current':>10}{'Value':>10}{'NLV%':>8}{'Underlying':>10}{'Delta':>10}{'DTE':>5}")
        self.logger.info("-" * 120)
        
        portfolio_value = portfolio_metrics['portfolio_value']
        for symbol, position in self.portfolio.positions.items():
            # Calculate position metrics
            is_option = isinstance(position, OptionPosition)
            pos_value = position.current_price * position.contracts * (100 if is_option else 1)
            pos_pct = pos_value / portfolio_value if portfolio_value > 0 else 0
            
            # Format the position info line
            self.logger.info(f"{symbol:<20}{position.contracts:>10d}${position.avg_entry_price:>6.2f}${position.current_price:>8.2f}${pos_value:>9.0f}{pos_pct:>7.1%}${position.underlying_price:>8.2f}{position.current_delta:>10.3f}{position.days_to_expiry if hasattr(position, 'days_to_expiry') else 0:>5d}")
        
        # Print table footer
        self.logger.info("-" * 120)
        self.logger.info(f"TOTAL{' ':>30}${self.portfolio.position_value:>9.0f}{exposure_pct:>7.1%}")
        self.logger.info("-" * 120)
        self.logger.info("==================================================")
        
    def _execute_trading_activities(self, current_date: datetime, daily_data: pd.DataFrame) -> None:
        """
        Execute all trading activities for the day including position management and signal execution.
        
        Args:
            current_date: Current trading date
            daily_data: Daily market data
        """
        # Log start of trading activities
        self.logger.info(f"[TradeManager] Processing trading activities for {current_date.strftime('%Y-%m-%d')}")
        
        # Check debug flags
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            self.logger.info(f"[DEBUG] CPD Hedging enabled: {self.hedging_manager.enabled}")
            self.logger.info(f"[DEBUG] Hedge mode: {self.hedging_manager.hedge_mode}")
            if hasattr(self.hedging_manager, 'target_delta_ratio'):
                self.logger.info(f"[DEBUG] Target Dollar Delta/NLV Ratio: {self.hedging_manager.target_delta_ratio:.4f}")
        
        # Manage existing positions (check exit conditions)
        # This includes checking for profit targets, stop losses, etc.
        self._manage_positions(current_date, daily_data)
        
        # Generate trading signals from strategy
        signals = self.strategy.generate_signals(current_date, daily_data)
        
        # Execute trading signals if any were generated
        if signals:
            self._execute_signals(signals, daily_data, current_date)
            
        # Calculate and store rolling performance metrics
        metrics = self.calculate_rolling_metrics()
        
        # Log risk scaling information if metrics are available
        if metrics and 'sharpe_ratio' in metrics:
            self.logger.info(f"[Risk Scaling] Current Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, Hist Mean: {metrics.get('mean_sharpe', 0):.2f}, z: {metrics.get('z_score', 0):.2f} => Scaling: {metrics.get('scaling_factor', 1.0):.2f}")
            scaling_factor = metrics.get('scaling_factor', 1.0)
            self.logger.info(f"  [Risk Scaling] {'High' if scaling_factor >= 0.8 else 'Low'} scaling factor ({scaling_factor:.2f}) - {'normal' if scaling_factor >= 0.8 else 'reduced'} position sizing")
            
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
            latest_return = self.portfolio.daily_returns[-1]
            daily_pnl = latest_return.get('pnl', 0)
            daily_return_pct = latest_return.get('return', 0)
            self.logger.info(f"Daily P&L: ${daily_pnl:.0f} ({daily_return_pct:.2%})")
            
            # Log option and hedge PnL
            option_pnl = latest_return.get('unrealized_pnl_change', 0)
            hedge_pnl = latest_return.get('realized_pnl', 0)
            self.logger.info(f"  Option PnL: ${option_pnl:.0f}")
            self.logger.info(f"  Hedge PnL: ${hedge_pnl:.0f}")
        
        # Portfolio position info
        self.logger.info(f"Open Trades: {len(self.portfolio.positions)}")
        
        # Calculate exposure
        exposure_pct = portfolio_metrics['position_value'] / portfolio_metrics['portfolio_value'] if portfolio_metrics['portfolio_value'] > 0 else 0
        self.logger.info(f"Total Position Exposure: {exposure_pct:.1%} of NLV")
        self.logger.info(f"Net Liq: ${portfolio_metrics['portfolio_value']:.0f}")
        self.logger.info(f"  Cash Balance: ${portfolio_metrics['cash_balance']:.0f}")
        self.logger.info(f"  Total Liability: ${-portfolio_metrics['position_value']:.0f}")
        
        # Add hedge info if available
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            hedge_pnl = latest_return.get('realized_pnl', 0)
            self.logger.info(f"  Self Hedge (Hedge PnL): ${hedge_pnl:.0f}")
        
        # Margin info
        self.logger.info(f"Total Margin Requirement: ${portfolio_metrics['total_margin']:.0f}")
        self.logger.info(f"Available Margin: ${portfolio_metrics['available_margin']:.0f}")
        self.logger.info(f"Margin-Based Leverage: {portfolio_metrics['current_leverage']:.2f}")
        
        # Portfolio Greeks section
        self.logger.info("\nPortfolio Greek Risk:")
        
        # Option delta
        self.logger.info(f"  Option Delta: {greeks['delta']:.3f} (${greeks['dollar_delta']:.2f})")
        
        # Hedge delta if available
        if hasattr(self, 'hedging_manager') and self.hedging_manager:
            hedge_delta = self.hedging_manager.current_hedge_delta if hasattr(self.hedging_manager, 'current_hedge_delta') else 0
            hedge_dollar_delta = self.hedging_manager.current_dollar_delta if hasattr(self.hedging_manager, 'current_dollar_delta') else 0
            self.logger.info(f"  Hedge Delta: {hedge_delta:.3f} (${hedge_dollar_delta:.2f})")
            
            # Total delta (options + hedge)
            total_delta = greeks['delta'] + hedge_delta
            total_dollar_delta = greeks['dollar_delta'] + hedge_dollar_delta
            self.logger.info(f"  Total Delta: {total_delta:.3f} (${total_dollar_delta:.2f})")
        
        # Other Greeks
        self.logger.info(f"  Gamma: {greeks['gamma']:.6f} (${greeks['dollar_gamma']:.2f} per 1% move)")
        self.logger.info(f"  Theta: ${greeks['dollar_theta']:.2f} per day")
        self.logger.info(f"  Vega: ${greeks['dollar_vega']:.2f} per 1% IV")
        
        # Performance metrics if we have enough history
        if hasattr(self.portfolio, 'daily_returns') and len(self.portfolio.daily_returns) >= 5:
            perf = self.portfolio.get_performance_metrics()
            self.logger.info("\nRolling Metrics:")
            self.logger.info(f"  Sharpe: {perf['sharpe_ratio']:.2f}, Volatility: {perf['volatility']:.2%}")
        
        # Open trades table
        self.logger.info("--------------------------------------------------")
        self.logger.info("\nOpen Trades Table:")
        self.logger.info("-" * 120)
        self.logger.info(f"{'Symbol':<20}{'Contracts':>10}{'Entry':>8}{'Current':>10}{'Value':>10}{'NLV%':>8}{'Underlying':>10}{'Delta':>10}{'DTE':>5}")
        self.logger.info("-" * 120)
        
        portfolio_value = portfolio_metrics['portfolio_value']
        for symbol, position in self.portfolio.positions.items():
            # Calculate position metrics
            is_option = isinstance(position, OptionPosition)
            pos_value = position.current_price * position.contracts * (100 if is_option else 1)
            pos_pct = pos_value / portfolio_value if portfolio_value > 0 else 0
            
            # Format the position info line
            self.logger.info(f"{symbol:<20}{position.contracts:>10d}${position.avg_entry_price:>6.2f}${position.current_price:>8.2f}${pos_value:>9.0f}{pos_pct:>7.1%}${position.underlying_price:>8.2f}{position.current_delta:>10.3f}{position.days_to_expiry if hasattr(position, 'days_to_expiry') else 0:>5d}")
        
        # Print table footer
        self.logger.info("-" * 120)
        self.logger.info(f"TOTAL{' ':>30}${self.portfolio.position_value:>9.0f}{exposure_pct:>7.1%}")
        self.logger.info("-" * 120)
        self.logger.info("==================================================")
        
        # Log completion of day processing
        self.logger.info(f"[TradeManager] Completed processing for {current_date.strftime('%Y-%m-%d')}")
    
    def _process_trading_day(self, current_date: datetime) -> None:
        """
        Process a single trading day in the backtest.
        
        Args:
            current_date: Current trading date
        """
        self.logger.info(f"[TradeManager] Processing day: {current_date.strftime('%Y-%m-%d')}")
        
        # Get data for the current day
        daily_data = self.data[self.data['DataDate'] == current_date]
        
        if len(daily_data) == 0:
            self.logger.warning(f"No data available for {current_date.strftime('%Y-%m-%d')}")
            return
            
        # Phase 1: PRE-TRADE - Update positions with market data and log pre-trade summary
        self._update_positions(current_date, daily_data)
        self._log_pre_trade_summary(current_date, daily_data)
        
        # Phase 2: TRADING ACTIVITIES - Manage positions, execute signals, calculate metrics
        self._execute_trading_activities(current_date, daily_data)
        
        # Phase 3: POST-TRADE - Log post-trade summary
        self._log_post_trade_summary(current_date)
            
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
        
        # Store pre-update position values for logging
        pre_update_values = {}
        for symbol, position in self.portfolio.positions.items():
            pre_update_values[symbol] = position.current_price * position.contracts * 100 if isinstance(position, OptionPosition) else position.current_price * position.contracts
        
        # Update portfolio positions
        self.portfolio.update_market_data(market_data_by_symbol, current_date)
            
        # Log position updates
        for symbol, position in self.portfolio.positions.items():
            if symbol in market_data_by_symbol:
                new_value = position.current_price * position.contracts * 100 if isinstance(position, OptionPosition) else position.current_price * position.contracts
                change = new_value - pre_update_values.get(symbol, 0)
                
                self.logger.info(f"[Position Update] {symbol}: ${new_value:.2f} (change: ${change:.2f})")
            
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
                self.logger.info(f"  P&L: ${pnl:.2f}")
                
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
        
        # Process each signal
        for signal in signals:
            action = signal.get('action')
            symbol = signal.get('symbol')
            
            # For open positions
            if action == 'open':
                # Get the required data for opening a position
                quantity = signal.get('quantity', 1)
                price = signal.get('price')
                position_type = signal.get('type', 'option')
                is_short = signal.get('is_short', True)
                
                # Log the signal
                self.logger.info(f"[TradeManager] Opening {quantity} {'short' if is_short else 'long'} {position_type} {symbol}")
                
                # Build execution data
                execution_data = {'date': current_date}
                if 'execution_data' in signal:
                    execution_data.update(signal['execution_data'])
                    
                # Get instrument data
                instrument_data = None
                if 'instrument_data' in signal:
                    instrument_data = signal['instrument_data']
                else:
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
                        self.logger.info(f"  Position value: ${position_value:.2f}")
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
                    self.logger.info(f"[TradeManager] Closed position - P&L: ${pnl:.2f}")
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
            "initial_capital": 100000,
            "max_position_size": 0.05,
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