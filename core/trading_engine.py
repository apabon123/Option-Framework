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
                    if "DELTA WARNING" in message:
                        return f"{self.COLORS['BOLD']}{self.COLORS['WARNING']}{message}{self.COLORS['ENDC']}"
                    else:
                        return f"{self.COLORS['WARNING']}{message}{self.COLORS['ENDC']}"
                elif "[INIT]" in message:
                    return f"{self.COLORS['BOLD']}{self.COLORS['INFO']}{message}{self.COLORS['ENDC']}"
                elif message.startswith("Progress:"):
                    # Clean up and highlight progress messages for better visibility
                    return f"{self.COLORS['BOLD']}{self.COLORS['INFO']}{message}{self.COLORS['ENDC']}"
                
                return message
        
        # Use colored formatter for console
        console_formatter = ColoredFormatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        if not verbose_console:
            # Filter out debug messages and some info messages from console
            console_handler.addFilter(self.StatusOnlyFilter())
            
        logger.addHandler(console_handler)
        
        # Add informative startup messages
        logger.info(f"[INIT] Trading Engine Initialized")
        logger.info(f"[INIT] Strategy: {config_dict.get('strategy', {}).get('name', 'Unknown')} with delta target {config_dict.get('strategy', {}).get('delta_target', 'N/A')}")
        logger.info(f"[INIT] Log file: {log_filename}")
        
        return logger

    class StatusOnlyFilter(logging.Filter):
        """Filter that allows only essential status update logs to console."""

        def filter(self, record):
            # Always filter out DEBUG level messages from console
            if record.levelno == logging.DEBUG:
                return False

            # For INFO level - only show essential status messages
            if record.levelno == logging.INFO:
                # Show processing date messages
                if record.msg.startswith("Processing ") or "[INIT]" in record.msg:
                    return True
                # Filter out most INFO messages
                if not "ERROR" in record.msg and not "CRITICAL" in record.msg:
                    return False

            # For WARNING level - only show critical warnings
            if record.levelno == logging.WARNING:
                # Only show severe warnings
                if "CRITICAL" in record.msg:
                    return True
                # Filter out common warnings
                if "DELTA WARNING" in record.msg:
                    return False

            # Show all ERROR and CRITICAL messages
            if record.levelno >= logging.ERROR:
                return True

            # Default - filter out
            return False
    
    def build_log_filename(self, config_dict: Dict[str, Any]) -> str:
        """
        Build a log filename based on configuration.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            str: Log filename
        """
        # Get strategy name from config
        strategy_name = config_dict.get('strategy', {}).get('name', 'Strategy')
        
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create log filename
        return f"{strategy_name}_log_{timestamp}.log"
        

class Strategy:
    """
    Base strategy class that defines the interface for trading strategies.
    
    All custom strategies should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, config_dict: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the strategy with configuration.
        
        Args:
            config_dict: Strategy configuration dictionary
            logger: Logger instance
        """
        # Ensure config is a dictionary to prevent attribute errors
        self.config = {} if config_dict is None else config_dict
        self.logger = logger or logging.getLogger('trading_engine')
        
        # Get name safely
        if isinstance(self.config, dict) and 'name' in self.config:
            self.name = self.config['name']
        else:
            self.name = 'BaseStrategy'
    
    def generate_signals(self, current_date: datetime, daily_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate trading signals for the current date.
        
        Args:
            current_date: Current date
            daily_data: Data for current date
            
        Returns:
            list: List of signal dictionaries
        """
        # Base implementation - no signals
        return []
    
    def check_exit_conditions(self, position: Position, market_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if exit conditions are met for a position.
        
        Args:
            position: Position to check
            market_data: Market data for position
            
        Returns:
            tuple: (exit_flag, reason) - Whether to exit and why
        """
        # Base implementation - no exits
        return False, None


class TradingEngine:
    """
    Trading engine for executing strategies against historical or live data.
    
    This class coordinates all aspects of the trading process,
    including data management, portfolio management, executing strategy
    signals, and performance reporting.
    """

    def __init__(self, config: Dict[str, Any], strategy: Strategy,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the trading engine.

        Args:
            config: Configuration dictionary
            strategy: Strategy instance
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger('trading_engine')
        self.strategy = strategy
        
        # Set default values for start_date and end_date
        self.start_date = None
        self.end_date = None
        
        # Load YAML configuration from config folder
        yaml_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.yaml')
        
        # Setup system components
        self._initialize_components()
        
        # Set up debug file if enabled
        if config.get('debug', {}).get('write_debug_file', False):
            debug_dir = config.get('debug', {}).get('debug_dir', 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            debug_file_path = os.path.join(debug_dir, f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            self.debug_file = open(debug_file_path, 'w')
            self.logger.info(f"Debug output will be written to {debug_file_path}")
        else:
            self.debug_file = None
    
    def _initialize_components(self):
        """
        Initialize all trading engine components.
        """
        # Initialize portfolio
        initial_capital = self.config.get('portfolio', {}).get('initial_capital', 100000.0)
        max_position_size_pct = self.config.get('portfolio', {}).get('max_position_size_pct', 0.25)
        max_portfolio_delta = self.config.get('portfolio', {}).get('max_portfolio_delta', 0.20)
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            max_position_size_pct=max_position_size_pct,
            max_portfolio_delta=max_portfolio_delta,
            logger=self.logger
        )
        
        # Initialize data manager
        data_config = self.config.get('data', {})
        self.data_manager = DataManager(data_config, logger=self.logger)
        
        # Initialize margin calculator
        margin_type = self.config.get('margin', {}).get('type', 'option')
        if margin_type.lower() == 'span':
            max_leverage = self.config.get('margin', {}).get('max_leverage', 1.0)
            self.margin_calculator = SPANMarginCalculator(max_leverage=max_leverage, logger=self.logger)
        elif margin_type.lower() == 'option':
            max_leverage = self.config.get('margin', {}).get('max_leverage', 1.0)
            otm_margin_multiplier = self.config.get('margin', {}).get('otm_margin_multiplier', 0.8)
            self.margin_calculator = OptionMarginCalculator(
                max_leverage=max_leverage,
                otm_margin_multiplier=otm_margin_multiplier,
                logger=self.logger
            )
        else:
            # Default to basic margin calculator
            max_leverage = self.config.get('margin', {}).get('max_leverage', 1.0)
            self.margin_calculator = MarginCalculator(max_leverage=max_leverage, logger=self.logger)
            
        # Initialize hedging manager if enabled
        hedging_config = self.config.get('hedging', {})
        if hedging_config.get('enabled', False):
            self.hedging_manager = HedgingManager(hedging_config, self.portfolio, logger=self.logger)
        else:
            self.hedging_manager = None
            
        # Initialize reporting system
        reporting_config = self.config.get('reporting', {})
        if reporting_config.get('enabled', True):
            self.reporting_system = ReportingSystem(
                reporting_config, 
                self.portfolio,
                logger=self.logger
            )
        else:
            self.reporting_system = None
    
    def write_debug_output(self, text: str) -> None:
        """
        Write debug information to the debug file.
        
        Args:
            text: Text to write to the debug file
        """
        # Skip if debug file is not set up
        if self.debug_file is None:
            return
        
        # Make sure logger is initialized
        logger = getattr(self, 'logger', None)
            
        try:
            self.debug_file.write(text)
            self.debug_file.flush()
        except Exception as e:
            if logger:
                logger.error(f"Failed to write to debug file: {e}")
            else:
                print(f"Failed to write to debug file: {e}")
    
    def calculate_rolling_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics based on portfolio return history.
        
        Uses expanding window initially (minimum 5 data points), then switches
        to rolling windows once enough data is available.
        
        Returns:
            dict: Dictionary of metrics including Sharpe ratio and volatility
        """
        # Minimum data points needed for initial calculations
        MIN_DATA_POINTS = 5
        
        # Check if we have enough daily returns for initial calculation
        if not hasattr(self.portfolio, 'daily_returns') or len(self.portfolio.daily_returns) < MIN_DATA_POINTS:
            # Not enough data points yet
            self.logger.info(f"No metrics available yet - need at least {MIN_DATA_POINTS} trading days")
            return {
                'sharpe': 0,
                'volatility': 0,
                'mean_return': 0,
                'drawdown': 0,
                'window_type': 'none',
                'data_points': len(self.portfolio.daily_returns) if hasattr(self.portfolio, 'daily_returns') else 0
            }
            
        # Extract return series
        returns = [entry.get('return', 0) for entry in self.portfolio.daily_returns]
        returns_series = pd.Series(returns)
        data_points = len(returns)
        
        # Get window sizes from config if available
        risk_config = self.config.get('risk', {})
        short_window = risk_config.get('short_window', 21)
        medium_window = risk_config.get('medium_window', 63)
        long_window = risk_config.get('long_window', 95)
        
        # Calculate expanding and rolling statistics
        metrics = {}
        
        # Expanding window (all data)
        mean_return = returns_series.mean() * 252  # Annualize
        volatility = returns_series.std() * np.sqrt(252)  # Annualize
        sharpe = mean_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
        
        # Store main metrics
        metrics['sharpe'] = sharpe
        metrics['volatility'] = volatility
        metrics['mean_return'] = mean_return
        metrics['drawdown'] = max_drawdown
        metrics['data_points'] = data_points
        
        # Store whether we're using expanding or rolling windows
        metrics['window_type'] = 'expanding' if data_points < short_window else 'rolling'
        
        # Calculate metrics for different windows
        windows = {}
        
        # SHORT WINDOW METRICS
        # If we have at least MIN_DATA_POINTS but less than short_window, use expanding window
        if MIN_DATA_POINTS <= data_points < short_window:
            # Use expanding window (all available data)
            windows['short'] = {
                'window': data_points,  # Expanding window size
                'window_type': 'expanding',
                'sharpe': sharpe,       # Use metrics from all available data
                'volatility': volatility,
                'mean_return': mean_return
            }
        # If we have enough data for a proper short window, use it
        elif data_points >= short_window:
            # Use actual short window
            short_data = returns_series.iloc[-short_window:]
            short_mean = short_data.mean() * 252
            short_vol = short_data.std() * np.sqrt(252)
            short_sharpe = short_mean / short_vol if short_vol > 0 else 0
            
            windows['short'] = {
                'window': short_window,
                'window_type': 'rolling',
                'sharpe': short_sharpe,
                'volatility': short_vol,
                'mean_return': short_mean
            }
        
        # MEDIUM WINDOW METRICS
        # If we have more than MIN_DATA_POINTS but less than medium_window, use expanding window
        if MIN_DATA_POINTS <= data_points < medium_window:
            # Use expanding window, same as overall metrics
            windows['medium'] = {
                'window': data_points,
                'window_type': 'expanding',
                'sharpe': sharpe,
                'volatility': volatility,
                'mean_return': mean_return
            }
        # If we have enough data for medium window, use it
        elif data_points >= medium_window:
            # Use actual medium window
            medium_data = returns_series.iloc[-medium_window:]
            medium_mean = medium_data.mean() * 252
            medium_vol = medium_data.std() * np.sqrt(252)
            medium_sharpe = medium_mean / medium_vol if medium_vol > 0 else 0
            
            windows['medium'] = {
                'window': medium_window,
                'window_type': 'rolling',
                'sharpe': medium_sharpe,
                'volatility': medium_vol,
                'mean_return': medium_mean
            }
        
        # LONG WINDOW METRICS
        # If we have more than MIN_DATA_POINTS but less than long_window, use expanding window
        if MIN_DATA_POINTS <= data_points < long_window:
            # Use expanding window, same as overall metrics
            windows['long'] = {
                'window': data_points,
                'window_type': 'expanding',
                'sharpe': sharpe,
                'volatility': volatility,
                'mean_return': mean_return
            }
        # If we have enough data for long window, use it
        elif data_points >= long_window:
            # Use actual long window
            long_data = returns_series.iloc[-long_window:]
            long_mean = long_data.mean() * 252
            long_vol = long_data.std() * np.sqrt(252)
            long_sharpe = long_mean / long_vol if long_vol > 0 else 0
            
            windows['long'] = {
                'window': long_window,
                'window_type': 'rolling',
                'sharpe': long_sharpe,
                'volatility': long_vol,
                'mean_return': long_mean
            }
        
        # Add windows to metrics
        metrics['windows'] = windows
        
        # Calculate historical mean and std of the Sharpe ratio
        # Minimum history required for z-score calculations
        MIN_HIST_POINTS = 10
        
        # Calculate historical stats based on available data
        if len(returns) >= MIN_HIST_POINTS:
            # Target history window size - use expanding window at first
            hist_window = max(MIN_HIST_POINTS, min(len(returns) // 2, 30))  # Cap at 30 days
            
            # Check if we have enough data for rolling window or need expanding window
            if len(returns) < hist_window + MIN_HIST_POINTS:
                # Not enough data for proper rolling window - use expanding window
                # Calculate simple historical stats using all data
                hist_mean = sharpe  # Use overall Sharpe as the historical mean
                hist_std = volatility / 4  # Estimate std dev based on volatility
                metrics['hist_window_type'] = 'expanding'
                metrics['hist_window'] = len(returns)
            else:
                # Enough data for rolling window
                # Calculate rolling Sharpe ratio series
                rolling_mean = returns_series.rolling(hist_window).mean() * 252
                rolling_std = returns_series.rolling(hist_window).std() * np.sqrt(252)
                rolling_sharpe = rolling_mean / rolling_std
                rolling_sharpe = rolling_sharpe.dropna()
                
                # Calculate historical stats
                hist_mean = rolling_sharpe.mean()
                hist_std = rolling_sharpe.std()
                metrics['hist_window_type'] = 'rolling'
                metrics['hist_window'] = hist_window
            
            # Calculate z-score for current Sharpe
            if hist_std > 0 and not pd.isna(hist_mean) and not pd.isna(sharpe):
                z_score = (sharpe - hist_mean) / hist_std
            else:
                z_score = 0
                
            # Add to metrics
            metrics['hist_mean'] = hist_mean
            metrics['hist_std'] = hist_std
            metrics['z_score'] = z_score
        
        return metrics
        
    def load_data(self) -> bool:
        """
        Load all required data for the backtesting process.
        
        This method loads option data, market data, and any other data needed for the backtest.
        
        Returns:
            bool: True if data loading was successful, False otherwise
        """
        self.logger.info("Loading data for backtesting...")
        
        # Get the input file path from the configuration
        paths_config = self.config.get('paths', {})
        input_file = paths_config.get('input_file')
        
        if not input_file:
            self.logger.error("No input file specified in configuration")
            return False
            
        # Handle date range from config
        dates_config = self.config.get('dates', {})
        start_date_str = dates_config.get('start_date')
        end_date_str = dates_config.get('end_date')
        
        # Parse dates if provided
        self.start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None
        self.end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None
        
        # Log date range and input file
        self.logger.info(f"Input file: {input_file}")
        if self.start_date and self.end_date:
            self.logger.info(f"Date range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        
        # Load option data
        try:
            # Load data using the data manager
            self.logger.info(f"Loading market data...")
            self.data = self.data_manager.load_option_data(input_file, self.start_date, self.end_date)
            
            if self.data is None or len(self.data) == 0:
                self.logger.error(f"Failed to load data from {input_file}")
                return False
                
            # Parse trading dates
            self.trading_dates = sorted(self.data['DataDate'].unique())
            
            if not self.trading_dates:
                self.logger.error("No trading dates found in data")
                return False
                
            # Update date range if not specified
            if not self.start_date:
                self.start_date = self.trading_dates[0]
            if not self.end_date:
                self.end_date = self.trading_dates[-1]
                
            self.logger.info(f"Successfully loaded data: {len(self.data)} rows, {len(self.trading_dates)} trading days")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.logger.error(traceback.format_exc())
            return False
            
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run a backtest using the loaded data and configured strategy.
        
        This method iterates through trading dates, generates signals,
        and executes trades according to the strategy.
        
        Returns:
            dict: Dictionary of backtest results and performance metrics
        """
        if not hasattr(self, 'data') or self.data is None or len(self.data) == 0:
            self.logger.error("No data loaded for backtest")
            return {"error": "No data loaded"}
            
        if not hasattr(self, 'trading_dates') or not self.trading_dates:
            self.logger.error("No trading dates available")
            return {"error": "No trading dates available"}
            
        self.logger.info(f"Running backtest: {self.start_date} to {self.end_date}")
        self.logger.info(f"Strategy: {self.strategy.name}")
        self.logger.info(f"Initial capital: ${self.portfolio.initial_capital:,.2f}")
        
        # Tracking variables
        processed_days = 0
        total_days = len(self.trading_dates)
        
        try:
            # Process each trading day
            for current_date in self.trading_dates:
                # Skip dates outside our range
                if (self.start_date and current_date < self.start_date) or \
                   (self.end_date and current_date > self.end_date):
                    continue
                    
                # Log progress
                processed_days += 1
                if processed_days % 5 == 0 or processed_days == total_days:
                    progress_pct = processed_days / total_days * 100
                    self.logger.info(f"Progress: {processed_days}/{total_days} days ({progress_pct:.1f}%)")
                    
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
            
    def _process_trading_day(self, current_date: datetime) -> None:
        """
        Process a single trading day in the backtest.
        
        Args:
            current_date: The current trading date
        """
        self.logger.debug(f"Processing {current_date.strftime('%Y-%m-%d')}")
        
        # Get data for the current day
        daily_data = self.data[self.data['DataDate'] == current_date]
        
        if len(daily_data) == 0:
            self.logger.warning(f"No data available for {current_date.strftime('%Y-%m-%d')}")
            return
            
        # Update existing positions with current market data
        self._update_positions(current_date, daily_data)
        
        # Manage existing positions (check exit conditions)
        self._manage_positions(current_date, daily_data)
        
        # Generate trading signals
        signals = self.strategy.generate_signals(current_date, daily_data)
        
        # Execute trading signals
        if signals:
            self._execute_signals(signals, daily_data, current_date)
            
        # Calculate and store rolling performance metrics
        self.calculate_rolling_metrics()
            
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
                
        # Update portfolio positions
        self.portfolio.update_market_data(market_data_by_symbol, current_date)
            
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
                    
                # Add execution data
                execution_data = {'date': current_date}
                
                # Close position
                self.portfolio.remove_position(symbol, price=price, execution_data=execution_data, reason=reason)
                
    def _execute_signals(self, signals: List[Dict[str, Any]], daily_data: pd.DataFrame, current_date: datetime) -> None:
        """
        Execute trading signals by adding or removing positions.
        
        Args:
            signals: List of signal dictionaries
            daily_data: Data for the current trading day
            current_date: Current trading date
        """
        if not signals:
            return
            
        self.logger.debug(f"Executing {len(signals)} trading signals")
        
        # Create a lookup for option data
        option_data_by_symbol = {}
        if 'OptionSymbol' in daily_data.columns:
            for _, row in daily_data.iterrows():
                symbol = row['OptionSymbol']
                option_data_by_symbol[symbol] = row
                
        # Process each signal
        for signal in signals:
            action = signal.get('action', '').upper()
            symbol = signal.get('symbol')
            quantity = signal.get('quantity', 1)
            
            # Skip invalid signals
            if not symbol or not action:
                self.logger.warning(f"Invalid signal: {signal}")
                continue
                
            # Skip if we don't have data for this symbol
            if symbol not in option_data_by_symbol:
                self.logger.warning(f"No data for signal symbol: {symbol}")
                continue
                
            # Get option data
            option_data = option_data_by_symbol[symbol]
            
            # Get price from signal or use mid price
            price = signal.get('price')
            if price is None and 'MidPrice' in option_data:
                price = option_data['MidPrice']
                
            # Skip if no price available
            if price is None or price <= 0:
                self.logger.warning(f"No valid price for {symbol}")
                continue
                
            # Execute the signal
            execution_data = {'date': current_date}
            
            if action in ('BUY', 'SELL'):
                # Determine if this is a short position
                is_short = action == 'SELL'
                
                # Add a new position
                self.portfolio.add_position(
                    symbol=symbol,
                    instrument_data=option_data,
                    quantity=quantity,
                    price=price,
                    position_type='option',
                    is_short=is_short,
                    execution_data=execution_data
                )
                
            elif action == 'CLOSE':
                # Close an existing position
                if symbol in self.portfolio.positions:
                    self.portfolio.remove_position(
                        symbol=symbol,
                        quantity=quantity,
                        price=price,
                        execution_data=execution_data,
                        reason=signal.get('reason', 'Signal')
                    )
                else:
                    self.logger.warning(f"Cannot close position {symbol} - not found in portfolio")
                    
            else:
                self.logger.warning(f"Unknown action in signal: {action}")
                
        # Calculate portfolio metrics after executing signals
        portfolio_metrics = self.portfolio.get_portfolio_metrics()
        self.logger.debug(f"Portfolio after signals: ${portfolio_metrics['portfolio_value']:,.2f}, {len(self.portfolio.positions)} positions")
        self.logger.debug(f"  Cash balance: ${portfolio_metrics['cash_balance']:,.2f}")
        self.logger.debug(f"  Delta: {portfolio_metrics['delta']:.2f} (${portfolio_metrics['dollar_delta']:,.2f})")