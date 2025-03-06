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
        
        # Print debugging info about the path
        if hasattr(self, 'logger'):
            self.logger.info(f"Looking for config.yaml at: {yaml_config_path}")
            self.logger.info(f"File exists: {os.path.exists(yaml_config_path)}")
        else:
            print(f"Looking for config.yaml at: {yaml_config_path}")
            print(f"File exists: {os.path.exists(yaml_config_path)}")
        
        if os.path.exists(yaml_config_path):
            try:
                import yaml
                with open(yaml_config_path, 'r') as yaml_file:
                    yaml_config = yaml.safe_load(yaml_file)
                    if yaml_config:
                        # Merge YAML config with existing config
                        self._merge_configs(yaml_config)
                        if hasattr(self, 'logger'):
                            self.logger.info(f"Loaded configuration from {yaml_config_path}")
                        else:
                            print(f"Loaded configuration from {yaml_config_path}")
                    else:
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"Config file was empty: {yaml_config_path}")
                        else:
                            print(f"Config file was empty: {yaml_config_path}")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Failed to load YAML config: {str(e)}")
                    self.logger.debug(f"Traceback: {tb}")
                else:
                    print(f"Failed to load YAML config: {str(e)}")
                    print(f"Traceback: {tb}")
                # Continue without failing
                
    def _merge_configs(self, yaml_config):
        """
        Merge YAML configuration with the existing config.
        
        Args:
            yaml_config: Configuration dictionary from YAML file
        """
        for key, value in yaml_config.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    # Recursively merge nested dictionaries
                    for nested_key, nested_value in value.items():
                        self.config[key][nested_key] = nested_value
                else:
                    # Override with YAML value
                    self.config[key] = value
            else:
                # Add new key from YAML
                self.config[key] = value
                
        # Set data_file attribute from config
        self.data_file = self.config.get('paths', {}).get('input_file')
        
        # Initialize start and end dates for the backtest from config
        dates_config = self.config.get('dates', {})
        if dates_config:
            self.start_date = dates_config.get('start_date')
            self.end_date = dates_config.get('end_date')
            
            # Parse dates if they're strings
            if isinstance(self.start_date, str):
                self.start_date = pd.to_datetime(self.start_date)
            if isinstance(self.end_date, str):
                self.end_date = pd.to_datetime(self.end_date)
            
            # Set default dates if not provided
            if self.start_date is None:
                self.start_date = pd.to_datetime('2023-01-01')
                self.logger.warning(f"No start date specified, using default: {self.start_date}")
            if self.end_date is None:
                self.end_date = pd.to_datetime('2025-12-31')
                self.logger.warning(f"No end date specified, using default: {self.end_date}")
        
        # Setup debug output file
        self.debug_file = None
        self.setup_debug_file()

        # Create component instances
        self.data_manager = DataManager(logger=self.logger)

        # Create portfolio
        portfolio_config = self.config.get('portfolio', {})
        initial_capital = portfolio_config.get('initial_capital', 100000)
        max_position_size = portfolio_config.get('max_position_size_pct', 0.25)
        max_portfolio_delta = portfolio_config.get('max_portfolio_delta', 0.20)

        # Prepare risk parameters
        risk_config = self.config.get('risk', {})
        max_leverage = risk_config.get('max_leverage', 6.0)

        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            max_position_size_pct=max_position_size,
            max_portfolio_delta=max_portfolio_delta,
            logger=self.logger
        )

        # Choose margin calculator based on config or default to SPAN for options
        margin_type = self.config.get('risk', {}).get('margin_type', 'SPAN').upper()
        
        if self.config.get('strategy', {}).get('instrument_type', '').lower() == 'option':
            # Create a simple correlation matrix for common ETFs
            correlation_matrix = pd.DataFrame(
                [[1.0, 0.7, 0.3, 0.5], 
                 [0.7, 1.0, 0.2, 0.4], 
                 [0.3, 0.2, 1.0, 0.1],
                 [0.5, 0.4, 0.1, 1.0]],
                index=['SPY', 'QQQ', 'GLD', 'TLT'],
                columns=['SPY', 'QQQ', 'GLD', 'TLT']
            )
            
            # Always log which margin calculator we're using
            if margin_type == 'OPTION':
                self.logger.info("[STATUS] Using Option Margin Calculator")
                self.margin_calculator = OptionMarginCalculator(
                    max_leverage=max_leverage,
                    otm_margin_multiplier=0.8,
                    logger=self.logger
                )
            else:  # Default to SPAN
                self.logger.info("[STATUS] Using SPAN Margin Calculator with delta hedging benefits")
                self.margin_calculator = SPANMarginCalculator(
                    max_leverage=max_leverage,
                    volatility_multiplier=1.2,  # Slightly higher for stress testing
                    correlation_matrix=correlation_matrix,
                    initial_margin_percentage=0.1,  # 10% initial margin
                    maintenance_margin_percentage=0.07,  # 7% maintenance
                    hedge_credit_rate=0.8,  # 80% credit for hedged positions
                    logger=self.logger
                )
        else:
            self.logger.info("[STATUS] Using Standard Margin Calculator")
            self.margin_calculator = MarginCalculator(
                max_leverage=max_leverage,
                logger=self.logger
            )

        # Initialize delta hedging manager
        strategy_config = self.config.get('strategy', {})
        self.hedging_manager = HedgingManager(
            config=strategy_config,
            portfolio=self.portfolio,
            logger=self.logger
        )

        # Create reporting system
        output_dir = self.config.get('paths', {}).get('output_dir', 'reports')
        self.reporting_system = ReportingSystem(  # Changed from self.reporting to self.reporting_system
            output_dir=output_dir,
            logger=self.logger
        )

        # Data file for the backtest is already set in _merge_configs
    
    def get_date_range(self) -> List[datetime]:
        """
        Get the date range for the backtest.
        
        Returns:
            list: List of dates in the range
        """
        # If dates are already in the data, use those
        if hasattr(self.data_manager, 'data') and self.data_manager.data is not None:
            if hasattr(self.data_manager.data, 'DataDate'):
                # Get unique dates from the data
                unique_dates = pd.Series(self.data_manager.data['DataDate'].unique())
                unique_dates = pd.to_datetime(unique_dates)
                unique_dates = unique_dates.sort_values()
                
                # Filter by start and end dates if provided
                if self.start_date:
                    unique_dates = unique_dates[unique_dates >= self.start_date]
                if self.end_date:
                    unique_dates = unique_dates[unique_dates <= self.end_date]
                    
                return unique_dates.tolist()
        
        # Otherwise, generate date range from start to end date
        if self.start_date and self.end_date:
            # Generate a range of trading days
            all_days = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
            return all_days.tolist()
            
        # Default - return empty list
        return []
    
    def data_for_date(self, date: datetime) -> pd.DataFrame:
        """
        Get data for a specific date.
        
        Args:
            date: Date to get data for
            
        Returns:
            DataFrame: Data for the date
        """
        if hasattr(self.data_manager, 'data') and self.data_manager.data is not None:
            # Filter data for the specific date
            if hasattr(self.data_manager.data, 'DataDate'):
                daily_data = self.data_manager.data[self.data_manager.data['DataDate'] == date]
                return daily_data
        
        # No data found
        return pd.DataFrame()
    
    def load_data(self) -> bool:
        """
        Load data for the backtest.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.data_file:
                self.logger.error("No data file specified")
                return False
                
            self.logger.info(f"[STATUS] Loading data from {self.data_file}...")
            loaded = self.data_manager.load_from_file(self.data_file)
            
            if not loaded:
                self.logger.error("Failed to load data")
                return False
                
            self.logger.info(f"[STATUS] Loaded {len(self.data_manager.data)} records")
            
            # Initialize strategy with data if it has an initialize method
            if hasattr(self.strategy, 'initialize'):
                self.strategy.initialize(self.data_manager)

            return True

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def run_backtest(self):
        """
        Run a backtest over the specified date range using the provided strategy.

        Returns:
            dict: Results of the backtest
        """
        # Clear, concise startup message that will be displayed to the user
        self.logger.info(
            f"[INIT] Starting backtest: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")

        # Initialize results tracking
        results = {
            'initial_capital': self.portfolio.initial_capital,
            'dates': [],
            'portfolio_values': [],
            'cash_values': [],
            'position_values': [],
            'trades': [],
            'positions': []
        }

        # Get the date range for the backtest
        date_range = self.get_date_range()
        total_dates = len(date_range)

        # Run the simulation for each date
        for i, current_date in enumerate(date_range):
            date_str = current_date.strftime('%Y-%m-%d')

            # Simple, clean progress indicator that shows date and percentage
            percent_complete = (i + 1) / total_dates * 100
            self.logger.info(f"Processing {date_str} [{percent_complete:.1f}%]")
            
            # Rearranged the flow to fix PnL calculation and pre-trade summary
            # Get data for the current date (moved this section up from below)
            daily_data = self.data_for_date(current_date)

            # Skip if no data
            if daily_data is None or daily_data.empty:
                self.logger.warning(f"[Engine] No data available for {date_str}, skipping")
                continue

            # Prepare option data with mid prices and days to expiry
            max_spread = self.config.get('data', {}).get('max_spread', 0.30)
            daily_data = self.data_manager.prepare_option_data(daily_data, current_date, max_spread)
            
            # Update positions with current market data BEFORE PRE-TRADE summary
            # This ensures the PRE-TRADE summary shows accurate current prices and PnL
            self.portfolio.update_positions(current_date, daily_data)
            
            # Write detailed PreTrade section
            portfolio_value = self.portfolio.get_portfolio_value()
            cash_balance = self.portfolio.cash_balance
            position_count = len(self.portfolio.positions)
            position_value = sum([pos.current_price * pos.contracts * 100 for pos in self.portfolio.positions.values()])
            
            # Format for table borders and separators
            separator = "=" * 50
            sub_separator = "-" * 50
            
            # Display a clean progress indicator in the log
            # This will be shown to the user through the console filter
            percent_complete = len(results['dates'])/len(date_range)*100
            self.logger.info(f"Progress: Processing: {date_str} [{percent_complete:.1f}%]")
            
            self.write_debug_output(f"\n{separator}\n")
            self.write_debug_output(f"PRE-TRADE Summary [{date_str}]:\n")
            
            # Calculate daily P&L for pre-trade summary
            daily_pnl = 0
            hedge_pnl = 0
            option_pnl = 0
            
            # Separate option and hedge PnL
            for symbol, pos in self.portfolio.positions.items():
                # Skip positions without previous price data
                if not hasattr(pos, 'prev_price') or pos.prev_price is None:
                    continue
                    
                # Calculate P&L for this position based on price movement
                prev_price = pos.prev_price
                curr_price = pos.current_price
                
                if pos.is_short:
                    pos_pnl = (prev_price - curr_price) * pos.contracts * 100
                else:
                    pos_pnl = (curr_price - prev_price) * pos.contracts * 100
                    
                # Add to appropriate category
                if isinstance(pos, OptionPosition):
                    option_pnl += pos_pnl
                else:
                    hedge_pnl += pos_pnl
                    
                daily_pnl += pos_pnl
            
            # Calculate percentage PnL if we have a valid portfolio value
            pnl_pct = 0
            if portfolio_value > 0:
                pnl_pct = daily_pnl / portfolio_value * 100
                
            self.write_debug_output(f"Daily P&L: ${daily_pnl:.2f} ({pnl_pct:.2f}%)\n")
            self.write_debug_output(f"  Option PnL: ${option_pnl:.2f}\n")
            self.write_debug_output(f"  Hedge PnL: ${hedge_pnl:.2f}\n")
            
            # Portfolio status 
            self.write_debug_output(f"Open Trades: {position_count}\n")
            if portfolio_value > 0:
                exposure_pct = position_value / portfolio_value * 100
                self.write_debug_output(f"Total Position Exposure: {exposure_pct:.1f}% of NLV\n")
            else:
                self.write_debug_output("Total Position Exposure: 0.0% of NLV\n")
                
            # Net liquidation value breakdown
            self.write_debug_output(f"Net Liq: ${portfolio_value:,.0f}\n")
            self.write_debug_output(f"  Cash Balance: ${cash_balance:,.0f}\n")
            
            # Current liability (unrealized P&L for short positions)
            liability = sum([pos.unrealized_pnl for pos in self.portfolio.positions.values() if pos.is_short])
            self.write_debug_output(f"  Total Liability: ${liability:,.0f}\n")
            self.write_debug_output(f"  Self Hedge (Hedge PnL): $0\n")
            
            # Calculate margin and leverage - with safe error handling
            try:
                if hasattr(self, 'margin_calculator'):
                    # Use detailed margin calculation with hedging benefits when possible
                    margin_method = "Unknown"
                    
                    if isinstance(self.margin_calculator, SPANMarginCalculator):
                        margin_method = "SPAN"
                        # Get the full margin breakdown
                        margin_details = self.margin_calculator.calculate_portfolio_margin(self.portfolio.positions)
                        margin_calc = margin_details['total_margin']
                        
                        # Display the hedging benefits in debug output
                        hedging_benefit = margin_details.get('hedging_benefits', 0)
                        inter_product_benefit = margin_details.get('inter_product_benefits', 0)
                        standalone_margin = sum(margin_details.get('margin_by_position', {}).values())
                        
                        self.write_debug_output(f"[{margin_method}] Margin Calculation:\n")
                        self.write_debug_output(f"  Standalone Margin: ${standalone_margin:,.0f}\n")
                        
                        if hedging_benefit > 0:
                            self.write_debug_output(f"  Delta Hedging Benefit: -${hedging_benefit:,.0f}\n")
                        
                        if inter_product_benefit > 0:
                            self.write_debug_output(f"  Inter-Product Correlation Benefit: -${inter_product_benefit:,.0f}\n")
                        
                        self.write_debug_output(f"  Total Margin Requirement: ${margin_calc:,.0f}\n")
                    elif isinstance(self.margin_calculator, OptionMarginCalculator):
                        margin_method = "Option"
                        # Option margin calculation
                        margin_calc = self.margin_calculator.calculate_total_margin(self.portfolio.positions)
                        self.write_debug_output(f"[{margin_method}] Margin Calculation:\n")
                        self.write_debug_output(f"  Total Margin Requirement: ${margin_calc:,.0f}\n")
                    else:
                        margin_method = "Standard"
                        # Simple margin calculation
                        margin_calc = self.margin_calculator.calculate_total_margin(self.portfolio.positions)
                        self.write_debug_output(f"[{margin_method}] Margin Calculation:\n")
                        self.write_debug_output(f"  Total Margin Requirement: ${margin_calc:,.0f}\n")
                else:
                    # Fallback basic margin calculation if margin calculator isn't available
                    margin_calc = sum([pos.avg_entry_price * pos.contracts * 100 * 0.5 for pos in self.portfolio.positions.values()])
                    self.write_debug_output(f"Total Margin Requirement: ${margin_calc:,.0f}\n")
                
                available_margin = cash_balance - margin_calc
                self.write_debug_output(f"Available Margin: ${available_margin:,.0f}\n")
            except Exception as e:
                self.write_debug_output(f"Error calculating margin: {e}\n")
                # Print full traceback for better debugging
                import traceback
                self.write_debug_output(f"Traceback: {traceback.format_exc()}\n")
                margin_calc = 0
                available_margin = cash_balance
                self.write_debug_output(f"Using fallback: Margin: $0, Available: ${cash_balance:,.0f}\n")
            
            leverage = 0
            if portfolio_value > 0:
                leverage = margin_calc / portfolio_value
                
            self.write_debug_output(f"Margin-Based Leverage: {leverage:.2f} \n\n")
            
            # Portfolio risk metrics
            portfolio_metrics = self.portfolio.get_portfolio_metrics()
            
            self.write_debug_output("Portfolio Greek Risk:\n")
            
            # Option delta
            delta = portfolio_metrics.get('delta', 0)
            dollar_delta = portfolio_metrics.get('dollar_delta', 0)
            self.write_debug_output(f"  Option Delta: {delta:.3f} (${dollar_delta:.2f})\n")
            
            # We don't have hedge delta yet, so show 0
            self.write_debug_output(f"  Hedge Delta: 0.000 ($0.00)\n")
            self.write_debug_output(f"  Total Delta: {delta:.3f} (${dollar_delta:.2f})\n")
            
            # Other Greeks
            gamma = portfolio_metrics.get('gamma', 0)
            dollar_gamma = gamma * 100  # Approximate dollar gamma per 1% move
            theta = portfolio_metrics.get('theta', 0)
            vega = portfolio_metrics.get('vega', 0)
            
            self.write_debug_output(f"  Gamma: {gamma:.6f} (${dollar_gamma:.2f} per 1% move)\n")
            self.write_debug_output(f"  Theta: ${theta:.2f} per day\n")
            self.write_debug_output(f"  Vega: ${vega:.2f} per 1% IV\n")
            
            # Add a section for current positions if they exist
            if position_count > 0:
                self.write_debug_output(f"{sub_separator}\n\n")
                self.write_debug_output("Current Positions:\n")
                
                # Column headers for positions table
                position_header = f"{'Symbol':<16} {'Contracts':>9} {'Entry':>7} {'Current':>8} {'Value':>10} "
                position_header += f"{'NLV%':>5} {'Delta':>8} {'DTE':>5}\n"
                position_divider = "-" * 76 + "\n"
                
                self.write_debug_output(position_divider)
                self.write_debug_output(position_header)
                self.write_debug_output(position_divider)
                
                # Output each position
                for symbol, pos in self.portfolio.positions.items():
                    nlv_pct = 0
                    if portfolio_value > 0:
                        nlv_pct = (pos.current_price * pos.contracts * 100) / portfolio_value * 100
                        
                    position_line = f"{symbol:<16} {pos.contracts:>9} ${pos.avg_entry_price:>5.2f} "
                    position_line += f"${pos.current_price:>6.2f} ${pos.current_price * pos.contracts * 100:>8,.0f} "
                    position_line += f"{nlv_pct:>5.1f}% {pos.current_delta:>8.3f} {pos.days_to_expiry:>5}\n"
                    
                    self.write_debug_output(position_line)
                    
                self.write_debug_output(position_divider)
            
            self.write_debug_output(f"{sub_separator}\n")
            
            # Get data for the current date
            daily_data = self.data_for_date(current_date)

            # Skip if no data
            if daily_data is None or daily_data.empty:
                self.logger.warning(f"[Engine] No data available for {date_str}, skipping")
                continue

            # Prepare option data with mid prices and days to expiry
            max_spread = self.config.get('data', {}).get('max_spread', 0.30)
            daily_data = self.data_manager.prepare_option_data(daily_data, current_date, max_spread)

            # Calculate rolling metrics for risk scaling and display
            rolling_metrics = self.calculate_rolling_metrics()
            
            # Generate signals from strategy
            signals = self.strategy.generate_signals(current_date, daily_data)
            
            # Log strategy signals to debug file
            if signals:
                self.write_debug_output("\nStrategy Trading Signals:\n")
                for signal in signals:
                    self.write_debug_output(f"  {signal['action']} {signal['quantity']} {signal['symbol']} @ ${signal['data'].get('MidPrice', 0):.2f}\n")
            
            # Generate hedging signals if enabled
            if self.hedging_manager.enable_hedging:
                self.write_debug_output("\nHedging Analysis:\n")
                
                # Create market data dictionary using the DataManager's more robust method
                market_data_by_symbol = self.data_manager.get_market_data_by_symbol(daily_data)
                
                # Generate hedging signals
                hedge_signals = self.hedging_manager.generate_hedge_signals(market_data_by_symbol, current_date)
                
                # Log hedge requirements
                requirements = self.hedging_manager.calculate_hedge_requirements(market_data_by_symbol)
                
                # Calculate acceptable delta range for display
                lower_bound = requirements['target_delta'] - requirements['tolerance']
                upper_bound = requirements['target_delta'] + requirements['tolerance']
                
                self.write_debug_output(f"  Current Delta: {requirements['current_delta']:.3f} (${requirements['delta_dollars']:.2f})\n")
                self.write_debug_output(f"  Target Delta: {requirements['target_delta']:.3f}\n")
                self.write_debug_output(f"  Acceptable Delta Range: {lower_bound:.3f} to {upper_bound:.3f}\n")
                self.write_debug_output(f"  Delta/NLV Ratio: {requirements['delta_ratio']:.2%}\n")
                self.write_debug_output(f"  Tolerance: ±{requirements['tolerance']:.3f}\n")
                
                # Add underlying price information
                underlying_price = self.hedging_manager._get_underlying_price(market_data_by_symbol)
                if underlying_price > 0:
                    self.write_debug_output(f"  Underlying Price: ${underlying_price:.2f}\n")
                else:
                    self.write_debug_output(f"  Underlying Price: Not available (hedging may not execute)\n")
                
                # Log hedging signals if any
                if hedge_signals:
                    self.write_debug_output("  Hedge Signals:\n")
                    for signal in hedge_signals:
                        self.write_debug_output(f"    {signal['action']} {signal['quantity']} {signal['symbol']} @ ${signal['price']:.2f} ({signal['reason']})\n")
                    
                    # Add hedging signals to main signals
                    signals.extend(hedge_signals)
                else:
                    self.write_debug_output("  No hedging required\n")
            
            # Process all signals (strategy + hedging)
            for signal in signals:
                self.process_signal(signal, current_date)

            # Update positions with current market data
            self.portfolio.update_positions(current_date, daily_data)
            
            # Write Risk Scaling and Hedging section
            self.write_debug_output(f"\n{separator}\n")
            # Use calculated rolling metrics for risk scaling display
            if rolling_metrics:
                # Get the window information
                window_type = rolling_metrics.get('window_type', 'expanding')
                risk_config = self.config.get('risk', {})
                risk_scaling_window = risk_config.get('risk_scaling_window', 'short')
                
                # Get the metrics
                sharpe = rolling_metrics.get('sharpe', 'N/A')
                hist_mean = rolling_metrics.get('hist_mean', 'N/A')
                hist_std = rolling_metrics.get('hist_std', 'N/A')
                z_score = rolling_metrics.get('z_score', 0)
                scaling = rolling_metrics.get('scaling', 1.0)
                hist_window_type = rolling_metrics.get('hist_window_type', 'none')
                
                # Get window data if available
                window_days = "?"
                if 'windows' in rolling_metrics and risk_scaling_window in rolling_metrics['windows']:
                    window_data = rolling_metrics['windows'][risk_scaling_window]
                    window_days = window_data.get('window', '?')
                    window_type = window_data.get('window_type', window_type)
                
                # Print risk scaling information with window type
                self.write_debug_output(f"[Risk Scaling] Using {window_type.upper()} window ({window_days}D) and scaling factor {scaling:.2f}\n")
                self.write_debug_output(f"[Risk Scaling] Current Sharpe: {sharpe}, Hist Data: {hist_window_type}, z: {z_score:.2f}\n")
                
                # Position sizing based on scaling
                if scaling < 0.5:
                    self.write_debug_output(f"  [Risk Scaling] Reduced position sizing ({scaling:.2f})\n")
                elif scaling >= 0.9:
                    self.write_debug_output(f"  [Risk Scaling] Normal position sizing ({scaling:.2f})\n")
                else:
                    self.write_debug_output(f"  [Risk Scaling] Moderate position sizing ({scaling:.2f})\n")
            else:
                # Default if no metrics available (less than 5 data points)
                self.write_debug_output("[Risk Scaling] Using default scaling factor (1.0) - insufficient data points\n")
                self.write_debug_output("[Risk Scaling] Need at least 5 data points for initial metrics calculation\n")
                self.write_debug_output("  [Risk Scaling] Normal position sizing (1.00)\n")
            
            portfolio_value = self.portfolio.get_portfolio_value()
            margin_buffer = self.config.get('risk', {}).get('margin_buffer', 0.1)
            max_margin = portfolio_value * (1 + margin_buffer)
            
            # Safe margin calculation with error handling
            try:
                if hasattr(self, 'margin_calculator'):
                    current_margin = sum([self.margin_calculator.calculate_position_margin(pos) for pos in self.portfolio.positions.values()])
                else:
                    # Fallback calculation
                    current_margin = sum([pos.avg_entry_price * pos.contracts * 100 * 0.5 for pos in self.portfolio.positions.values()])
            except Exception as e:
                self.write_debug_output(f"Error calculating position margins: {e}\n")
                current_margin = 0
                
            available_margin = max_margin - current_margin
            
            self.write_debug_output("[Portfolio Rebalancer Analysis]\n")
            self.write_debug_output(f"  Current NLV: ${portfolio_value:,.0f}\n")
            self.write_debug_output(f"  Risk Scaling Factor: 1.00\n")
            self.write_debug_output(f"  Maximum Margin Allowed: ${max_margin:,.0f} (with {margin_buffer*100}% buffer)\n")
            self.write_debug_output(f"  Current Margin: ${current_margin:,.0f}\n")
            
            if portfolio_value > 0:
                available_pct = available_margin / portfolio_value * 100
                self.write_debug_output(f"  Available Margin: ${available_margin:,.0f} ({available_pct:.2f}% of NLV)\n")
            else:
                self.write_debug_output(f"  Available Margin: ${available_margin:,.0f} (0.00% of NLV)\n")
                
            # Add enhanced margin analysis section with method information
            margin_method = "Standard"
            if isinstance(self.margin_calculator, SPANMarginCalculator):
                margin_method = "SPAN"
            elif isinstance(self.margin_calculator, OptionMarginCalculator):
                margin_method = "Option"
                
            self.write_debug_output(f"[{margin_method} Margin Analysis] Current: ${current_margin:,.2f}, Maximum: ${max_margin:,.2f}\n")
            
            if portfolio_value > 0:
                current_exposure = current_margin / portfolio_value
                max_allowed = max_margin / portfolio_value
                self.write_debug_output(f"  Current Exposure: {current_exposure:.2f}x, Maximum Allowed: {max_allowed:.2f}x\n")
            else:
                self.write_debug_output("  Current Exposure: 0.00x, Maximum Allowed: 0.00x\n")
                
            self.write_debug_output(f"  Remaining Margin Capacity: ${available_margin:,.2f}\n")
            
            # If we have hedging benefits, show them
            if isinstance(self.margin_calculator, SPANMarginCalculator) and len(self.portfolio.positions) > 1:
                try:
                    # Calculate hedging benefits
                    margin_details = self.margin_calculator.calculate_portfolio_margin(self.portfolio.positions)
                    hedging_benefit = margin_details.get('hedging_benefits', 0)
                    if hedging_benefit > 0:
                        self.write_debug_output(f"  Delta Hedging Benefit: ${hedging_benefit:,.2f}\n")
                except Exception:
                    pass
            
            # Manage existing positions
            self.manage_positions(current_date, daily_data)

            # Record daily results
            results['dates'].append(current_date)
            portfolio_value = self.portfolio.get_portfolio_value()
            results['portfolio_values'].append(portfolio_value)
            results['cash_values'].append(self.portfolio.cash_balance)
            results['position_values'].append(self.portfolio.position_value)
            
            # Write Trading Activity Section
            self.write_debug_output("\nToday's Trading Activity:\n\n")
            
            # Get the day's transactions
            todays_transactions = [t for t in self.portfolio.transactions if t.get('date', datetime.now()).date() == current_date.date()]
            
            # Group transactions by action (Buy/Sell)
            added_positions = [t for t in todays_transactions if t.get('action') == 'SELL']
            closed_positions = [t for t in todays_transactions if t.get('action') == 'BUY']
            
            # Display Added Positions
            if added_positions:
                self.write_debug_output("Added Positions:\n")
                table_header = "-" * 120 + "\n"
                header_line = f"{'Symbol':<16} {'Contracts':>9} {'Price':>9} {'Value':>10} {'Delta':>8} {'DTE':>7} {'Margin':>10}\n"
                
                self.write_debug_output(table_header)
                self.write_debug_output(header_line)
                self.write_debug_output(table_header)
                
                for tx in added_positions:
                    symbol = tx.get('symbol', '')
                    position = self.portfolio.positions.get(symbol)
                    
                    if position:
                        delta = position.current_delta
                        dte = position.days_to_expiry
                        margin = 0
                        # Safely calculate margin
                        try:
                            if hasattr(self, 'margin_calculator'):
                                margin = self.margin_calculator.calculate_position_margin(position)
                            else:
                                margin = position.avg_entry_price * position.contracts * 100 * 0.5
                        except Exception:
                            margin = 0
                    else:
                        delta = 0
                        dte = 0
                        margin = 0
                        
                    line = f"{symbol:<16} {tx.get('quantity', 0):>9} ${tx.get('price', 0):>7.2f} "
                    line += f"${tx.get('value', 0):>8.2f} {delta:>7.3f} {dte:>7} ${margin:>10,.0f}\n"
                    
                    self.write_debug_output(line)
                    
                self.write_debug_output(table_header)
            
            # Write PostTrade section to debug file
            self.write_debug_output(f"\n{separator}\n")
            self.write_debug_output(f"POST-TRADE Summary [{date_str}]:\n")
            
            # Get updated portfolio values
            portfolio_value = self.portfolio.get_portfolio_value()
            cash_balance = self.portfolio.cash_balance
            position_count = len(self.portfolio.positions)
            position_value = sum([pos.current_price * pos.contracts * 100 for pos in self.portfolio.positions.values()])
            
            # Calculate daily P&L for post-trade using the same method as pre-trade
            daily_pnl = 0
            hedge_pnl = 0
            option_pnl = 0
            portfolio_value = self.portfolio.get_portfolio_value()
            
            # Separate option and hedge PnL - use same calculation as PRE-TRADE
            for symbol, pos in self.portfolio.positions.items():
                # Skip positions without previous price data
                if not hasattr(pos, 'prev_price') or pos.prev_price is None:
                    continue
                    
                # Calculate P&L for this position based on price movement
                prev_price = pos.prev_price
                curr_price = pos.current_price
                
                if pos.is_short:
                    pos_pnl = (prev_price - curr_price) * pos.contracts * 100
                else:
                    pos_pnl = (curr_price - prev_price) * pos.contracts * 100
                    
                # Add to appropriate category
                if isinstance(pos, OptionPosition):
                    option_pnl += pos_pnl
                else:
                    hedge_pnl += pos_pnl
                    
                daily_pnl += pos_pnl
            
            # Calculate percentage PnL if we have a valid portfolio value
            pnl_pct = 0
            if portfolio_value > 0:
                pnl_pct = daily_pnl / portfolio_value * 100
                
            self.write_debug_output(f"Daily P&L: ${daily_pnl:.2f} ({pnl_pct:.2f}%)\n")
            self.write_debug_output(f"  Option PnL: ${option_pnl:.2f}\n")
            self.write_debug_output(f"  Hedge PnL: ${hedge_pnl:.2f}\n")
            
            # Portfolio status
            self.write_debug_output(f"Open Trades: {position_count}\n")
            if portfolio_value > 0:
                exposure_pct = position_value / portfolio_value * 100
                self.write_debug_output(f"Total Position Exposure: {exposure_pct:.1f}% of NLV\n")
            else:
                self.write_debug_output("Total Position Exposure: 0.0% of NLV\n")
                
            # Net liquidation value breakdown
            self.write_debug_output(f"Net Liq: ${portfolio_value:,.0f}\n")
            self.write_debug_output(f"  Cash Balance: ${cash_balance:,.0f}\n")
            
            # Current liability (unrealized P&L for short positions)
            liability = sum([pos.unrealized_pnl for pos in self.portfolio.positions.values() if pos.is_short])
            self.write_debug_output(f"  Total Liability: ${liability:,.0f}\n")
            self.write_debug_output(f"  Self Hedge (Hedge PnL): $0\n")
            
            # Calculate margin and leverage with safe error handling
            try:
                if hasattr(self, 'margin_calculator'):
                    margin_calc = self.margin_calculator.calculate_total_margin(self.portfolio.positions)
                else:
                    # Fallback basic margin calculation
                    margin_calc = sum([pos.avg_entry_price * pos.contracts * 100 * 0.5 for pos in self.portfolio.positions.values()])
                
                self.write_debug_output(f"Total Margin Requirement: ${margin_calc:,.0f}\n")
                
                available_margin = cash_balance - margin_calc
                self.write_debug_output(f"Available Margin: ${available_margin:,.0f}\n")
            except Exception as e:
                self.write_debug_output(f"Error calculating margin: {e}\n")
                margin_calc = 0
                available_margin = cash_balance
                self.write_debug_output(f"Using fallback: Margin: $0, Available: ${cash_balance:,.0f}\n")
            
            leverage = 0
            if portfolio_value > 0:
                leverage = margin_calc / portfolio_value
                
            self.write_debug_output(f"Margin-Based Leverage: {leverage:.2f} \n\n")
            
            # Portfolio risk metrics
            portfolio_metrics = self.portfolio.get_portfolio_metrics()
            
            self.write_debug_output("Portfolio Greek Risk:\n")
            
            # Option delta
            delta = portfolio_metrics.get('delta', 0)
            dollar_delta = portfolio_metrics.get('dollar_delta', 0)
            self.write_debug_output(f"  Option Delta: {delta:.3f} (${dollar_delta:.2f})\n")
            
            # We don't have hedge delta yet, so show 0
            self.write_debug_output(f"  Hedge Delta: 0.000 ($0.00)\n")
            self.write_debug_output(f"  Total Delta: {delta:.3f} (${dollar_delta:.2f})\n")
            
            # Other Greeks
            gamma = portfolio_metrics.get('gamma', 0)
            dollar_gamma = gamma * 100  # Approximate dollar gamma per 1% move
            theta = portfolio_metrics.get('theta', 0)
            vega = portfolio_metrics.get('vega', 0)
            
            self.write_debug_output(f"  Gamma: {gamma:.6f} (${dollar_gamma:.2f} per 1% move)\n")
            self.write_debug_output(f"  Theta: ${theta:.2f} per day\n")
            self.write_debug_output(f"  Vega: ${vega:.2f} per 1% IV\n\n")
            
            # Rolling metrics section with actual data
            self.write_debug_output("Metrics Analysis:\n")
            
            # Display actual rolling/expanding metrics from calculation
            if rolling_metrics:
                # Show overall window type
                window_type = rolling_metrics.get('window_type', 'unknown')
                self.write_debug_output(f"  Window Type: {window_type.upper()} (minimum 5 data points required)\n")
                
                # Determine which window is used for risk scaling from config
                risk_config = self.config.get('risk', {})
                risk_scaling_window = risk_config.get('risk_scaling_window', 'short')
                
                # Get metrics for the selected risk scaling window
                if 'windows' in rolling_metrics and risk_scaling_window in rolling_metrics['windows']:
                    window_data = rolling_metrics['windows'][risk_scaling_window]
                    window_days = window_data['window']
                    window_type = window_data.get('window_type', 'unknown')
                    self.write_debug_output(f"  Risk Scaling Window ({window_days}D - {risk_scaling_window.upper()}, {window_type}): Sharpe: {window_data['sharpe']:.2f}, Vol: {window_data.get('volatility', 0):.2%}\n")
                
                # Display overall metrics
                self.write_debug_output("\n  Overall Metrics:\n")
                if 'sharpe' in rolling_metrics:
                    self.write_debug_output(f"    Sharpe Ratio: {rolling_metrics['sharpe']:.2f}\n")
                if 'volatility' in rolling_metrics:
                    self.write_debug_output(f"    Volatility (annualized): {rolling_metrics['volatility']:.2%}\n")
                if 'drawdown' in rolling_metrics:
                    self.write_debug_output(f"    Max Drawdown: {rolling_metrics['drawdown']:.2%}\n")
                
                # Display historical stats
                hist_window_type = rolling_metrics.get('hist_window_type', 'none')
                hist_window = rolling_metrics.get('hist_window', 0)
                if hist_window > 0:
                    self.write_debug_output(f"\n  Historical Stats ({hist_window}D, {hist_window_type}):\n")
                    if 'hist_mean' in rolling_metrics:
                        self.write_debug_output(f"    Historical Mean Sharpe: {rolling_metrics['hist_mean']:.2f}\n")
                    if 'hist_std' in rolling_metrics:
                        self.write_debug_output(f"    Historical Std Dev: {rolling_metrics['hist_std']:.2f}\n")
                    if 'z_score' in rolling_metrics:
                        self.write_debug_output(f"    Current Z-Score: {rolling_metrics['z_score']:.2f}\n")
                    if 'scaling' in rolling_metrics:
                        self.write_debug_output(f"    Position Scaling: {rolling_metrics['scaling']:.2f}\n")
                
                # Display all window metrics if available
                if 'windows' in rolling_metrics:
                    windows = rolling_metrics['windows']
                    self.write_debug_output("\n  Window Analysis:\n")
                    
                    for window_name, window_data in windows.items():
                        if 'sharpe' in window_data and 'window' in window_data:
                            window_type = window_data.get('window_type', 'unknown')
                            self.write_debug_output(f"    {window_name.capitalize()} Window ({window_data['window']} days, {window_type}): Sharpe: {window_data['sharpe']:.2f}, Vol: {window_data.get('volatility', 0):.2%}\n")
            else:
                self.write_debug_output("  No metrics available yet - need at least 5 trading days\n")
                
            self.write_debug_output(f"{sub_separator}\n\n")
            
            # Print open positions table
            if position_count > 0:
                self.write_debug_output("Open Trades Table:\n")
                
                table_header = "-" * 120 + "\n"
                header_line = f"{'Symbol':<16} {'Contracts':>9} {'Entry':>7} {'Current':>8} {'Value':>10} "
                header_line += f"{'NLV%':>5} {'Underlying':>10} {'Delta':>8} {'Gamma':>8} {'Theta':>9} {'Vega':>9} {'Margin':>10} {'DTE':>5}\n"
                
                self.write_debug_output(table_header)
                self.write_debug_output(header_line)
                self.write_debug_output(table_header)
                
                # Total values for the final row
                total_value = 0
                total_margin = 0
                
                # Output each position
                for symbol, pos in self.portfolio.positions.items():
                    position_value = pos.current_price * pos.contracts * 100
                    total_value += position_value
                    
                    margin = 0
                    # Safely calculate margin
                    try:
                        if hasattr(self, 'margin_calculator'):
                            margin = self.margin_calculator.calculate_position_margin(pos)
                        else:
                            margin = pos.avg_entry_price * pos.contracts * 100 * 0.5
                    except Exception:
                        margin = 0
                        
                    total_margin += margin
                    
                    nlv_pct = 0
                    if portfolio_value > 0:
                        nlv_pct = position_value / portfolio_value * 100
                        
                    position_line = f"{symbol:<16} {pos.contracts:>9} ${pos.avg_entry_price:>5.2f} "
                    position_line += f"${pos.current_price:>6.2f} ${position_value:>8,.0f} "
                    position_line += f"{nlv_pct:>5.1f}% ${pos.underlying_price:>9.2f} "
                    position_line += f"{pos.current_delta:>8.3f} {pos.current_gamma:>8.6f} "
                    position_line += f"{pos.current_theta:>9.2f} {pos.current_vega:>9.2f} "
                    position_line += f"${margin:>9,.0f} {pos.days_to_expiry:>5}\n"
                    
                    self.write_debug_output(position_line)
                
                # Add a total line
                self.write_debug_output(table_header)
                total_line = f"{'TOTAL':<16} {'':<9} {'':<7} {'':<8} ${total_value:>8,.0f} "
                if portfolio_value > 0:
                    total_line += f"{total_value/portfolio_value*100:>5.1f}% {'':<10} {'':<8} {'':<8} {'':<9} {'':<9} "
                else:
                    total_line += f"{0:>5.1f}% {'':<10} {'':<8} {'':<8} {'':<9} {'':<9} "
                total_line += f"${total_margin:>9,.0f} {'':<5}\n"
                self.write_debug_output(total_line)
                self.write_debug_output(table_header)
            
            self.write_debug_output(f"{separator}\n")
            
            # We've already displayed a clean progress indicator
            # All the detailed output should go to the debug file instead of console
            pass

            # Add trades from this day
            if hasattr(self.portfolio, 'transactions'):
                results['trades'].extend([t for t in self.portfolio.transactions if t['date'] == current_date])

        # Calculate performance metrics
        perf_metrics = self.calculate_performance_metrics(results)
        results.update(perf_metrics)

        # Generate report if requested
        if self.config.get('backtest', {}).get('generate_report', True):
            try:
                # Create equity history dictionary with proper datetime keys
                equity_history = {}
                dates = results.get('dates', [])
                portfolio_values = results.get('portfolio_values', [])
                
                # Only include data if we have both dates and values
                if dates and portfolio_values and len(dates) == len(portfolio_values):
                    for i, date in enumerate(dates):
                        equity_history[date] = portfolio_values[i]
                
                # Get risk metrics
                risk_metrics = {
                    'initial_capital': results.get('initial_value', 0),
                    'final_value': results.get('final_value', 0),
                    'total_return': results.get('total_return', 0),
                    'sharpe_ratio': results.get('sharpe_ratio', 0),
                    'max_drawdown': results.get('max_drawdown', 0)
                }
                
                # Generate the report
                report_path = self.reporting_system.generate_html_report(
                    equity_history=equity_history,
                    risk_metrics=risk_metrics,
                    config=self.config
                )
                results['report_path'] = report_path
            except Exception as e:
                self.logger.error(f"Error generating HTML report: {e}")
                traceback_str = traceback.format_exc()
                self.logger.debug(traceback_str)
        
        # Add final summary to debug file
        self.write_debug_output("\n\n=== BACKTEST SUMMARY ===\n")
        self.write_debug_output(f"Initial Capital: ${results.get('initial_value', 0):,.2f}\n")
        self.write_debug_output(f"Final Value: ${results.get('final_value', 0):,.2f}\n")
        self.write_debug_output(f"Total Return: {results.get('total_return', 0):.2%}\n")
        self.write_debug_output(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}\n")
        self.write_debug_output(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}\n")
        
        # Close debug file
        self._cleanup()
        
        return results
        
    def _cleanup(self):
        """Close open resources when done."""
        if self.debug_file:
            try:
                # Add a final message
                self.write_debug_output("\n\n==== SIMULATION COMPLETED ====\n")
                self.write_debug_output(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # Close the file
                self.debug_file.close()
                
                # Log completion
                if hasattr(self, 'logger'):
                    self.logger.info("Debug log file closed successfully")
                    
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error closing debug file: {e}")
            finally:
                self.debug_file = None
    
    def setup_debug_file(self) -> None:
        """
        Set up a debug file for detailed trading information.
        """
        # Skip if config doesn't exist or debug output is disabled
        if not hasattr(self, 'config') or not self.config.get('backtest', {}).get('debug_output', True):
            return
            
        # Create output directory if it doesn't exist
        output_dir = self.config.get('paths', {}).get('output_dir', 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Build debug filename with strategy parameters
        strategy_config = self.config.get('strategy', {})
        
        # Get important strategy parameters for filename
        delta_target = strategy_config.get('delta_target', -0.2)
        delta_str = f"DT{abs(delta_target):.2f}".replace('.', '')
        leverage = self.config.get('risk', {}).get('max_leverage', 6.0)
        dte_max = strategy_config.get('days_to_expiry_max', 45)
        hedge_enabled = strategy_config.get('enable_hedging', False)
        
        filename_parts = [
            strategy_config.get('name', 'Strategy')[:3].upper(),
            datetime.now().strftime('%y%m%d'),
            f"EH{hedge_enabled}",
            delta_str,
            f"DTol{strategy_config.get('delta_tolerance', 0.05)}",
            f"L{int(leverage)}"
        ]
        
        debug_filename = os.path.join(output_dir, f"{''.join(filename_parts)}.log")
        
        report_filename = f"{''.join(filename_parts)}.html"
        report_path = os.path.join(output_dir, report_filename)
        
        try:
            self.debug_file = open(debug_filename, 'w')
            
            # Write enhanced header with detailed configuration
            self.debug_file.write("=== TRADING ENGINE STRATEGY EXECUTION ===\n")
            self.debug_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.debug_file.write(f"Log file: {os.path.abspath(debug_filename)}\n")
            self.debug_file.write(f"Log level: {self.config.get('backtest', {}).get('log_level', 'INFO')}\n")
            self.debug_file.write(f"Report will be saved as: {report_filename}\n\n")
            
            # Main strategy settings section
            self.debug_file.write("Main Strategy Settings:\n")
            for key, value in sorted(strategy_config.items()):
                self.debug_file.write(f"  {key}: {value}\n")
            self.debug_file.write("=" * 50 + "\n\n")
            
            # Risk and margin settings
            self.debug_file.write("Margin Management Settings:\n")
            risk_config = self.config.get('risk', {})
            self.debug_file.write(f"  Margin Buffer: {risk_config.get('margin_buffer', 10)}%\n")
            self.debug_file.write(f"  Maximum Leverage: {risk_config.get('max_leverage', 6.0)}x\n")
            self.debug_file.write(f"  Maximum Position Size: {self.config.get('portfolio', {}).get('max_position_size_pct', 0.05) * 100}% of NLV\n")
            self.debug_file.write(f"  Maximum Portfolio Delta: {self.config.get('portfolio', {}).get('max_portfolio_delta', 0.2) * 100}% of NLV\n")
            self.debug_file.write("Loading data...\n")
            self.debug_file.flush()
            
            # Make sure logger is initialized
            logger = getattr(self, 'logger', None)
            
            if logger:
                logger.info(f"Debug output file: {debug_filename}")
            else:
                print(f"Debug output file: {debug_filename}")
        except Exception as e:
            # Make sure logger is initialized
            logger = getattr(self, 'logger', None)
            
            if logger:
                logger.error(f"Failed to create debug file: {e}")
            else:
                print(f"Failed to create debug file: {e}")
            self.debug_file = None
    
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
            return None
            
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
            
            # Get risk parameters from config
            min_z = risk_config.get('min_z', -2.0)
            target_z = risk_config.get('target_z', 0.0)
            min_scaling = risk_config.get('min_investment', 0.25)
            
            # Calculate scaling factor based on z-score
            # Higher z-scores mean better performance, leading to more aggressive sizing
            if z_score >= target_z:
                # Above target z-score - full exposure
                scaling = 1.0
            elif z_score <= min_z:
                # Below minimum z-score - minimum exposure
                scaling = min_scaling
            else:
                # Between min and target z-scores - linear scaling
                scaling = min_scaling + (z_score - min_z) / (target_z - min_z) * (1 - min_scaling)
                
            metrics['scaling'] = scaling
        else:
            # Not enough data for historical stats - use default values
            metrics['hist_mean'] = 0
            metrics['hist_std'] = 0
            metrics['z_score'] = 0
            metrics['scaling'] = 1.0  # Use full scaling by default
            metrics['hist_window_type'] = 'none'
            metrics['hist_window'] = 0
            
        return metrics
        
    def calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            results: Dictionary of backtest results
            
        Returns:
            dict: Performance metrics
        """
        # Extract data
        portfolio_values = results.get('portfolio_values', [])
        dates = results.get('dates', [])
        
        # Need at least 2 data points
        if len(portfolio_values) < 2:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'volatility': 0
            }
        
        # Calculate returns
        returns = []
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i-1] > 0:
                returns.append((portfolio_values[i] / portfolio_values[i-1]) - 1)
        
        # Convert to numpy array for calculations
        returns = np.array(returns)
        
        # Calculate total return
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value / initial_value) - 1
        
        # Calculate annualized return
        days = (dates[-1] - dates[0]).days
        years = days / 365.0 if days > 0 else 1
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Calculate volatility
        daily_volatility = np.std(returns)
        annualized_volatility = daily_volatility * np.sqrt(252)  # Annualize for trading days
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        avg_daily_return = np.mean(returns)
        sharpe_ratio = 0
        if daily_volatility > 0:
            sharpe_ratio = (avg_daily_return / daily_volatility) * np.sqrt(252)
        
        # Calculate max drawdown
        highwater = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - highwater) / highwater
        max_drawdown = abs(min(drawdown))
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': annualized_volatility
        }
    
    def process_signal(self, signal: Dict[str, Any], current_date: datetime) -> None:
        """
        Process a trading signal.
        
        Args:
            signal: Signal to process
            current_date: Current date for the trade
        """
        action = signal.get('action', '')
        symbol = signal.get('symbol', '')
        quantity = signal.get('quantity', 0)
        
        # Skip invalid signals
        if not symbol or quantity <= 0:
            return
            
        # Get price from signal data
        price = 0
        signal_data = signal.get('data', {})
        if hasattr(signal_data, 'get'):
            price = signal_data.get('MidPrice', 0)
        elif hasattr(signal_data, 'iloc'):
            price = signal_data['MidPrice'] if 'MidPrice' in signal_data else 0
            
        # Set execution data
        execution_data = {
            'date': current_date,
            'signal': signal
        }
        
        # Process buy or sell
        if action.upper() == 'BUY':
            # For buy signals, use a long position
            is_short = False
            
            # Add position
            position = self.portfolio.add_position(
                symbol=symbol,
                instrument_data=signal_data,
                quantity=quantity,
                price=price,
                position_type=signal.get('type', 'stock'),
                is_short=is_short,
                execution_data=execution_data
            )
            
            self.logger.info(f"[Engine] Added LONG position: {quantity} {symbol} @ ${price:.2f}")
            
        elif action.upper() == 'SELL':
            # For sell signals, use a short position
            is_short = True
            
            # Add position
            position = self.portfolio.add_position(
                symbol=symbol,
                instrument_data=signal_data,
                quantity=quantity,
                price=price,
                position_type=signal.get('type', 'stock'),
                is_short=is_short,
                execution_data=execution_data
            )
            
            self.logger.info(f"[Engine] Added SHORT position: {quantity} {symbol} @ ${price:.2f}")
            
        elif action.upper() == 'CLOSE':
            # Close position
            if symbol in self.portfolio.positions:
                pnl = self.portfolio.remove_position(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    execution_data=execution_data,
                    reason=signal.get('reason', 'Signal')
                )
                
                self.logger.info(f"[Engine] Closed position: {quantity} {symbol} @ ${price:.2f}")
                self.logger.info(f"  P&L: ${pnl:.2f}")
                
    def manage_positions(self, current_date: datetime, daily_data: pd.DataFrame) -> None:
        """
        Check exit conditions and manage existing positions.
        
        Args:
            current_date: Current simulation date
            daily_data: DataFrame containing market data for the current date
        """
        # Create a dictionary of market data by symbol using the DataManager's method
        market_data = self.data_manager.get_market_data_by_symbol(daily_data)
        
        # Check each position against exit conditions
        for symbol, position in list(self.portfolio.positions.items()):
            # Skip if position has no market data
            if symbol not in market_data:
                self.logger.warning(f"No market data for position {symbol}, skipping exit check")
                continue
            
            # Get current market data
            current_data = market_data[symbol]
            
            # Check exit conditions
            exit_flag, reason = self.strategy.check_exit_conditions(position, current_data)
            
            # Exit position if conditions met
            if exit_flag:
                self.logger.info(f"[Engine] Exit signal for {symbol}: {reason}")
                
                # Close the position
                self.portfolio.remove_position(
                    symbol=symbol,
                    price=position.current_price,
                    execution_data={'date': current_date},
                    reason=reason
                )


def load_configuration(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a file (JSON, YAML, etc.)
    
    Args:
        config_file: Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    import json
    import os
    
    # Check if file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
        
    # Load JSON config
    with open(config_file, 'r') as f:
        config = json.load(f)
        
    return config


def main(config_file: str, strategy_class=None):
    """
    Main entry point for the trading engine.
    
    Args:
        config_file: Path to config file
        strategy_class: Strategy class to use (optional)
        
    Returns:
        dict: Results of the backtest
    """
    # Load configuration
    loaded_config = load_configuration(config_file)
    
    # Create logging
    logging_manager = LoggingManager()
    logger = logging_manager.setup_logging(loaded_config)
    
    # Create strategy instance (if provided)
    if strategy_class:
        strategy = strategy_class(loaded_config.get('strategy', {}), logger)
    else:
        # Dynamically load strategy class from config
        strategy_name = loaded_config.get('strategy', {}).get('name', 'BaseStrategy')
        strategy_module = __import__('strategies')
        strategy_class = getattr(strategy_module, strategy_name)
        strategy = strategy_class(loaded_config.get('strategy', {}), logger)
    
    # Create trading engine
    engine = TradingEngine(loaded_config, strategy, logger)
    
    # Load data
    logger.info("[INIT] Loading data...")
    if not engine.load_data():
        logger.error("Failed to load data")
        return None
        
    # Run backtest - the detailed messages will be shown by the run_backtest method
    results = engine.run_backtest()
    
    # Return results
    return results


# Allow this file to be run directly for testing purposes
if __name__ == "__main__":
    print("This module is designed to be imported, not run directly.")
    print("See example_strategy.py for a usage example.")
    print("Running a simple test...")
    
    # Create a minimal test configuration
    test_config = {
        'paths': {
            'input_file': 'test_data.csv',
            'output_dir': 'test_output'
        },
        'dates': {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        },
        'strategy': {
            'name': 'TestStrategy'
        }
    }
    
    # Create a simple strategy for testing
    class TestStrategy(Strategy):
        def generate_signals(self, current_date, daily_data):
            return []
        
        def check_exit_conditions(self, position, market_data):
            return False, None
    
    # Create a trading engine instance
    test_strategy = TestStrategy(test_config['strategy'])
    engine = TradingEngine(test_config, test_strategy)
    
    print(f"Engine created successfully with strategy: {test_strategy.name}")
    print("Path handling is working correctly.")