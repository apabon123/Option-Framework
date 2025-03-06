"""
Theta Engine Options Trading Strategy

This version always uses compounding returns for position sizing.
Note: A non‐compounded version is not possible in this design because the
trading signals and available margin calculations are based on the current
(compounded) net liquidation value. As a result, new trades are added based on
the dynamic account state rather than a fixed initial allocation.

Key Features:
  - Iterative position sizing based on available margin.
  - Integrated hedging adjustments.
  - Detailed daily reporting and rolling performance metrics.
  - Warnings suppressed for a cleaner log output.
"""
import os
import sys
import logging
from datetime import datetime, timedelta
import io
import base64
import warnings
import math

# Third-party library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import ffn

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)


class LoggingManager:
    """
    Manages logging configuration and operations for the Theta Engine.

    This class centralizes all logging functionality including setup,
    filtering, and redirection of output streams.
    """

    def __init__(self):
        """Initialize the LoggingManager"""
        self.logger = None
        self.log_file = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def status_filter(self, record):
        """
        Modified filter to only show status messages in console without separators.

        Args:
            record: Log record to filter

        Returns:
            bool: True if record should be displayed, False otherwise
        """
        # Always filter out DEBUG level messages from console
        if record.levelno == logging.DEBUG:
            return False

        # Check if this is a status update message but exclude separator lines
        is_status = (
                record.getMessage().startswith('[STATUS]') or
                'Progress:' in record.getMessage() or
                'Strategy execution completed' in record.getMessage() or
                'Error:' in record.getMessage() or
                'WARNING:' in record.getMessage() or
                'ERROR:' in record.getMessage()
        )

        # Filter out separator lines (lines with only = characters)
        if record.getMessage().startswith('==='):
            return False

        return is_status

    def setup_logging(self, config, verbose_console=False, debug_mode=False, clean_format=True):
        """
        Set up logging for the Theta Engine with modified progress format.

        Args:
            config: Configuration dictionary
            verbose_console: If True, all logs go to console. If False, only status updates go to console.
            debug_mode: If True, enables DEBUG level logging. If False, uses INFO level.
            clean_format: If True, uses a clean format without timestamps and log levels.

        Returns:
            logger: Configured logger instance
        """
        # Store original stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Create output directory if it doesn't exist
        output_dir = config['paths'].get('output_dir', 'logs')
        os.makedirs(output_dir, exist_ok=True)

        # Build log filename based on configuration
        log_filename = os.path.join(output_dir, self.build_log_filename(config))
        self.log_file = log_filename

        # Create the logger
        logger = logging.getLogger('theta_engine')

        # Set the root logger level first - this is crucial
        logging.getLogger().setLevel(logging.WARNING)  # Set root logger to WARNING by default

        # Set this specific logger's level
        logger.setLevel(logging.INFO if not debug_mode else logging.DEBUG)
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

        # Console handler - can be filtered and should never show DEBUG unless explicitly requested
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Always set console to INFO level at minimum
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        if not verbose_console:
            console_handler.addFilter(self.status_filter)
        logger.addHandler(console_handler)

        # Also set up sys.stdout redirection for any print statements
        sys.stdout = self.StreamToLogger(logger, logging.INFO)
        sys.stderr = self.StreamToLogger(logger, logging.ERROR)

        # Log initial messages
        logger.info(f"=== THETA ENGINE STRATEGY EXECUTION ===")
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Log file: {log_filename}")
        logger.info(f"Log level: {'DEBUG' if debug_mode else 'INFO'}")

        # Get delta target from config for display in progress messages
        delta_target = config.get('strategy', {}).get('delta_target', 0)
        delta_target_str = f"DT{abs(delta_target):.2f}".replace('0.', '.')

        # Store delta target string for use in progress messages
        self.delta_target_str = delta_target_str

        console_only_message = f"[STATUS] Strategy execution started - detailed logs being written to file"
        print(console_only_message)

        self.logger = logger
        return logger

    def teardown_logging(self):
        """Restore original stdout and stderr"""
        if hasattr(self, 'original_stdout') and self.original_stdout:
            sys.stdout = self.original_stdout

        if hasattr(self, 'original_stderr') and self.original_stderr:
            sys.stderr = self.original_stderr

    def build_log_filename(self, config):
        """
        Build a log filename based on configuration parameters.

        Args:
            config: The configuration dictionary

        Returns:
            str: Formatted log filename
        """
        # Format start date as yymmdd
        start_date_str = config['dates']['start_date'].strftime('%y%m%d')

        # Get strategy parameters
        strategy_config = config.get('strategy', {})
        hedge_mode = strategy_config.get('hedge_mode', 'ratio').lower()
        enable_hedging = strategy_config.get('enable_hedging', False)

        # Get portfolio parameters
        portfolio_config = config.get('portfolio', {})
        max_leverage = portfolio_config.get('max_leverage', 12)

        # Get delta target and tolerance
        dt = strategy_config.get('delta_target', 0)
        dtol = strategy_config.get('delta_tolerance', 0.2)

        if hedge_mode == "constant":
            cp_delta = strategy_config.get('constant_portfolio_delta', 0.05)
            # Ensure correct sign based on is_short flag
            cp_delta = -abs(cp_delta) if strategy_config.get('is_short', True) else abs(cp_delta)
            cp_delta_str = f"{cp_delta:.2f}"
            prefix = f"CPD{cp_delta_str}"
            return f"{prefix}-{start_date_str}-EH{enable_hedging}-DT{dt}-DTol{dtol}-L{max_leverage}.log"
        else:
            prefix = "HTD"
            return f"{prefix}-{start_date_str}-EH{enable_hedging}-DT{dt}-DTol{dtol}-L{max_leverage}.log"

    def build_html_report_filename(self, config):
        """
        Build an HTML report filename based on configuration parameters.

        Args:
            config: The configuration dictionary

        Returns:
            str: Formatted HTML report filename
        """
        # Format start date as yymmdd
        start_date_str = config['dates']['start_date'].strftime('%y%m%d')

        # Get strategy parameters
        strategy_config = config.get('strategy', {})
        hedge_mode = strategy_config.get('hedge_mode', 'ratio').lower()
        enable_hedging = strategy_config.get('enable_hedging', False)

        # Get portfolio parameters
        portfolio_config = config.get('portfolio', {})
        max_leverage = portfolio_config.get('max_leverage', 12)

        # Get delta target and tolerance
        dt = strategy_config.get('delta_target', 0)
        dtol = strategy_config.get('delta_tolerance', 0.2)

        if hedge_mode == "constant":
            cp_delta = strategy_config.get('constant_portfolio_delta', 0.05)
            # Ensure correct sign based on is_short flag
            cp_delta = -abs(cp_delta) if strategy_config.get('is_short', True) else abs(cp_delta)
            cp_delta_str = f"{cp_delta:.2f}"
            prefix = f"CPD{cp_delta_str}"
            return f"{prefix}-{start_date_str}-EH{enable_hedging}-DT{dt}-DTol{dtol}-L{max_leverage}.html"
        else:
            prefix = "HTD"
            return f"{prefix}-{start_date_str}-EH{enable_hedging}-DT{dt}-DTol{dtol}-L{max_leverage}.html"

    def log_status(self, message):
        """
        Log a status message that will appear in both log file and console.

        Args:
            message: Message to log
        """
        if self.logger:
            self.logger.info(f"[STATUS] {message}")
        else:
            print(f"[STATUS] {message}")

    def log_error(self, message, exception=None):
        """
        Log an error message that will appear in both log file and console.

        Args:
            message: Error message
            exception: Optional exception object
        """
        error_message = f"ERROR: {message}"
        if exception:
            error_message += f" - {str(exception)}"

        if self.logger:
            self.logger.error(error_message)
        else:
            print(error_message)

        # Log traceback if exception provided
        if exception:
            import traceback
            trace_str = traceback.format_exc()
            if self.logger:
                self.logger.error(trace_str)
            else:
                print(trace_str)

    def print_strategy_settings(self, config):
        """
        Print main strategy settings for logging.

        Args:
            config: Configuration dictionary
        """
        if self.logger:
            self.logger.info("\nMain Strategy Settings:")
            for key, value in config['strategy'].items():
                self.logger.info(f"  {key}: {value}")
            self.logger.info("=" * 50)
        else:
            print("\nMain Strategy Settings:")
            for key, value in config['strategy'].items():
                print(f"  {key}: {value}")
            print("=" * 50)

    def print_console_summary(self, metrics):
        """
        Print a concise summary of results to the console.

        Args:
            metrics: Dictionary of performance metrics
        """
        # Temporarily restore original stdout
        old_stdout = sys.stdout
        sys.stdout = self.original_stdout

        print("\n=== THETA ENGINE STRATEGY SUMMARY ===")
        print(f"Initial Capital: ${metrics.get('start_value', 0):,.2f}")
        print(f"Final Value: ${metrics.get('end_value', 0):,.2f}")
        print(f"Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"CAGR: {metrics.get('cagr', 0):.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"Volatility: {metrics.get('volatility', 0):.2%}")
        print(f"Avg Risk Scaling: {metrics.get('avg_risk_scaling', 0):.2f}")
        print(f"HTML Report: {self.get_html_report_path()}")
        print("=" * 40)

        # Restore logger stdout
        sys.stdout = old_stdout

    def get_html_report_path(self):
        """Get the path to the HTML report"""
        log_path = self.log_file if self.log_file else ""
        if log_path.endswith(".log"):
            return log_path.replace(".log", ".html")
        return ""

    class StreamToLogger:
        """
        Redirects print statements to the logger with better Unicode handling.
        """

        def __init__(self, logger, log_level=logging.INFO):
            self.logger = logger
            self.log_level = log_level
            self.buffer = ''

        def write(self, message):
            try:
                if message and message.strip():
                    self.buffer += message
                    if message.endswith('\n'):
                        # Handle potential Unicode encoding issues
                        try:
                            self.logger.log(self.log_level, self.buffer.rstrip())
                        except UnicodeEncodeError:
                            # Replace problematic characters with ASCII equivalents
                            safe_message = self.buffer.rstrip().encode('ascii', 'replace').decode('ascii')
                            self.logger.log(self.log_level, safe_message)
                        self.buffer = ''
            except Exception as e:
                # Last resort error handling
                try:
                    self.logger.error(f"Error in StreamToLogger: {str(e)}")
                except:
                    pass
                self.buffer = ''

        def flush(self):
            if self.buffer:
                try:
                    self.logger.log(self.log_level, self.buffer.rstrip())
                except UnicodeEncodeError:
                    # Replace problematic characters with ASCII equivalents
                    safe_message = self.buffer.rstrip().encode('ascii', 'replace').decode('ascii')
                    self.logger.log(self.log_level, safe_message)
                except Exception as e:
                    # Last resort error handling
                    try:
                        self.logger.error(f"Error in StreamToLogger flush: {str(e)}")
                    except:
                        pass
                self.buffer = ''

# ========================
# Strategy Class
# ========================
class ThetaEngineStrategy:
    """
    Defines the trading logic for the Theta Engine strategy.

    This class contains the rules for selecting entry candidates, choosing
    the best trades, and determining when to exit positions.
    """

    def __init__(self, config, logger=None):
        """
        Initialize the strategy with configuration parameters.

        Args:
            config: Strategy configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger('theta_engine')

        # Extract key strategy parameters
        self.is_short = config.get('is_short', True)
        self.delta_target = config.get('delta_target', -0.2)
        self.delta_tolerance = config.get('delta_tolerance', 0.2)
        self.days_to_expiry_min = config.get('days_to_expiry_min', 60)
        self.days_to_expiry_max = config.get('days_to_expiry_max', 90)
        self.profit_target = config.get('profit_target', 0.65)
        self.stop_loss_threshold = config.get('stop_loss_threshold', 2.5)
        self.close_days_to_expiry = config.get('close_days_to_expiry', 14)

        # Log strategy initialization
        self.logger.info("[Strategy] Initializing ThetaEngineStrategy")
        self.logger.info(f"  Delta Target: {self.delta_target}")
        self.logger.info(f"  DTE Range: {self.days_to_expiry_min}-{self.days_to_expiry_max} days")
        self.logger.info(f"  Position Type: {'Short' if self.is_short else 'Long'}")
        self.logger.info(f"  Profit Target: {self.profit_target * 100:.0f}% of premium")
        self.logger.info(f"  Stop Loss: {self.stop_loss_threshold * 100:.0f}% of premium")
        self.logger.info(f"  Time-Based Exit: {self.close_days_to_expiry} DTE")

    def filter_candidates(self, daily_data):
        """
        Filter for valid trade entry candidates based on delta target sign.

        Args:
            daily_data: DataFrame of daily options data

        Returns:
            DataFrame: Filtered candidates sorted by preference
        """
        self.logger.debug(f"[Strategy] Filtering candidates from {len(daily_data)} options")

        # Determine option type based on delta target sign
        option_type = 'call' if self.delta_target > 0 else 'put'

        # Apply initial filters
        candidates = daily_data[
            (daily_data['Type'].str.lower() == option_type) &  # Option type based on delta_target sign
            (daily_data['DaysToExpiry'] >= self.days_to_expiry_min) &  # Min DTE
            (daily_data['DaysToExpiry'] <= self.days_to_expiry_max) &  # Max DTE
            (daily_data['MidPrice'] > 0)  # Valid price
            ].copy()

        if not candidates.empty:
            # Calculate delta distance for each candidate
            candidates['delta_distance'] = abs(candidates['Delta'] - self.delta_target)

            # Sort by closest to target delta
            candidates = candidates.sort_values('delta_distance')

            self.logger.debug(f"[Strategy] Found {len(candidates)} suitable {option_type} candidates")
        else:
            self.logger.debug(f"[Strategy] No suitable {option_type} candidates found")

        return candidates

    def select_best_candidate(self, candidates):
        """
        Select the best trade candidate from filtered options.

        Args:
            candidates: DataFrame of filtered candidates

        Returns:
            Series or None: Best candidate or None if no suitable candidates
        """
        if candidates.empty:
            self.logger.debug("[Strategy] No candidates available for selection")
            return None

        # Select the candidate closest to target delta
        best_candidate = candidates.iloc[0]

        self.logger.info(f"[Strategy] Selected best candidate: {best_candidate['OptionSymbol']}")
        self.logger.info(f"  Strike: {best_candidate['Strike']}, DTE: {best_candidate['DaysToExpiry']}")
        self.logger.info(f"  Delta: {best_candidate['Delta']}, Price: ${best_candidate['MidPrice']:.2f}")

        return best_candidate

    def check_exit_conditions(self, position, current_data):
        """
        Check if position meets exit criteria.

        Args:
            position: Position object
            current_data: Current market data for the position

        Returns:
            tuple: (exit_flag, reason) - Whether to exit and why
        """
        # Time-based exit - close by DTE
        if position.days_to_expiry <= self.close_days_to_expiry:
            self.logger.debug(f"[Strategy] Exit signal: DTE {position.days_to_expiry} ≤ {self.close_days_to_expiry}")
            return True, f"Close by DTE {self.close_days_to_expiry}"

        # Calculate profit percentage
        if self.is_short:
            # For short options: positive profit when entry_price > current_price
            profit_pct = (position.avg_entry_price - position.current_price) / position.avg_entry_price if position.avg_entry_price > 0 else 0
        else:
            # For long options: positive profit when current_price > entry_price
            profit_pct = (position.current_price - position.avg_entry_price) / position.avg_entry_price if position.avg_entry_price > 0 else 0

        # Profit target exit
        if profit_pct >= self.profit_target:
            self.logger.debug(f"[Strategy] Exit signal: Profit {profit_pct:.2%} ≥ Target {self.profit_target:.2%}")
            return True, 'Profit Target'

        # Stop loss exit
        if profit_pct <= -self.stop_loss_threshold:
            self.logger.debug(
                f"[Strategy] Exit signal: Loss {profit_pct:.2%} ≤ Stop Loss -{self.stop_loss_threshold:.2%}")
            return True, 'Stop Loss'

        # No exit condition met
        self.logger.debug(f"[Strategy] No exit signal: Current profit {profit_pct:.2%}")
        return False, None

    def calculate_risk_metrics(self, position):
        """
        Calculate risk metrics for a position.

        Args:
            position: Position object

        Returns:
            dict: Risk metrics dictionary
        """
        metrics = {}

        # Calculate current profit/loss percentage
        if self.is_short:
            profit_pct = (position.avg_entry_price - position.current_price) / position.avg_entry_price
        else:
            profit_pct = (position.current_price - position.avg_entry_price) / position.avg_entry_price

        metrics['profit_pct'] = profit_pct

        # Calculate distance to stop loss in delta points
        if self.is_short:
            delta_to_stop = abs(position.current_delta - (position.current_delta * (1 + self.stop_loss_threshold)))
        else:
            delta_to_stop = abs(position.current_delta - (position.current_delta * (1 - self.stop_loss_threshold)))

        metrics['delta_to_stop'] = delta_to_stop

        # Calculate time decay per day as percentage of current price
        if position.current_theta != 0 and position.current_price > 0:
            theta_decay_pct = (position.current_theta / position.current_price) * 100
            metrics['theta_decay_pct'] = theta_decay_pct
        else:
            metrics['theta_decay_pct'] = 0

        # Log calculated metrics
        self.logger.debug(f"[Strategy] Risk metrics for {position.option_symbol}:")
        self.logger.debug(f"  Current P/L: {profit_pct:.2%}")
        self.logger.debug(f"  Delta to Stop: {delta_to_stop:.4f}")
        self.logger.debug(f"  Theta Decay: {metrics.get('theta_decay_pct', 0):.2f}% per day")

        return metrics

    def evaluate_market_conditions(self, market_data):
        """
        Evaluate current market conditions for trading decisions.

        Args:
            market_data: DataFrame of market data

        Returns:
            dict: Market condition assessment
        """
        if market_data.empty:
            self.logger.warning("[Strategy] Cannot evaluate market conditions: No data available")
            return {'suitable_for_trading': False}

        # Get a sample of the market data (first row)
        sample = market_data.iloc[0]

        # Extract underlying price and basic metrics
        underlying_price = sample.get('UnderlyingPrice', 0)

        # For demonstration purposes, let's say we calculate some market metrics
        # In a real implementation, you would analyze IV, term structure, etc.
        result = {
            'underlying_price': underlying_price,
            'suitable_for_trading': True,
            'market_regime': 'normal'  # Could be 'high_vol', 'low_vol', etc.
        }

        self.logger.debug(f"[Strategy] Market evaluation: {result['market_regime']} regime")

        return result

    def validate_trade_params(self, option_data):
        """
        Validate option parameters for trading.

        Args:
            option_data: Option data (Series or dict-like)

        Returns:
            dict: Validation result with 'is_valid' flag and 'issues' list
        """
        # Initialize with default values for handling both dict and Series
        symbol = "Unknown"
        price = 0
        delta = 0
        dte = 0
        bid_ask_spread = 0

        # Extract data with appropriate handling for both dict and Series
        if hasattr(option_data, 'get') and not hasattr(option_data, 'iloc'):
            # Dictionary style access
            symbol = option_data.get('OptionSymbol', 'Unknown')
            price = option_data.get('MidPrice', 0)
            delta = option_data.get('Delta', 0)
            dte = option_data.get('DaysToExpiry', 0)

            bid = option_data.get('Bid', 0)
            ask = option_data.get('Ask', 0)
            if bid > 0 and ask > 0:
                bid_ask_spread = (ask - bid) / ((bid + ask) / 2)
        else:
            # Series style access
            symbol = option_data['OptionSymbol'] if 'OptionSymbol' in option_data else 'Unknown'
            price = option_data['MidPrice'] if 'MidPrice' in option_data else 0
            delta = option_data['Delta'] if 'Delta' in option_data else 0
            dte = option_data['DaysToExpiry'] if 'DaysToExpiry' in option_data else 0

            bid = option_data['Bid'] if 'Bid' in option_data else 0
            ask = option_data['Ask'] if 'Ask' in option_data else 0
            if bid > 0 and ask > 0:
                bid_ask_spread = (ask - bid) / ((bid + ask) / 2)

        # Define validation thresholds
        max_acceptable_spread = self.config.get('normal_spread', 0.20)
        min_price = 0.10

        # Validate parameters
        is_valid = True
        issues = []

        if price < min_price:
            is_valid = False
            issues.append(f"Price (${price:.2f}) below minimum (${min_price:.2f})")

        if bid_ask_spread > max_acceptable_spread:
            is_valid = False
            issues.append(f"Bid-ask spread ({bid_ask_spread:.2%}) exceeds maximum ({max_acceptable_spread:.2%})")

        if not self.days_to_expiry_min <= dte <= self.days_to_expiry_max:
            is_valid = False
            issues.append(f"DTE ({dte}) outside allowed range ({self.days_to_expiry_min}-{self.days_to_expiry_max})")

        if abs(delta - self.delta_target) > self.delta_tolerance:
            is_valid = False
            issues.append(
                f"Delta ({delta:.3f}) too far from target ({self.delta_target:.3f}±{self.delta_tolerance:.3f})")

        # Log validation results
        if is_valid:
            self.logger.debug(f"[Strategy] Validated {symbol}: All parameters within acceptable ranges")
        else:
            self.logger.debug(f"[Strategy] Validation failed for {symbol}: {', '.join(issues)}")

        return {'is_valid': is_valid, 'issues': issues}


# ========================
# Position Class
# ========================
class Position:
    """
    Represents a position in a single option contract with PnL tracking.

    This class manages a position's lifecycle including adding and removing
    contracts, tracking market data, and calculating performance metrics.
    """

    def __init__(self, option_symbol, contract_data, initial_contracts=0, strategy_config=None, logger=None):
        """
        Initialize a position with option contract data.

        Args:
            option_symbol: Unique identifier for the option
            contract_data: Initial data for the option contract
            initial_contracts: Starting number of contracts
            strategy_config: Strategy configuration dictionary
            logger: Logger instance
        """
        self.option_symbol = option_symbol
        self.logger = logger or logging.getLogger('theta_engine')
        self.strategy_config = strategy_config or {}

        # Handle pandas Series objects correctly
        if hasattr(contract_data, 'get') and not hasattr(contract_data, 'iloc'):
            # This is a dictionary
            self.strike = contract_data.get('Strike')
            self.expiration = contract_data.get('Expiration')
            self.type = contract_data.get('Type')
        else:
            # This is a pandas Series
            self.strike = contract_data['Strike'] if 'Strike' in contract_data else None
            self.expiration = contract_data['Expiration'] if 'Expiration' in contract_data else None
            self.type = contract_data['Type'] if 'Type' in contract_data else None

        self.contracts = initial_contracts

        # Determine if this is a short position based on the is_short flag in strategy config
        # rather than assuming puts are always short
        self.is_short = self.strategy_config.get('is_short', True)

        # Log the position type for debugging
        option_type = "unknown"
        if self.type is not None:
            option_type = self.type.lower() if isinstance(self.type, str) else str(self.type).lower()

        self.logger.debug(f"[Position] Created {option_type} position, is_short={self.is_short}")

        # For average price calculation
        self.total_value = 0
        self.avg_entry_price = 0

        # For P&L tracking
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.max_drawdown = 0

        # For market data
        self.current_price = 0
        self.current_delta = 0
        self.current_gamma = 0
        self.current_theta = 0
        self.current_vega = 0
        self.underlying_price = 0
        self.days_to_expiry = 0

        # Transaction history
        self.transactions = []
        self.daily_data = []

        # CPD reference data (for Constant Portfolio Delta hedging)
        self.cpd_reference_symbol = None
        self.cpd_reference_price = None
        self.cpd_reference_delta = None
        self.cpd_reference_expiry = None
        self.cpd_effective_contracts = 0

        # Update with initial data if provided
        if contract_data is not None:
            self.update_market_data(contract_data)

    def add_contracts(self, quantity, price, execution_data=None):
        """
        Add contracts to the position with proper cost basis tracking.

        Args:
            quantity: Number of contracts to add
            price: Price per contract
            execution_data: Additional data about the execution

        Returns:
            float: Updated average entry price
        """
        if quantity <= 0:
            return self.avg_entry_price

        # Record transaction - handle both dict and Series
        transaction_date = None
        if execution_data is not None:
            try:
                # Check if execution_data is a Pandas Series
                if hasattr(execution_data, 'get') and not hasattr(execution_data, 'iloc'):
                    # Dictionary style access
                    transaction_date = execution_data.get('DataDate')
                elif hasattr(execution_data, 'iloc'):
                    # Series style access
                    transaction_date = execution_data['DataDate'] if 'DataDate' in execution_data.index else None
                else:
                    transaction_date = None
            except (TypeError, ValueError) as e:
                # Log the error but continue with None date
                if self.logger:
                    self.logger.warning(f"Error extracting transaction date: {e}")
                transaction_date = None

        transaction = {
            'date': transaction_date,
            'action': 'SELL' if self.is_short else 'BUY',
            'contracts': quantity,
            'price': price,
            'value': price * quantity * 100
        }
        self.transactions.append(transaction)

        # Calculate weighted average entry price
        old_value = self.avg_entry_price * self.contracts
        new_value = price * quantity
        total_contracts = self.contracts + quantity

        if total_contracts > 0:
            self.avg_entry_price = (old_value + new_value) / total_contracts

        # Update contract count
        old_contracts = self.contracts
        self.contracts = total_contracts

        # Log the transaction
        if self.logger:
            self.logger.info(f"[Position] Added {quantity} contracts of {self.option_symbol} at ${price:.2f}")
            self.logger.info(f"  Previous: {old_contracts} at ${self.avg_entry_price:.2f}")
            self.logger.info(f"  New position: {self.contracts} contracts at avg price ${self.avg_entry_price:.2f}")

        # Update total trade value (for accounting)
        self.total_value = self.avg_entry_price * self.contracts * 100

        return self.avg_entry_price

    def remove_contracts(self, quantity, price, execution_data=None, reason="Close"):
        """
        Remove contracts from position and calculate realized PnL.

        Args:
            quantity: Number of contracts to remove
            price: Current price per contract
            execution_data: Additional data about the execution
            reason: Reason for closing (e.g., "Profit Target", "Stop Loss")

        Returns:
            float: Realized PnL for this closure
        """
        if quantity <= 0 or quantity > self.contracts:
            return 0

        # Store current average entry price
        entry_price = self.avg_entry_price

        # Calculate realized PnL
        # For short positions: entry_price - exit_price is the P&L
        # For long positions: exit_price - entry_price is the P&L
        if self.is_short:
            pnl = (entry_price - price) * quantity * 100
        else:
            pnl = (price - entry_price) * quantity * 100

        # Record transaction - handle different types properly
        transaction_date = None

        # Safely extract transaction date from execution_data
        if execution_data is not None:
            try:
                # Check if execution_data is a dictionary-like object
                if hasattr(execution_data, 'get') and not hasattr(execution_data, 'iloc'):
                    transaction_date = execution_data.get('DataDate')
                # Check if execution_data is a pandas Series
                elif hasattr(execution_data, 'iloc'):
                    transaction_date = execution_data['DataDate'] if 'DataDate' in execution_data.index else None
                else:
                    transaction_date = None
            except (TypeError, ValueError) as e:
                # Log the error but continue with None date
                if self.logger:
                    self.logger.warning(f"Error extracting transaction date: {e}")
                transaction_date = None

        transaction = {
            'date': transaction_date,
            'action': 'BUY' if self.is_short else 'SELL',
            'contracts': quantity,
            'price': price,
            'pnl': pnl,
            'reason': reason
        }
        self.transactions.append(transaction)

        # Update realized PnL total
        self.realized_pnl += pnl

        # Update contract count
        old_contracts = self.contracts
        self.contracts -= quantity

        # Reset average price if position closed completely
        if self.contracts == 0:
            self.avg_entry_price = 0
            self.total_value = 0
        else:
            # No need to adjust avg_entry_price when removing contracts
            self.total_value = self.avg_entry_price * self.contracts * 100

        # Log the transaction
        if self.logger:
            self.logger.info(f"[Position] Removed {quantity} contracts of {self.option_symbol} at ${price:.2f}")
            self.logger.info(f"  Entry price: ${entry_price:.2f}, Exit price: ${price:.2f}")
            self.logger.info(
                f"  P&L: ${pnl:.2f} ({'+' if pnl >= 0 else ''}{pnl / (entry_price * quantity * 100) * 100:.2f}%)")
            self.logger.info(f"  Remaining: {self.contracts} contracts")

        return pnl

    def update_market_data(self, market_data):
        """
        Update position with latest market data and calculate unrealized PnL.

        Args:
            market_data: Current market data for this option

        Returns:
            float: Updated unrealized PnL
        """
        # Store the data
        try:
            # If market_data has a copy method (DataFrame or Series), use it
            if hasattr(market_data, 'copy'):
                self.daily_data.append(market_data.copy())
            else:
                # Otherwise just append the data as is
                self.daily_data.append(market_data)
        except Exception as e:
            # If there's an error, just append the data as is and log the error
            self.daily_data.append(market_data)
            if self.logger:
                self.logger.warning(f"Error copying market data: {e}")

        # Update key metrics - handle different types of market_data
        try:
            # Check if market_data is a dictionary-like object
            if hasattr(market_data, 'get') and not hasattr(market_data, 'iloc'):
                # Dictionary style access
                self.current_price = market_data.get('MidPrice', 0)
                self.current_delta = market_data.get('Delta', 0)
                self.current_gamma = market_data.get('Gamma', 0)
                self.current_theta = market_data.get('Theta', 0)
                self.current_vega = market_data.get('Vega', 0)
                self.underlying_price = market_data.get('UnderlyingPrice', 0)

                # Calculate days to expiry if possible
                if 'DataDate' in market_data and self.expiration:
                    self.days_to_expiry = (self.expiration - market_data.get('DataDate')).days
            # Check if market_data is a pandas Series
            elif hasattr(market_data, 'iloc'):
                # Series style access
                self.current_price = market_data['MidPrice'] if 'MidPrice' in market_data.index else 0
                self.current_delta = market_data['Delta'] if 'Delta' in market_data.index else 0
                self.current_gamma = market_data['Gamma'] if 'Gamma' in market_data.index else 0
                self.current_theta = market_data['Theta'] if 'Theta' in market_data.index else 0
                self.current_vega = market_data['Vega'] if 'Vega' in market_data.index else 0
                self.underlying_price = market_data['UnderlyingPrice'] if 'UnderlyingPrice' in market_data.index else 0

                # Calculate days to expiry if possible
                if 'DataDate' in market_data.index and self.expiration is not None:
                    self.days_to_expiry = (self.expiration - market_data['DataDate']).days
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error updating market data: {e}")

        # Calculate unrealized PnL (positive values mean profit)
        if self.contracts > 0:
            if self.is_short:
                # For short options: entry_price - current_price is the P&L per unit
                # If current_price goes down, that's profit for short options
                pnl_per_contract = self.avg_entry_price - self.current_price
            else:
                # For long options: current_price - entry_price is the P&L per unit
                pnl_per_contract = self.current_price - self.avg_entry_price

            # Calculate total PnL based on contract size and count
            self.unrealized_pnl = pnl_per_contract * self.contracts * 100
        else:
            self.unrealized_pnl = 0

        # Update max drawdown tracking (drawdown is negative PnL)
        if self.unrealized_pnl < -self.max_drawdown:
            self.max_drawdown = -self.unrealized_pnl

        return self.unrealized_pnl

    def calculate_margin_requirement(self, max_leverage):
        """
        Calculate margin requirement for this position, accounting for unrealized PnL.
        Initial margin is based on entry price, and unrealized PnL (both gains and losses) is factored in.

        Args:
            max_leverage: Maximum leverage allowed

        Returns:
            float: Margin requirement in dollars
        """
        if self.contracts <= 0:
            return 0

        # Initial margin calculation based on entry price
        initial_margin = self.avg_entry_price * self.contracts * 100 * max_leverage

        # Adjust margin for unrealized PnL (both gains and losses)
        # For short positions, add losses (negative PnL) and subtract gains (positive PnL)
        # For long positions, the same logic applies

        # Calculate adjusted margin: initial margin +/- unrealized PnL
        adjusted_margin = initial_margin - self.unrealized_pnl

        # No safety floor - return directly adjusted margin
        return adjusted_margin

    def get_greeks(self):
        """
        Get position Greeks

        Returns:
            dict: Dictionary of position Greeks
        """
        sign = -1 if self.is_short else 1
        return {
            'delta': sign * self.current_delta * self.contracts,
            'gamma': sign * self.current_gamma * self.contracts,
            # Remove the extra 100 multiplier for theta and vega if market data is already per contract
            'theta': sign * self.current_theta * self.contracts,
            'vega': sign * self.current_vega * self.contracts,
            'dollar_delta': sign * self.current_delta * self.contracts * self.underlying_price * 100,
            'dollar_gamma': sign * self.current_gamma * self.contracts * (self.underlying_price ** 2) * 0.01,
            'dollar_theta': sign * self.current_theta * self.contracts,  # previously multiplied by 100
            'dollar_vega': sign * self.current_vega * self.contracts  # previously multiplied by 100
        }

    def get_position_summary(self):
        """
        Return a summary of the position for reporting

        Returns:
            dict: Position summary dictionary
        """
        return {
            'symbol': self.option_symbol,
            'contracts': self.contracts,
            'entry_price': self.avg_entry_price,
            'current_price': self.current_price,
            'underlying_price': self.underlying_price,
            'value': self.current_price * self.contracts * 100,
            'delta': self.current_delta,
            'gamma': self.current_gamma,
            'theta': self.current_theta,
            'vega': self.current_vega,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'dte': self.days_to_expiry,
            'is_short': self.is_short
        }

    def set_cpd_reference(self, reference_symbol, reference_price, reference_delta, reference_expiry):
        """
        Set the CPD reference information for this position

        Args:
            reference_symbol: Reference option symbol
            reference_price: Reference option price
            reference_delta: Reference option delta
            reference_expiry: Reference option expiration

        Returns:
            float: Calculated effective contracts
        """
        self.cpd_reference_symbol = reference_symbol
        self.cpd_reference_price = reference_price
        self.cpd_reference_delta = reference_delta
        self.cpd_reference_expiry = reference_expiry

        # Calculate effective contracts based on price ratio - only done once
        if reference_price > 0:
            price_ratio = self.avg_entry_price / reference_price
            self.cpd_effective_contracts = self.contracts * price_ratio

            self.logger.info(
                f"[CPD Reference] Position {self.option_symbol}: Set reference price ${reference_price:.2f}")
            self.logger.info(f"  Price ratio: {price_ratio:.2f}")
            self.logger.info(f"  Effective contracts: {self.cpd_effective_contracts:.2f}")

            return self.cpd_effective_contracts

        return 0

    def update_cpd_effective_contracts_for_closure(self, closed_contracts):
        """
        Update effective contracts when some contracts are closed

        Args:
            closed_contracts: Number of contracts closed

        Returns:
            float: Change in effective contracts
        """
        if self.contracts <= 0 or closed_contracts <= 0:
            return 0

        # Calculate the proportion of contracts being closed
        proportion_closed = closed_contracts / (self.contracts + closed_contracts)

        # Calculate the effective contracts being reduced
        reduced_effective = self.cpd_effective_contracts * proportion_closed

        # Update the effective contracts
        self.cpd_effective_contracts -= reduced_effective

        self.logger.info(f"[CPD Reference] Position {self.option_symbol}: Reduced effective contracts")
        self.logger.info(f"  Closed {closed_contracts} contracts ({proportion_closed:.2%})")
        self.logger.info(f"  Reduced effective contracts by {reduced_effective:.2f}")
        self.logger.info(f"  Remaining effective contracts: {self.cpd_effective_contracts:.2f}")

        return reduced_effective


class PortfolioRebalancer:
    """
    Handles portfolio rebalancing based on risk scaling and margin constraints.
    """

    def __init__(self, config, position_manager, risk_manager, cpd_hedge_manager, logger=None):
        """Initialize the PortfolioRebalancer"""
        self.config = config
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.cpd_hedge_manager = cpd_hedge_manager
        self.logger = logger or logging.getLogger('theta_engine')

        # New parameters for margin management
        self.margin_buffer_pct = config.get('margin_buffer_pct', 0.10)  # 10% buffer by default
        self.negative_margin_threshold = config.get('negative_margin_threshold', -0.05)  # -5% of NLV
        self.rebalance_cooldown_days = config.get('rebalance_cooldown_days', 3)  # Cooldown period for rebalancing
        self.last_rebalance_date = None
        self.forced_rebalance_threshold = config.get('forced_rebalance_threshold', -0.10)  # -10% of NLV

    def analyze_portfolio(self, risk_scaling, current_date=None):
        """
        Analyze portfolio for rebalancing needs based on risk scaling with improved logic.
        Distinguishes between scaling rebalance and negative margin rebalance.

        Args:
            risk_scaling: Risk scaling factor
            current_date: Current date (for cooldown tracking)

        Returns:
            dict: Analysis results
        """
        metrics = self.position_manager.get_portfolio_metrics()
        current_nlv = metrics['net_liquidation_value']
        current_margin = metrics['total_margin']
        available_margin = metrics['available_margin']

        # Maximum margin allowed based on risk scaling with buffer
        max_margin = risk_scaling * current_nlv * (1 + self.margin_buffer_pct)

        # Calculate negative margin as percentage of NLV
        negative_margin_pct = available_margin / current_nlv if current_nlv > 0 else 0

        # Check if we're in cooldown period - but only for negative margin rebalances
        in_cooldown = False
        days_since_rebalance = None

        if current_date and self.last_rebalance_date:
            days_since_rebalance = (current_date - self.last_rebalance_date).days
            # Only consider cooldown for negative margin situations
            if available_margin < 0:
                in_cooldown = days_since_rebalance < self.rebalance_cooldown_days

                # Always log cooldown status at INFO level for better visibility
                if in_cooldown:
                    self.logger.info(
                        f"[Portfolio Rebalancer] In rebalance cooldown period for negative margin: day {days_since_rebalance} of {self.rebalance_cooldown_days}")
                    cooldown_end = self.last_rebalance_date + pd.Timedelta(days=self.rebalance_cooldown_days)
                    self.logger.info(f"[Portfolio Rebalancer] Cooldown ends: {cooldown_end.strftime('%Y-%m-%d')}")

        # Determine if rebalancing is needed
        needs_rebalancing = False
        needs_scaling_rebalance = False
        needs_negative_margin_rebalance = False

        # Force rebalancing if margin is extremely negative (regardless of cooldown)
        if negative_margin_pct < self.forced_rebalance_threshold:
            needs_rebalancing = True
            needs_negative_margin_rebalance = True
            self.logger.info(
                f"[Portfolio Rebalancer] CRITICAL: Forced rebalancing triggered - Extremely negative margin ({negative_margin_pct:.2%} of NLV)")

        # Check for regular rebalancing conditions
        else:
            # Scaling rebalancing - always performed regardless of cooldown
            if current_margin > max_margin:
                needs_rebalancing = True
                needs_scaling_rebalance = True
                self.logger.info(
                    f"[Portfolio Rebalancer] Margin exceeds risk-adjusted maximum with buffer: ${current_margin:.0f} > ${max_margin:.0f}")

            # Negative margin rebalancing - subject to cooldown
            elif negative_margin_pct < self.negative_margin_threshold and not in_cooldown:
                needs_rebalancing = True
                needs_negative_margin_rebalance = True
                self.logger.info(
                    f"[Portfolio Rebalancer] Negative margin ({negative_margin_pct:.2%} of NLV) below threshold ({self.negative_margin_threshold:.2%})")

            # Negative margin but in cooldown
            elif negative_margin_pct < self.negative_margin_threshold and in_cooldown:
                self.logger.info(
                    f"[Portfolio Rebalancer] Negative margin detected (${available_margin:.0f}) but in rebalance cooldown period ({days_since_rebalance}/{self.rebalance_cooldown_days} days)")

        # Always log margin metrics at INFO level for better visibility
        self.logger.info("[Portfolio Rebalancer Analysis]")
        self.logger.info(f"  Current NLV: ${current_nlv:.0f}")
        self.logger.info(f"  Risk Scaling Factor: {risk_scaling:.2f}")
        self.logger.info(f"  Maximum Margin Allowed: ${max_margin:.0f} (with {self.margin_buffer_pct:.0%} buffer)")
        self.logger.info(f"  Current Margin: ${current_margin:.0f}")
        self.logger.info(f"  Available Margin: ${available_margin:.0f} ({negative_margin_pct:.2%} of NLV)")

        # Calculate excess margin
        excess_margin = current_margin - max_margin if current_margin > max_margin else 0

        # If we need rebalancing but are in cooldown, log a clear message
        if current_margin > max_margin:
            self.logger.info(
                f"[Portfolio Rebalancer] Risk scaling rebalance needed due to excess margin (${excess_margin:.0f})")

        # If negative margin but we're in cooldown, log a clear message
        if available_margin < 0 and in_cooldown:
            self.logger.info(
                f"[Portfolio Rebalancer] Would need rebalancing due to negative margin (${available_margin:.0f}), but in cooldown period")

        return {
            'needs_rebalancing': needs_rebalancing,
            'needs_scaling_rebalance': needs_scaling_rebalance,
            'needs_negative_margin_rebalance': needs_negative_margin_rebalance,
            'current_nlv': current_nlv,
            'current_margin': current_margin,
            'max_margin': max_margin,
            'excess_margin': excess_margin,
            'available_margin': available_margin,
            'negative_margin_pct': negative_margin_pct,
            'in_cooldown': in_cooldown,
            'days_since_rebalance': days_since_rebalance
        }

    def rebalance_portfolio(self, current_date, market_data_by_symbol):
        """
        Rebalance portfolio based on risk parameters with improved logic.
        Cooldown only applies to negative margin rebalances, not to risk scaling rebalances.

        Args:
            current_date: Current date
            market_data_by_symbol: Market data dictionary
        """
        # Calculate risk scaling factor
        risk_scaling, current_sharpe, z_score = self.risk_manager.calculate_risk_scaling(
            self.position_manager.daily_returns
        )

        # Get current portfolio metrics
        analysis = self.analyze_portfolio(risk_scaling, current_date)

        # If no rebalancing needed, return
        if not analysis['needs_rebalancing']:
            if analysis.get('in_cooldown', False):
                self.logger.info(
                    f"[PortfolioRebalancer] In cooldown period ({analysis.get('days_since_rebalance', 0)}/{self.rebalance_cooldown_days} days) - skipping rebalance")
            else:
                self.logger.info("[PortfolioRebalancer] No rebalancing needed per analysis")
            return

        # Get current metrics from the analysis
        net_liq = analysis['current_nlv']
        current_margin = analysis['current_margin']
        current_exposure = current_margin / net_liq if net_liq > 0 else 0
        available_margin = analysis['available_margin']
        negative_margin_pct = analysis.get('negative_margin_pct', 0)

        # Clear reason text and set flags for different rebalance types
        reason_text = []
        is_negative_margin_rebalance = False
        is_scaling_rebalance = False

        # Record why rebalancing was triggered
        if current_margin > analysis['max_margin']:
            reason_text.append(
                f"Current margin (${current_margin:,.0f}) exceeds maximum allowed (${analysis['max_margin']:,.0f})")
            is_scaling_rebalance = True

        if available_margin < 0:
            reason_text.append(f"Negative margin (${available_margin:,.0f}, {negative_margin_pct:.2%} of NLV)")
            is_negative_margin_rebalance = True

            # Check if this is a forced rebalance due to extreme negative margin
            if negative_margin_pct < self.forced_rebalance_threshold:
                reason_text.append(
                    f"FORCED REBALANCE: Extreme negative margin below threshold ({self.forced_rebalance_threshold:.2%})")

        # Log detailed rebalancing analysis
        self.logger.info("\n[Portfolio Rebalancing Analysis]")
        self.logger.info(f"  Current NLV: ${net_liq:,.0f}")
        self.logger.info(f"  Current Margin: ${current_margin:,.0f}")
        self.logger.info(f"  Current Exposure: {current_exposure:.2f}x")
        self.logger.info(f"  Risk Scaling Factor: {risk_scaling:.2f}")
        self.logger.info(f"  Maximum Margin Allowed: ${analysis['max_margin']:,.0f}")
        self.logger.info(f"  Available Margin: ${available_margin:,.0f}")
        self.logger.info(f"  Rebalancing triggered by: {', '.join(reason_text)}")
        self.logger.info(
            f"  Scaling rebalance: {is_scaling_rebalance}, Negative margin rebalance: {is_negative_margin_rebalance}")

        # Calculate target for reduction - more conservative approach
        if current_margin > analysis['max_margin']:
            # Use a smaller reduction target when we're just above max margin
            # This helps avoid over-reducing positions
            excess_pct = (current_margin - analysis['max_margin']) / analysis['max_margin']
            if excess_pct < 0.10:  # If less than 10% over max margin
                # Reduce just enough to get slightly below max margin
                excess_margin = (current_margin - analysis['max_margin']) * 1.1  # Add 10% buffer to the reduction
            else:
                # For larger excesses, be more aggressive
                excess_margin = (current_margin - analysis['max_margin']) * 1.2  # Add 20% buffer to the reduction
        elif available_margin < 0:
            # For negative available margin, calculate a moderate reduction target
            # aiming to restore a small positive margin buffer
            target_buffer = net_liq * 0.02  # Target 2% of NLV as buffer
            excess_margin = -available_margin + target_buffer
        else:
            excess_margin = 0

        self.logger.info(f"  Target Margin Reduction: ${excess_margin:,.0f}")

        # If no margin reduction needed, exit
        if excess_margin <= 0:
            self.logger.info("  No margin reduction needed.")
            return

        # Log the positions before rebalancing
        self.logger.info("\n[Positions Before Rebalancing]")
        header_len = 130  # Increased length to accommodate all columns
        self.logger.info("-" * header_len)
        self.logger.info(
            f"{'Symbol':<16} {'Contracts':>9} {'Entry':>8} {'Current':>8} {'PnL':>10} {'PnL %':>8} {'Margin':>12} {'DTE':>5} {'Delta':>8}")
        self.logger.info("-" * header_len)

        position_data = []
        for symbol, position in self.position_manager.positions.items():
            position_margin = position.calculate_margin_requirement(self.risk_manager.max_leverage)

            # Calculate PnL percentage
            if position.avg_entry_price > 0:
                if position.is_short:
                    pnl_pct = (position.avg_entry_price - position.current_price) / position.avg_entry_price * 100
                else:
                    pnl_pct = (position.current_price - position.avg_entry_price) / position.avg_entry_price * 100
            else:
                pnl_pct = 0

            position_data.append({
                'symbol': symbol,
                'position': position,
                'margin': position_margin,
                'dte': position.days_to_expiry,
                'pnl_pct': pnl_pct,
                'delta': position.current_delta,
                'unrealized_pnl': position.unrealized_pnl
            })

            self.logger.info(f"{symbol:<16} {position.contracts:>9} "
                             f"${position.avg_entry_price:>7.2f} ${position.current_price:>7.2f} "
                             f"${position.unrealized_pnl:>9.2f} {pnl_pct:>7.1f}% "
                             f"${position_margin:>11,.0f} {position.days_to_expiry:>5} {position.current_delta:>8.3f}")

        self.logger.info("-" * header_len)

        # Improved position selection strategy
        # 1. First, sort by PnL (largest negative first)
        # 2. For positions with similar PnL, sort by DTE (lowest first)
        sorted_positions = sorted(position_data, key=lambda item: (
            item['unrealized_pnl'] if item['unrealized_pnl'] < 0 else 9999999,  # Sort negative PnL first
            item['dte']  # Then by DTE for positions with similar PnL
        ))

        # Log the reduction plan
        self.logger.info("\n[Reduction Plan]")
        self.logger.info(f"  Target Margin Reduction: ${excess_margin:,.0f}")
        self.logger.info(f"  Reduction Strategy: Prioritize losing positions, then by DTE")

        # Track positions and contracts to close
        positions_to_close = []

        # Reduce positions until excess margin is eliminated
        remaining_excess = excess_margin
        for position_info in sorted_positions:
            if remaining_excess <= 0:
                break

            symbol = position_info['symbol']
            position = position_info['position']
            position_margin = position_info['margin']
            margin_per_contract = position_margin / position.contracts if position.contracts > 0 else 0

            if margin_per_contract <= 0:
                continue

            # Calculate what percentage of the position we need to close
            # More conservative approach - don't close more than 25% at a time unless urgently needed
            max_pct_to_close = self.config.get('max_position_reduction_pct', 0.25)  # Default to 25%

            # For positions in loss, we might want to close a larger percentage
            if position_info['unrealized_pnl'] < 0:
                pnl_severity = abs(position_info['unrealized_pnl']) / (
                            position.avg_entry_price * position.contracts * 100)
                # Increase max closure percentage based on loss severity
                if pnl_severity > 0.1:  # If losing more than 10% of position value
                    losing_max_reduction = self.config.get('losing_position_max_reduction_pct', 0.40)  # Default to 40%
                    max_pct_to_close = min(losing_max_reduction,
                                           pnl_severity * 2)  # Close up to configured max based on loss severity

            # For extremely negative margin situations, allow closing up to the urgent reduction percentage
            if negative_margin_pct < self.forced_rebalance_threshold:
                max_pct_to_close = self.config.get('urgent_reduction_pct', 0.50)  # Default to 50%

            # Calculate contracts to close
            contracts_needed_for_full_reduction = math.ceil(remaining_excess / margin_per_contract)
            contracts_allowed_by_pct = math.ceil(position.contracts * max_pct_to_close)

            # Don't close more than needed or allowed by percentage
            contracts_to_close = min(contracts_needed_for_full_reduction, contracts_allowed_by_pct, position.contracts)

            if contracts_to_close <= 0:
                continue

            # Calculate margin freed
            margin_freed = contracts_to_close * margin_per_contract

            # Store position to close
            positions_to_close.append({
                'symbol': symbol,
                'contracts': contracts_to_close,
                'margin_freed': margin_freed,
                'full_close': contracts_to_close == position.contracts,
                'pnl': position_info['unrealized_pnl'] * (contracts_to_close / position.contracts)
            })

            # Update remaining excess margin
            remaining_excess -= margin_freed

            # Log this reduction
            self.logger.info(
                f"  Position {symbol}: Close {contracts_to_close} of {position.contracts} contracts ({contracts_to_close / position.contracts:.0%})")
            self.logger.info(f"    Margin per contract: ${margin_per_contract:,.0f}")
            self.logger.info(f"    Margin freed: ${margin_freed:,.0f}")
            self.logger.info(
                f"    Unrealized P&L: ${position_info['unrealized_pnl']:.0f} ({position_info['pnl_pct']:.1f}%)")
            self.logger.info(f"    Remaining excess margin: ${remaining_excess:,.0f}")

        # Execute the position closures
        self.logger.info("\n[Executing Position Closures]")
        total_margin_freed = 0
        total_pnl = 0

        for close_info in positions_to_close:
            symbol = close_info['symbol']
            contracts = close_info['contracts']

            # Skip if no market data for this symbol
            if symbol not in market_data_by_symbol:
                self.logger.warning(f"  Warning: No market data available for {symbol}, skipping closure")
                continue

            # Get the position
            position = self.position_manager.get_position(symbol)

            # Double check that position still exists and has contracts
            if position is None or position.contracts == 0:
                self.logger.info(f"  Warning: Position {symbol} no longer exists or has no contracts")
                continue

            # Before closing, log CPD effective contracts
            if hasattr(position, 'cpd_effective_contracts') and position.cpd_effective_contracts > 0:
                self.logger.info(
                    f"  [CPD Before] Position {symbol}: {position.cpd_effective_contracts:.2f} effective contracts")

            # Store original entry price for validation
            original_entry_price = position.avg_entry_price

            # Close the position
            pnl = self.position_manager.close_position(
                symbol,
                contracts,
                market_data_by_symbol[symbol],
                "Rebalance"
            )
            total_pnl += pnl

            # Calculate actual margin freed
            if close_info['full_close']:
                margin_freed = close_info['margin_freed']
            else:
                # Recalculate for partial closure
                remaining_position = self.position_manager.get_position(symbol)
                if remaining_position:
                    margin_before = close_info['margin_freed'] + remaining_position.calculate_margin_requirement(
                        self.risk_manager.max_leverage)
                    margin_after = remaining_position.calculate_margin_requirement(self.risk_manager.max_leverage)
                    margin_freed = margin_before - margin_after
                else:
                    # Position was completely closed despite expecting partial closure
                    margin_freed = close_info['margin_freed']

            total_margin_freed += margin_freed

            # Log the closure
            self.logger.info(f"  Closed {contracts} contracts of {symbol}")
            current_price = market_data_by_symbol[symbol]['MidPrice'] if 'MidPrice' in market_data_by_symbol[
                symbol] else 0
            self.logger.info(f"    Entry Price: ${original_entry_price:.2f}, Exit Price: ${current_price:.2f}")
            self.logger.info(f"    PnL: ${pnl:,.0f}")
            self.logger.info(f"    Margin Freed: ${margin_freed:,.0f}")

        # Summary after rebalancing
        self.logger.info("\n[Rebalancing Summary]")
        self.logger.info(f"  Total Positions Affected: {len(positions_to_close)}")
        self.logger.info(f"  Total Margin Freed: ${total_margin_freed:,.0f}")
        self.logger.info(f"  Total Realized PnL: ${total_pnl:,.0f}")

        # Recalculate metrics after rebalancing
        new_metrics = self.position_manager.get_portfolio_metrics()
        new_exposure = new_metrics['total_margin'] / new_metrics['net_liquidation_value'] if new_metrics[
                                                                                                 'net_liquidation_value'] > 0 else 0
        self.logger.info(f"  New Exposure: {new_exposure:.2f}x (Max Allowed: {risk_scaling:.2f}x)")
        self.logger.info(f"  New Available Margin: ${new_metrics['available_margin']:,.0f}")

        # Update last rebalance date ONLY if this was a negative margin rebalance
        # (Do NOT update for scaling rebalances)
        if is_negative_margin_rebalance:
            self.last_rebalance_date = current_date
            cooldown_end_date = current_date + pd.Timedelta(days=self.rebalance_cooldown_days)
            self.logger.info(
                f"  Negative margin rebalance - cooldown active until: {cooldown_end_date.strftime('%Y-%m-%d')}")
        else:
            self.logger.info(f"  Risk scaling rebalance only - no cooldown period applied")



# ========================
# RiskManager Class
# ========================
class RiskManager:
    """
    Manages position sizing and risk calculations based on performance metrics.

    This class calculates risk scaling factors based on performance metrics,
    determines appropriate position sizes, and calculates margin requirements.
    """

    def __init__(self, config, logger=None):
        """Initialize the RiskManager with configuration parameters"""
        self.config = config
        self.logger = logger or logging.getLogger('theta_engine')

        # Extract key risk parameters
        self.max_leverage = config.get('portfolio', {}).get('max_leverage', 12)
        self.max_nlv_percent = config.get('portfolio', {}).get('max_nlv_percent', 0.25)

        # Extract risk scaling parameters
        risk_config = config.get('risk', {})
        self.rolling_window = risk_config.get('rolling_window', 21)
        self.target_z = risk_config.get('target_z', 0)  # z-score at which full exposure is reached
        self.min_z = risk_config.get('min_z', -2.0)  # z-score for minimum exposure
        self.min_investment = risk_config.get('min_investment', 0.25)  # Minimum investment level

        # Track risk scaling history for analysis
        self.risk_scaling_history = []

        # Log initialization
        self.logger.info("[RiskManager] Initialized")
        self.logger.info(f"  Max Leverage: {self.max_leverage}x")
        self.logger.info(f"  Max NLV Percent: {self.max_nlv_percent:.2%}")
        self.logger.info(f"  Rolling Window: {self.rolling_window} days")
        self.logger.info(f"  Target Z-Score: {self.target_z}")
        self.logger.info(f"  Min Z-Score: {self.min_z}")
        self.logger.info(f"  Min Investment: {self.min_investment:.2%}")

    def calculate_position_size(self, option_data, portfolio_metrics, risk_scaling=1.0):
        """
        Calculate position size based on risk parameters with proper risk scaling and max_nlv_percent limit.

        Args:
            option_data: Option data with price information
            portfolio_metrics: Dictionary of portfolio metrics (NLV, available margin, etc.)
            risk_scaling: Risk scaling factor (default 1.0)

        Returns:
            int: Number of contracts to trade
        """
        net_liq = portfolio_metrics.get('net_liquidation_value', 0)
        available_margin = portfolio_metrics.get('available_margin', 0)
        current_margin = portfolio_metrics.get('total_margin', 0)

        # Handle NaN risk scaling
        if pd.isna(risk_scaling):
            self.logger.warning("[RiskManager] Risk scaling is NaN, using default value 1.0")
            risk_scaling = 1.0

        # Maximum margin allocation allowed based on risk scaling and current NLV
        max_margin_alloc = risk_scaling * net_liq

        # Calculate remaining margin capacity
        remaining_margin_capacity = max(max_margin_alloc - current_margin, 0)

        # Get option price and calculate margin per contract
        if hasattr(option_data, 'get') and not hasattr(option_data, 'iloc'):
            option_price = option_data.get('MidPrice', 0)
            option_symbol = option_data.get('OptionSymbol', 'Unknown')
        else:
            option_price = option_data['MidPrice'] if 'MidPrice' in option_data else 0
            option_symbol = option_data['OptionSymbol'] if 'OptionSymbol' in option_data else 'Unknown'

        margin_per_contract = option_price * 100 * self.max_leverage
        if margin_per_contract <= 0:
            self.logger.warning(
                f"[RiskManager] Invalid margin per contract for {option_symbol}: ${margin_per_contract:.2f}")
            return 0

        # Calculate maximum contracts based on remaining margin capacity
        capacity_max_contracts = int(remaining_margin_capacity / margin_per_contract) if margin_per_contract > 0 else 0

        # Also check against available margin (to avoid going negative)
        available_max_contracts = int(available_margin / margin_per_contract) if margin_per_contract > 0 else 0

        # Apply max_nlv_percent limit - maximum position size as percentage of NLV
        max_position_margin = net_liq * self.max_nlv_percent
        max_position_contracts = int(max_position_margin / margin_per_contract) if margin_per_contract > 0 else 0

        # Ensure all values are non-negative
        capacity_max_contracts = max(capacity_max_contracts, 0)
        available_max_contracts = max(available_max_contracts, 0)
        max_position_contracts = max(max_position_contracts, 0)

        # Take the most conservative (lowest) limit
        contracts = min(capacity_max_contracts, available_max_contracts, max_position_contracts)

        # NEW: Override when capacity exists but NLV percent would prevent any trading
        if (contracts == 0 and max_position_contracts == 0 and
                min(capacity_max_contracts, available_max_contracts) >= 1):
            contracts = 1
            self.logger.info(
                f"[Position Sizing] Override: Max NLV percent would yield 0 contracts, but capacity exists. Setting to 1 contract.")

        # Ensure at least the minimum position size if adding any contracts
        min_position_size = self.config.get('strategy', {}).get('min_position_size', 1)
        if contracts > 0 and contracts < min_position_size:
            contracts = min_position_size

        # Enhanced logging including the max_nlv_percent constraint
        self.logger.info(f"[Position Sizing] Option: {option_symbol}")
        self.logger.info(f"  Price: ${option_price:.2f}, Margin per contract: ${margin_per_contract:.2f}")
        self.logger.info(
            f"  NLV: ${net_liq:.2f}, Maximum Margin: ${max_margin_alloc:.2f}, Current Margin: ${current_margin:.2f}")
        self.logger.info(
            f"  Remaining Capacity: ${remaining_margin_capacity:.2f}, Available Margin: ${available_margin:.2f}")
        self.logger.info(
            f"  Max NLV Percent: {self.max_nlv_percent:.2%}, Position limit: {max_position_contracts} contracts")
        self.logger.info(
            f"  Capacity limit: {capacity_max_contracts} contracts, Available margin limit: {available_max_contracts} contracts")
        self.logger.info(f"  Risk scaling: {risk_scaling:.2f}")
        self.logger.info(f"  Final position size: {contracts} contracts")

        return contracts

    def calculate_margin_requirement(self, position):
        """
        Calculate margin required for a position

        Args:
            position: Position object

        Returns:
            float: Margin requirement in dollars
        """
        return position.calculate_margin_requirement(self.max_leverage)

    def calculate_risk_scaling(self, returns):
        """
        Calculate risk scaling factor based on Sharpe ratio z-score.

        Args:
            returns: List of daily return dictionaries with 'date' and 'return' keys

        Returns:
            tuple: (scaling_factor, current_sharpe, z_score)
        """
        # If no returns data, use neutral scaling
        if not returns or len(returns) == 0:
            self.logger.info("[Risk Scaling] No returns data, using neutral scaling (1.0)")
            return 1.0, None, None

        # Convert to pandas Series for statistical calculations
        returns_series = pd.Series([entry['return'] for entry in returns])

        # Handle different window sizes based on available data
        if len(returns_series) < self.rolling_window:
            # If we have fewer observations than window, use expanding window
            self.logger.debug(
                f"[Risk Scaling] Using expanding window (n={len(returns_series)}) instead of rolling window ({self.rolling_window})")

            # Calculate annualized mean and standard deviation
            mean_val = returns_series.mean() * 252  # Annualize
            std_val = returns_series.std() * np.sqrt(252)  # Annualize

            # Calculate current Sharpe ratio
            current_sharpe = mean_val / std_val if std_val != 0 else np.nan

            # Calculate historical Sharpe series using expanding window
            sharpe_series = returns_series.expanding(min_periods=1).apply(
                lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() != 0 else np.nan
            )
        else:
            # Use rolling window as configured
            self.logger.debug(f"[Risk Scaling] Using {self.rolling_window}-day rolling window")

            # Calculate rolling statistics
            rolling_mean = returns_series.rolling(self.rolling_window).mean() * 252
            rolling_std = returns_series.rolling(self.rolling_window).std() * np.sqrt(252)

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
        curr_sharpe_str = f"{current_sharpe:.2f}" if (
                    current_sharpe is not None and not pd.isna(current_sharpe)) else "N/A"
        hist_mean_str = f"{hist_mean:.2f}" if (hist_mean is not None and not pd.isna(hist_mean)) else "N/A"
        hist_std_str = f"{hist_std:.2f}" if (hist_std is not None and not pd.isna(hist_std)) else "N/A"
        z_str = f"{z_score:.2f}" if (z_score is not None and not pd.isna(z_score)) else "N/A"

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

        # Store risk scaling history
        self.risk_scaling_history.append({
            'date': returns[-1]['date'] if returns else None,
            'sharpe': current_sharpe if not pd.isna(current_sharpe) else 0,
            'z_score': z_score if not pd.isna(z_score) else 0,
            'scaling': scaling,
            'hist_mean': hist_mean if not pd.isna(hist_mean) else 0,
            'hist_std': hist_std if not pd.isna(hist_std) else 0
        })

        return scaling, current_sharpe, z_score

    def get_target_exposure(self, risk_scaling):
        """
        Calculate the target exposure based on risk scaling

        Args:
            risk_scaling: Risk scaling factor

        Returns:
            float: Target exposure as a multiple of net liquidation value
        """
        return self.max_leverage * risk_scaling

    def get_exposure_bounds(self, target_exposure, tolerance=0.1):
        """
        Calculate acceptable exposure bounds based on target exposure

        Args:
            target_exposure: Target exposure level
            tolerance: Tolerance percentage (default 0.1 = 10%)

        Returns:
            tuple: (min_exposure, max_exposure)
        """
        min_exposure = target_exposure * (1 - tolerance)
        max_exposure = target_exposure * (1 + tolerance)
        return min_exposure, max_exposure

    def calculate_multi_window_sharpes(self, returns_series):
        """
        Calculate Sharpe ratios for multiple rolling windows

        Args:
            returns_series: Series of daily returns

        Returns:
            dict: Sharpe ratios for short, medium, and long windows
        """
        if len(returns_series) < 21:  # Need at least some data for shortest window
            return None

        # Get window sizes from config
        risk_config = self.config.get('risk', {})
        short_window = risk_config.get('short_window', 21)
        medium_window = risk_config.get('medium_window', 63)
        long_window = risk_config.get('long_window', 252)

        # Calculate Sharpes for each window
        result = {}

        # Short window (if enough data)
        if len(returns_series) >= short_window:
            short_mean = returns_series.rolling(short_window).mean() * 252
            short_std = returns_series.rolling(short_window).std() * np.sqrt(252)
            short_sharpe = short_mean.iloc[-1] / short_std.iloc[-1] if short_std.iloc[-1] != 0 else np.nan
            result['short'] = {'window': short_window, 'sharpe': short_sharpe}

        # Medium window (if enough data)
        if len(returns_series) >= medium_window:
            med_mean = returns_series.rolling(medium_window).mean() * 252
            med_std = returns_series.rolling(medium_window).std() * np.sqrt(252)
            med_sharpe = med_mean.iloc[-1] / med_std.iloc[-1] if med_std.iloc[-1] != 0 else np.nan
            result['medium'] = {'window': medium_window, 'sharpe': med_sharpe}

        # Long window (if enough data)
        if len(returns_series) >= long_window:
            long_mean = returns_series.rolling(long_window).mean() * 252
            long_std = returns_series.rolling(long_window).std() * np.sqrt(252)
            long_sharpe = long_mean.iloc[-1] / long_std.iloc[-1] if long_std.iloc[-1] != 0 else np.nan
            result['long'] = {'window': long_window, 'sharpe': long_sharpe}

        return result


class CPDHedgeManager:
    """
    Manages Constant Portfolio Delta (CPD) hedging for the Theta Engine strategy.

    This class handles finding reference contracts, calculating effective contract counts,
    and determining appropriate hedge positions.
    """

    def __init__(self, config, risk_manager, position_manager=None, logger=None):
        """
        Initialize the CPD Hedge Manager

        Args:
            config: Strategy configuration
            risk_manager: RiskManager instance
            position_manager: PositionManager instance (needed for ratio mode)
            logger: Logger instance
        """
        self.config = config
        self.risk_manager = risk_manager
        self.position_manager = position_manager  # Store reference to position manager for ratio mode
        self.logger = logger or logging.getLogger('theta_engine')

        # Extract key CPD parameters
        self.enable_hedging = config.get('enable_hedging', False)
        self.hedge_mode = config.get('hedge_mode', 'constant')
        self.constant_portfolio_delta = config.get('constant_portfolio_delta', 0.05)
        self.hedge_target_ratio = config.get('hedge_target_ratio', 1.75)
        self.delta_tolerance = config.get('delta_tolerance', 0.2)

        # Track reference contracts and effective contract counts
        self.reference_contracts = {}  # {expiry: {symbol: contract_data}}
        self.position_effective_contracts = {}  # {symbol: effective_contract_count}
        self.total_effective_contracts = 0

        # Log initialization
        self.logger.info("[CPDHedgeManager] Initialized")
        self.logger.info(f"  Hedging Enabled: {self.enable_hedging}")
        self.logger.info(f"  Hedge Mode: {self.hedge_mode}")
        if self.hedge_mode.lower() == 'constant':
            self.logger.info(f"  Target CPD: {self.constant_portfolio_delta}")
        elif self.hedge_mode.lower() == 'ratio':
            self.logger.info(f"  Target Dollar Delta/NLV Ratio: {self.hedge_target_ratio}%")
        self.logger.info(f"  Delta Tolerance: {self.delta_tolerance}")

    def find_reference_contract(self, daily_data, target_delta=None, position_expiry=None):
        """
        Find an appropriate reference contract with delta close to target,
        preferably with the same expiration as the position.

        The reference contract type is determined by the delta_target sign:
        - For positive delta_target (trading calls): Use calls for reference
        - For negative delta_target (trading puts): Use puts for reference

        The sign of the constant_portfolio_delta determines if hedge is long or short.

        Args:
            daily_data: Daily option data
            target_delta: Target delta value (if None, uses constant_portfolio_delta from config)
            position_expiry: Expiration date of the position (optional)

        Returns:
            Series or dict: Reference contract data or None if not found
        """
        # If target_delta is not specified, use absolute value of constant_portfolio_delta
        if target_delta is None:
            # Get the strategy's delta_target to determine option type
            strategy_delta_target = self.config.get('delta_target', 0)

            # Use the absolute value of CPD, but preserve sign for logging
            cpd = self.constant_portfolio_delta

            # Determine the option type based on the strategy's delta_target
            option_type = 'call' if strategy_delta_target > 0 else 'put'

            # Use the absolute value of CPD for search, but maintain the sign for reference
            search_delta = abs(cpd)
            if option_type == 'put':
                search_delta = -search_delta  # Puts have negative delta

            target_delta = search_delta

        self.logger.info(f"[DEBUG] find_reference_contract called with target_delta={target_delta}")

        # Determine option type based on the sign of target_delta
        option_type = 'call' if target_delta > 0 else 'put'

        # Convert daily_data to DataFrame if it's not already
        if isinstance(daily_data, dict):
            # Create a list of Series for all items in the dict
            data_list = list(daily_data.values())
            if data_list:
                daily_data = pd.DataFrame(data_list)
            else:
                self.logger.warning("[DEBUG] Daily data dictionary is empty")
                return None

        if daily_data is None or len(daily_data) == 0:
            self.logger.warning("[CPDHedgeManager] No data available for reference contract selection")
            return None

        # Log daily_data structure for debugging
        self.logger.info(f"[DEBUG] Daily data type: {type(daily_data)}")
        self.logger.info(
            f"[DEBUG] Daily data columns: {daily_data.columns.tolist() if hasattr(daily_data, 'columns') else 'No columns'}")
        self.logger.info(
            f"[DEBUG] Daily data shape: {daily_data.shape if hasattr(daily_data, 'shape') else 'No shape'}")

        # Filter for options with delta close to target
        tolerance = 0.02  # How close to target delta
        try:
            candidates = daily_data[
                (daily_data['Type'].str.lower() == option_type) &
                (daily_data['Delta'] >= target_delta - tolerance) &
                (daily_data['Delta'] <= target_delta + tolerance) &
                (daily_data['MidPrice'] > 0.2)  # Ensure reasonable price
                ].copy()

            self.logger.info(
                f"[DEBUG] Found {len(candidates)} {option_type} candidates with delta close to {target_delta}")
        except Exception as e:
            self.logger.error(f"[DEBUG] Error filtering for reference contracts: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

        if candidates.empty:
            self.logger.debug(f"[CPDHedgeManager] No reference candidates found with delta {target_delta}±{tolerance}")
            return None

        # If position_expiry is provided, prioritize contracts with matching expiry
        if position_expiry is not None:
            matching_expiry = candidates[candidates['Expiration'] == position_expiry]
            if not matching_expiry.empty:
                candidates = matching_expiry
                self.logger.info(f"[DEBUG] Found {len(matching_expiry)} candidates with matching expiry")

        # Calculate delta distance and sort
        candidates['delta_distance'] = abs(candidates['Delta'] - target_delta)
        candidates = candidates.sort_values('delta_distance')

        # Select best candidate
        best_candidate = candidates.iloc[0]

        # Log selection
        self.logger.info(f"[CPD Contract] Selected: {best_candidate['OptionSymbol']}")
        self.logger.info(f"  Delta: {best_candidate['Delta']:.3f} (target: {target_delta:.3f})")
        self.logger.info(f"  Price: ${best_candidate['MidPrice']:.2f}")

        if position_expiry is not None:
            expiry_match = "same as position" if best_candidate[
                                                     'Expiration'] == position_expiry else "different from position"
            self.logger.info(f"  Expiry: {best_candidate['Expiration'].strftime('%Y-%m-%d')} ({expiry_match})")

        # Store in reference contracts dictionary
        expiry = best_candidate['Expiration']
        if expiry not in self.reference_contracts:
            self.reference_contracts[expiry] = {}

        self.reference_contracts[expiry][best_candidate['OptionSymbol']] = best_candidate

        return best_candidate

    # Updated calculate_effective_contracts method for CPDHedgeManager class
    def calculate_effective_contracts(self, positions, daily_data):
        """
        Calculate the total effective contract count across all positions with improved tabular output.

        When finding reference contracts:
        - For positions with positive delta (calls), reference contracts will be calls
        - For positions with negative delta (puts), reference contracts will be puts
        - The sign of constant_portfolio_delta determines if the hedging is long or short

        Args:
            positions: Dictionary of positions
            daily_data: Daily option data

        Returns:
            float: Total effective contracts
        """
        if not self.enable_hedging or self.hedge_mode != 'constant':
            return 0

        if not positions:
            return 0

        # Reset tracking
        self.position_effective_contracts = {}
        total_effective = 0

        self.logger.info("[Total Effective Contracts Calculation]")

        # Start table for effective contracts
        self.logger.info("-" * 115)
        self.logger.info(
            f"{'Position':<20} {'Contracts':>9} {'Entry Price':>11} {'Ref. Contract':<20} {'Ref. Price':>10} {'Price Ratio':>11} {'Effective':>10}")
        self.logger.info("-" * 115)

        # Process each position to calculate effective contracts
        for symbol, position in positions.items():
            # Skip positions with no contracts
            if position.contracts <= 0:
                continue

            # Get position expiry
            position_expiry = position.expiration

            # Check if position already has a CPD reference set
            has_reference = (hasattr(position, 'cpd_reference_symbol') and
                             position.cpd_reference_symbol is not None and
                             position.cpd_reference_price is not None)

            if has_reference:
                # Use existing reference for existing positions
                ref_symbol = position.cpd_reference_symbol
                ref_price = position.cpd_reference_price
                price_ratio = position.avg_entry_price / ref_price if ref_price > 0 else 1
                effective_contracts = position.cpd_effective_contracts

                # Display in table format
                self.logger.info(
                    f"{symbol:<20} {position.contracts:>9} ${position.avg_entry_price:>10.2f} {ref_symbol:<20} ${ref_price:>9.2f} {price_ratio:>11.2f} {effective_contracts:>10.2f}")

                # Store for this position
                self.position_effective_contracts[symbol] = effective_contracts

                # Add to total
                total_effective += effective_contracts

            else:
                # This is a new position - find a reference contract for it
                self.logger.info(
                    f"{symbol:<20} {position.contracts:>9} ${position.avg_entry_price:>10.2f} {'Finding reference...':40}")

                # Let find_reference_contract determine the correct option type and target delta
                # based on the delta_target sign and CPD value
                reference_contract = self.find_reference_contract(
                    daily_data,
                    target_delta=None,  # Use None to let the method handle option type and target delta
                    position_expiry=position_expiry
                )

                if reference_contract is None:
                    # Skip position if no reference contract found
                    self.logger.warning(
                        f"No reference contract found for position {symbol} with expiry {position_expiry}")
                    continue

                # Calculate price ratio (position price to reference price)
                position_price = position.avg_entry_price

                # Extract reference price and symbol handling both Series and dict
                if hasattr(reference_contract, 'get') and not hasattr(reference_contract, 'iloc'):
                    # Dictionary style access
                    reference_price = reference_contract.get('MidPrice', 0)
                    reference_symbol = reference_contract.get('OptionSymbol', 'Unknown')
                    reference_delta = reference_contract.get('Delta', 0)
                else:
                    # Series style access
                    reference_price = reference_contract['MidPrice']
                    reference_symbol = reference_contract['OptionSymbol']
                    reference_delta = reference_contract['Delta']

                price_ratio = position_price / reference_price if reference_price > 0 else 1

                # Calculate effective contracts
                effective_contracts = position.contracts * price_ratio

                # Store for this position
                self.position_effective_contracts[symbol] = effective_contracts

                # Add to total
                total_effective += effective_contracts

                # Log the newly created reference in table format
                self.logger.info(
                    f"{symbol:<20} {position.contracts:>9} ${position_price:>10.2f} {reference_symbol:<20} ${reference_price:>9.2f} {price_ratio:>11.2f} {effective_contracts:>10.2f}")

                # Set CPD reference in position for this new position
                if position.contracts > 0:
                    # Set reference data in position
                    position.set_cpd_reference(reference_symbol, reference_price, reference_delta, position_expiry)

        # Close the table
        self.logger.info("-" * 115)
        self.logger.info(f"Total Effective Contracts: {total_effective:.2f}")

        # Save total
        self.total_effective_contracts = total_effective

        return total_effective

    def update_effective_contracts(self, symbol, old_contracts, new_contracts):
        """
        Update the effective contract count when a position is partially closed.

        Args:
            symbol: Option symbol
            old_contracts: Previous contract count
            new_contracts: New contract count

        Returns:
            float: Change in effective contracts
        """
        if not self.enable_hedging or self.hedge_mode != 'constant':
            return 0

        if symbol not in self.position_effective_contracts:
            return 0

        # Skip if no change in contracts
        if old_contracts == new_contracts:
            return 0

        # Calculate percentage of position closed
        pct_closed = (old_contracts - new_contracts) / old_contracts if old_contracts > 0 else 0

        # Get current effective contracts
        current_effective = self.position_effective_contracts[symbol]

        # Calculate reduction in effective contracts
        reduced_effective = current_effective * pct_closed

        # Calculate new effective contracts
        new_effective = current_effective - reduced_effective

        # Log the update
        self.logger.info(f"[CPD Reference] Position {symbol}: Reduced effective contracts")
        self.logger.info(f"  Closed {old_contracts - new_contracts} of {old_contracts} contracts ({pct_closed:.2%})")
        self.logger.info(f"  Reduced effective contracts by {reduced_effective:.2f}")
        self.logger.info(f"  Remaining effective contracts: {new_effective:.2f}")

        if new_contracts <= 0:
            # Position fully closed
            self.logger.info(f"  Position closed - removed {current_effective:.2f} effective contracts")
            self.position_effective_contracts.pop(symbol)
            self.total_effective_contracts -= current_effective
            return -current_effective
        else:
            # Position partially closed
            self.position_effective_contracts[symbol] = new_effective
            self.total_effective_contracts -= reduced_effective
            return -reduced_effective

    def calculate_hedge_position(self, portfolio_greeks, current_underlying_price):
        """
        Calculate the required hedge position based on the selected hedge mode:
        - 'constant': Maintains a constant portfolio delta
        - 'ratio': Maintains a target dollar delta to NLV ratio using NLV-based tolerance bands

        Args:
            portfolio_greeks: Portfolio Greeks
            current_underlying_price: Current underlying price

        Returns:
            int: Required hedge position in shares
        """
        if not self.enable_hedging or current_underlying_price <= 0:
            return 0

        # Get current option delta and dollar delta
        option_delta = portfolio_greeks['delta']
        option_dollar_delta = portfolio_greeks['dollar_delta']

        # Get current hedge delta
        current_hedge_delta = portfolio_greeks['hedge_delta']
        current_total_delta = option_delta + current_hedge_delta

        # Log common information
        self.logger.info(f"[Hedge] Option Delta: {option_delta:.2f}, Current Hedge Delta: {current_hedge_delta:.2f}")
        self.logger.info(f"[Hedge] Current Total Delta: {current_total_delta:.2f}")

        # Handle different hedge modes
        if self.hedge_mode.lower() == 'constant':
            # Constant Portfolio Delta mode
            target_delta = self.constant_portfolio_delta
            total_effective = self.total_effective_contracts

            self.logger.info(f"[Constant Hedge] CPD Target Delta: {target_delta:.3f}")
            self.logger.info(f"[Constant Hedge] Total Effective Contracts: {total_effective:.2f}")

            # Calculate target portfolio delta
            target_total_delta = total_effective * target_delta
            self.logger.info(f"[Constant Hedge] Target Total Delta: {target_total_delta:.2f}")

            # Calculate acceptable range with tolerance
            lower_bound = target_total_delta * (1 - self.delta_tolerance)
            upper_bound = target_total_delta * (1 + self.delta_tolerance)

            # Ensure the bounds are ordered correctly (lower < upper) regardless of sign
            if target_total_delta < 0:
                # For negative targets, the "lower" bound is actually the more negative number
                lower_bound, upper_bound = upper_bound, lower_bound

            self.logger.info(f"[Constant Hedge] Acceptable Delta Range: {lower_bound:.2f} to {upper_bound:.2f}")

            # Check if adjustment needed
            if lower_bound <= current_total_delta <= upper_bound:
                self.logger.info("[Constant Hedge] Total Delta within tolerance range - no hedge adjustment required")
                return int(round(current_hedge_delta * 100))

            # Calculate required hedge delta
            if current_total_delta < lower_bound:
                desired_hedge_delta = lower_bound - option_delta
                self.logger.info(
                    f"[Constant Hedge] Total Delta below range - need to increase by {lower_bound - current_total_delta:.2f}")
            elif current_total_delta > upper_bound:
                desired_hedge_delta = upper_bound - option_delta
                self.logger.info(
                    f"[Constant Hedge] Total Delta above range - need to decrease by {current_total_delta - upper_bound:.2f}")
            else:
                desired_hedge_delta = current_hedge_delta
                self.logger.info("[Constant Hedge] Delta calculation edge case - maintaining current hedge")

        elif self.hedge_mode.lower() == 'ratio':
            # Dollar Delta to NLV Ratio mode
            # Get current net liquidation value from the position manager
            from_position_manager = getattr(self, 'position_manager', None)
            if from_position_manager and hasattr(from_position_manager, 'get_portfolio_metrics'):
                metrics = from_position_manager.get_portfolio_metrics()
                net_liq = metrics.get('net_liquidation_value', 0)
            else:
                # Fallback if position manager not available - get this from config or other source
                net_liq = self.config.get('portfolio', {}).get('initial_capital', 100000)
                self.logger.warning(f"[Ratio Hedge] Using fallback NLV value: ${net_liq:,.2f}")

            # Calculate target dollar delta based on target ratio
            target_ratio = self.hedge_target_ratio  # Already a decimal value
            target_dollar_delta = net_liq * target_ratio

            # Calculate current dollar delta
            current_dollar_delta = option_dollar_delta + portfolio_greeks['dollar_hedge_delta']

            self.logger.info(f"[Ratio Hedge] NLV: ${net_liq:,.2f}")
            self.logger.info(f"[Ratio Hedge] Target Ratio: {target_ratio:.4f}")
            self.logger.info(f"[Ratio Hedge] Target Dollar Delta: ${target_dollar_delta:,.2f}")
            self.logger.info(f"[Ratio Hedge] Current Dollar Delta: ${current_dollar_delta:,.2f}")
            self.logger.info(
                f"[Ratio Hedge] Current Ratio: {(current_dollar_delta / net_liq if net_liq > 0 else 0):.2%}")

            # Calculate acceptable range with tolerance based on NLV
            tolerance_amount = self.delta_tolerance * net_liq
            lower_bound = target_dollar_delta - tolerance_amount
            upper_bound = target_dollar_delta + tolerance_amount

            self.logger.info(f"[Ratio Hedge] Acceptable Dollar Delta Range: ${lower_bound:,.2f} to ${upper_bound:,.2f}")

            # Check if adjustment needed
            if lower_bound <= current_dollar_delta <= upper_bound:
                self.logger.info("[Ratio Hedge] Dollar Delta within tolerance range - no hedge adjustment required")
                return int(round(current_hedge_delta * 100))

            # Calculate required dollar hedge delta
            if current_dollar_delta < lower_bound:
                required_dollar_hedge = lower_bound - option_dollar_delta
                self.logger.info(
                    f"[Ratio Hedge] Dollar Delta below range - need to increase by ${lower_bound - current_dollar_delta:,.2f}")
            elif current_dollar_delta > upper_bound:
                required_dollar_hedge = upper_bound - option_dollar_delta
                self.logger.info(
                    f"[Ratio Hedge] Dollar Delta above range - need to decrease by ${current_dollar_delta - upper_bound:,.2f}")
            else:
                required_dollar_hedge = portfolio_greeks['dollar_hedge_delta']
                self.logger.info("[Ratio Hedge] Dollar Delta calculation edge case - maintaining current hedge")

            # Convert dollar hedge to delta
            desired_hedge_delta = required_dollar_hedge / (current_underlying_price * 100)

        else:
            self.logger.warning(f"[Hedge] Unknown hedge mode: {self.hedge_mode}")
            return int(round(current_hedge_delta * 100))

        # Convert to shares (100 shares per 1.0 delta) - ensure integer value
        desired_hedge_position = int(round(desired_hedge_delta * 100))

        # Get current hedge position as integer
        current_hedge_position = int(round(current_hedge_delta * 100))

        # Calculate adjustment
        adjustment = desired_hedge_position - current_hedge_position

        if adjustment != 0:
            action = "Buy" if adjustment > 0 else "Sell"
            self.logger.info(f"[Hedge Adjustment] {action} {abs(adjustment)} shares @ ${current_underlying_price:.2f}")
            self.logger.info(f"  Previous: {current_hedge_position} shares -> New: {desired_hedge_position} shares")
        else:
            self.logger.info("[Hedge Adjustment] No adjustment needed - maintaining current position")

        return desired_hedge_position

# ========================
# PositionManager Class
# ========================
class PositionManager:
    """
    Manages a collection of positions with portfolio-level metrics.

    This class manages all positions, handles cash balance, calculates
    portfolio-level metrics and greeks, and manages hedging.
    """

    def __init__(self, initial_capital, config, risk_manager, logger=None):
        """
        Initialize the PositionManager.

        Args:
            initial_capital: Initial capital amount
            config: Configuration dictionary
            risk_manager: RiskManager instance
            logger: Logger instance
        """
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.config = config
        self.risk_manager = risk_manager
        self.logger = logger or logging.getLogger('theta_engine')

        # Track positions by option symbol
        self.positions = {}

        # Tracking for today's activity
        self.today_added_positions = {}
        self.today_closed_positions = {}
        self.today_realized_pnl = 0

        # Performance tracking
        self.equity_history = {}
        self.daily_returns = []
        self.current_metrics = None

        # Hedge position
        self.hedge_position = 0
        self.hedge_pnl = 0
        self.previous_underlying_price = None

        # Track daily metrics for reporting
        self.dollar_delta_to_nlv_history = []

        self.logger.info(f"[PositionManager] Initialized with ${initial_capital:,.2f} initial capital")

    def get_position(self, option_symbol):
        """
        Get an existing position or return None.

        Args:
            option_symbol: Option symbol to look up

        Returns:
            Position: Position object or None if not found
        """
        return self.positions.get(option_symbol)

    def add_position(self, option_data, contracts):
        """
        Add a new position or add to existing position.

        Args:
            option_data: Option data
            contracts: Number of contracts to add

        Returns:
            tuple: (position, option_symbol) - The position and its symbol
        """
        # Handle both pandas Series and dict-like objects
        if hasattr(option_data, 'get') and not hasattr(option_data, 'iloc'):
            # This is a dictionary
            option_symbol = option_data.get('OptionSymbol')
            price = option_data.get('MidPrice')
        else:
            # This is a pandas Series
            option_symbol = option_data['OptionSymbol']
            price = option_data['MidPrice']

        # Get or create position
        position = self.get_position(option_symbol)
        if position is None:
            position = Position(option_symbol, option_data, 0, logger=self.logger)
            self.positions[option_symbol] = position

        # Add contracts to the position
        position.add_contracts(contracts, price, option_data)

        # Update cash balance
        # For short positions, we receive premium
        if position.is_short:
            premium_received = price * contracts * 100
            self.cash_balance += premium_received
            self.logger.info(f"  Premium received: ${premium_received:.2f}")
        else:
            # For long positions, we pay premium
            premium_paid = price * contracts * 100
            self.cash_balance -= premium_paid
            self.logger.info(f"  Premium paid: ${premium_paid:.2f}")

        # Record for today's activity
        self.today_added_positions[option_symbol] = {
            'contracts': contracts,
            'price': price,
            'value': price * contracts * 100,
            'data': option_data
        }

        return position, option_symbol

    def close_position(self, option_symbol, contracts, market_data, reason="Close"):
        """
        Close all or part of a position.

        Args:
            option_symbol: Option symbol to close
            contracts: Number of contracts to close
            market_data: Current market data
            reason: Reason for closing (e.g., "Profit Target", "Stop Loss")

        Returns:
            float: Realized PnL for this closure
        """
        position = self.get_position(option_symbol)
        if position is None or position.contracts <= 0:
            return 0

        # Limit to available contracts
        contracts_to_close = min(contracts, position.contracts)
        if contracts_to_close <= 0:
            return 0

        # Store the current entry price and contract count before making changes
        entry_price = position.avg_entry_price
        old_contracts = position.contracts

        # Get current price - handle both dict and Series
        if hasattr(market_data, 'get') and not hasattr(market_data, 'iloc'):
            current_price = market_data.get('MidPrice', 0)
        else:
            current_price = market_data['MidPrice'] if 'MidPrice' in market_data else 0

        # Close the position and get realized PnL
        pnl = position.remove_contracts(contracts_to_close, current_price, market_data, reason)

        # Update cash balance based on short/long position
        if position.is_short:
            # For short positions being closed, we pay the current price
            self.cash_balance -= current_price * contracts_to_close * 100
        else:
            # For long positions being closed, we receive the current price
            self.cash_balance += current_price * contracts_to_close * 100

        # Add to today's realized PnL
        self.today_realized_pnl += pnl

        # Record for today's activity
        if option_symbol not in self.today_closed_positions:
            self.today_closed_positions[option_symbol] = {
                'contracts': 0,
                'price': 0,
                'entry_price': entry_price,
                'pnl': 0,
                'reason': reason,
                'data': market_data
            }

        self.today_closed_positions[option_symbol]['contracts'] += contracts_to_close
        self.today_closed_positions[option_symbol]['price'] = current_price
        self.today_closed_positions[option_symbol]['pnl'] += pnl

        # Get the new contract count after the closure
        new_contracts = position.contracts

        # Update CPD effective contracts if there is a CPD hedge manager attached
        if hasattr(self, 'cpd_hedge_manager'):
            position.update_cpd_effective_contracts_for_closure(contracts_to_close)

        # Remove empty positions
        if position.contracts == 0:
            del self.positions[option_symbol]

        return pnl

    def update_positions(self, current_date, market_data_by_symbol):
        """
        Update all positions with latest market data.

        Args:
            current_date: Current date for the update
            market_data_by_symbol: Dictionary of {symbol: market_data}
        """
        # Store previous day's position values for accurate PnL calculation
        previous_position_values = {}
        for symbol, position in self.positions.items():
            previous_position_values[symbol] = {
                'price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl
            }

        # Reset today's tracking
        self.today_added_positions = {}
        self.today_closed_positions = {}
        self.today_realized_pnl = 0

        # Daily PnL tracking
        daily_unrealized_pnl_change = 0

        # Update each position with latest market data
        for symbol, position in list(self.positions.items()):
            if symbol in market_data_by_symbol:
                # Store pre-update values
                pre_update_unrealized_pnl = position.unrealized_pnl

                # Update position with new market data
                position.update_market_data(market_data_by_symbol[symbol])

                # Calculate daily PnL change
                pnl_change = position.unrealized_pnl - pre_update_unrealized_pnl
                daily_unrealized_pnl_change += pnl_change

                # Debug log
                self.logger.info(
                    f"[Position Update] {symbol}: ${position.unrealized_pnl:.2f} (change: ${pnl_change:.2f})")
            else:
                self.logger.warning(f"Warning: No market data found for {symbol}")

        # Update hedge PnL if applicable
        previous_hedge_pnl = self.hedge_pnl
        if self.hedge_position != 0 and self.previous_underlying_price is not None:
            # Get the current underlying price (using any position's data)
            current_underlying = None
            for symbol, data in market_data_by_symbol.items():
                if hasattr(data, 'get') and not hasattr(data, 'iloc'):
                    # Dictionary style access
                    if 'UnderlyingPrice' in data:
                        current_underlying = data.get('UnderlyingPrice')
                        break
                else:
                    # Series style access
                    if 'UnderlyingPrice' in data:
                        current_underlying = data['UnderlyingPrice']
                        break

            if current_underlying is not None:
                # Calculate hedge PnL
                price_change = current_underlying - self.previous_underlying_price
                hedge_pnl_change = self.hedge_position * price_change
                self.hedge_pnl += hedge_pnl_change
                self.previous_underlying_price = current_underlying

                self.logger.info(
                    f"[Hedge Update] P&L change: ${hedge_pnl_change:.2f} (Price change: ${price_change:.2f})")

        # Calculate daily hedge PnL change
        daily_hedge_pnl_change = self.hedge_pnl - previous_hedge_pnl

        # Recalculate portfolio metrics
        self.current_metrics = self.get_portfolio_metrics()

        # Store net liquidation value in equity history
        current_nlv = self.current_metrics['net_liquidation_value']
        self.equity_history[current_date] = current_nlv

        # Calculate daily return if we have previous data
        if len(self.equity_history) > 1:
            dates = sorted(self.equity_history.keys())
            previous_date = dates[-2]
            previous_nlv = self.equity_history[previous_date]

            if previous_nlv > 0:
                # Calculate daily return components
                total_daily_pnl = current_nlv - previous_nlv
                daily_return = total_daily_pnl / previous_nlv

                # Store daily return with PnL breakdown
                self.daily_returns.append({
                    'date': current_date,
                    'return': daily_return,
                    'pnl': total_daily_pnl,
                    'unrealized_pnl_change': daily_unrealized_pnl_change,
                    'realized_pnl': self.today_realized_pnl,
                    'hedge_pnl_change': daily_hedge_pnl_change
                })

                # Debug log
                self.logger.info(f"[Daily Return] {daily_return:.2%} (${total_daily_pnl:.2f})")
                self.logger.info(f"  Unrealized PnL Change: ${daily_unrealized_pnl_change:.2f}")
                self.logger.info(f"  Realized PnL: ${self.today_realized_pnl:.2f}")
                self.logger.info(f"  Hedge PnL Change: ${daily_hedge_pnl_change:.2f}")

        # Store dollar delta to NLV ratio for tracking
        greeks = self.get_portfolio_greeks()
        dollar_delta = greeks.get('dollar_total_delta', 0)
        delta_ratio = dollar_delta / current_nlv if current_nlv > 0 else 0

        self.dollar_delta_to_nlv_history.append({
            'date': current_date,
            'dollar_delta': dollar_delta,
            'nlv': current_nlv,
            'dollar_delta_ratio': delta_ratio
        })

    def get_portfolio_metrics(self):
        """
        Calculate portfolio-level metrics with proper margin accounting.

        Returns:
            dict: Portfolio metrics
        """
        # Calculate position metrics
        total_margin = 0
        total_unrealized_pnl = 0
        total_position_value = 0  # Current total market value of all positions

        for symbol, position in self.positions.items():
            # Calculate margin requirement, which now includes unrealized losses
            position_margin = self.risk_manager.calculate_margin_requirement(position)
            total_margin += position_margin

            # Sum unrealized PnL
            total_unrealized_pnl += position.unrealized_pnl

            # Add current position value (what it would cost to close the position)
            position_value = position.current_price * position.contracts * 100
            # For short positions, this is what we'd pay to close (a liability)
            # For long positions, this is what we'd receive if we closed (an asset)
            # We'll handle this in the NLV calculation below
            if position.is_short:
                total_position_value -= position_value
            else:
                total_position_value += position_value

        # Calculate net liquidation value
        # For short options: NLV = Cash - Current position value (the liability)
        # For long options: NLV = Cash + Current position value (the asset)
        # Since we've already handled the sign in total_position_value calculation,
        # we can just add it here
        net_liq = self.cash_balance + total_position_value + self.hedge_pnl

        # Calculate available margin
        available_margin = net_liq - total_margin

        # Calculate leverage
        leverage = total_margin / net_liq if net_liq > 0 else 0

        return {
            'cash_balance': self.cash_balance,
            'total_margin': total_margin,
            'net_liquidation_value': net_liq,
            'available_margin': available_margin,
            'leverage': leverage,
            'total_unrealized_pnl': total_unrealized_pnl,
            'hedge_pnl': self.hedge_pnl,
            'total_position_value': total_position_value
        }

    def get_portfolio_greeks(self):
        """
        Calculate aggregated portfolio Greeks.

        Returns:
            dict: Portfolio-level Greeks
        """
        portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'dollar_delta': 0.0,
            'dollar_gamma': 0.0,
            'dollar_theta': 0.0,
            'dollar_vega': 0.0,
        }

        # Aggregate Greeks from all positions
        for symbol, position in self.positions.items():
            position_greeks = position.get_greeks()

            for greek, value in position_greeks.items():
                portfolio_greeks[greek] += value

        # Ensure hedge position is an integer
        hedge_position_int = int(round(self.hedge_position))

        # Convert to delta equivalent (100 shares = 1.0 delta)
        hedge_delta = hedge_position_int / 100.0

        # Get current underlying price
        current_underlying = 0
        if self.positions:
            # Use the first position's underlying price as a proxy
            current_underlying = next(iter(self.positions.values())).underlying_price

        dollar_hedge_delta = hedge_delta * current_underlying * 100

        portfolio_greeks['hedge_delta'] = hedge_delta
        portfolio_greeks['dollar_hedge_delta'] = dollar_hedge_delta
        portfolio_greeks['total_delta'] = portfolio_greeks['delta'] + hedge_delta
        portfolio_greeks['dollar_total_delta'] = portfolio_greeks['dollar_delta'] + dollar_hedge_delta

        return portfolio_greeks

    def adjust_hedge_with_cpd(self, current_underlying_price, cpd_hedge_manager, daily_data):
        """
        Adjust hedge position using CPD hedge manager, supporting both 'constant' and 'ratio' modes.
        For ratio mode, uses NLV-based tolerance rather than percentage of target delta.

        Args:
            current_underlying_price: Current underlying price
            cpd_hedge_manager: CPDHedgeManager instance
            daily_data: Daily option data
        """
        self.logger.info(f"[Hedge] Adjusting hedge position with underlying price: {current_underlying_price}")

        # Check if hedging is enabled in the CPD hedge manager
        if not cpd_hedge_manager.enable_hedging:
            self.logger.info(f"[Hedge] Hedging not enabled")
            return

        # Get current portfolio Greeks
        greeks = self.get_portfolio_greeks()
        self.logger.info(f"[Hedge] Portfolio delta before hedge adjustment: {greeks['delta']:.4f}")

        # Log dollar delta values
        self.logger.info(f"[Hedge] Dollar delta: ${greeks['dollar_delta']:,.2f}")
        if 'dollar_hedge_delta' in greeks:
            self.logger.info(f"[Hedge] Current dollar hedge delta: ${greeks['dollar_hedge_delta']:,.2f}")
        if 'dollar_total_delta' in greeks:
            self.logger.info(f"[Hedge] Current dollar total delta: ${greeks['dollar_total_delta']:,.2f}")

        # Calculate desired hedge position based on hedge mode
        if cpd_hedge_manager.hedge_mode.lower() == 'constant':
            # For 'constant' mode - calculate effective contracts first
            total_effective = cpd_hedge_manager.calculate_effective_contracts(self.positions, daily_data)
            self.logger.info(f"[Hedge] Total effective contracts: {total_effective}")

            # Then calculate hedge position
            desired_hedge_position = cpd_hedge_manager.calculate_hedge_position(
                greeks, current_underlying_price
            )
        elif cpd_hedge_manager.hedge_mode.lower() == 'ratio':
            # For 'ratio' mode - calculate hedge position based on dollar delta to NLV ratio
            metrics = self.get_portfolio_metrics()
            net_liq = metrics['net_liquidation_value']

            # Get the target ratio from config
            target_ratio = cpd_hedge_manager.hedge_target_ratio  # Already a decimal value

            # Calculate target dollar delta
            target_dollar_delta = net_liq * target_ratio

            # Get current dollar deltas
            current_dollar_delta = greeks['dollar_delta']
            current_hedge_dollar_delta = greeks.get('dollar_hedge_delta', 0)
            current_total_dollar_delta = greeks.get('dollar_total_delta',
                                                    current_dollar_delta + current_hedge_dollar_delta)

            # Calculate the current ratio for logging
            current_ratio = current_total_dollar_delta / net_liq if net_liq > 0 else 0

            self.logger.info(f"[Ratio Hedge] NLV: ${net_liq:,.2f}")
            self.logger.info(f"[Ratio Hedge] Target Ratio: {target_ratio:.4f}")
            self.logger.info(f"[Ratio Hedge] Target Dollar Delta: ${target_dollar_delta:,.2f}")
            self.logger.info(f"[Ratio Hedge] Current Dollar Delta: ${current_total_dollar_delta:,.2f}")
            self.logger.info(f"[Ratio Hedge] Current Ratio: {current_ratio:.2%}")

            # Determine acceptable range with tolerance based on NLV
            tolerance_amount = cpd_hedge_manager.delta_tolerance * net_liq
            lower_bound = target_dollar_delta - tolerance_amount
            upper_bound = target_dollar_delta + tolerance_amount

            self.logger.info(f"[Ratio Hedge] Acceptable Dollar Delta Range: ${lower_bound:,.2f} to ${upper_bound:,.2f}")

            # Check if adjustment needed
            if lower_bound <= current_total_dollar_delta <= upper_bound:
                self.logger.info("[Ratio Hedge] Dollar Delta within tolerance range - no hedge adjustment required")
                return

            # Calculate required dollar hedge
            if current_total_dollar_delta < lower_bound:
                required_dollar_hedge = lower_bound - current_dollar_delta
                self.logger.info(
                    f"[Ratio Hedge] Dollar Delta below range - need to increase by ${lower_bound - current_total_dollar_delta:,.2f}")
            elif current_total_dollar_delta > upper_bound:
                required_dollar_hedge = upper_bound - current_dollar_delta
                self.logger.info(
                    f"[Ratio Hedge] Dollar Delta above range - need to decrease by ${current_total_dollar_delta - upper_bound:,.2f}")
            else:
                required_dollar_hedge = current_hedge_dollar_delta
                self.logger.info("[Ratio Hedge] Dollar Delta calculation edge case - maintaining current hedge")

            # Convert dollar hedge to shares
            desired_hedge_delta = required_dollar_hedge / (current_underlying_price * 100)
            desired_hedge_position = int(round(desired_hedge_delta * 100))

        else:
            self.logger.warning(f"[Hedge] Unknown hedge mode: {cpd_hedge_manager.hedge_mode}")
            return

        # Ensure current hedge position is an integer
        current_hedge_position = int(round(self.hedge_position))

        # Skip if no change needed
        if desired_hedge_position == current_hedge_position:
            self.logger.info(f"[Hedge] No position adjustment needed - maintaining {current_hedge_position} shares")
            # Update to ensure integer value
            self.hedge_position = current_hedge_position
            return

        # Execute hedge adjustment
        adjustment = desired_hedge_position - current_hedge_position
        old_position = current_hedge_position
        self.hedge_position = desired_hedge_position  # Set directly to integer value

        # Store current price for future PnL calculation
        self.previous_underlying_price = current_underlying_price

        # Log the adjustment
        action = "Bought" if adjustment > 0 else "Sold"
        self.logger.info(f"[Hedge] {action} {abs(adjustment)} shares @ ${current_underlying_price:.2f}")
        self.logger.info(f"  Previous hedge: {old_position} shares")
        self.logger.info(f"  New hedge: {self.hedge_position} shares")

        # Calculate new delta values after adjustment
        new_hedge_delta = desired_hedge_position / 100
        new_total_delta = greeks['delta'] + new_hedge_delta
        new_dollar_hedge_delta = new_hedge_delta * current_underlying_price * 100
        new_dollar_total_delta = greeks['dollar_delta'] + new_dollar_hedge_delta

        self.logger.info(f"  New total delta: {new_total_delta:.4f}")
        self.logger.info(f"  New dollar hedge delta: ${new_dollar_hedge_delta:,.2f}")
        self.logger.info(f"  New dollar total delta: ${new_dollar_total_delta:,.2f}")

    def adjust_hedge(self, current_underlying_price):
        """
        Adjust hedge position to maintain target portfolio delta.

        Args:
            current_underlying_price: Current underlying price
        """
        if not self.config.get('strategy', {}).get('enable_hedging', False):
            return

        hedge_mode = self.config.get('strategy', {}).get('hedge_mode', 'constant').lower()
        target_delta = self.config.get('strategy', {}).get('constant_portfolio_delta', 0.05)
        delta_tolerance = self.config.get('strategy', {}).get('delta_tolerance', 0.2)

        # Get current portfolio Greeks
        greeks = self.get_portfolio_greeks()
        option_delta = greeks['delta']

        # Ensure hedge_delta is treated consistently as a number
        hedge_delta = greeks['hedge_delta']
        current_total_delta = option_delta + hedge_delta

        # Calculate target total delta
        if hedge_mode == 'constant':
            # For short puts, we typically want a small positive delta
            # to partially hedge the negative delta
            abs_target = abs(target_delta)
            target_total_delta = abs_target if option_delta < 0 else -abs_target
        else:
            # Alternative hedging modes could be implemented here
            target_total_delta = 0

        # Determine acceptable range with tolerance
        lower_bound = target_total_delta * (1 - delta_tolerance)
        upper_bound = target_total_delta * (1 + delta_tolerance)

        # Check if adjustment needed
        if lower_bound <= current_total_delta <= upper_bound:
            # Ensure current position is an integer
            self.hedge_position = int(round(self.hedge_position))
            self.logger.info(f"[Hedge] No adjustment needed. Current delta: {current_total_delta:.3f}")
            return

        # Calculate required hedge position
        # Adjust to boundary, not to target
        if current_total_delta < lower_bound:
            # Need to increase delta - adjust to lower bound
            desired_hedge_delta = lower_bound - option_delta
        elif current_total_delta > upper_bound:
            # Need to decrease delta - adjust to upper bound
            desired_hedge_delta = upper_bound - option_delta
        else:
            # This shouldn't happen
            desired_hedge_delta = hedge_delta

        # Convert delta to shares (integer number of shares)
        desired_hedge_position = int(round(desired_hedge_delta * 100))

        # Ensure current position is an integer
        current_hedge_position = int(round(self.hedge_position))

        # Calculate adjustment
        adjustment = desired_hedge_position - current_hedge_position

        if adjustment != 0:
            # Execute hedge adjustment
            self.hedge_position = desired_hedge_position  # Set directly to integer

            # Store current price for future PnL calculation
            self.previous_underlying_price = current_underlying_price

            # Log the adjustment
            action = "Bought" if adjustment > 0 else "Sold"
            self.logger.info(f"[Hedge] {action} {abs(adjustment)} shares @ ${current_underlying_price:.2f}")
            self.logger.info(f"  Previous hedge: {current_hedge_position} shares")
            self.logger.info(f"  New hedge: {self.hedge_position} shares")
            self.logger.info(
                f"  Target delta range: {lower_bound:.3f} to {upper_bound:.3f}, Option delta: {option_delta:.3f}")

    def can_add_position(self):
        """
        Check if portfolio can add more positions.

        Returns:
            bool: True if portfolio can add more positions, False otherwise
        """
        metrics = self.current_metrics or self.get_portfolio_metrics()
        return metrics['available_margin'] > 0

    def get_sharpe_scaling_z(self):
        """
        Calculate risk scaling factor based on Sharpe ratio using RiskManager.

        Returns:
            tuple: (scaling_factor, current_sharpe, z_score)
        """
        return self.risk_manager.calculate_risk_scaling(self.daily_returns)

    # Updated print_daily_summary method with fixed Greek normalization
    def print_daily_summary(self, current_date, label="Daily Summary", include_rolling_metrics=False):
        """
        Print daily portfolio summary with cleaner format and optional rolling metrics.

        Args:
            current_date: Current date
            label: Label for the summary section (e.g., "PRE-TRADE Summary", "POST-TRADE Summary")
            include_rolling_metrics: Whether to include rolling metrics (typically only for POST-TRADE)
        """
        self.logger.info("=" * 50)
        self.logger.info(f"{label} [{current_date.strftime('%Y-%m-%d')}]:")

        # Find daily return data for this date
        daily_return_data = None
        for entry in self.daily_returns:
            if entry['date'] == current_date:
                daily_return_data = entry
                break

        # Print PnL summary
        if daily_return_data:
            pnl = daily_return_data['pnl']
            pct = daily_return_data['return'] * 100
            unrealized_change = daily_return_data.get('unrealized_pnl_change', 0)
            hedge_change = daily_return_data.get('hedge_pnl_change', 0)
            realized_pnl = daily_return_data.get('realized_pnl', 0)

            self.logger.info(f"Daily P&L: ${pnl:.0f} ({pct:.2f}%)")
            self.logger.info(f"  Option PnL: ${unrealized_change + realized_pnl:.0f}")
            self.logger.info(f"  Hedge PnL: ${hedge_change:.0f}")
        else:
            self.logger.info("No daily return data available")

        # Print portfolio metrics
        metrics = self.get_portfolio_metrics()

        # Calculate total position value as percentage of NLV
        current_nlv = metrics['net_liquidation_value']
        total_position_value = abs(metrics.get('total_position_value', 0))
        position_exposure_pct = (total_position_value / current_nlv * 100) if current_nlv > 0 else 0

        self.logger.info(f"Open Trades: {len(self.positions)}")
        self.logger.info(f"Total Position Exposure: {position_exposure_pct:.1f}% of NLV")
        self.logger.info(f"Net Liq: ${metrics['net_liquidation_value']:.0f}")
        self.logger.info(f"  Cash Balance: ${metrics['cash_balance']:.0f}")
        self.logger.info(f"  Total Liability: ${metrics['total_position_value']:.0f}")
        self.logger.info(f"  Self Hedge (Hedge PnL): ${metrics['hedge_pnl']:.0f}")
        self.logger.info(f"Total Margin Requirement: ${metrics['total_margin']:.0f}")
        self.logger.info(f"Available Margin: ${metrics['available_margin']:.0f}")
        self.logger.info(f"Margin-Based Leverage: {metrics['leverage']:.2f} ")

        # Print Greeks in a more concise form
        greeks = self.get_portfolio_greeks()
        self.logger.info("\nPortfolio Greek Risk:")
        self.logger.info(f"  Option Delta: {greeks['delta']:.3f} (${greeks['dollar_delta']:.2f})")
        self.logger.info(f"  Hedge Delta: {greeks['hedge_delta']:.3f} (${greeks['dollar_hedge_delta']:.2f})")
        self.logger.info(f"  Total Delta: {greeks['total_delta']:.3f} (${greeks['dollar_total_delta']:.2f})")
        self.logger.info(f"  Gamma: {greeks['gamma']:.6f} (${greeks['dollar_gamma']:.2f} per 1% move)")
        self.logger.info(f"  Theta: ${greeks['dollar_theta']:.2f} per day")
        self.logger.info(f"  Vega: ${greeks['dollar_vega']:.2f} per 1% IV")

        # Only print rolling metrics if requested (typically only for POST-TRADE)
        if include_rolling_metrics:
            # Print rolling metrics directly here
            if hasattr(self, 'risk_manager') and hasattr(self.risk_manager, 'rolling_window') and len(
                    self.daily_returns) > 0:
                # Convert returns for sharpe calculation
                returns_series = pd.Series([r['return'] for r in self.daily_returns])

                self.logger.info("\nRolling Metrics:")

                # Calculate expanding window metrics (all data)
                all_returns = returns_series
                if len(all_returns) >= 5:  # Minimum observations for meaningful stats
                    all_mean = all_returns.mean() * 252  # Annualize
                    all_std = all_returns.std() * np.sqrt(252)  # Annualize
                    all_sharpe = all_mean / all_std if all_std > 0 else 0
                    self.logger.info(
                        f"  Expanding Window (all obs, min 5 required): Sharpe: {all_sharpe:.2f}, Volatility: {all_std * 100:.2f}%")

                # Calculate multi-window Sharpes if available
                if hasattr(self.risk_manager, 'calculate_multi_window_sharpes'):
                    sharpe_data = self.risk_manager.calculate_multi_window_sharpes(returns_series)

                    if sharpe_data:
                        # Print metrics from different windows
                        if 'short' in sharpe_data:
                            short_sharpe = sharpe_data['short']['sharpe']
                            short_window = sharpe_data['short']['window']
                            # Calculate volatility for this window
                            if len(returns_series) >= short_window:
                                short_vol = returns_series.rolling(short_window).std().iloc[-1] * np.sqrt(252) * 100
                                self.logger.info(
                                    f"  Short Window ({short_window} days, rolling): Sharpe: {short_sharpe:.2f}, Volatility: {short_vol:.2f}%")

                        if 'medium' in sharpe_data:
                            medium_sharpe = sharpe_data['medium']['sharpe']
                            medium_window = sharpe_data['medium']['window']
                            # Calculate volatility for this window
                            if len(returns_series) >= medium_window:
                                medium_vol = returns_series.rolling(medium_window).std().iloc[-1] * np.sqrt(252) * 100
                                self.logger.info(
                                    f"  Medium Window ({medium_window} days, rolling): Sharpe: {medium_sharpe:.2f}, Volatility: {medium_vol:.2f}%")

                        if 'long' in sharpe_data:
                            long_sharpe = sharpe_data['long']['sharpe']
                            long_window = sharpe_data['long']['window']
                            # Calculate volatility for this window
                            if len(returns_series) >= long_window:
                                long_vol = returns_series.rolling(long_window).std().iloc[-1] * np.sqrt(252) * 100
                                self.logger.info(
                                    f"  Long Window ({long_window} days, rolling): Sharpe: {long_sharpe:.2f}, Volatility: {long_vol:.2f}%")

        self.logger.info("-" * 50)

        # Print open positions with improved format
        if self.positions:
            # Get current NLV for percentage calculation
            current_nlv = metrics['net_liquidation_value']

            self.logger.info("\nOpen Trades Table:")
            header_len = 170  # Increased length to accommodate all columns
            self.logger.info("-" * header_len)
            self.logger.info(
                f"{'Symbol':<16} {'Contracts':>9} {'Entry':>8} {'Current':>8} {'Value':>10} {'NLV%':>6} {'Underlying':>10} "
                f"{'Delta':>9} {'Gamma':>9} {'Theta':>9} {'Vega':>9} {'Margin':>12} {'DTE':>5}")
            self.logger.info("-" * header_len)

            total_value = 0
            total_margin = 0

            for symbol, position in self.positions.items():
                # Calculate position value (absolute value) - for display purposes
                position_value = abs(position.current_price * position.contracts * 100)

                # Calculate position as percentage of NLV
                nlv_percent = (position_value / current_nlv * 100) if current_nlv > 0 else 0

                # Calculate margin for the position
                position_margin = position.calculate_margin_requirement(self.risk_manager.max_leverage)

                # Get raw Greeks data for proper normalization
                # These should be PER CONTRACT values
                if position.contracts > 0:
                    # Delta and Gamma are typically already per-contract values
                    delta_per_contract = position.current_delta
                    gamma_per_contract = position.current_gamma

                    # Properly normalize Theta and Vega to per-contract values
                    # The sign is handled in display based on position type (long/short)
                    theta_per_contract = position.current_theta / position.contracts
                    vega_per_contract = position.current_vega / position.contracts
                else:
                    delta_per_contract = gamma_per_contract = theta_per_contract = vega_per_contract = 0

                # Adjust sign for display - short positions have negative theta benefits
                display_sign = -1 if position.is_short else 1

                # Add to totals
                total_value += position_value
                total_margin += position_margin

                # Format the row with properly normalized Greeks
                self.logger.info(
                    f"{symbol:<16} {position.contracts:>9} "
                    f"${position.avg_entry_price:>7.2f} ${position.current_price:>7.2f} "
                    f"${position_value:>9,.0f} {nlv_percent:>5.1f}% ${position.underlying_price:>9,.2f} "
                    f"{delta_per_contract:>8.3f} {gamma_per_contract:>8.6f} "
                    f"{display_sign * theta_per_contract:>8.2f} {display_sign * vega_per_contract:>8.2f} "
                    f"${position_margin:>11,.0f} {position.days_to_expiry:>5}")

            # Print totals
            self.logger.info("-" * header_len)
            self.logger.info(f"{'TOTAL':<16} {'':<9} {'':>8} {'':>8} "
                             f"${total_value:>9,.0f} {(total_value / current_nlv * 100) if current_nlv > 0 else 0:>5.1f}% {'':>10} "
                             f"{'':>9} {'':>9} {'':>9} {'':>9} "
                             f"${total_margin:>11,.0f}")
            self.logger.info("-" * header_len)

        self.logger.info("=" * 50)

    def _print_rolling_metrics(self):
        """
        Print rolling metrics for different time windows.
        """
        if hasattr(self.risk_manager, 'calculate_multi_window_sharpes') and len(self.daily_returns) > 20:
            # Convert returns for sharpe calculation
            returns_series = pd.Series([r['return'] for r in self.daily_returns])

            # Calculate multi-window Sharpes
            sharpe_data = self.risk_manager.calculate_multi_window_sharpes(returns_series)

            if sharpe_data:
                self.logger.info("\nRolling Metrics:")

                # Calculate expanding window metrics (all data)
                all_returns = returns_series
                if len(all_returns) >= 5:  # Minimum observations for meaningful stats
                    all_mean = all_returns.mean() * 252  # Annualize
                    all_std = all_returns.std() * np.sqrt(252)  # Annualize
                    all_sharpe = all_mean / all_std if all_std > 0 else 0
                    self.logger.info(
                        f"  Expanding Window (all obs, min 5 required): Sharpe: {all_sharpe:.2f}, Volatility: {all_std * 100:.2f}%")

                # Print metrics from different windows
                if 'short' in sharpe_data:
                    short_sharpe = sharpe_data['short']['sharpe']
                    short_window = sharpe_data['short']['window']
                    # Calculate volatility for this window
                    if len(returns_series) >= short_window:
                        short_vol = returns_series.rolling(short_window).std().iloc[-1] * np.sqrt(252) * 100
                        self.logger.info(
                            f"  Short Window ({short_window} days, rolling): Sharpe: {short_sharpe:.2f}, Volatility: {short_vol:.2f}%")

                if 'medium' in sharpe_data:
                    medium_sharpe = sharpe_data['medium']['sharpe']
                    medium_window = sharpe_data['medium']['window']
                    # Calculate volatility for this window
                    if len(returns_series) >= medium_window:
                        medium_vol = returns_series.rolling(medium_window).std().iloc[-1] * np.sqrt(252) * 100
                        self.logger.info(
                            f"  Medium Window ({medium_window} days, rolling): Sharpe: {medium_sharpe:.2f}, Volatility: {medium_vol:.2f}%")

                if 'long' in sharpe_data:
                    long_sharpe = sharpe_data['long']['sharpe']
                    long_window = sharpe_data['long']['window']
                    # Calculate volatility for this window
                    if len(returns_series) >= long_window:
                        long_vol = returns_series.rolling(long_window).std().iloc[-1] * np.sqrt(252) * 100
                        self.logger.info(
                            f"  Long Window ({long_window} days, rolling): Sharpe: {long_sharpe:.2f}, Volatility: {long_vol:.2f}%")

    def print_trade_activity(self):
        """
        Print summary of today's trading activity with enhanced format including DTE and Margin.
        """
        if not self.today_added_positions and not self.today_closed_positions:
            return

        self.logger.info("\nToday's Trading Activity:")

        # Print closed positions
        if self.today_closed_positions:
            self.logger.info(f"\nClosed Positions: (Total P&L: ${self.today_realized_pnl:.2f})")
            self.logger.info("-" * 100)
            self.logger.info(f"{'Symbol':<16} {'Contracts':>9} {'Entry':>8} {'Exit':>8} {'P&L':>10} {'Reason':<20}")
            self.logger.info("-" * 100)

            for symbol, data in self.today_closed_positions.items():
                self.logger.info(f"{symbol:<16} {data['contracts']:>9} "
                                 f"${data['entry_price']:>7.2f} ${data['price']:>7.2f} "
                                 f"${data['pnl']:>9.2f} {data['reason']:<20}")

            self.logger.info("-" * 100)

        # Print added positions with enhanced columns (DTE and Margin)
        if self.today_added_positions:
            self.logger.info(f"\nAdded Positions:")
            self.logger.info("-" * 120)
            self.logger.info(
                f"{'Symbol':<16} {'Contracts':>9} {'Price':>8} {'Value':>10} {'Delta':>8} {'DTE':>5} {'Margin':>12}")
            self.logger.info("-" * 120)

            for symbol, data in self.today_added_positions.items():
                # Handle both dict and Series for extraction
                delta = 0
                dte = 0
                margin = 0

                # Extract data with appropriate handling for both dict and Series
                if hasattr(data['data'], 'get') and not hasattr(data['data'], 'iloc'):
                    # Dictionary style access
                    delta = data['data'].get('Delta', 0)

                    # Calculate DTE if we have DataDate and Expiration
                    if 'DataDate' in data['data'] and 'Expiration' in data['data']:
                        dte = (data['data']['Expiration'] - data['data']['DataDate']).days

                    # Calculate margin (simplified approximation)
                    margin = data['price'] * data['contracts'] * 100 * self.risk_manager.max_leverage
                else:
                    # Series style access
                    delta = data['data']['Delta'] if 'Delta' in data['data'] else 0

                    # Calculate DTE if we have DataDate and Expiration
                    if 'DataDate' in data['data'].index and 'Expiration' in data['data'].index:
                        dte = (data['data']['Expiration'] - data['data']['DataDate']).days

                    # Calculate margin (simplified approximation)
                    margin = data['price'] * data['contracts'] * 100 * self.risk_manager.max_leverage

                self.logger.info(f"{symbol:<16} {data['contracts']:>9} "
                                 f"${data['price']:>7.2f} ${data['value']:>9.2f} "
                                 f"{delta:>8.3f} {dte:>5} ${margin:>11,.0f}")

            self.logger.info("-" * 120)

    def monitor_positions(self, strategy_config):
        """
        Check all positions for exit criteria.

        Args:
            strategy_config: Strategy configuration

        Returns:
            list: Closed positions information
        """
        closed_positions = []

        self.logger.debug(f"[PositionManager] Monitoring {len(self.positions)} positions for exit conditions")

        # Check each position against exit criteria
        for symbol, position in list(self.positions.items()):
            # Skip if no market data available
            if not position.daily_data:
                self.logger.warning(f"[PositionManager] No market data available for {symbol}")
                continue

            # Check exit conditions
            exit_flag, reason = False, None

            # Time-based exit
            if position.days_to_expiry <= strategy_config.get('close_days_to_expiry', 14):
                exit_flag, reason = True, f"Close by DTE {strategy_config.get('close_days_to_expiry', 14)}"

            # Profit target exit
            profit_pct = (
                                     position.avg_entry_price - position.current_price) / position.avg_entry_price if position.is_short else (
                                                                                                                                                         position.current_price - position.avg_entry_price) / position.avg_entry_price
            if profit_pct >= strategy_config.get('profit_target', 0.5):
                exit_flag, reason = True, 'Profit Target'

            # Stop loss exit
            if profit_pct <= -strategy_config.get('stop_loss_threshold', 2.0):
                exit_flag, reason = True, 'Stop Loss'

            if exit_flag:
                self.logger.debug(f"[PositionManager] Exit signal for {symbol}: {reason}")
                # Close the position
                pnl = self.close_position(
                    symbol,
                    position.contracts,
                    position.daily_data[-1],
                    reason
                )

                # Record the closure
                closed_positions.append({
                    'symbol': symbol,
                    'contracts': position.contracts,
                    'reason': reason,
                    'pnl': pnl
                })

                self.logger.info(f"[PositionManager] Closed position {symbol}: {reason}")
                self.logger.info(f"  P&L: ${pnl:.2f}")
            else:
                self.logger.debug(f"[PositionManager] No exit signal for {symbol}")

        return closed_positions

# ========================
# TradeManager Class
# ========================
class TradeManager:
    """
    Handles trade execution and lifecycle management.

    This class selects trades to enter, monitors existing positions for
    exit conditions, and processes daily trading operations.
    """

    def __init__(self, strategy, position_manager, risk_manager, portfolio_rebalancer=None, logger=None):
        """
        Initialize the TradeManager.

        Args:
            strategy: ThetaEngineStrategy instance
            position_manager: PositionManager instance
            risk_manager: RiskManager instance
            portfolio_rebalancer: PortfolioRebalancer instance (optional)
            logger: Logger instance
        """
        self.strategy = strategy
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.config = strategy.config
        self.logger = logger or logging.getLogger('theta_engine')

        # Store the portfolio rebalancer if provided
        self.portfolio_rebalancer = portfolio_rebalancer

        # Log initialization
        self.logger.info("[TradeManager] Initialized")
        self.logger.debug(f"[TradeManager] Strategy: {type(strategy).__name__}")
        self.logger.debug(f"[TradeManager] Config: {self.config}")

    def process_day(self, current_date, daily_data, market_data_by_symbol):
        """
        Process a single trading day with improved margin management and labeled summaries.

        Args:
            current_date: Current date
            daily_data: Daily option data
            market_data_by_symbol: Dictionary of {symbol: market_data}
        """
        self.logger.info(f"[TradeManager] Processing day: {current_date.strftime('%Y-%m-%d')}")

        # Update existing positions with current market data
        self.logger.debug("[TradeManager] Updating positions with current market data")
        self.position_manager.update_positions(current_date, market_data_by_symbol)

        # Print pre-trade summary with PRE-TRADE label
        self.position_manager.print_daily_summary(current_date, "PRE-TRADE Summary")

        # Calculate risk scaling factor for position sizing
        risk_scaling, current_sharpe, z_score = self.risk_manager.calculate_risk_scaling(
            self.position_manager.daily_returns
        )

        # Get current metrics to check for margin issues
        metrics = self.position_manager.get_portfolio_metrics()

        # Initialize portfolio rebalancer if not already created
        if not hasattr(self, 'portfolio_rebalancer') or self.portfolio_rebalancer is None:
            self.logger.info("[TradeManager] Creating new PortfolioRebalancer")
            self.portfolio_rebalancer = PortfolioRebalancer(
                self.config,
                self.position_manager,
                self.risk_manager,
                getattr(self, 'cpd_hedge_manager', None),
                logger=self.logger
            )
        else:
            self.logger.debug("[TradeManager] Using existing PortfolioRebalancer")
            # Log the current values of margin buffer and thresholds
            margin_buffer = getattr(self.portfolio_rebalancer, 'margin_buffer_pct', 'Not set')
            neg_threshold = getattr(self.portfolio_rebalancer, 'negative_margin_threshold', 'Not set')
            cooldown = getattr(self.portfolio_rebalancer, 'rebalance_cooldown_days', 'Not set')
            self.logger.debug(
                f"[TradeManager] Margin buffer: {margin_buffer}, Neg threshold: {neg_threshold}, Cooldown: {cooldown}")

        # Check for rebalancing needs - now passing current_date for cooldown tracking
        analysis = self.portfolio_rebalancer.analyze_portfolio(risk_scaling, current_date)

        # Log whether cooldown is active
        in_cooldown = analysis.get('in_cooldown', False)
        if in_cooldown:
            self.logger.info("[TradeManager] In rebalance cooldown period - limited rebalancing")

        if analysis['needs_rebalancing']:
            reason = "negative margin" if metrics['available_margin'] < 0 else "risk analysis"
            self.logger.info(f"[TradeManager] Portfolio rebalancing needed based on {reason}")
            # Pass current_date to rebalance_portfolio method
            self.portfolio_rebalancer.rebalance_portfolio(current_date, market_data_by_symbol)
        else:
            self.logger.info("[TradeManager] No rebalancing needed based on current settings")

        # Check exit conditions for existing positions
        self.logger.debug("[TradeManager] Checking exit conditions for existing positions")
        self.monitor_positions(market_data_by_symbol)

        # Find and execute new trade opportunities with risk scaling
        self.logger.debug("[TradeManager] Finding new trade opportunities")
        self.find_trade_opportunities(daily_data, risk_scaling)

        # Adjust hedge position if enabled
        if self.config.get('strategy', {}).get('enable_hedging',
                                               False) and daily_data is not None and not daily_data.empty:
            self.logger.debug("[TradeManager] Adjusting hedge position")
            # Get underlying price from the first row of daily data
            first_row = daily_data.iloc[0] if len(daily_data) > 0 else None
            if first_row is not None and 'UnderlyingPrice' in first_row:
                if hasattr(self, 'cpd_hedge_manager') and self.cpd_hedge_manager.enable_hedging:
                    # Use CPD hedge manager if available
                    self.position_manager.adjust_hedge_with_cpd(
                        first_row['UnderlyingPrice'],
                        self.cpd_hedge_manager,
                        daily_data
                    )
                else:
                    # Use standard hedge adjustment
                    self.position_manager.adjust_hedge(first_row['UnderlyingPrice'])
            else:
                self.logger.warning("[TradeManager] Cannot adjust hedge: No underlying price available")

        # Print trade activity for the day
        self.position_manager.print_trade_activity()

        # Print post-trade summary with POST-TRADE label and include rolling metrics
        self.position_manager.print_daily_summary(current_date, "POST-TRADE Summary", include_rolling_metrics=True)

        self.logger.info(f"[TradeManager] Completed processing for {current_date.strftime('%Y-%m-%d')}")

    # Also update process_day_with_cpd method similarly
    def process_day_with_cpd(self, current_date, daily_data, market_data_by_symbol, cpd_hedge_manager):
        """
        Process a single trading day with CPD hedging, improved margin management and labeled summaries.
        Handles both 'constant' and 'ratio' hedge modes.

        Args:
            current_date: Current date
            daily_data: Daily option data
            market_data_by_symbol: Dictionary of {symbol: market_data}
            cpd_hedge_manager: CPDHedgeManager instance
        """
        self.logger.info(f"[TradeManager] Processing day: {current_date.strftime('%Y-%m-%d')}")

        # Debug CPD settings
        self.logger.info(f"[DEBUG] CPD Hedging enabled: {cpd_hedge_manager.enable_hedging}")
        self.logger.info(f"[DEBUG] Hedge mode: {cpd_hedge_manager.hedge_mode}")
        if cpd_hedge_manager.hedge_mode.lower() == 'constant':
            self.logger.info(f"[DEBUG] Target CPD: {cpd_hedge_manager.constant_portfolio_delta}")
        elif cpd_hedge_manager.hedge_mode.lower() == 'ratio':
            self.logger.info(f"[DEBUG] Target Dollar Delta/NLV Ratio: {cpd_hedge_manager.hedge_target_ratio:.4f}")

        # Update the portfolio rebalancer's CPD hedge manager reference
        self.portfolio_rebalancer.cpd_hedge_manager = cpd_hedge_manager

        # Update existing positions with current market data
        self.logger.debug("[TradeManager] Updating positions with current market data")
        self.position_manager.update_positions(current_date, market_data_by_symbol)

        # Print pre-trade summary with PRE-TRADE label
        self.position_manager.print_daily_summary(current_date, "PRE-TRADE Summary")

        # Calculate risk scaling factor for position sizing
        risk_scaling, current_sharpe, z_score = self.risk_manager.calculate_risk_scaling(
            self.position_manager.daily_returns
        )

        # Get current metrics to check for margin issues
        metrics = self.position_manager.get_portfolio_metrics()

        # Check for rebalancing needs - now passing current_date for cooldown tracking
        analysis = self.portfolio_rebalancer.analyze_portfolio(risk_scaling, current_date)
        if analysis['needs_rebalancing']:
            self.logger.info("[TradeManager] Portfolio rebalancing needed based on risk analysis")
            # Pass current_date to rebalance_portfolio method
            self.portfolio_rebalancer.rebalance_portfolio(current_date, market_data_by_symbol)

        # Check exit conditions for existing positions
        self.logger.debug("[TradeManager] Checking exit conditions for existing positions")
        self.monitor_positions(market_data_by_symbol)

        # Find and execute new trade opportunities with risk scaling
        self.logger.debug("[TradeManager] Finding new trade opportunities")
        self.find_trade_opportunities(daily_data, risk_scaling)

        # Adjust hedge position if enabled
        if daily_data is not None and len(daily_data) > 0:
            self.logger.info("[TradeManager] Adjusting hedge position")
            # Get underlying price from the first row of daily data
            first_row = daily_data.iloc[0] if len(daily_data) > 0 else None

            # Safe extraction of underlying price
            underlying_price = None
            if first_row is not None and 'UnderlyingPrice' in first_row:
                underlying_price = first_row['UnderlyingPrice']

            if underlying_price is not None:
                # Adjust hedge using CPD hedge manager
                self.position_manager.adjust_hedge_with_cpd(
                    underlying_price,
                    cpd_hedge_manager,
                    daily_data
                )
            else:
                self.logger.warning("[TradeManager] Cannot adjust hedge: No underlying price available")
        else:
            self.logger.warning("[TradeManager] Cannot adjust hedge: No daily data available")

        # Print trade activity for the day
        self.position_manager.print_trade_activity()

        # Print post-trade summary with POST-TRADE label and include rolling metrics
        self.position_manager.print_daily_summary(current_date, "POST-TRADE Summary", include_rolling_metrics=True)

        self.logger.info(f"[TradeManager] Completed processing for {current_date.strftime('%Y-%m-%d')}")

    def find_trade_opportunities(self, daily_data, risk_scaling=1.0):
        """
        Find and execute new trade opportunities with risk scaling applied.

        Args:
            daily_data: Daily option data
            risk_scaling: Risk scaling factor (default 1.0)
        """
        # Check if daily_data is empty
        if daily_data is None or len(daily_data) == 0 or daily_data.empty:
            self.logger.info("[TradeManager] No data available for trade selection")
            return

        # Check if we can add more positions
        if not self.position_manager.can_add_position():
            self.logger.info("[TradeManager] No capacity for new positions")
            return

        # Get current portfolio metrics
        metrics = self.position_manager.get_portfolio_metrics()
        net_liq = metrics['net_liquidation_value']
        current_margin = metrics['total_margin']
        available_margin = metrics['available_margin']

        # Log risk scaling factor
        self.logger.info(f"[TradeManager] Risk scaling factor: {risk_scaling:.2f}")

        # Calculate maximum allowed margin based on risk scaling
        max_allowed_margin = risk_scaling * net_liq

        # Calculate remaining margin capacity
        remaining_margin_capacity = max(max_allowed_margin - current_margin, 0)

        # If remaining capacity is negligible or we're already at or above max, don't add more
        if remaining_margin_capacity < 0.05 * net_liq:  # Less than 5% of NLV available
            # Enhanced message to explain why margin capacity is insufficient
            if current_margin > max_allowed_margin:
                self.logger.info(f"[TradeManager] Insufficient margin capacity: ${remaining_margin_capacity:,.2f} - "
                                 f"Current margin (${current_margin:,.2f}) exceeds maximum allowed (${max_allowed_margin:,.2f})")
            else:
                self.logger.info(f"[TradeManager] Insufficient margin capacity: ${remaining_margin_capacity:,.2f} - "
                                 f"Remaining capacity below minimum threshold ({0.05 * net_liq:,.2f})")

            # Check if we're in a rebalance cooldown period
            if hasattr(self, 'portfolio_rebalancer') and self.portfolio_rebalancer:
                current_date = None
                for date in self.position_manager.equity_history.keys():
                    current_date = date

                if current_date and self.portfolio_rebalancer.last_rebalance_date:
                    days_since = (current_date - self.portfolio_rebalancer.last_rebalance_date).days
                    if days_since < self.portfolio_rebalancer.rebalance_cooldown_days:
                        self.logger.info(f"[TradeManager] Currently in rebalance cooldown period: "
                                         f"day {days_since} of {self.portfolio_rebalancer.rebalance_cooldown_days}")
                        self.logger.info(f"[TradeManager] Cooldown ends: "
                                         f"{(self.portfolio_rebalancer.last_rebalance_date + pd.Timedelta(days=self.portfolio_rebalancer.rebalance_cooldown_days)).strftime('%Y-%m-%d')}")
            return

        # Current exposure for logging
        current_exposure = current_margin / net_liq if net_liq > 0 else 0

        self.logger.info(f"[Margin Analysis] Current: ${current_margin:,.2f}, Maximum: ${max_allowed_margin:,.2f}")
        self.logger.info(f"  Current Exposure: {current_exposure:.2f}x, Maximum Allowed: {risk_scaling:.2f}x")
        self.logger.info(f"  Remaining Margin Capacity: ${remaining_margin_capacity:,.2f}")

        # Filter for valid entry candidates
        candidates = self.strategy.filter_candidates(daily_data)

        if candidates.empty:
            self.logger.info("[TradeManager] No suitable candidates found")
            return

        # Select the best candidate
        best_candidate = self.strategy.select_best_candidate(candidates)

        if best_candidate is None:
            self.logger.info("[TradeManager] No suitable candidates after filtering")
            return

        # Log the selected candidate
        self.logger.info(f"[TradeManager] Selected: {best_candidate['OptionSymbol']}")
        self.logger.info(f"  Delta: {best_candidate['Delta']:.3f} (target: {self.strategy.delta_target:.3f})")
        self.logger.info(f"  DTE: {best_candidate['DaysToExpiry']} days")
        self.logger.info(f"  Price: ${best_candidate['MidPrice']:.2f}")

        # Calculate margin per contract
        margin_per_contract = best_candidate['MidPrice'] * 100 * self.risk_manager.max_leverage

        # Calculate how many contracts we can add with remaining margin capacity
        max_contracts = int(remaining_margin_capacity / margin_per_contract) if margin_per_contract > 0 else 0

        # Use the risk manager to calculate base position size, but cap by available margin
        base_contracts = self.risk_manager.calculate_position_size(
            best_candidate,
            metrics,
            risk_scaling
        )

        # Take the minimum of max_contracts and base_contracts
        contracts = min(base_contracts, max_contracts)

        # Ensure at least minimum position size if we're adding any
        min_position_size = self.config.get('min_position_size', 1)
        if contracts > 0 and contracts < min_position_size:
            contracts = min_position_size

        # Log detailed position sizing
        self.logger.info(f"[Position Sizing] Using risk scaling: {risk_scaling:.2f}")
        self.logger.info(f"  Remaining margin capacity: ${remaining_margin_capacity:,.2f}")
        self.logger.info(f"  Margin per contract: ${margin_per_contract:,.2f}")
        self.logger.info(f"  Max contracts by margin: {max_contracts}, Base position size: {base_contracts}")
        self.logger.info(f"  Final contract count: {contracts}")

        if contracts <= 0:
            self.logger.info("[TradeManager] No capacity to add position")
            return

        # Execute the trade
        position, symbol = self.position_manager.add_position(best_candidate, contracts)
        self.logger.info(f"[TradeManager] Added {contracts} contracts of {symbol}")

    def monitor_positions(self, market_data_by_symbol):
        """
        Monitor existing positions for exit conditions.

        Args:
            market_data_by_symbol: Dictionary of {symbol: market_data}

        Returns:
            list: Closed positions information
        """
        positions_to_close = []

        self.logger.debug(
            f"[TradeManager] Monitoring {len(self.position_manager.positions)} positions for exit conditions")

        # Check each position against exit criteria
        for symbol, position in list(self.position_manager.positions.items()):
            # Skip if no market data available
            if symbol not in market_data_by_symbol:
                self.logger.warning(f"[TradeManager] No market data available for {symbol}")
                continue

            # Check exit conditions - market_data is not used in this function anymore
            # to avoid Series truth value issues
            exit_flag, reason = self.strategy.check_exit_conditions(
                position,
                None  # Don't pass current_data, use position's current values instead
            )

            if exit_flag:
                self.logger.debug(f"[TradeManager] Exit signal for {symbol}: {reason}")
                positions_to_close.append({
                    'symbol': symbol,
                    'position': position,
                    'reason': reason,
                    'market_data': market_data_by_symbol[symbol]
                })
            else:
                self.logger.debug(f"[TradeManager] No exit signal for {symbol}")

        # Execute position closures
        for close_info in positions_to_close:
            symbol = close_info['symbol']
            position = close_info['position']
            reason = close_info['reason']
            market_data = close_info['market_data']

            # Close the position
            pnl = self.position_manager.close_position(
                symbol,
                position.contracts,
                market_data,
                reason
            )

            self.logger.info(f"[TradeManager] Closed position {symbol}: {reason}")
            self.logger.info(f"  Contracts: {position.contracts}")
            self.logger.info(f"  P&L: ${pnl:.2f}")

        return positions_to_close

    def validate_trade_params(self, option_data):
        """
        Validate option parameters for trading.

        Args:
            option_data: Option data

        Returns:
            dict: Validation result with 'is_valid' flag and 'issues' list
        """
        return self.strategy.validate_trade_params(option_data)

# ========================
# DataManager Class
# ========================
class DataManager:
    """
    Handles data loading and preprocessing for the Theta Engine strategy.

    This class manages loading option data from files, preprocessing the data,
    and providing filtered data for specific dates.
    """

    def __init__(self, logger=None):
        """
        Initialize the DataManager with a logger.

        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger('theta_engine')

    def load_option_data(self, file_path, start_date, end_date):
        """
        Load and preprocess option data from a CSV file.

        Args:
            file_path: Path to the option data file
            start_date: Start date for filtering data
            end_date: End date for filtering data

        Returns:
            DataFrame: Processed option data
        """
        self.logger.info(f"Loading option data from {file_path}...")

        try:
            # Read the CSV file
            df = pd.read_csv(file_path, parse_dates=['DataDate', 'Expiration'])

            # Filter by date range
            df = df[(df['DataDate'] >= start_date) & (df['DataDate'] <= end_date)]

            # Reset index
            df.reset_index(inplace=True, drop=True)

            # Calculate days to expiry
            df['DaysToExpiry'] = (df['Expiration'] - df['DataDate']).dt.days

            # Validate required columns
            required_columns = ['OptionSymbol', 'DataDate', 'Expiration', 'Strike',
                                'Delta', 'Bid', 'Ask', 'IV', 'Type', 'UnderlyingPrice']

            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column {col} missing from data")

            self.logger.info(f"Data loaded successfully: {len(df)} records")
            self.logger.info(f"Date range: {df['DataDate'].min()} to {df['DataDate'].max()}")
            self.logger.info(
                f"Underlying price range: ${df['UnderlyingPrice'].min():.2f} to ${df['UnderlyingPrice'].max():.2f}")

            return df

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def calculate_mid_prices(self, df, normal_spread):
        """
        Calculate option mid prices with spread validation.

        Args:
            df: DataFrame of option data
            normal_spread: Maximum acceptable bid-ask spread percentage

        Returns:
            DataFrame: DataFrame with mid prices calculated
        """
        self.logger.info("Calculating mid prices...")

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Check if 'Last' column exists, if not, try to create it
        if 'Last' not in result_df.columns and 'Close' in result_df.columns:
            result_df['Last'] = result_df['Close']
        elif 'Last' not in result_df.columns:
            result_df['Last'] = 0.0

        # Calculate mid prices
        bid = result_df['Bid']
        ask = result_df['Ask']
        last = result_df['Last']

        # Create a mask for valid bid/ask
        valid_mask = (bid > 0) & (ask > 0)

        # Calculate mid price where valid
        mid = (bid + ask) / 2

        # Calculate spread percentage where mid > 0
        spread_pct = pd.Series(index=result_df.index)
        spread_pct.loc[valid_mask & (mid > 0)] = (ask - bid) / mid

        # Create MidPrice column with default from Last
        result_df['MidPrice'] = last

        # Update where spread is normal
        normal_spread_mask = valid_mask & (spread_pct <= normal_spread)
        result_df.loc[normal_spread_mask, 'MidPrice'] = mid.loc[normal_spread_mask]

        # Log some statistics
        valid_prices = result_df[result_df['MidPrice'] > 0]
        self.logger.info(f"Mid prices calculated: {len(valid_prices)} valid prices")

        if not valid_prices.empty:
            self.logger.info(
                f"Price range: ${valid_prices['MidPrice'].min():.2f} to ${valid_prices['MidPrice'].max():.2f}")
            avg_spread = (valid_prices['Ask'] - valid_prices['Bid']).mean()
            self.logger.info(f"Average bid-ask spread: ${avg_spread:.4f}")

        return result_df

    def get_daily_data(self, df, date):
        """
        Get data for a specific date.

        Args:
            df: DataFrame of option data
            date: Date to filter data for

        Returns:
            DataFrame: Option data for the specified date
        """
        daily_data = df[df['DataDate'] == date].copy()
        self.logger.debug(f"Retrieved {len(daily_data)} records for {date}")
        return daily_data

    def get_market_data_by_symbol(self, daily_data):
        """
        Create a dictionary of market data by option symbol.

        Args:
            daily_data: DataFrame of daily option data

        Returns:
            dict: Dictionary of {symbol: data_row}
        """
        return {row['OptionSymbol']: row for _, row in daily_data.iterrows()}

    def filter_by_expiration(self, df, min_dte=None, max_dte=None):
        """
        Filter options data by days to expiration.

        Args:
            df: DataFrame of option data
            min_dte: Minimum days to expiration (optional)
            max_dte: Maximum days to expiration (optional)

        Returns:
            DataFrame: Filtered option data
        """
        filtered_df = df.copy()

        if min_dte is not None:
            filtered_df = filtered_df[filtered_df['DaysToExpiry'] >= min_dte]

        if max_dte is not None:
            filtered_df = filtered_df[filtered_df['DaysToExpiry'] <= max_dte]

        return filtered_df

    def filter_by_option_type(self, df, option_type=None):
        """
        Filter options data by option type (put/call).

        Args:
            df: DataFrame of option data
            option_type: 'put', 'call', or None for all

        Returns:
            DataFrame: Filtered option data
        """
        if option_type:
            return df[df['Type'].str.lower() == option_type.lower()]
        return df

    def filter_by_delta(self, df, min_delta=None, max_delta=None):
        """
        Filter options data by delta range.

        Args:
            df: DataFrame of option data
            min_delta: Minimum delta value (optional)
            max_delta: Maximum delta value (optional)

        Returns:
            DataFrame: Filtered option data
        """
        filtered_df = df.copy()

        if min_delta is not None:
            filtered_df = filtered_df[filtered_df['Delta'] >= min_delta]

        if max_delta is not None:
            filtered_df = filtered_df[filtered_df['Delta'] <= max_delta]

        return filtered_df

    def calculate_option_greeks(self, df):
        """
        Calculate additional option Greeks if not present in the data.

        This is a placeholder method. In a real implementation, you would
        use an options pricing model to calculate missing Greeks.

        Args:
            df: DataFrame of option data

        Returns:
            DataFrame: Option data with calculated Greeks
        """
        # This is just a placeholder. In a real implementation, you would
        # use the Black-Scholes model or another option pricing model.
        self.logger.info("Calculating option Greeks...")

        # Check if all Greeks are already present
        required_greeks = ['Delta', 'Gamma', 'Theta', 'Vega']
        missing_greeks = [greek for greek in required_greeks if greek not in df.columns]

        if not missing_greeks:
            self.logger.info("All required Greeks already present in data")
            return df

        self.logger.warning(f"Missing Greeks: {missing_greeks}")
        self.logger.warning("Greek calculation not implemented - using dummy values for missing Greeks")

        # Add dummy values for missing Greeks
        result_df = df.copy()
        for greek in missing_greeks:
            if greek == 'Delta':
                # Dummy Delta calculation based on moneyness
                result_df['Delta'] = (result_df['Strike'] - result_df['UnderlyingPrice']) / result_df['UnderlyingPrice']
            elif greek == 'Gamma':
                result_df['Gamma'] = 0.001  # Dummy value
            elif greek == 'Theta':
                result_df['Theta'] = -0.01  # Dummy value
            elif greek == 'Vega':
                result_df['Vega'] = 0.1  # Dummy value

        return result_df


class ReportingSystem:
    """
    Handles reporting and visualization for the Theta Engine strategy.

    This class creates performance reports, charts, and trade logs.
    """

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger('theta_engine')
        os.makedirs(config['paths']['output_dir'], exist_ok=True)

    def generate_html_report(self, equity_history, risk_scaling_history, greeks_history, output_file=None):
        """
        Generate an HTML performance report with interactive charts and detailed statistics.

        Args:
            equity_history: Dictionary of {date: equity_value}
            risk_scaling_history: List of risk scaling dictionaries
            greeks_history: Dictionary of {date: greeks_data}
            output_file: Output file path (optional)

        Returns:
            str: Path to the saved report
        """
        try:
            if output_file is None:
                output_file = os.path.join(
                    self.config['paths']['output_dir'],
                    f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                )

            # Convert equity history to DataFrame
            equity_df = pd.DataFrame.from_dict(equity_history, orient='index', columns=['Equity'])
            equity_df.index.name = 'Date'
            equity_df.index = pd.to_datetime(equity_df.index)

            # Convert risk scaling history to DataFrame
            risk_df = pd.DataFrame(risk_scaling_history)
            if 'date' in risk_df.columns:
                risk_df['date'] = pd.to_datetime(risk_df['date'])
                risk_df.set_index('date', inplace=True)

            # Generate charts
            equity_chart = self.generate_equity_chart(equity_df)
            risk_scaling_chart = self.generate_risk_scaling_chart(risk_df)

            delta_nlv_chart = None
            if greeks_history:
                delta_nlv_chart = self.generate_delta_nlv_chart(greeks_history, equity_df)

            # Build performance metrics
            performance_metrics = self.calculate_performance_metrics(equity_df, risk_df)

            # Add equity history to metrics for use in the heatmap
            performance_metrics['equity_history'] = equity_history

            # Calculate detailed statistics using ffn and convert to HTML table
            stats = ffn.calc_stats(equity_df['Equity'])
            stats_series = stats.stats  # Extract the stats Series
            stats_df = stats_series.to_frame(name="Value")
            stats_df.index.name = "Metric"
            stats_html = stats_df.to_html(classes="table table-bordered", border=0)

            # Build HTML report content, including the detailed stats section
            html_content = self._build_html_report(
                performance_metrics,
                equity_chart,
                risk_scaling_chart,
                delta_nlv_chart,
                stats_html
            )

            with open(output_file, 'w') as f:
                f.write(html_content)

            self.logger.info(f"HTML report saved to {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def generate_equity_chart(self, equity_df):
        """
        Generate a base64-encoded image of the equity curve.

        Args:
            equity_df: DataFrame with equity history

        Returns:
            str: Base64-encoded PNG image
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import io
        import base64

        # Create figure
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Plot equity curve
        ax.plot(equity_df.index, equity_df['Equity'], linewidth=2)

        # Add horizontal line at starting equity
        ax.axhline(y=equity_df['Equity'].iloc[0], color='gray', linestyle='--', alpha=0.7)

        # Format x-axis to show dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)

        # Add grid, title and labels
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('Equity Curve', fontsize=14)
        ax.set_ylabel('Net Liquidation Value ($)')
        ax.set_xlabel('Date')

        # Ensure tight layout
        plt.tight_layout()

        # Convert plot to PNG image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)

        # Encode PNG image to base64 string
        img_str = base64.b64encode(buf.read()).decode('utf-8')

        # Close plot
        plt.close(fig)

        return img_str

    def generate_risk_scaling_chart(self, risk_df):
        """
        Generate a base64-encoded image of the risk scaling history.

        Args:
            risk_df: DataFrame with risk scaling history

        Returns:
            str: Base64-encoded PNG image
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import io
        import base64

        # If DataFrame is empty, return None
        if risk_df.empty:
            self.logger.warning("Risk scaling history is empty, cannot generate chart")
            return None

        # Create figure
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Ensure we have the necessary columns
        required_columns = ['risk_scaling', 'sharpe', 'z_score']
        for col in required_columns:
            if col not in risk_df.columns:
                risk_df[col] = 0
                self.logger.warning(f"Risk scaling chart: missing column '{col}', using zeros")

        # Plot risk scaling
        ax.plot(risk_df.index, risk_df['risk_scaling'], linewidth=2, color='blue', label='Risk Scaling')

        # Add a second y-axis for Sharpe and z-score
        ax2 = ax.twinx()
        ax2.plot(risk_df.index, risk_df['sharpe'], linewidth=1.5, color='green', linestyle='--', label='Sharpe Ratio')
        ax2.plot(risk_df.index, risk_df['z_score'], linewidth=1.5, color='red', linestyle=':', label='Z-Score')

        # Format x-axis to show dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)

        # Add grid, title and labels
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('Risk Scaling History', fontsize=14)
        ax.set_ylabel('Risk Scaling Factor', color='blue')
        ax2.set_ylabel('Sharpe Ratio / Z-Score', color='green')
        ax.set_xlabel('Date')

        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Ensure tight layout
        plt.tight_layout()

        # Convert plot to PNG image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)

        # Encode PNG image to base64 string
        img_str = base64.b64encode(buf.read()).decode('utf-8')

        # Close plot
        plt.close(fig)

        return img_str

    def generate_delta_nlv_chart(self, greeks_history, equity_df):
        """
        Generate a base64-encoded image of the Dollar Delta/NLV ratio.

        Args:
            greeks_history: Dictionary of {date: greeks_data}
            equity_df: DataFrame with equity history

        Returns:
            str: Base64-encoded PNG image
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd
        import io
        import base64

        try:
            # Create DataFrame from greeks history
            dates = []
            dollar_deltas = []

            for date, greeks in greeks_history.items():
                dates.append(date)
                dollar_deltas.append(greeks.get('dollar_total_delta', 0))

            greeks_df = pd.DataFrame({'date': dates, 'dollar_delta': dollar_deltas})
            greeks_df['date'] = pd.to_datetime(greeks_df['date'])
            greeks_df.set_index('date', inplace=True)
            greeks_df.sort_index(inplace=True)

            # Align dates with equity_df
            aligned_df = pd.merge(greeks_df, equity_df, left_index=True, right_index=True, how='inner')

            # Calculate delta/NLV ratio
            if not aligned_df.empty:
                aligned_df['delta_nlv_ratio'] = aligned_df['dollar_delta'] / aligned_df['Equity']

                # Create figure
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111)

                # Plot delta/NLV ratio
                ax.plot(aligned_df.index, aligned_df['delta_nlv_ratio'], linewidth=2, color='purple')

                # Add horizontal line at zero
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

                # Format x-axis to show dates nicely
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.xticks(rotation=45)

                # Add grid, title and labels
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_title('Dollar Delta / NLV Ratio', fontsize=14)
                ax.set_ylabel('Delta/NLV Ratio')
                ax.set_xlabel('Date')

                # Calculate statistics for reference lines
                avg_ratio = aligned_df['delta_nlv_ratio'].mean()
                min_ratio = aligned_df['delta_nlv_ratio'].min()
                max_ratio = aligned_df['delta_nlv_ratio'].max()

                # Add reference lines for min, max, and average
                ax.axhline(y=avg_ratio, color='green', linestyle='--', alpha=0.7, label=f'Avg: {avg_ratio:.4f}')
                ax.axhline(y=min_ratio, color='red', linestyle='--', alpha=0.7, label=f'Min: {min_ratio:.4f}')
                ax.axhline(y=max_ratio, color='blue', linestyle='--', alpha=0.7, label=f'Max: {max_ratio:.4f}')

                # Add legend
                ax.legend(loc='best')

                # Ensure tight layout
                plt.tight_layout()

                # Convert plot to PNG image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)

                # Encode PNG image to base64 string
                img_str = base64.b64encode(buf.read()).decode('utf-8')

                # Close plot
                plt.close(fig)

                return img_str

            return None
        except Exception as e:
            self.logger.error(f"Error generating delta/NLV chart: {e}")
            return None

    def calculate_performance_metrics(self, equity_df, risk_df):
        """
        Calculate performance metrics from equity history.

        Args:
            equity_df: DataFrame with equity history
            risk_df: DataFrame with risk scaling history

        Returns:
            dict: Performance metrics
        """
        try:
            # Basic performance stats
            start_value = equity_df['Equity'].iloc[0]
            end_value = equity_df['Equity'].iloc[-1]
            trading_days = len(equity_df)
            years = trading_days / 252

            # Calculate returns
            total_return = (end_value / start_value) - 1
            cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else 0

            # Calculate daily returns
            equity_df['daily_return'] = equity_df['Equity'].pct_change()

            # Calculate risk metrics
            volatility = equity_df['daily_return'].std() * np.sqrt(252)  # Annualized

            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = (equity_df['daily_return'].mean() * 252) / volatility if volatility > 0 else 0

            # Maximum drawdown
            equity_df['cumulative_return'] = (1 + equity_df['daily_return']).cumprod()
            equity_df['running_max'] = equity_df['cumulative_return'].cummax()
            equity_df['drawdown'] = (equity_df['cumulative_return'] - equity_df['running_max']) / equity_df[
                'running_max']
            max_drawdown = equity_df['drawdown'].min()

            # Risk scaling metrics
            if not risk_df.empty and 'risk_scaling' in risk_df.columns:
                avg_risk_scaling = risk_df['risk_scaling'].mean()
                min_risk_scaling = risk_df['risk_scaling'].min()
                max_risk_scaling = risk_df['risk_scaling'].max()
            else:
                avg_risk_scaling = min_risk_scaling = max_risk_scaling = 0

            return {
                'total_return': total_return,
                'cagr': cagr,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'start_value': start_value,
                'end_value': end_value,
                'avg_risk_scaling': avg_risk_scaling,
                'min_risk_scaling': min_risk_scaling,
                'max_risk_scaling': max_risk_scaling
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {
                'total_return': 0,
                'cagr': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'start_value': 0,
                'end_value': 0,
                'avg_risk_scaling': 0,
                'min_risk_scaling': 0,
                'max_risk_scaling': 0
            }

    def _generate_config_table(self):
        """
        Generate HTML for configuration settings tables grouped by category with a more compact layout.

        Returns:
            str: HTML for configuration tables
        """
        if not hasattr(self, 'config') or not self.config:
            return "<p>Configuration settings not available</p>"

        html = """
        <div class="accordion">
        """

        # Define categories and their display names
        categories = [
            ('portfolio', 'Portfolio Settings'),
            ('risk', 'Risk Management Settings'),
            ('strategy', 'Strategy Parameters'),
            ('trading', 'Trading Parameters'),
            ('paths', 'File Paths'),
            ('dates', 'Date Range')
        ]

        # Create accordion sections for each category
        for category_id, (category_key, category_name) in enumerate(categories):
            if category_key in self.config:
                category_config = self.config[category_key]

                html += f"""
                <div class="accordion-item">
                    <h3 class="accordion-header" id="heading{category_id}">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                                data-bs-target="#collapse{category_id}" aria-expanded="false" aria-controls="collapse{category_id}">
                            {category_name}
                        </button>
                    </h3>
                    <div id="collapse{category_id}" class="accordion-collapse collapse" aria-labelledby="heading{category_id}">
                        <div class="accordion-body">
                            <table class="config-table table table-sm table-striped">
                                <thead>
                                    <tr>
                                        <th>Parameter</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                """

                # Add rows for each parameter in this category
                for param, value in category_config.items():
                    # Format dates specially
                    if isinstance(value, pd.Timestamp) or isinstance(value, datetime):
                        formatted_value = value.strftime('%Y-%m-%d')
                    # Format numeric values
                    elif isinstance(value, float):
                        if 0 < abs(value) < 1:  # Small values like 0.25 should be shown as percentages
                            formatted_value = f"{value:.2%}"
                        else:
                            formatted_value = f"{value:,.2f}"
                    elif isinstance(value, int):
                        formatted_value = f"{value:,}"
                    # Format file paths to be shorter
                    elif isinstance(value, str) and ('\\' in value or '/' in value):
                        path_parts = value.split('\\' if '\\' in value else '/')
                        if len(path_parts) > 3:
                            # Show only the last 3 parts of the path
                            formatted_value = ".../" + "/".join(path_parts[-3:])
                        else:
                            formatted_value = value
                    else:
                        formatted_value = str(value)

                    html += f"""
                                    <tr>
                                        <td>{param}</td>
                                        <td>{formatted_value}</td>
                                    </tr>
                    """

                html += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                """

        html += """
        </div>
        """
        return html

    def generate_returns_heatmap(self, equity_df):
        """
        Generate an HTML returns heatmap by month and year.

        Args:
            equity_df: DataFrame with equity history

        Returns:
            str: HTML for the returns heatmap
        """
        try:
            # Check if we have sufficient data
            if equity_df is None or len(equity_df) < 5 or 'Equity' not in equity_df.columns:
                return "<p>Insufficient data for returns heatmap</p>"

            # Calculate daily returns
            equity_df['daily_return'] = equity_df['Equity'].pct_change()

            # Convert index to datetime if it's not already
            if not isinstance(equity_df.index, pd.DatetimeIndex):
                equity_df.index = pd.to_datetime(equity_df.index)

            # Resample to monthly returns
            monthly_returns = equity_df['daily_return'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )

            # Check if we have enough monthly returns
            if len(monthly_returns) < 2:
                return "<p>Insufficient monthly data for returns heatmap</p>"

            # Extract year and month directly from the datetime index
            returns_data = []
            for date, value in monthly_returns.items():
                returns_data.append({
                    'Year': date.year,
                    'Month': date.month,
                    'Return': value
                })

            # Convert to DataFrame
            monthly_df = pd.DataFrame(returns_data)

            # Check if we have enough unique years/months
            if len(monthly_df['Year'].unique()) < 1 or len(monthly_df['Month'].unique()) < 2:
                return "<p>Insufficient data variation for returns heatmap</p>"

            # Create pivot table
            returns_pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')

            # Define month names for columns
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            # Map column indices to month names
            col_mapping = {i: month_names[i - 1] for i in range(1, 13) if i in returns_pivot.columns}
            returns_pivot = returns_pivot.rename(columns=col_mapping)

            # Generate HTML for the heatmap
            html = """
            <div class="returns-heatmap">
                <table class="heatmap-table">
                    <thead>
                        <tr>
                            <th>Year</th>
            """

            # Add month headers
            for month in returns_pivot.columns:
                html += f"<th>{month}</th>"

            # Add annual returns column
            html += "<th>Annual</th></tr></thead><tbody>"

            # Calculate max absolute return for color scaling
            max_abs_return = max(
                abs(returns_pivot.fillna(0).values.min()),
                abs(returns_pivot.fillna(0).values.max())
            )

            if pd.isna(max_abs_return) or max_abs_return == 0:
                max_abs_return = 0.05  # Default to 5% if no valid returns

            # Add data rows
            for year in returns_pivot.index:
                html += f"<tr><td>{year}</td>"

                # Calculate annual return
                annual_return = (1 + returns_pivot.loc[year].fillna(0)).prod() - 1

                # Add monthly returns with color coding
                for month in returns_pivot.columns:
                    if month in returns_pivot.columns and not pd.isna(returns_pivot.loc[year, month]):
                        value = returns_pivot.loc[year, month]
                        # Scale color intensity based on return magnitude
                        intensity = min(abs(value) / max_abs_return * 0.8, 0.8)
                        if value > 0:
                            bgcolor = f"rgba(0, 128, 0, {intensity})"  # Green for positive
                            color = "white" if intensity > 0.5 else "black"
                        else:
                            bgcolor = f"rgba(220, 0, 0, {intensity})"  # Red for negative
                            color = "white" if intensity > 0.5 else "black"

                        html += f'<td style="background-color: {bgcolor}; color: {color}">{value:.2%}</td>'
                    else:
                        html += '<td></td>'

                # Add annual return
                if not pd.isna(annual_return):
                    # Color annual return
                    intensity = min(abs(annual_return) / max_abs_return * 0.8, 0.8)
                    if annual_return > 0:
                        bgcolor = f"rgba(0, 128, 0, {intensity})"  # Green for positive
                        color = "white" if intensity > 0.5 else "black"
                    else:
                        bgcolor = f"rgba(220, 0, 0, {intensity})"  # Red for negative
                        color = "white" if intensity > 0.5 else "black"

                    html += f'<td style="background-color: {bgcolor}; color: {color}"><strong>{annual_return:.2%}</strong></td>'
                else:
                    html += '<td></td>'

                html += "</tr>"

            # Calculate and add average monthly returns
            html += "<tr><td><strong>Avg</strong></td>"

            # Add monthly averages
            for month in returns_pivot.columns:
                monthly_avg = returns_pivot[month].mean()
                if not pd.isna(monthly_avg):
                    # Color monthly average
                    intensity = min(abs(monthly_avg) / max_abs_return * 0.8, 0.8)
                    if monthly_avg > 0:
                        bgcolor = f"rgba(0, 128, 0, {intensity})"  # Green for positive
                        color = "white" if intensity > 0.5 else "black"
                    else:
                        bgcolor = f"rgba(220, 0, 0, {intensity})"  # Red for negative
                        color = "white" if intensity > 0.5 else "black"

                    html += f'<td style="background-color: {bgcolor}; color: {color}"><strong>{monthly_avg:.2%}</strong></td>'
                else:
                    html += '<td></td>'

            # Calculate and add average annual return
            avg_annual = returns_pivot.fillna(0).apply(lambda x: (1 + x).prod() - 1, axis=1).mean()

            if not pd.isna(avg_annual):
                intensity = min(abs(avg_annual) / max_abs_return * 0.8, 0.8)
                if avg_annual > 0:
                    bgcolor = f"rgba(0, 128, 0, {intensity})"  # Green for positive
                    color = "white" if intensity > 0.5 else "black"
                else:
                    bgcolor = f"rgba(220, 0, 0, {intensity})"  # Red for negative
                    color = "white" if intensity > 0.5 else "black"

                html += f'<td style="background-color: {bgcolor}; color: {color}"><strong>{avg_annual:.2%}</strong></td>'
            else:
                html += '<td></td>'

            html += "</tr>"

            html += """
                    </tbody>
                </table>
            </div>
            """

            return html
        except Exception as e:
            self.logger.error(f"Error generating returns heatmap: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return f"<p>Error generating returns heatmap: {str(e)}</p>"

    def _build_html_report(self, metrics, equity_chart, risk_scaling_chart, delta_nlv_chart, stats_html):
        """
        Build HTML report content with improved layout including returns heatmap and collapsible config settings.

        Args:
            metrics: Dictionary of performance metrics
            equity_chart: Base64-encoded equity chart image
            risk_scaling_chart: Base64-encoded risk scaling chart image
            delta_nlv_chart: Base64-encoded delta/NLV chart image
            stats_html: HTML string for detailed statistics (generated by ffn)

        Returns:
            str: HTML content
        """
        # Generate configuration settings table
        config_html = self._generate_config_table()

        # Get equity data from the passed parameters instead of trying to access it as an instance attribute
        # We need to convert the equity_df in the calling function and pass it here if needed
        try:
            # Try to generate returns heatmap using the equity data available from ThetaEngine
            equity_df = pd.DataFrame.from_dict(metrics.get('equity_history', {}), orient='index', columns=['Equity'])
            equity_df.index.name = 'Date'
            returns_heatmap_html = self.generate_returns_heatmap(equity_df)
        except (AttributeError, KeyError, TypeError) as e:
            self.logger.warning(f"Could not generate returns heatmap: {e}")
            returns_heatmap_html = "<p>Returns heatmap not available: equity history data is missing or invalid.</p>"

        # Define JavaScript separately to avoid issues with f-strings
        js_code = """
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const tables = document.querySelectorAll('table:not(.heatmap-table)');
                tables.forEach(table => {
                    if (!table.parentElement.classList.contains('table-responsive')) {
                        const wrapper = document.createElement('div');
                        wrapper.classList.add('table-responsive');
                        table.parentNode.insertBefore(wrapper, table);
                        wrapper.appendChild(table);

                        if (!table.classList.contains('table')) {
                            table.classList.add('table', 'table-striped', 'table-hover', 'table-sm');
                        }
                    }
                });
            });
        </script>
        """

        # Create the HTML content
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Theta Engine Performance Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; background-color: #f8f9fa; }}
                .container {{ max-width: 1200px; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; margin-top: 1.5rem; }}
                .card {{ border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }}
                .card-header {{ background-color: #f8f9fa; font-weight: bold; }}
                .metrics-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }}
                .metric-card {{ flex: 1; min-width: 200px; text-align: center; padding: 15px; }}
                .metric-title {{ font-weight: bold; margin-bottom: 5px; color: #6c757d; }}
                .metric-value {{ font-size: 24px; color: #0066cc; }}
                .chart-container {{ margin: 20px 0; }}
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
                .config-table {{ font-size: 0.85rem; }}
                .config-section {{ margin-bottom: 15px; }}
                .accordion-button {{ padding: 0.5rem 1rem; }}
                .accordion-body {{ padding: 1rem; }}
                table.heatmap-table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
                .heatmap-table th, .heatmap-table td {{ padding: 6px 8px; text-align: center; border: 1px solid #dee2e6; }}
                .heatmap-table th {{ background-color: #f8f9fa; font-weight: bold; }}
                .heatmap-table td {{ width: 7%; }}
                .heatmap-table tr td:first-child {{ text-align: center; font-weight: bold; background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="mb-4">Theta Engine Performance Report</h1>

                <div class="card">
                    <div class="card-header">Performance Summary</div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-4">
                                <div class="metric-card">
                                    <div class="metric-title">Total Return</div>
                                    <div class="metric-value {('positive' if metrics['total_return'] >= 0 else 'negative')}">{metrics['total_return']:.2%}</div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="metric-card">
                                    <div class="metric-title">CAGR</div>
                                    <div class="metric-value {('positive' if metrics['cagr'] >= 0 else 'negative')}">{metrics['cagr']:.2%}</div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="metric-card">
                                    <div class="metric-title">Sharpe Ratio</div>
                                    <div class="metric-value">{metrics['sharpe_ratio']:.2f}</div>
                                </div>
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-md-4">
                                <div class="metric-card">
                                    <div class="metric-title">Volatility (Ann.)</div>
                                    <div class="metric-value">{metrics['volatility']:.2%}</div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="metric-card">
                                    <div class="metric-title">Max Drawdown</div>
                                    <div class="metric-value {('positive' if metrics['max_drawdown'] >= 0 else 'negative')}">{metrics['max_drawdown']:.2%}</div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="metric-card">
                                    <div class="metric-title">Avg Risk Scaling</div>
                                    <div class="metric-value">{metrics['avg_risk_scaling']:.2f}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">Monthly Returns Heatmap</div>
                    <div class="card-body">
                        {returns_heatmap_html}
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">Equity Curve</div>
                    <div class="card-body chart-container text-center">
                        {"<img src='data:image/png;base64," + equity_chart + "' alt='Equity Curve' class='img-fluid'>" if equity_chart else "<p>Equity chart not available</p>"}
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">Risk Scaling History</div>
                    <div class="card-body chart-container text-center">
                        {"<img src='data:image/png;base64," + risk_scaling_chart + "' alt='Risk Scaling History' class='img-fluid'>" if risk_scaling_chart else "<p>Risk scaling chart not available</p>"}
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">Dollar Delta / NLV Ratio</div>
                    <div class="card-body chart-container text-center">
                        {"<img src='data:image/png;base64," + delta_nlv_chart + "' alt='Dollar Delta / NLV Ratio' class='img-fluid'>" if delta_nlv_chart else "<p>Dollar Delta / NLV ratio chart not available</p>"}
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">Configuration Settings</div>
                    <div class="card-body">
                        {config_html}
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">Detailed Statistics</div>
                    <div class="card-body">
                        {stats_html}
                    </div>
                </div>
            </div>
            {js_code}
        </body>
        </html>
        """
        return html

    def save_trade_log(self, trade_log, output_file=None):
        """
        Save the trade log to a CSV file.

        Args:
            trade_log: DataFrame of trade log
            output_file: Output file path (optional)

        Returns:
            str: Path to the saved file
        """
        try:
            # If output file not specified, generate a filename
            if output_file is None:
                output_file = os.path.join(
                    self.config['paths']['output_dir'],
                    f"trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )

            # Save to CSV
            trade_log.to_csv(output_file, index=False)
            self.logger.info(f"Trade log saved to {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Error saving trade log: {e}")
            return None

    def print_performance_metrics(self, metrics):
        """
        Print performance metrics to the log.

        Args:
            metrics: Dictionary of performance metrics
        """
        self.logger.info("\n=== Performance Metrics ===")

        # Format metrics for display
        for key, value in metrics.items():
            if isinstance(value, float):
                if key in ['total_return', 'cagr', 'volatility', 'max_drawdown']:
                    self.logger.info(f"{key.replace('_', ' ').title()}: {value:.2%}")
                elif key in ['sharpe_ratio', 'avg_risk_scaling', 'min_risk_scaling', 'max_risk_scaling']:
                    self.logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}")
                elif key in ['start_value', 'end_value']:
                    self.logger.info(f"{key.replace('_', ' ').title()}: ${value:,.2f}")
                else:
                    self.logger.info(f"{key.replace('_', ' ').title()}: {value}")
            else:
                self.logger.info(f"{key.replace('_', ' ').title()}: {value}")

        self.logger.info("=" * 30)

# ========================
# ThetaEngine Class
# ========================
class ThetaEngine:
    """
    Main orchestration class for the Theta Engine strategy.

    This class coordinates all components of the strategy, including
    data loading, position management, trade execution, and reporting.
    """

    def __init__(self, config, logger=None):
        """Initialize the ThetaEngine with configuration."""
        self.config = config
        self.logger = logger or logging.getLogger('theta_engine')

        # Create component instances
        self.data_manager = DataManager(logger=self.logger)
        self.strategy = ThetaEngineStrategy(config['strategy'], logger=self.logger)
        self.risk_manager = RiskManager(config, logger=self.logger)

        # Create position manager first
        self.position_manager = PositionManager(
            config['portfolio']['initial_capital'],
            config,
            self.risk_manager,
            logger=self.logger
        )

        # Now create CPD hedge manager with position_manager reference
        self.cpd_hedge_manager = CPDHedgeManager(
            config['strategy'],
            self.risk_manager,
            position_manager=self.position_manager,  # Pass position_manager reference
            logger=self.logger
        )

        # Create portfolio rebalancer with full configuration
        self.portfolio_rebalancer = PortfolioRebalancer(
            config,
            self.position_manager,
            self.risk_manager,
            self.cpd_hedge_manager,
            logger=self.logger
        )

        # Create trade manager and pass portfolio_rebalancer to it
        self.trade_manager = TradeManager(
            self.strategy,
            self.position_manager,
            self.risk_manager,
            logger=self.logger
        )

        # Set the portfolio_rebalancer for the trade manager
        self.trade_manager.portfolio_rebalancer = self.portfolio_rebalancer

        # For tracking trades and performance
        self.trade_log = []

        # For tracking risk scaling history
        self.risk_scaling_history = []

        # For tracking Greeks history
        self.greeks_history = {}

        self.logger.info(
            f"[ThetaEngine] Initialized with {config['portfolio']['initial_capital']:,.2f} initial capital")
        self.logger.debug(f"[ThetaEngine] Configuration: {config}")

    def run(self, data):
        """
        Execute the strategy day by day with improved margin management and custom progress logging.

        Args:
            data: DataFrame of option data for the entire period

        Returns:
            tuple: (final_metrics, equity_history, trade_log, risk_scaling_history)
        """
        # Check for required columns
        required_columns = ['DataDate', 'OptionSymbol', 'MidPrice', 'Expiration', 'Strike', 'Delta']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' is missing from input data")

        # Get unique trading dates from the data
        unique_dates = sorted(data['DataDate'].unique())
        total_dates = len(unique_dates)

        # Get delta target for display
        delta_target = self.config.get('strategy', {}).get('delta_target', 0)
        delta_target_str = f"DT{abs(delta_target):.2f}".replace('0.', '.')

        self.logger.info(f"[ThetaEngine] Starting execution over {total_dates} trading days")
        self.logger.info(f"  Date range: {unique_dates[0]} to {unique_dates[-1]}")

        # Initialize previous NLV from starting capital
        prev_nlv = self.position_manager.initial_capital
        # Optionally, record the starting capital with a date one day before the first trading day.
        initial_date = unique_dates[0] - pd.Timedelta(days=1)
        self.position_manager.equity_history[initial_date] = prev_nlv
        self.logger.info(f"Initial equity set on {initial_date.strftime('%Y-%m-%d')}: ${prev_nlv:,.2f}")

        # Reset risk scaling history for reporting
        self.risk_scaling_history = []

        # Process each trading day
        for i, current_date in enumerate(unique_dates):
            progress_pct = (i / total_dates) * 100
            # Modified progress message with delta target and without separator lines
            self.logger.info(
                f"Progress: {i}/{total_dates} days ({progress_pct:.1f}%) - Processing {current_date.strftime('%Y-%m-%d')} - {delta_target_str}"
            )

            # Filter market data for the current day
            daily_data = data[data['DataDate'] == current_date].copy()
            market_data_by_symbol = {row['OptionSymbol']: row for _, row in daily_data.iterrows()}

            # Process the day (with CPD hedging if enabled) - now passing current_date explicitly
            if self.cpd_hedge_manager.enable_hedging:
                self.trade_manager.process_day_with_cpd(
                    current_date,  # Explicitly pass current_date
                    daily_data,
                    market_data_by_symbol,
                    self.cpd_hedge_manager
                )
            else:
                self.trade_manager.process_day(
                    current_date,  # Explicitly pass current_date
                    daily_data,
                    market_data_by_symbol
                )

            # After processing, recalculate portfolio metrics and update equity history
            current_nlv = self.position_manager.get_portfolio_metrics()['net_liquidation_value']
            self.position_manager.equity_history[current_date] = current_nlv

            # For days beyond the first trading day, compute daily return relative to prev_nlv
            if i > 0:
                daily_pnl = current_nlv - prev_nlv
                daily_return = daily_pnl / prev_nlv if prev_nlv != 0 else 0

                # Store daily return with breakdown details
                self.position_manager.daily_returns.append({
                    'date': current_date,
                    'return': daily_return,
                    'pnl': daily_pnl,
                    # Optional breakdown components
                    'unrealized_pnl_change': self.position_manager.daily_returns[-1].get('unrealized_pnl_change',
                                                                                         0) if self.position_manager.daily_returns else 0,
                    'realized_pnl': self.position_manager.today_realized_pnl,
                    'hedge_pnl_change': 0  # Adjust as needed if hedge pnl is computed separately
                })

                self.logger.info(f"[Daily Return] {daily_return:.2%} (${daily_pnl:.2f})")
                self.logger.info(f"  (Previous NLV: ${prev_nlv:,.2f} -> Current NLV: ${current_nlv:,.2f})")

            # Update prev_nlv for the next iteration
            prev_nlv = current_nlv

            # Also store risk scaling and Greeks history
            risk_scaling, sharpe, z_score = self.risk_manager.calculate_risk_scaling(
                self.position_manager.daily_returns
            )
            self.risk_scaling_history.append({
                'date': current_date,
                'risk_scaling': risk_scaling,
                'sharpe': sharpe if sharpe is not None and not pd.isna(sharpe) else 0,
                'z_score': z_score if z_score is not None and not pd.isna(z_score) else 0
            })
            self.greeks_history[current_date] = self.position_manager.get_portfolio_greeks()

        # End-of-simulation summary
        position_count = len(self.position_manager.positions)
        final_metrics = self.position_manager.get_portfolio_metrics()

        self.logger.info("\n=== Strategy Execution Summary ===")
        self.logger.info(f"Initial Capital: ${self.position_manager.initial_capital:,.2f}")
        self.logger.info(f"Final Net Liquidation Value: ${final_metrics['net_liquidation_value']:,.2f}")
        self.logger.info(
            f"Total Return: {(final_metrics['net_liquidation_value'] / self.position_manager.initial_capital - 1):.2%}")
        self.logger.info(f"Open Positions: {position_count}")
        self.logger.info(f"Trading Days: {total_dates}")

        # Build trade log DataFrame from transactions in each position
        trade_log_data = []
        for symbol, position in self.position_manager.positions.items():
            for trade in position.transactions:
                trade_log_data.append({
                    'symbol': symbol,
                    **trade
                })
        trade_log_df = pd.DataFrame(trade_log_data)
        risk_scaling_df = pd.DataFrame(self.risk_scaling_history)

        return final_metrics, self.position_manager.equity_history, trade_log_df, risk_scaling_df

    def generate_report(self, output_file=None):
        """
        Generate a performance report.

        Args:
            output_file: Output file path (optional)

        Returns:
            dict: Performance metrics
        """
        if not self.position_manager.daily_returns:
            self.logger.warning("No performance data available for reporting")
            return None

        try:
            # Create a DataFrame from equity history
            equity_df = pd.DataFrame.from_dict(self.position_manager.equity_history, orient='index', columns=['Equity'])
            equity_df.index.name = 'Date'

            # Calculate basic performance metrics
            start_value = equity_df['Equity'].iloc[0]
            end_value = equity_df['Equity'].iloc[-1]
            trading_days = len(equity_df)
            years = trading_days / 252

            # Calculate returns
            total_return = (end_value / start_value) - 1
            cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else 0

            # Calculate volatility and Sharpe ratio
            if len(self.position_manager.daily_returns) > 1:
                returns_series = pd.Series([r['return'] for r in self.position_manager.daily_returns])
                volatility = returns_series.std() * np.sqrt(252)  # Annualized
                avg_return = returns_series.mean() * 252  # Annualized

                # Sharpe ratio (assuming 0% risk-free rate)
                sharpe_ratio = avg_return / volatility if volatility > 0 else 0

                # Maximum drawdown
                cumulative = (1 + returns_series).cumprod()
                peak = cumulative.cummax()
                drawdown = (cumulative - peak) / peak
                max_drawdown = drawdown.min()
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0

            # Log performance metrics
            self.logger.info("\n=== Performance Metrics ===")
            self.logger.info(f"Total Return: {total_return:.2%}")
            self.logger.info(f"CAGR: {cagr:.2%}")
            self.logger.info(f"Volatility (annualized): {volatility:.2%}")
            self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            self.logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")

            # Save report to file if requested
            if output_file:
                # Create a nicer formatted report using HTML
                self._save_html_report(
                    equity_df,
                    pd.DataFrame(self.risk_scaling_history),
                    pd.DataFrame(self.position_manager.dollar_delta_to_nlv_history),
                    output_file
                )

            return {
                'total_return': total_return,
                'cagr': cagr,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }

        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _save_html_report(self, equity_df, risk_scaling_df, delta_ratio_df, output_file):
        """
        Save an HTML performance report.

        Args:
            equity_df: Equity history DataFrame
            risk_scaling_df: Risk scaling history DataFrame
            delta_ratio_df: Dollar delta to NLV ratio history DataFrame
            output_file: Output file path
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Generate the HTML content
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Theta Engine Performance Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1, h2, h3 { color: #333; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .metric-box { background-color: #f5f5f5; border-radius: 5px; padding: 15px; margin-bottom: 10px; }
                    .metric-title { font-weight: bold; margin-bottom: 5px; }
                    .metric-value { font-size: 24px; color: #0066cc; }
                    .metrics-row { display: flex; gap: 15px; margin-bottom: 20px; }
                    .metric-box { flex: 1; }
                    .chart-container { margin: 30px 0; }
                    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                    th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Theta Engine Performance Report</h1>
            """

            # Add strategy details
            html_content += """
                    <h2>Strategy Configuration</h2>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
            """

            for section in ['strategy', 'risk', 'portfolio']:
                if section in self.config:
                    html_content += f"<tr><th colspan='2'>{section.capitalize()} Parameters</th></tr>"
                    for key, value in self.config[section].items():
                        html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"

            html_content += """
                    </table>
            """

            # Add performance metrics
            start_value = equity_df['Equity'].iloc[0]
            end_value = equity_df['Equity'].iloc[-1]
            total_return = (end_value / start_value) - 1
            trading_days = len(equity_df)
            years = trading_days / 252
            cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else 0

            # Calculate volatility, Sharpe ratio and max drawdown
            if len(self.position_manager.daily_returns) > 1:
                returns_series = pd.Series([r['return'] for r in self.position_manager.daily_returns])
                volatility = returns_series.std() * np.sqrt(252)  # Annualized
                avg_return = returns_series.mean() * 252  # Annualized
                sharpe_ratio = avg_return / volatility if volatility > 0 else 0

                cumulative = (1 + returns_series).cumprod()
                peak = cumulative.cummax()
                drawdown = (cumulative - peak) / peak
                max_drawdown = drawdown.min()
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0

            html_content += """
                    <h2>Performance Metrics</h2>
                    <div class="metrics-row">
                        <div class="metric-box">
                            <div class="metric-title">Total Return</div>
                            <div class="metric-value">{:.2%}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-title">CAGR</div>
                            <div class="metric-value">{:.2%}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-title">Volatility</div>
                            <div class="metric-value">{:.2%}</div>
                        </div>
                    </div>
                    <div class="metrics-row">
                        <div class="metric-box">
                            <div class="metric-title">Sharpe Ratio</div>
                            <div class="metric-value">{:.2f}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-title">Max Drawdown</div>
                            <div class="metric-value">{:.2%}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-title">Starting Capital</div>
                            <div class="metric-value">${:,.0f}</div>
                        </div>
                    </div>
            """.format(
                total_return, cagr, volatility, sharpe_ratio, max_drawdown,
                self.position_manager.initial_capital
            )

            # Add charts (placeholder instructions)
            html_content += """
                    <h2>Equity Curve</h2>
                    <div class="chart-container">
                        <p>Chart would be embedded here. Use a library like Plotly or Chart.js to create interactive charts.</p>
                    </div>

                    <h2>Risk Scaling History</h2>
                    <div class="chart-container">
                        <p>Risk scaling chart would be embedded here.</p>
                    </div>

                    <h2>Dollar Delta/NLV Ratio</h2>
                    <div class="chart-container">
                        <p>Dollar delta/NLV ratio chart would be embedded here.</p>
                    </div>
            """

            # Close the HTML
            html_content += """
                </div>
            </body>
            </html>
            """

            # Save the HTML file
            with open(output_file, 'w') as f:
                f.write(html_content)

            self.logger.info(f"HTML report saved to {output_file}")

        except Exception as e:
            self.logger.error(f"Error saving HTML report: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


def load_configuration():
    """
    Create the configuration dictionary with all settings including improved margin management.

    Returns:
        dict: Configuration dictionary
    """
    config = {
        # File paths
        'paths': {
            'input_file': r"C:\Users\alexp\OneDrive\Gdrive\Trading\Data Outputs\GLD_Combined.csv",
            # CSV file with options data
            'output_dir': r"C:\Users\alexp\OneDrive\Gdrive\Trading\Data Outputs\scenario_results",
            # Directory for output files
            'trades_output_file': r"C:\Users\alexp\OneDrive\Gdrive\Trading\Data Outputs\SPY_Theta_Engine_Trades.csv"
            # Trade log path
        },

        # Date range for backtesting
        'dates': {
            'start_date': pd.to_datetime("2024-01-01"),  # Start of backtest period
            'end_date': pd.to_datetime("2024-12-31")  # End of backtest period
        },

        # Portfolio settings
        'portfolio': {
            'initial_capital': 100000,  # Starting capital in dollars
            'max_leverage': 12,  # Leverage multiplier for margin requirements
            'max_nlv_percent': 1  # Maximum single position size as fraction of NLV
        },

        # Risk management settings
        'risk': {
            'rolling_window': 21,  # Days for rolling performance metrics calculation
            'target_z': 0,  # Z-score for full exposure
            'min_z': -2.0,  # Z-score for minimum exposure
            'min_investment': 0.25  # Minimum exposure level during poor performance
        },

        # Strategy parameters
        'strategy': {
            'enable_hedging': True,  # Whether to use delta hedging
            'hedge_mode': 'ratio',  # Hedging approach 'constant' or 'ratio'
            'constant_portfolio_delta': 0.05,  # Target delta for the entire portfolio
            'hedge_target_ratio': 1.75,  # Multiplier for alternative hedge modes
            'days_to_expiry_min': 60,  # Minimum DTE for new positions
            'days_to_expiry_max': 90,  # Maximum DTE for new positions
            'is_short': True,  # Whether to sell (True) or buy options
            'delta_target': -0.05,  # Target delta for option selection
            'profit_target': 0.65,  # Take profit at this percentage of premium
            'stop_loss_threshold': 2.5,  # Stop loss at this multiple of premium
            'close_days_to_expiry': 14,  # Close positions when reaching this DTE
            'delta_tolerance': 1.5,  # Allowable deviation from target delta
            'min_position_size': 1  # Minimum contracts per position
        },

        # Trading parameters
        'trading': {
            'normal_spread': 0.60  # Maximum acceptable bid-ask spread as percentage
        },

        # Margin management parameters
        'margin_buffer_pct': 0.10,  # Buffer over risk-scaled max margin
        'negative_margin_threshold': -0.05,  # Trigger rebalancing when margin < -5% of NLV
        'rebalance_cooldown_days': 3,  # Days to wait after rebalancing
        'forced_rebalance_threshold': -0.10,  # Force rebalance if margin < -10% of NLV

        # Position reduction parameters
        'max_position_reduction_pct': 0.25,  # Max position reduction during normal rebalancing
        'losing_position_max_reduction_pct': 0.40,  # Max reduction for losing positions
        'urgent_reduction_pct': 0.50,  # Max reduction during urgent rebalancing
    }

    return config


def main(custom_config=None, preloaded_data=None):
    """
    Main function to run the Theta Engine strategy.
    If preloaded_data is provided, it will be used instead of reloading from disk.

    Args:
        custom_config (dict, optional): Custom configuration dictionary.
        preloaded_data (DataFrame, optional): Preloaded option data.

    Returns:
        tuple: (final_metrics, equity_history, trade_log, risk_scaling_history)
               where final_metrics is a dictionary with keys:
               'total_return', 'cagr', 'volatility', 'sharpe_ratio',
               'max_drawdown', 'start_value', 'end_value',
               'avg_risk_scaling', 'min_risk_scaling', 'max_risk_scaling'
    """
    # Load configuration
    config = custom_config or load_configuration()

    # Set up logging
    logging_manager = LoggingManager()
    logger = logging_manager.setup_logging(
        config,
        verbose_console=False,
        debug_mode=False,
        clean_format=True
    )

    try:
        report_filename = logging_manager.build_html_report_filename(config)
        logger.info(f"Report will be saved as: {report_filename}")

        logging_manager.print_strategy_settings(config)
        logger.info("\nMargin Management Settings:")
        logger.info(f"  Margin Buffer: {config.get('margin_buffer_pct', 0.10) * 100:.0f}%")
        logger.info(f"  Negative Margin Threshold: {config.get('negative_margin_threshold', -0.05) * 100:.0f}% of NLV")
        logger.info(f"  Rebalance Cooldown: {config.get('rebalance_cooldown_days', 3)} days")
        logger.info(
            f"  Forced Rebalance Threshold: {config.get('forced_rebalance_threshold', -0.10) * 100:.0f}% of NLV")
        logger.info(
            f"  Max Position Reduction: {config.get('max_position_reduction_pct', 0.25) * 100:.0f}% per rebalance")

        logger.info("Loading data...")
        logging_manager.log_status("Loading data...")
        data_manager = DataManager(logger=logger)
        if preloaded_data is None:
            data = data_manager.load_option_data(
                file_path=config['paths']['input_file'],
                start_date=config['dates']['start_date'],
                end_date=config['dates']['end_date']
            )
        else:
            data = preloaded_data

        required_columns = ['DataDate', 'OptionSymbol', 'Expiration', 'Strike', 'Delta', 'Bid', 'Ask', 'Type',
                            'UnderlyingPrice']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        logger.info("Calculating mid prices...")
        logging_manager.log_status("Calculating mid prices...")
        data = data_manager.calculate_mid_prices(data, config['trading']['normal_spread'])
        if 'MidPrice' not in data.columns:
            raise ValueError("Failed to calculate MidPrice column")
        if 'DataDate' not in data.columns:
            raise ValueError("DataDate column is missing after mid price calculation")

        logger.info("Initializing trading engine...")
        logging_manager.log_status("Initializing trading engine...")
        engine = ThetaEngine(config, logger=logger)

        if not hasattr(engine.trade_manager,
                       'portfolio_rebalancer') or engine.trade_manager.portfolio_rebalancer is None:
            logger.warning("Portfolio rebalancer not initialized in trade manager - fixing...")
            engine.trade_manager.portfolio_rebalancer = engine.portfolio_rebalancer

        logger.info("Running strategy simulation...")
        logging_manager.log_status("Running strategy simulation...")
        start_time = datetime.now()

        final_metrics, equity_history, trade_log, risk_scaling_history = engine.run(data)

        execution_time = datetime.now() - start_time
        logger.info(f"Strategy execution completed in {execution_time}")
        logging_manager.log_status(f"Strategy execution completed in {execution_time}")

        # Generate performance report and save outputs
        logger.info("Generating performance report...")
        logging_manager.log_status("Generating performance report...")
        reporting_system = ReportingSystem(config, logger=logger)

        trade_log_path = reporting_system.save_trade_log(
            trade_log,
            os.path.join(config['paths']['output_dir'], "trade_log.csv")
        )
        html_report_path = reporting_system.generate_html_report(
            equity_history,
            risk_scaling_history,
            engine.greeks_history,
            os.path.join(config['paths']['output_dir'], report_filename)
        )

        equity_df = pd.DataFrame.from_dict(equity_history, orient='index', columns=['Equity'])
        equity_df.index.name = 'Date'
        equity_csv_path = os.path.join(config['paths']['output_dir'], "equity_history.csv")
        equity_df.to_csv(equity_csv_path)
        logger.info(f"Equity history saved to {equity_csv_path}")

        risk_scaling_csv_path = os.path.join(config['paths']['output_dir'], "risk_scaling_history.csv")
        pd.DataFrame(risk_scaling_history).to_csv(risk_scaling_csv_path, index=False)
        logger.info(f"Risk scaling history saved to {risk_scaling_csv_path}")

        # Calculate performance metrics into a dictionary with the required keys.
        metrics = reporting_system.calculate_performance_metrics(
            equity_df,
            pd.DataFrame(risk_scaling_history)
        )
        reporting_system.print_performance_metrics(metrics)
        logging_manager.print_console_summary(metrics)

        # Optionally, you can reassign final_metrics to metrics if you want to ensure it
        # contains the keys: total_return, cagr, volatility, sharpe_ratio, max_drawdown,
        # start_value, end_value, avg_risk_scaling, min_risk_scaling, and max_risk_scaling.
        final_metrics = metrics

        logging_manager.teardown_logging()
        return final_metrics, equity_history, trade_log, risk_scaling_history

    except Exception as e:
        logging_manager.log_error("An error occurred during strategy execution", e)
        logging_manager.teardown_logging()
        raise


if __name__ == "__main__":
    try:
        final_metrics, equity_history, trade_log, risk_scaling_history = main()
    except Exception as e:
        print("Check the log file for details.")
        print(f"Error: {str(e)}")


