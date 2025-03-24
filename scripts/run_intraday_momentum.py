#!/usr/bin/env python
"""
Intraday Momentum Strategy Runner

This script runs the Intraday Momentum Strategy with the specified configuration.
It supports command-line arguments for configuration path, date range, and more.
"""

import argparse
import logging
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path

# Add the root directory to the Python path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from option_framework.strategy.intraday_momentum_strategy import IntradayMomentumStrategy
from option_framework.backtest.backtest_engine import BacktestEngine
from option_framework.utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Intraday Momentum Strategy")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/strategy/intraday_momentum_config.yaml",
        help="Path to the strategy configuration file",
    )
    
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date for backtest (YYYY-MM-DD), overrides config file",
    )
    
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date for backtest (YYYY-MM-DD), overrides config file",
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to input data file, overrides config file",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory for output files, overrides config file",
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level, overrides config file",
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def validate_and_update_config(config, args):
    """Validate configuration and update with command line arguments."""
    # Override config with command line arguments if provided
    if args.start_date:
        config["dates"]["start_date"] = args.start_date
    
    if args.end_date:
        config["dates"]["end_date"] = args.end_date
    
    if args.input_file:
        config["intraday"]["file_path"] = args.input_file
    
    if args.output_dir:
        config["output"]["directory"] = args.output_dir
    
    if args.log_level:
        config["logging"]["level"] = args.log_level
    
    # Validate essential settings
    if not config.get("strategy", {}).get("name"):
        raise ValueError("Strategy name not specified in configuration")
    
    if not config.get("intraday", {}).get("file_path"):
        raise ValueError("Intraday data file path not specified in configuration")
    
    # Ensure input file exists
    intraday_file = config["intraday"]["file_path"]
    if not os.path.exists(intraday_file):
        raise FileNotFoundError(f"Intraday data file not found: {intraday_file}")
    
    return config


def run_strategy(config):
    """Initialize and run the Intraday Momentum Strategy."""
    # Setup logging
    log_level = config["logging"].get("level", "INFO")
    log_file = None
    if config["logging"].get("file", False):
        output_dir = config["output"].get("directory", "output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"intraday_momentum_{timestamp}.log")
    
    setup_logging(level=log_level, log_file=log_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Intraday Momentum Strategy run with configuration: {config['strategy']['name']}")
    
    # Extract strategy parameters
    strategy_params = config["strategy"]
    
    # Initialize the strategy
    strategy = IntradayMomentumStrategy(
        lookback_days=strategy_params.get("lookback_days", 20),
        volatility_multiplier=strategy_params.get("volatility_multiplier", 1.5),
        entry_times=strategy_params.get("entry_times", []),
        min_holding_period_minutes=strategy_params.get("min_holding_period_minutes", 15),
        invert_signals=strategy_params.get("invert_signals", False),
        use_prev_close=strategy_params.get("use_prev_close", False),
        min_history_days=strategy_params.get("min_history_days", 10),
        max_position_size_pct=strategy_params.get("max_position_size_pct", 0.1),
        max_leverage=strategy_params.get("max_leverage", 2.0),
        stop_loss_pct=strategy_params.get("stop_loss_pct", 0.01),
        take_profit_pct=strategy_params.get("take_profit_pct", 0.02),
        use_trailing_stop=strategy_params.get("use_trailing_stop", True),
        trailing_stop_pct=strategy_params.get("trailing_stop_pct", 0.005),
    )
    
    # Initialize backtest engine
    backtest = BacktestEngine(
        strategy=strategy,
        data_file=config["intraday"]["file_path"],
        start_date=config["dates"]["start_date"],
        end_date=config["dates"]["end_date"],
        initial_capital=config["portfolio"].get("initial_capital", 100000),
        timezone=config["intraday"].get("timezone", "America/New_York"),
        market_open=config["intraday"].get("market_open", "09:30:00"),
        market_close=config["intraday"].get("market_close", "16:00:00"),
        interval_minutes=config["intraday"].get("interval_minutes", 1),
    )
    
    # Run backtest
    results = backtest.run()
    
    # Save and display results
    output_dir = config["output"].get("directory", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    if config["output"].get("save_trades", True):
        trades_file = os.path.join(output_dir, f"trades_{timestamp}.csv")
        results.trades.to_csv(trades_file)
        logger.info(f"Trade data saved to {trades_file}")
    
    if config["output"].get("save_equity_curve", True):
        equity_file = os.path.join(output_dir, f"equity_curve_{timestamp}.csv")
        results.equity_curve.to_csv(equity_file)
        logger.info(f"Equity curve saved to {equity_file}")
    
    if config["output"].get("generate_charts", True):
        chart_file = os.path.join(output_dir, f"performance_chart_{timestamp}.png")
        results.plot_performance(save_path=chart_file)
        logger.info(f"Performance chart saved to {chart_file}")
    
    # Display summary statistics
    logger.info("Backtest Results Summary:")
    logger.info(f"Total Return: {results.total_return:.2%}")
    logger.info(f"Annualized Return: {results.annualized_return:.2%}")
    logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
    logger.info(f"Total Trades: {results.total_trades}")
    logger.info(f"Win Rate: {results.win_rate:.2%}")
    
    return results


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Validate and update configuration
        config = validate_and_update_config(config, args)
        
        # Run the strategy
        results = run_strategy(config)
        
        print("Strategy execution completed successfully.")
        return 0
    
    except Exception as e:
        print(f"Error running strategy: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 