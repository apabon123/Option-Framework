#!/usr/bin/env python
"""
Volatility Breakout Strategy Runner

This script runs the Volatility Breakout Strategy with the specified configuration.
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

from option_framework.strategy.volatility_breakout_strategy import VolatilityBreakoutStrategy
from option_framework.backtest.backtest_engine import BacktestEngine
from option_framework.utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Volatility Breakout Strategy")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/strategy/volatility_breakout_config.yaml",
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
        config["data"]["file_path"] = args.input_file
    
    if args.output_dir:
        config["output"]["directory"] = args.output_dir
    
    if args.log_level:
        config["logging"]["level"] = args.log_level
    
    # Validate essential settings
    if not config.get("strategy", {}).get("name"):
        raise ValueError("Strategy name not specified in configuration")
    
    if not config.get("data", {}).get("file_path"):
        raise ValueError("Data file path not specified in configuration")
    
    # Ensure input file exists
    data_file = config["data"]["file_path"]
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    return config


def run_strategy(config):
    """Initialize and run the Volatility Breakout Strategy."""
    # Setup logging
    log_level = config["logging"].get("level", "INFO")
    log_file = None
    if config["logging"].get("file", False):
        output_dir = config["output"].get("directory", "output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"volatility_breakout_{timestamp}.log")
    
    setup_logging(level=log_level, log_file=log_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Volatility Breakout Strategy run with configuration: {config['strategy']['name']}")
    
    # Extract strategy parameters
    strategy_params = config["strategy"]
    
    # Initialize the strategy
    strategy = VolatilityBreakoutStrategy(
        atr_period=strategy_params.get("atr_period", 14),
        atr_multiplier=strategy_params.get("atr_multiplier", 1.5),
        enter_long=strategy_params.get("enter_long", True),
        enter_short=strategy_params.get("enter_short", True),
        use_prev_day_reference=strategy_params.get("use_prev_day_reference", False),
        opening_range_minutes=strategy_params.get("opening_range_minutes", 30),
        position_sizing=strategy_params.get("position_sizing", "volatility"),
        risk_per_trade=strategy_params.get("risk_per_trade", 0.02),
        fixed_size=strategy_params.get("fixed_size", 100),
        max_position_size_pct=strategy_params.get("max_position_size_pct", 0.1),
        stop_loss_atr=strategy_params.get("stop_loss_atr", 2.0),
        take_profit_atr=strategy_params.get("take_profit_atr", 3.0),
        use_trailing_stop=strategy_params.get("use_trailing_stop", True),
        trailing_start_atr=strategy_params.get("trailing_start_atr", 1.0),
        trailing_stop_atr=strategy_params.get("trailing_stop_atr", 1.5),
        max_trades_per_day=strategy_params.get("max_trades_per_day", 2),
        time_exit_minutes=strategy_params.get("time_exit_minutes", 240),
        filters=strategy_params.get("filters", {}),
    )
    
    # Initialize backtest engine
    backtest = BacktestEngine(
        strategy=strategy,
        data_file=config["data"]["file_path"],
        start_date=config["dates"]["start_date"],
        end_date=config["dates"]["end_date"],
        initial_capital=config["portfolio"].get("initial_capital", 100000),
        max_leverage=config["portfolio"].get("max_leverage", 1.0),
        commission=config["portfolio"].get("commission", 0.005),
        slippage_model=config["portfolio"].get("slippage_model", "percent"),
        slippage_amount=config["portfolio"].get("slippage_amount", 0.0005),
        timezone=config["data"].get("timezone", "America/New_York"),
        is_daily=config["data"].get("daily", False),
        interval_minutes=config["data"].get("interval_minutes", 5),
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
    
    if config["output"].get("save_tearsheet", True):
        tearsheet_file = os.path.join(output_dir, f"tearsheet_{timestamp}.html")
        results.generate_tearsheet(save_path=tearsheet_file)
        logger.info(f"Performance tearsheet saved to {tearsheet_file}")
    
    # Display summary statistics
    logger.info("Backtest Results Summary:")
    logger.info(f"Total Return: {results.total_return:.2%}")
    logger.info(f"Annualized Return: {results.annualized_return:.2%}")
    logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
    logger.info(f"Total Trades: {results.total_trades}")
    logger.info(f"Win Rate: {results.win_rate:.2%}")
    logger.info(f"Profit Factor: {results.profit_factor:.2f}")
    
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