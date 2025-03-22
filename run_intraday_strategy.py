#!/usr/bin/env python
"""
Intraday Momentum Strategy Execution Script

This script executes the IntradayMomentumStrategy for intraday trading,
using the Option-Framework infrastructure.
"""
import sys
import os
import logging
import argparse
import yaml
from datetime import datetime
from typing import Dict, Any, Optional

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies import IntradayMomentumStrategy
from core.trading_engine import TradingEngine
from simple_logging import SimpleLoggingManager


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Intraday Momentum Strategy")
    parser.add_argument('--config', type=str, default='config/intraday_momentum_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with additional logging')
    parser.add_argument('--margin-log-level', choices=['debug', 'info', 'warning', 'error', 'verbose'],
                      default='info', help='Set the margin log level')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate the configuration."""
    required_sections = ['portfolio', 'dates', 'data', 'strategy']
    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required section '{section}' in config")
            return False
    
    # Check for strategy-specific parameters
    if config['strategy'].get('name') != 'IntradayMomentumStrategy':
        print("Warning: This script is designed for IntradayMomentumStrategy.")
        
    return True


def setup_logging(config: Dict[str, Any], args) -> logging.Logger:
    """Set up logging based on configuration and arguments."""
    # Apply command line arguments to config
    if args.debug:
        config['debug_mode'] = True
    if args.margin_log_level:
        config['margin_log_level'] = args.margin_log_level
    
    # Set up logger
    logging_manager = SimpleLoggingManager()
    logger = logging_manager.setup_logging(
        config,
        verbose_console=True,
        debug_mode=config.get('debug_mode', False)
    )
    
    return logger


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate configuration
    if not validate_config(config):
        sys.exit(1)
    
    # Set up logging
    logger = setup_logging(config, args)
    logger.info("Starting Intraday Momentum Strategy execution")
    
    try:
        # Create strategy instance
        strategy = IntradayMomentumStrategy(
            name="IntradayMomentumStrategy",
            config=config,
            logger=logger
        )
        
        # Create and initialize trading engine
        engine = TradingEngine(
            config=config,
            strategy=strategy,
            logger=logger
        )
        
        # Initialize components
        logger.info("Initializing trading engine components...")
        engine.initialize()
        
        # Load data
        if not engine.load_data():
            logger.error("Failed to load data. Exiting.")
            sys.exit(1)
        
        # Run backtest
        logger.info("Running backtest...")
        results = engine.run_backtest()
        
        # Print results
        print("\n----- Backtest Results -----")
        print(f"Initial Value: ${results['initial_value']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:,.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:,.2%}")
        print("---------------------------\n")
        
        # Generate reports
        if config.get('reporting', {}).get('enable_charts', True):
            logger.info("Generating reports...")
            engine.generate_reports()
        
        logger.info("Execution completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 