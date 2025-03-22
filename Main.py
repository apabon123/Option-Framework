#!/usr/bin/env python
"""
Main entry point for the trading system.

This module provides a command-line interface for running backtests
with different strategies and configuration options.
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime
import importlib

from core.trading_engine import TradingEngine, Strategy
from simple_logging import SimpleLoggingManager


def load_yaml_config(config_path):
    """Load configuration from a YAML file."""
    print(f"Loading configuration from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(f"Loaded configuration: {config}")
        return config


def get_strategy_class(strategy_name):
    """
    Dynamically import and return the strategy class by name.

    Args:
        strategy_name: Name of the strategy class to import

    Returns:
        Strategy class
    """
    try:
        # First check if we can directly import it from example_strategy
        # This is a direct fallback for SimpleOptionStrategy
        if strategy_name == "SimpleOptionStrategy":
            from strategies.example_strategy import SimpleOptionStrategy
            return SimpleOptionStrategy

        # Add specific handling for PutSellStrat
        if strategy_name == "PutSellStrat":
            try:
                print(f"Attempting to import PutSellStrat from strategies.put_sell_strat")
                from strategies.put_sell_strat import PutSellStrat
                print(f"Successfully imported PutSellStrat")
                return PutSellStrat
            except Exception as e:
                print(f"Error importing PutSellStrat: {e}")
                import traceback
                print(traceback.format_exc())
                raise
                
        # Add specific handling for CallPutStrat
        if strategy_name == "CallPutStrat":
            try:
                print(f"Attempting to import CallPutStrat from strategies.call_put_strat")
                from strategies.call_put_strat import CallPutStrat
                print(f"Successfully imported CallPutStrat")
                return CallPutStrat
            except Exception as e:
                print(f"Error importing CallPutStrat: {e}")
                import traceback
                print(traceback.format_exc())
                raise
                
        # Add specific handling for ThetaDecayStrategy
        if strategy_name == "ThetaDecayStrategy":
            try:
                print(f"Attempting to import ThetaDecayStrategy from strategies.theta_strategy")
                from strategies.theta_strategy import ThetaDecayStrategy
                print(f"Successfully imported ThetaDecayStrategy")
                return ThetaDecayStrategy
            except Exception as e:
                print(f"Error importing ThetaDecayStrategy: {e}")
                import traceback
                print(traceback.format_exc())
                raise

        # Try the general approach for other strategies
        try:
            # First try to import from strategies package
            module_name = f"strategies.{strategy_name.lower()}"
            print(f"Trying to import from {module_name}")
            module = importlib.import_module(module_name)
            return getattr(module, strategy_name)
        except (ImportError, AttributeError) as e:
            print(f"Error importing from {module_name}: {e}")
            try:
                # Fall back to direct import if it's a fully qualified name
                module_name, class_name = strategy_name.rsplit('.', 1)
                print(f"Trying direct import from {module_name} for class {class_name}")
                module = importlib.import_module(module_name)
                return getattr(module, class_name)
            except (ValueError, ImportError, AttributeError) as e:
                print(f"Direct import failed: {e}")
                raise ImportError(f"Could not import strategy class: {strategy_name}")
    except Exception as e:
        print(f"Error importing strategy: {e}")
        import traceback
        print(traceback.format_exc())
        raise ImportError(f"Could not import strategy class: {strategy_name}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run trading strategy backtest")

    parser.add_argument(
        "-c", "--config",
        default="config/config.yaml",
        help="Path to configuration YAML file (default: config/config.yaml)"
    )

    parser.add_argument(
        "-s", "--strategy",
        help="Override strategy name from config"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode with extra logging"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Set the logging level (overrides config file)"
    )
    
    parser.add_argument(
        "--margin-log-level",
        choices=["minimal", "standard", "verbose", "debug"],
        default=None,
        help="Set the verbosity level for margin calculations"
    )

    parser.add_argument(
        "--portfolio-log-level",
        choices=["minimal", "standard", "verbose", "debug"],
        default=None,
        help="Set the verbosity level for portfolio operations"
    )

    parser.add_argument(
        "--start-date",
        help="Override start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date",
        help="Override end date (YYYY-MM-DD)"
    )

    return parser.parse_args()


def main():
    """Main entry point for the application."""
    print("\n=== Starting Option Framework ===")
    args = parse_args()
    print(f"Using config file: {args.config}")

    try:
        # Load configuration
        config = load_yaml_config(args.config)

        # Override configuration with command-line arguments
        if args.strategy:
            if 'strategy' not in config:
                config['strategy'] = {}
            config['strategy']['name'] = args.strategy
            print(f"Strategy override: {args.strategy}")

        if args.start_date:
            if 'dates' not in config:
                config['dates'] = {}
            config['dates']['start_date'] = args.start_date
            print(f"Start date override: {args.start_date}")

        if args.end_date:
            if 'dates' not in config:
                config['dates'] = {}
            config['dates']['end_date'] = args.end_date
            print(f"End date override: {args.end_date}")

        # Configure logging
        print("Setting up logging...")
        logging_manager = SimpleLoggingManager()
        
        # Override log level if specified in command line
        if args.log_level:
            if 'logging' not in config:
                config['logging'] = {}
            config['logging']['level'] = args.log_level
            print(f"Log level override: {args.log_level}")
            
        # Override margin log level if specified in command line
        if args.margin_log_level:
            # First, update the new structure
            if 'logging' not in config:
                config['logging'] = {}
            if 'components' not in config['logging']:
                config['logging']['components'] = {}
            if 'margin' not in config['logging']['components']:
                config['logging']['components']['margin'] = {}
            config['logging']['components']['margin']['level'] = args.margin_log_level
            
            # Also update the old structure for backward compatibility
            if 'margin' not in config:
                config['margin'] = {}
            if 'logging' not in config['margin']:
                config['margin']['logging'] = {}
            config['margin']['logging']['level'] = args.margin_log_level
            print(f"Margin log level override: {args.margin_log_level}")
            
        # Override portfolio log level if specified in command line
        if args.portfolio_log_level:
            # First, update the new structure
            if 'logging' not in config:
                config['logging'] = {}
            if 'components' not in config['logging']:
                config['logging']['components'] = {}
            if 'portfolio' not in config['logging']['components']:
                config['logging']['components']['portfolio'] = {}
            config['logging']['components']['portfolio']['level'] = args.portfolio_log_level
            
            # Also update the old structure for backward compatibility
            if 'portfolio' not in config:
                config['portfolio'] = {}
            if 'logging' not in config['portfolio']:
                config['portfolio']['logging'] = {}
            config['portfolio']['logging']['level'] = args.portfolio_log_level
            print(f"Portfolio log level override: {args.portfolio_log_level}")
        
        try:
            logger = logging_manager.setup_logging(
                config,
                verbose_console=args.verbose,
                debug_mode=args.debug,
                clean_format=not args.debug
            )
        except Exception as e:
            import traceback
            print(f"ERROR in setup_logging: {e}")
            print(traceback.format_exc())
            return 1

        # Get strategy class and create instance
        strategy_name = config.get('strategy', {}).get('name')
        if not strategy_name:
            logger.error("No strategy name specified in configuration")
            return 1

        print(f"Initializing strategy: {strategy_name}")
        logger.info(f"[INIT] Using strategy: {strategy_name}")

        try:
            strategy_class = get_strategy_class(strategy_name)
            strategy = strategy_class(config.get('strategy', {}), logger)
        except Exception as e:
            import traceback
            print(f"ERROR loading strategy: {e}")
            print(traceback.format_exc())
            return 1

        # Create trading engine
        try:
            print("Creating trading engine...")
            engine = TradingEngine(config, strategy)
        except Exception as e:
            import traceback
            print(f"ERROR creating trading engine: {e}")
            print(traceback.format_exc())
            return 1

        # Load data
        print("Loading market data...")
        logger.info("[INIT] Loading market data...")
        try:
            if not engine.load_data():
                logger.error("Failed to load data")
                return 1
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 1

        # Run backtest
        try:
            print("\n=== Starting backtest ===")
            results = engine.run_backtest()
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 1

        # Print summary
        print("\n=== Backtest Results ===")
        logger.info("\n=== Backtest Results ===")
        print(f"Initial Capital: ${results.get('initial_value', 0):,.2f}")
        logger.info(f"Initial Capital: ${results.get('initial_value', 0):,.2f}")
        print(f"Final Value: ${results.get('final_value', 0):,.2f}")
        logger.info(f"Final Value: ${results.get('final_value', 0):,.2f}")
        print(f"Total Return: {results.get('total_return', 0):.2%}")
        logger.info(f"Total Return: {results.get('total_return', 0):.2%}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        logger.info(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")

        # Print report path
        if 'report_path' in results:
            print(f"Detailed report saved to: {results['report_path']}")
            logger.info(f"Detailed report saved to: {results['report_path']}")

        print("\n=== Backtest completed successfully ===")
        return 0

    except Exception as e:
        logging.error(f"Error running backtest: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())