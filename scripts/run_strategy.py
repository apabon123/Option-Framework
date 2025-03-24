#!/usr/bin/env python
"""
Unified Strategy Runner

This script provides a unified interface for running any strategy registered
in the strategy registry configuration. It supports passing arguments to the
specific strategy runner script.
"""

import argparse
import logging
import os
import subprocess
import sys
import yaml
from datetime import datetime
from pathlib import Path

# Add the root directory to the Python path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from option_framework.utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a trading strategy")
    
    parser.add_argument(
        "strategy_id",
        type=str,
        help="ID of the strategy to run (from the strategy registry)",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the strategy configuration file (overrides registry config)",
    )
    
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date for backtest (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date for backtest (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to input data file",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory for output files",
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available strategies",
    )
    
    parser.add_argument(
        "--list_category",
        type=str,
        help="List strategies in a specific category",
    )
    
    return parser.parse_args()


def load_registry(registry_path="config/strategy_registry.yaml"):
    """Load the strategy registry configuration."""
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Strategy registry file not found: {registry_path}")
    
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)
    
    return registry


def list_strategies(registry, category=None):
    """List all available strategies, optionally filtered by category."""
    strategies = registry.get("strategies", {})
    categories = registry.get("categories", {})
    
    print("\n=== AVAILABLE STRATEGIES ===\n")
    
    if category:
        if category not in categories:
            print(f"Category '{category}' not found in registry. Available categories:")
            for cat_id, cat_info in categories.items():
                print(f"  - {cat_id}: {cat_info.get('name', '')}")
            return
        
        category_info = categories[category]
        print(f"Category: {category_info.get('name', category)}\n")
        strategy_ids = category_info.get("strategies", [])
        for strategy_id in strategy_ids:
            if strategy_id in strategies:
                strategy = strategies[strategy_id]
                print(f"ID: {strategy_id}")
                print(f"Name: {strategy.get('name', 'Unnamed')}")
                print(f"Description: {strategy.get('description', 'No description')}")
                print(f"Tags: {', '.join(strategy.get('tags', []))}")
                print()
    else:
        # List all strategies
        for strategy_id, strategy in strategies.items():
            print(f"ID: {strategy_id}")
            print(f"Name: {strategy.get('name', 'Unnamed')}")
            print(f"Description: {strategy.get('description', 'No description')}")
            print(f"Tags: {', '.join(strategy.get('tags', []))}")
            print()
        
        # List categories
        print("=== STRATEGY CATEGORIES ===\n")
        for cat_id, cat_info in categories.items():
            print(f"Category: {cat_info.get('name', cat_id)}")
            print(f"Strategies: {', '.join(cat_info.get('strategies', []))}")
            print()


def run_strategy(registry, args):
    """Run the specified strategy with the provided arguments."""
    strategies = registry.get("strategies", {})
    
    if args.strategy_id not in strategies:
        print(f"Strategy '{args.strategy_id}' not found in registry. Use --list to see available strategies.")
        return 1
    
    strategy = strategies[args.strategy_id]
    script_path = strategy.get("script_path")
    
    if not script_path:
        print(f"No script path specified for strategy '{args.strategy_id}'.")
        return 1
    
    script_path = os.path.join(root_dir, script_path)
    if not os.path.exists(script_path):
        print(f"Strategy script not found: {script_path}")
        return 1
    
    # Build command arguments
    cmd = [sys.executable, script_path]
    
    if args.config:
        cmd.extend(["--config", args.config])
    
    if args.start_date:
        cmd.extend(["--start_date", args.start_date])
    
    if args.end_date:
        cmd.extend(["--end_date", args.end_date])
    
    if args.input_file:
        cmd.extend(["--input_file", args.input_file])
    
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    
    if args.log_level:
        cmd.extend(["--log_level", args.log_level])
    
    # Setup logging for this runner
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    logger.info(f"Running strategy: {strategy.get('name', args.strategy_id)}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run the strategy script
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running strategy: {e}")
        return e.returncode
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Load strategy registry
        registry = load_registry()
        
        # List strategies if requested
        if args.list or args.list_category:
            list_strategies(registry, args.list_category)
            return 0
        
        # Run the specified strategy
        if not args.strategy_id:
            print("No strategy ID specified. Use --list to see available strategies.")
            return 1
        
        return run_strategy(registry, args)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 