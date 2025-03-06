"""
Main entry point for the trading system.

This module provides the main application entry point with configuration
setup and module initialization.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import core components
from core.position import Position, OptionPosition
from core.margin import MarginCalculator, OptionMarginCalculator, SPANMarginCalculator
from core.portfolio import Portfolio
from core.options_analysis import VolatilitySurface, OptionsAnalyzer
from core.data_manager import DataManager
from core.reporting import ReportingSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('main')


def load_config():
    """
    Load configuration settings for the application.
    
    Returns:
        dict: Configuration dictionary
    """
    # This is a placeholder - in a real application, this would
    # load from a YAML/JSON file or environment variables
    config = {
        'portfolio': {
            'initial_capital': 100000,
            'max_position_size_pct': 0.10,
            'max_portfolio_delta': 0.20
        },
        'risk': {
            'max_leverage': 10,
            'volatility_multiplier': 1.2,
            'margin_buffer_pct': 0.10
        },
        'options': {
            'min_days_to_expiry': 30,
            'max_days_to_expiry': 90,
            'z_score_threshold': 1.5
        },
        'data': {
            'input_file': r"C:\Users\alexp\OneDrive\Gdrive\Trading\Data Outputs\GLD_Combined.csv",
            'date_format': '%Y-%m-%d',
            'normal_spread': 0.60  # Maximum acceptable bid-ask spread as percentage
        },
        'paths': {
            'output_dir': 'output',
            'report_dir': 'reports'
        }
    }
    
    return config


def initialize_system(config):
    """
    Initialize the trading system with core components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Dictionary of initialized components
    """
    logger.info("Initializing trading system...")
    
    # Initialize directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['report_dir'], exist_ok=True)
    
    # Initialize data manager
    data_manager = DataManager(logger=logger)
    
    # Initialize volatility surface and options analyzer
    vol_surface = VolatilitySurface(logger=logger)
    options_analyzer = OptionsAnalyzer(
        vol_surface=vol_surface,
        z_score_threshold=config['options']['z_score_threshold'],
        logger=logger
    )
    
    # Initialize margin calculator
    margin_calculator = SPANMarginCalculator(
        max_leverage=config['risk']['max_leverage'],
        volatility_multiplier=config['risk']['volatility_multiplier'],
        logger=logger
    )
    
    # Initialize portfolio
    portfolio = Portfolio(
        initial_capital=config['portfolio']['initial_capital'],
        max_position_size_pct=config['portfolio']['max_position_size_pct'],
        max_portfolio_delta=config['portfolio']['max_portfolio_delta'],
        logger=logger
    )
    
    # Initialize reporting system
    reporting_system = ReportingSystem(
        output_dir=config['paths']['report_dir'],
        logger=logger
    )
    
    return {
        'data_manager': data_manager,
        'vol_surface': vol_surface,
        'options_analyzer': options_analyzer,
        'margin_calculator': margin_calculator,
        'portfolio': portfolio,
        'reporting_system': reporting_system
    }


def run_backtest(components, config):
    """
    Run a backtest using actual data from the GLD_Combined.csv file.
    
    Args:
        components: Dictionary of system components
        config: Configuration dictionary
    """
    logger.info("Running backtest with real option data...")
    
    # Extract components
    data_manager = components['data_manager']
    vol_surface = components['vol_surface']
    options_analyzer = components['options_analyzer']
    portfolio = components['portfolio']
    reporting_system = components['reporting_system']
    
    # Load actual data
    data_file = config['data']['input_file']
    logger.info(f"Loading data from {data_file}")
    
    # Define a date range (can be adjusted as needed)
    start_date = pd.to_datetime('2023-06-01')  # Adjust as needed
    end_date = pd.to_datetime('2023-08-31')    # Adjust as needed
    
    # Load and preprocess data
    try:
        data = data_manager.load_option_data(data_file, start_date, end_date)
        data = data_manager.calculate_mid_prices(data, config['data']['normal_spread'])
        logger.info(f"Loaded and processed {len(data)} records")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Get unique trading dates
    trading_dates = data_manager.get_dates_list()
    logger.info(f"Backtest period: {trading_dates[0]} to {trading_dates[-1]} ({len(trading_dates)} trading days)")
    
    # Track Greeks history for reporting
    greeks_history = {}
    
    # Process each trading day
    for current_date in trading_dates:
        logger.info(f"\nProcessing date: {current_date.strftime('%Y-%m-%d')}")
        
        # Get data for current day
        daily_data = data_manager.get_daily_data(current_date)
        if daily_data.empty:
            logger.warning(f"No data available for {current_date}, skipping")
            continue
            
        # Create market data by symbol dictionary
        market_data_by_symbol = data_manager.get_market_data_by_symbol(daily_data)
        
        # Update volatility surface
        vol_surface.update_surface(daily_data)
        
        # Update existing positions
        portfolio.update_market_data(market_data_by_symbol, current_date)
        
        # Store portfolio Greeks for this date
        greeks_history[current_date] = portfolio.get_portfolio_greeks()
        
        # Analyze options to find opportunities
        analyzed_options = options_analyzer.analyze_options_chain(daily_data)
        
        # Filter for potential trades (this is a simple example strategy)
        cheap_puts = analyzed_options[
            (analyzed_options['IsCheap'] == True) & 
            (analyzed_options['Type'] == 'put') &
            (analyzed_options['DaysToExpiry'] >= config['options']['min_days_to_expiry']) &
            (analyzed_options['DaysToExpiry'] <= config['options']['max_days_to_expiry'])
        ]
        
        # Check if portfolio has capacity and we found opportunities
        if (portfolio.get_portfolio_metrics()['portfolio_value'] > 0 and 
            not portfolio.positions and not cheap_puts.empty):
            
            # Select a trade
            selected_put = cheap_puts.iloc[0]
            
            # Add position to portfolio
            portfolio.add_position(
                symbol=selected_put['OptionSymbol'],
                instrument_data=selected_put.to_dict(),
                quantity=1,
                price=selected_put['MidPrice'],
                position_type='option',
                is_short=True,
                execution_data={'date': current_date}
            )
            
            logger.info(f"Added position: {selected_put['OptionSymbol']} at ${selected_put['MidPrice']:.2f}")
        
        # Check if we should close any positions
        positions_to_close = []
        for symbol, position in list(portfolio.positions.items()):
            # Exit conditions (this is a simple example)
            if position.days_to_expiry <= 14:  # Close short-dated options
                positions_to_close.append(symbol)
            elif position.is_short and position.unrealized_pnl > 0 and position.avg_entry_price > 0 and position.unrealized_pnl / (position.avg_entry_price * position.contracts * 100) > 0.5:
                # Take profit at 50% of premium for short positions
                positions_to_close.append(symbol)
        
        # Close positions
        for symbol in positions_to_close:
            if symbol in market_data_by_symbol:
                portfolio.remove_position(
                    symbol=symbol,
                    price=market_data_by_symbol[symbol]['MidPrice'],
                    execution_data={'date': current_date},
                    reason="Time-based exit" if portfolio.positions[symbol].days_to_expiry <= 14 else "Profit target"
                )
                logger.info(f"Closed position: {symbol}")
    
    # Generate performance report
    report_path = reporting_system.generate_html_report(
        equity_history=portfolio.equity_history,
        greeks_history=greeks_history,
        config=config
    )
    
    # Final performance summary
    metrics = portfolio.get_performance_metrics()
    logger.info("\nBacktest Results:")
    logger.info(f"  Initial Capital: ${portfolio.initial_capital:,.2f}")
    logger.info(f"  Final Value: ${portfolio.get_portfolio_value():,.2f}")
    logger.info(f"  Total Return: {metrics['return']:.2%}")
    logger.info(f"  CAGR: {metrics['cagr']:.2%}")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"  Report: {report_path}")


def main():
    """
    Main application entry point.
    """
    logger.info("Starting trading system...")
    
    # Load configuration
    config = load_config()
    logger.info(f"Configuration loaded: {len(config)} sections")
    
    # Initialize system
    components = initialize_system(config)
    
    # Run backtest with real data
    run_backtest(components, config)
    
    logger.info("Trading system execution completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())