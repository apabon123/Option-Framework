"""
Example of using the SSVI strategy to identify relative value trades.

This example demonstrates:
1. Loading option market data
2. Configuring the SSVI strategy
3. Running the strategy to identify trades
4. Running a simple backtest
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import strategy modules
from strategies.ssvi_strategy import SSVIStrategy
from strategies.config.ssvi_strategy_config import load_config, create_custom_config
from data_managers.option_data_manager import OptionDataManager
from analysis.relative_value.ssvi_model import SSVIModel
from core.portfolio import Portfolio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ssvi_example.log')
    ]
)
logger = logging.getLogger('ssvi_example')

def load_example_data(data_file=None):
    """
    Load example option chain data.
    
    If no data file is provided, generate synthetic data.
    
    Args:
        data_file: Path to data file (optional)
        
    Returns:
        DataFrame: Option chain data
    """
    if data_file and os.path.exists(data_file):
        logger.info(f"Loading option data from {data_file}")
        return pd.read_csv(data_file)
    
    # Generate synthetic option data
    logger.info("Generating synthetic option data")
    
    # Sample parameters
    underlying_price = 100.0
    vol_level = 0.25
    rfr = 0.02
    
    # Generate strikes
    strikes = np.arange(70, 131, 5)
    
    # Generate expiry days
    days_to_expiry = [7, 14, 30, 60, 90, 180]
    
    # Generate option data
    rows = []
    
    for dte in days_to_expiry:
        for strike in strikes:
            # Calculate moneyness
            moneyness = np.log(strike / underlying_price)
            
            # Calculate IV with smile (higher for OTM options)
            iv_skew = 0.05 * moneyness**2
            
            # Add term structure (higher vol for longer dated options)
            term_structure = 0.02 * np.log(dte / 30) if dte > 0 else 0
            
            # Combine to get IV
            iv = vol_level + iv_skew + term_structure
            
            # Add some random noise for realism
            iv_noise = np.random.normal(0, 0.02)
            iv = max(0.05, iv + iv_noise)
            
            # Calculate a simple delta approximation
            t = dte / 365.0
            call_delta = np.exp(-rfr * t) * norm_cdf((np.log(underlying_price / strike) + (rfr + 0.5 * iv**2) * t) / (iv * np.sqrt(t)))
            put_delta = call_delta - np.exp(-rfr * t)
            
            # Calculate simple gamma and vega approximations
            gamma = np.exp(-rfr * t) * norm_pdf((np.log(underlying_price / strike) + (rfr + 0.5 * iv**2) * t) / (iv * np.sqrt(t))) / (underlying_price * iv * np.sqrt(t))
            vega = 0.01 * underlying_price * np.sqrt(t) * norm_pdf((np.log(underlying_price / strike) + (rfr + 0.5 * iv**2) * t) / (iv * np.sqrt(t)))
            
            # Calculate simple theta approximation (daily)
            theta_call = -underlying_price * iv * norm_pdf((np.log(underlying_price / strike) + (rfr + 0.5 * iv**2) * t) / (iv * np.sqrt(t))) / (2 * 365 * np.sqrt(t))
            theta_put = theta_call + rfr * strike * np.exp(-rfr * t) / 365
            
            # Call option
            call_row = {
                'Symbol': f"SAMPLE{dte}C{strike}",
                'Underlying': 'SAMPLE',
                'UnderlyingPrice': underlying_price,
                'Strike': strike,
                'DTE': dte,
                'Type': 'CALL',
                'IV': iv,
                'Delta': call_delta,
                'Gamma': gamma,
                'Vega': vega,
                'Theta': theta_call,
                'Volume': int(np.random.exponential(500) * np.exp(-0.01 * abs(moneyness * 100))),
                'OpenInterest': int(np.random.exponential(2000) * np.exp(-0.01 * abs(moneyness * 100))),
                'Bid': 0,  # Will calculate based on IV
                'Ask': 0,  # Will calculate based on IV
                'Last': 0  # Will calculate based on IV
            }
            
            # Put option
            put_row = {
                'Symbol': f"SAMPLE{dte}P{strike}",
                'Underlying': 'SAMPLE',
                'UnderlyingPrice': underlying_price,
                'Strike': strike,
                'DTE': dte,
                'Type': 'PUT',
                'IV': iv,
                'Delta': put_delta,
                'Gamma': gamma,
                'Vega': vega,
                'Theta': theta_put,
                'Volume': int(np.random.exponential(500) * np.exp(-0.01 * abs(moneyness * 100))),
                'OpenInterest': int(np.random.exponential(2000) * np.exp(-0.01 * abs(moneyness * 100))),
                'Bid': 0,  # Will calculate based on IV
                'Ask': 0,  # Will calculate based on IV
                'Last': 0  # Will calculate based on IV
            }
            
            # Calculate option prices using Black-Scholes
            call_row['Last'] = bs_price(underlying_price, strike, dte/365, rfr, iv, 'c')
            put_row['Last'] = bs_price(underlying_price, strike, dte/365, rfr, iv, 'p')
            
            # Add bid-ask spread (wider for less liquid options)
            spread_factor = 0.05 + 0.1 * abs(moneyness) + 0.05 * (1 - np.exp(-0.01 * dte))
            call_row['Bid'] = call_row['Last'] * (1 - spread_factor)
            call_row['Ask'] = call_row['Last'] * (1 + spread_factor)
            put_row['Bid'] = put_row['Last'] * (1 - spread_factor)
            put_row['Ask'] = put_row['Last'] * (1 + spread_factor)
            
            rows.append(call_row)
            rows.append(put_row)
    
    # Convert to DataFrame
    option_chain = pd.DataFrame(rows)
    
    # Add some rich/cheap distortions for testing
    # Make some options artificially rich (higher IV)
    rich_indices = np.random.choice(len(option_chain), size=int(len(option_chain) * 0.1), replace=False)
    option_chain.loc[rich_indices, 'IV'] = option_chain.loc[rich_indices, 'IV'] * 1.15
    
    # Make some options artificially cheap (lower IV)
    cheap_indices = np.random.choice(len(option_chain), size=int(len(option_chain) * 0.1), replace=False)
    option_chain.loc[cheap_indices, 'IV'] = option_chain.loc[cheap_indices, 'IV'] * 0.85
    
    # Recalculate prices for distorted IVs
    for idx, row in option_chain.iterrows():
        option_type = 'c' if row['Type'] == 'CALL' else 'p'
        price = bs_price(row['UnderlyingPrice'], row['Strike'], row['DTE']/365, rfr, row['IV'], option_type)
        spread_factor = 0.05 + 0.1 * abs(np.log(row['Strike'] / row['UnderlyingPrice'])) + 0.05 * (1 - np.exp(-0.01 * row['DTE']))
        
        option_chain.loc[idx, 'Last'] = price
        option_chain.loc[idx, 'Bid'] = price * (1 - spread_factor)
        option_chain.loc[idx, 'Ask'] = price * (1 + spread_factor)
    
    return option_chain

# Helper functions for option pricing/calculations
def norm_cdf(x):
    """Standard normal CDF approximation"""
    return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * x * (0.044715 * x**2 + 1)))

def norm_pdf(x):
    """Standard normal PDF"""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def bs_price(s, k, t, r, sigma, option_type):
    """Simple Black-Scholes price calculation"""
    if t <= 0:
        if option_type == 'c':
            return max(0, s - k)
        else:
            return max(0, k - s)
    
    d1 = (np.log(s/k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    if option_type == 'c':
        return s * norm_cdf(d1) - k * np.exp(-r * t) * norm_cdf(d2)
    else:
        return k * np.exp(-r * t) * norm_cdf(-d2) - s * norm_cdf(-d1)

def run_strategy_example():
    """Run a simple example of the SSVI strategy"""
    logger.info("Starting SSVI strategy example")
    
    # Load option data
    option_chain = load_example_data()
    underlying_price = option_chain['UnderlyingPrice'].iloc[0]
    
    logger.info(f"Loaded option chain with {len(option_chain)} options")
    
    # Load strategy configuration
    config = load_config('default')
    
    # Create custom configuration for this example
    custom_config = create_custom_config({
        'trade_params': {
            'zscore_threshold': 1.2,  # Lower threshold for example
            'min_dte': 7,
            'max_dte': 180,
        }
    })
    
    # Initialize strategy components
    option_data_manager = OptionDataManager()
    ssvi_model = SSVIModel()
    portfolio = Portfolio("SSVI Example Portfolio", initial_capital=100000.0)
    
    # Initialize strategy with custom configuration
    strategy = SSVIStrategy(
        config=custom_config,
        option_data_manager=option_data_manager,
        ssvi_model=ssvi_model,
        portfolio=portfolio,
        logger=logger
    )
    
    # Update strategy with option data
    result = strategy.update(option_chain, underlying_price)
    
    if not result:
        logger.error("Strategy update failed")
        return
    
    # Generate trades
    trades = strategy.generate_trades()
    
    logger.info(f"Generated {len(trades)} potential trades")
    
    # Print top trades
    if trades:
        logger.info("Top 5 recommended trades:")
        for i, trade in enumerate(trades[:5]):
            trade_type = trade['type']
            
            if trade_type == 'single_leg':
                logger.info(f"#{i+1}: {trade['direction']} {trade['symbol']} (Z-score: {trade['zscore']:.2f}, Return/Risk: {trade['expected_return_per_risk']:.2f})")
            elif trade_type == 'vertical_spread':
                leg1 = trade['legs'][0]
                leg2 = trade['legs'][1]
                logger.info(f"#{i+1}: {trade_type} - {leg1['direction']} {leg1['symbol']} + {leg2['direction']} {leg2['symbol']} (Return/Risk: {trade['expected_return_per_risk']:.2f})")
            elif trade_type == 'butterfly':
                logger.info(f"#{i+1}: {trade['strategy']} on {trade['underlying']} {trade['dte']}d (Z-score: {trade['zscore']:.2f})")
    else:
        logger.info("No trades generated")
    
    # Execute top trades
    executed_trades = strategy.execute_trades(trades, max_trades=3)
    
    logger.info(f"Executed {len(executed_trades)} trades")
    
    # Print trade details
    for i, trade in enumerate(executed_trades):
        logger.info(f"Trade #{i+1} details: {trade}")
    
    return strategy, option_chain

def run_simple_backtest():
    """Run a simple backtest of the SSVI strategy"""
    logger.info("Starting SSVI strategy backtest")
    
    # Load configuration
    config = load_config('default')
    
    # Initialize strategy
    strategy = SSVIStrategy(config=config, logger=logger)
    
    # Generate synthetic historical data
    backtest_days = 30
    historical_data = []
    
    # Generate a base underlying price path
    base_price = 100.0
    vol = 0.2
    rfr = 0.02
    price_path = [base_price]
    
    # Simulate price path with daily returns
    for day in range(1, backtest_days):
        daily_return = np.random.normal(rfr/252, vol/np.sqrt(252))
        new_price = price_path[-1] * (1 + daily_return)
        price_path.append(new_price)
    
    # Generate option data for each day
    for day, price in enumerate(price_path):
        # Create synthetic option chain for this day
        option_chain = load_example_data()
        
        # Adjust underlying price
        price_ratio = price / option_chain['UnderlyingPrice'].iloc[0]
        option_chain['UnderlyingPrice'] = price
        
        # Adjust strikes proportionally (as if it's a new chain each day)
        option_chain['Strike'] = option_chain['Strike'] * price_ratio
        
        # Create data point
        data_point = {
            'date': (datetime.now() - timedelta(days=backtest_days-day)).strftime('%Y-%m-%d'),
            'option_chain': option_chain,
            'underlying_price': price,
            'current_prices': {
                # Simulate current prices for portfolio valuation
                # In a real implementation, this would be actual market prices
                row['Symbol']: row['Last'] for _, row in option_chain.iterrows()
            }
        }
        
        historical_data.append(data_point)
    
    # Run backtest
    results = strategy.backtest(historical_data, initial_capital=100000.0)
    
    # Print backtest results
    logger.info(f"Backtest Results:")
    logger.info(f"Total Return: {results['total_return']:.2%}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"Number of Trades: {len(results['trade_log'])}")
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(results['dates'], results['equity_curve'][1:])
    plt.title('SSVI Strategy Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ssvi_strategy_equity_curve.png')
    
    return results

if __name__ == "__main__":
    # Run single update example
    strategy, option_chain = run_strategy_example()
    
    # Run simple backtest
    backtest_results = run_simple_backtest()
    
    logger.info("SSVI strategy example completed") 