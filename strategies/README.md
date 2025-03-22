# Options Trading Strategies

This directory contains implementations of various options trading strategies that can be run against market data using the Option-Framework.

## Available Strategies

### SSVI-based Relative Value Strategy

A strategy based on the Surface Stochastic Volatility Inspired (SSVI) model that identifies and trades relative value opportunities in the options market.

#### Strategy Overview

The SSVI strategy works by:

1. Fitting the SSVI parameterization to current options market data
2. Identifying options that deviate significantly from the model (via Z-scores)
3. Creating trades based on the expectation that option prices revert to the model values
4. Managing risk through portfolio constraints and hedging

The strategy can use single-leg trades (buying cheap options, selling rich options) or multi-leg trades (vertical spreads, butterflies) to take advantage of relative value opportunities.

#### Features

- Adaptive parameters based on market regime (high/low volatility conditions)
- Adjustable Z-score thresholds for signal generation
- Risk management through position sizing and exposure limits
- Comprehensive backtesting capabilities
- Flexible configuration system for strategy customization

#### Configuration

The strategy is configured through a configuration file in `config/ssvi_strategy_config.py`. Key parameters include:

- `zscore_threshold`: Defines the minimum deviation required to generate a signal
- `min_dte` / `max_dte`: Defines the expiration range to consider
- Risk parameters: Maximum position size, delta/vega exposure limits
- SSVI model parameters: Fitting method, parameter bounds, etc.

Example configurations are provided for different market regimes (high volatility, low volatility).

#### Usage

```python
from strategies.ssvi_strategy import SSVIStrategy
from strategies.config.ssvi_strategy_config import load_config
from data_managers.option_data_manager import OptionDataManager

# Load configuration
config = load_config('default')  # or 'high_vol', 'low_vol', 'multi_leg'

# Initialize strategy
strategy = SSVIStrategy(config=config)

# Update with market data
option_chain = option_data_manager.get_current_options('SPY')
underlying_price = option_data_manager.get_current_price('SPY')
strategy.update(option_chain, underlying_price)

# Generate and execute trades
trades = strategy.generate_trades()
executed_trades = strategy.execute_trades(trades, max_trades=3)
```

See `examples/ssvi_strategy_example.py` for a complete working example.

### Other Strategies

Additional strategies that leverage the Option-Framework include:

- Delta hedging strategies
- Options portfolio optimization
- Volatility arbitrage

## Creating New Strategies

To create a new strategy:

1. Create a new strategy file in the strategies directory
2. Implement the strategy class with at minimum the following methods:
   - `__init__`: Initialize the strategy with configuration
   - `update`: Update the strategy with new market data
   - `generate_trades`: Generate trade recommendations
   - `execute_trades`: Execute recommended trades

3. Create appropriate configuration files in the `config` directory
4. Create an example script in the `examples` directory

## Backtesting

Strategies can be backtested using historical market data. The backtesting framework supports:

- Performance metrics (Sharpe ratio, max drawdown, etc.)
- Transaction costs and slippage modeling
- Portfolio simulation

Example:

```python
from strategies.ssvi_strategy import SSVIStrategy
from data_managers.option_data_manager import OptionDataManager

# Load historical data
option_data_manager = OptionDataManager()
historical_data = option_data_manager.load_historical_options('SPY', '2020-01-01', '2020-12-31')

# Initialize strategy
strategy = SSVIStrategy(config=config)

# Run backtest
results = strategy.backtest(historical_data, initial_capital=100000.0)
```

## Performance Considerations

- Options data processing can be computationally intensive
- The SSVI model fitting process can take time for large option chains
- Consider using cached SSVI parameters for frequently traded underlyings 