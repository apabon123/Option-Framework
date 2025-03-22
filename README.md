# Option Trading Framework

A comprehensive options trading framework designed for backtesting and simulating options trading strategies with sophisticated risk management and dynamic delta hedging capabilities.

## Overview

This framework provides a complete environment for developing, testing, and evaluating options trading strategies. It features accurate Greek calculations, dynamic delta hedging, position management, and detailed performance reporting.

## Recent Changes

- **Enhanced SPAN Margin Calculator**: Greatly improved the SPAN margin calculation system with configurable parameters, vega risk incorporation, proper delta hedging benefits, and partial hedge handling
- **Configuration-Driven Architecture**: Added YAML configuration for all margin parameters, allowing easy adjustment to match different broker requirements
- **Greek Sign Convention Fixes**: Implemented proper handling of signs for delta, gamma, theta, and vega for both short and long positions in calculation and display contexts
- **Position Storage Improvements**: Centralized position storage in the Position Inventory as a single source of truth
- **Hedging Logic Refinement**: Fixed boundary adjustments in the hedging algorithm to prevent over-hedging
- **Display Formatting**: Improved display of Greeks in trading reports and logs for better readability
- **Enhanced SPAN margin calculator with realistic hedging benefits**
- **Improved greeks sign handling throughout the codebase**
- **Added option to configure which margin calculator to use (SPAN, Simple, or Option-specific)**
- **Configuration-driven architecture with comprehensive YAML configuration**
- **Implemented portfolio margin calculation with proper delta hedging**
- **Added advanced volatility bootstrapping for option pricer**
- **Better handling for short options and long stock positions**

## Architecture

The framework is built with a modular design:

### Core Components

- **Position Management (`position.py`)**:
  - `Position`: Base class for all financial instrument positions
  - `OptionPosition`: Specialized class for options with Greek calculations
  - Handles accurate tracking of P&L, risk metrics, and position lifecycle
  - Implements `get_greeks(for_display=False)` with proper sign conventions

- **Position Inventory (`position_inventory.py`)**:
  - Central repository for all positions as the single source of truth
  - Aggregates portfolio-level Greeks and risk metrics
  - Provides consistent access to position data for all components

- **Margin Calculator System (`margin.py`)**:
  - Implements multiple margin calculation methods including SPAN methodology
  - Features configurable parameters for matching different broker requirements
  - Accurately calculates portfolio margin with proper hedging benefits
  - Supports both perfect and partial delta hedging scenarios
  - Incorporates comprehensive risk factors (delta, gamma, vega) in scan risk calculations

- **Hedging System (`hedging.py`)**:
  - Implements dynamic delta hedging strategies
  - Supports ratio-based and constant-delta approaches
  - Uses a "hedge to boundary" approach to prevent over-hedging

- **Trading Engine (`trading_engine.py`)**:
  - Coordinates the entire trading simulation
  - Processes market data updates
  - Executes trades and manages the portfolio
  - Formats and displays position data with proper Greek signs

### Key Features

- **Accurate Greek Calculations**: Proper handling of Greek signs for both display and calculation purposes
- **Sophisticated Hedging**: Dynamic delta hedging with configurable tolerance bands
- **SPAN Margin Calculations**: Advanced portfolio margin calculations with proper hedging benefits
- **Position Management**: Complete position lifecycle tracking with P&L calculation
- **Risk Management**: Comprehensive risk metrics and exposure monitoring
- **Performance Reporting**: Detailed reports on strategy performance

## Position and Greek Sign Handling

The framework implements a sophisticated approach to handling option positions and their Greeks:

### Option Position Storage

1. **Position Creation**:
   - Option positions are created from market data with their original Greek values
   - The system tracks both long and short positions with proper accounting

2. **Greek Sign Storage**:
   - Greeks are initially stored with their raw values from the input data
   - For short positions, signs are adjusted during position initialization
   - The `update_market_data` method adjusts signs based on position type when new data arrives

3. **Greek Retrieval System**:
   - The `get_greeks(for_display)` method provides Greeks with appropriate signs based on context
   - `for_display=False`: Returns Greeks with mathematically correct signs for calculations
   - `for_display=True`: Returns Greeks with conventional signs for display purposes

### Greek Sign Conventions

#### Calculation Mode (`for_display=False`):
- **Delta**: 
  - Long Calls: Positive
  - Long Puts: Negative
  - Short Calls: Negative
  - Short Puts: Positive (critical for hedging calculations)
- **Gamma**:
  - Long Positions: Positive
  - Short Positions: Negative
- **Theta**:
  - Long Positions: Negative
  - Short Positions: Positive
- **Vega**:
  - Long Positions: Positive
  - Short Positions: Negative

#### Display Mode (`for_display=True`):
Additional sign adjustments for clearer reporting of risk exposure in UI and logs.

## SPAN Margin Calculation

The framework includes a sophisticated SPAN margin calculator that:
- Properly accounts for delta hedging benefits
- Applies industry-standard scan risk calculations
- Includes volatility shifts in margin calculations
- Scales margins based on option moneyness
- Adjusts margins based on regulatory minimums

The margin calculator type is configurable through the YAML configuration file, allowing you to choose between:
- `span`: Advanced SPAN-style portfolio margining with delta hedging benefits (default)
- `option`: Option-specific margin calculations without portfolio-level benefits
- `simple`: Basic margin calculations based on maximum leverage

### Key Components

- **Scan Risk Calculation**: Combines delta, gamma, and vega risk factors to estimate potential losses under various market scenarios
- **Portfolio Margin**: Calculates margin requirements across a portfolio with proper hedging offsets
- **Configuration Driven**: All parameters are configurable through YAML configuration files

### Configurable Parameters

```yaml
margin:
  span:
    # Maximum leverage allowed (higher means less margin required)
    max_leverage: 12.0
    
    # Initial margin as percentage of notional value
    initial_margin_percentage: 0.1
    
    # Maintenance margin as percentage of notional value
    maintenance_margin_percentage: 0.07
    
    # Credit rate applied to hedged positions (0.0 to 1.0)
    hedge_credit_rate: 0.8
    
    # Price move percentage for risk scenarios
    price_move_pct: 0.05
    
    # Volatility shift for risk scenarios
    vol_shift_pct: 0.3
    
    # Scaling factor applied to gamma effects
    gamma_scaling_factor: 0.3
    
    # Minimum scan risk as percentage of option premium
    min_scan_risk_percentage: 0.25
    
    # Maximum ratio of margin to option premium
    max_margin_to_premium_ratio: 20.0
    
    # Whether to scale margin lower for out-of-the-money options
    otm_scaling_enabled: true
    
    # Minimum scaling for far out-of-the-money options
    otm_minimum_scaling: 0.1
```

### Hedging Benefits

The margin calculator properly accounts for delta hedging in the portfolio:
- Perfect hedges receive the full hedge credit rate reduction (e.g., 80% margin reduction)
- Partial hedges receive proportional benefits based on hedge quality
- The system properly handles hedging across different position types (options and stocks)

## Hedging Strategy

The framework implements a "hedge to boundary" approach:
- Maintains portfolio delta within a target range defined by tolerance
- When delta exceeds the upper boundary: Hedges down to the upper boundary
- When delta falls below the lower boundary: Hedges up to the lower boundary
- Within boundaries: No hedging required

This prevents over-hedging and reduces unnecessary trading costs.

## Usage

```bash
# Run a backtest with verbose output
python main.py -v

# Run with specific configuration
python main.py --config custom_config.yaml

# Generate a performance report
python main.py --report
```

## Configuration

The framework is fully configurable through the `config/config.yaml` file. Key configuration sections include:

### Margin Management Configuration
```yaml
margin_management:
  # High and target margin usage thresholds (as percentage of available margin)
  high_margin_threshold: 0.98
  target_margin_threshold: 0.95
  
  # Type of margin calculator to use (span, option, simple)
  margin_calculator_type: "span"
  
  # Method for margin calculation (portfolio, simple)
  margin_calculation_method: "portfolio"
  
  # Maximum leverage allowed for portfolio
  max_leverage: 12.0
  
  #... other margin settings
```

### Portfolio Settings
```yaml
portfolio:
  initial_capital: 100000
  max_leverage: 12
  max_nlv_percent: 1.0
  max_position_size_pct: 0.25

# Hedging configuration
hedging:
  enabled: true
  mode: "ratio"
  hedge_with_underlying: true
  target_delta_ratio: 0.1     # Target delta as percentage of NLV
  delta_tolerance: 0.3        # Tolerance in percentage of NLV
  hedge_symbol: "SPY"
  max_hedge_ratio: 3.0

# Strategy parameters
strategy:
  name: "ThetaDecayStrategy"
  days_to_expiry_min: 60
  days_to_expiry_max: 90
  is_short: true
  delta_target: -0.2
  delta_tolerance: 0.01
  profit_target: 0.65
  stop_loss_threshold: 2.5
  close_days_to_expiry: 14
```

## Input Data Format

The framework expects option data with the following format:
- Puts have negative delta (market standard for long puts)
- Calls have positive delta (market standard for long calls)
- The system handles adjusting signs for short positions internally

## Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib (for visualization)
- tabulate (for formatted output)

## Development

When modifying the codebase, ensure:
- The `get_greeks(for_display=False)` method is used for all calculation contexts
- The `get_greeks(for_display=True)` method is used for display purposes
- The hedging system has the correct Greek signs for accurate delta calculations

## Project Structure

```
option-framework/
├── config/                     # YAML configuration files
│   └── config.yaml             # Main configuration file
├── core/                       # Core components and business logic
│   ├── data_manager.py         # Data loading and preprocessing
│   ├── hedging.py              # Delta hedging implementation
│   ├── margin.py               # SPAN margin calculations
│   ├── options_analysis.py     # Options pricing and volatility analysis
│   ├── portfolio.py            # Portfolio management
│   ├── position.py             # Position tracking
│   ├── reporting.py            # Visualization and reporting
│   └── trading_engine.py       # Main backtesting engine
├── strategies/                 # Strategy implementations
│   ├── example_strategy.py     # Example option strategy
│   └── theta_strategy.py       # Theta decay focused strategy
├── tests/                      # Unit tests for validating functionality
├── main.py                     # Main entry point
└── README.md                   # This file
```

## Creating Custom Strategies

Create new strategies by extending the base Strategy class:

```python
from core.trading_engine import Strategy

class MyCustomStrategy(Strategy):
    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        # Initialize strategy-specific parameters
        
    def generate_signals(self, current_date, daily_data):
        signals = []  # Implement signal generation logic here
        return signals
        
    def check_exit_conditions(self, position, market_data):
        # Determine exit conditions
        return False, None
```

## Business Logic Verification

The framework produces detailed verification files in a structured format that includes:

- Daily P&L breakdown
- Position and risk information
- Portfolio Greek metrics (Delta, Gamma, Theta, Vega)
- Detailed trade tables
- Risk management logs

These files allow for thorough verification of the framework's calculations and decision-making process.

## License

[MIT License](LICENSE)

## Acknowledgments

- Option Pricing Models: Based on the Black-Scholes-Merton framework
- SPAN Margin Calculations: Inspired by methodologies used by CME Group

# Intraday Momentum Strategy

The Option-Framework now includes an implementation of the Intraday Momentum Strategy, based on the paper "Beat the Market - An Effective Intraday Momentum Strategy".

## Strategy Overview

The Intraday Momentum Strategy identifies breakouts from a "noise range" established around the opening price. The strategy:

1. Calculates a volatility-based noise range around the opening price (or previous close)
2. Generates long signals when price breaks above the upper bound
3. Generates short signals when price breaks below the lower bound
4. Uses trailing stops based on the noise range boundaries

## Configuration

The strategy can be configured through the `config/intraday_momentum_config.yaml` file. Key parameters include:

- `lookback_days`: Number of days to look back for volatility calculation
- `volatility_multiplier`: Multiplier applied to average range to determine the noise range
- `entry_times`: List of minute marks to check for entry signals (e.g., [0, 15, 30, 45])
- `min_holding_period_minutes`: Minimum time to hold a position before considering exit
- `invert_signals`: Whether to invert the strategy signals (false by default)

## Running the Strategy

To run the Intraday Momentum Strategy:

```bash
python run_intraday_strategy.py --config config/intraday_momentum_config.yaml
```

Optional parameters:
- `--debug`: Enable debug mode with additional logging
- `--margin-log-level`: Set the margin log level (debug, info, warning, error, verbose)

## Data Requirements

The strategy works best with intraday data (minute bars) but can also work with daily data in a simplified mode. For intraday trading, ensure your data has proper timestamp indexing with timezone information.

## Risk Management Module

The Option-Framework includes a comprehensive risk management system that allows for dynamic position sizing and risk control across different market conditions. The risk module provides several risk management approaches:

### Available Risk Managers

- **VolatilityTargetRiskManager**: Scales position sizes to target a specific annualized volatility, automatically adjusting for changing market conditions.
- **SharpeRatioRiskManager**: Adjusts position sizes based on risk-adjusted returns, scaling up for strategies with better Sharpe ratios.
- **AdaptiveRiskManager**: Implements a "heat" system that reduces position sizes after losses and gradually increases them during positive performance periods.
- **CombinedRiskManager**: Uses a weighted combination of multiple risk approaches for more robust position sizing.

### Key Features

- Configurable risk limits including maximum position sizes, daily loss limits, and concentration limits
- Dynamic position sizing based on performance metrics and market conditions
- Comprehensive risk metrics calculation including drawdowns, Sharpe ratios, and Value at Risk
- Full integration with trading strategies and backtesting systems

### Configuration

Risk managers are configured through YAML configuration files. See `config/risk_config.yaml` for a detailed example with comments explaining each parameter.

```yaml
# Example of basic risk configuration
risk:
  risk_limits:
    max_position_size: 10
    max_daily_loss: 0.02
  manager_type: "volatility"
  volatility:
    target_volatility: 0.15
```

### Usage

To use a risk manager in your strategy:

```python
from core.risk.factory import RiskManagerFactory

# Create a risk manager from config
risk_manager = RiskManagerFactory.create(config, risk_metrics, logger)

# Calculate position size
position_size = risk_manager.calculate_position_size(data, portfolio_metrics)
```