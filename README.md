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

The framework implements a sophisticated SPAN (Standard Portfolio Analysis of Risk) margin calculator with the following features:

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

### Configuration

Main configuration is handled through `config/config.yaml`:

```yaml
# Portfolio settings
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