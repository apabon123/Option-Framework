# Python Option Framework

A comprehensive backtesting framework for options trading strategies with a focus on theta decay and delta hedging.

## Overview

This system provides a modular, object-oriented framework for developing, testing, and analyzing quantitative trading strategies. It has been designed primarily for options trading with particular emphasis on:

- Theta decay strategies
- Delta hedging mechanisms
- Risk management
- Position tracking
- Performance analysis

## Features

- **Modular Design**: Clear separation of concerns with independent components
- **Flexible Strategy Pattern**: Implement custom strategies with a common interface
- **Comprehensive Position Management**: Track all aspects of trading positions
- **Advanced Risk Management**: Including SPAN margin calculations with delta hedging offsets
- **Detailed Performance Reporting**: HTML reports with interactive charts
- **Configurable Parameters**: YAML-based configuration for easy customization

## Project Structure

```
trading-system/
├── config/                     # Configuration files
│   └── config.yaml             # Main configuration file
├── core/                       # Core components
│   ├── data_manager.py         # Data loading and preprocessing
│   ├── hedging.py              # Delta hedging implementation
│   ├── margin.py               # Margin calculation including SPAN
│   ├── options_analysis.py     # Options pricing and volatility analysis
│   ├── portfolio.py            # Portfolio management
│   ├── position.py             # Position tracking
│   ├── reporting.py            # Visualization and reporting
│   └── trading_engine.py       # Main backtesting engine
├── strategies/                 # Strategy implementations
│   └── example_strategy.py     # Example option strategy
├── tests/                      # Unit tests
│   ├── test_portfolio.py       # Portfolio tests
│   ├── test_position.py        # Position tests
│   └── test_config.py          # Configuration tests
├── main.py                     # Main entry point
├── RunStrategies.py            # Strategy runner for parameter sweeps
└── theta_engine_2.py           # Theta strategy implementation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trading-system.git
   cd trading-system
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your environment:
   - Edit `config/config.yaml` to customize your backtesting parameters
   - Ensure your data files are in the correct format and location

## Usage

### Running a Backtest

To run a backtest with the default theta strategy:

```bash
python main.py
```

To run a specific strategy:

```bash
python RunStrategies.py
```

### Creating a Custom Strategy

1. Create a new Python file in the `strategies` directory
2. Inherit from the `Strategy` base class
3. Implement the required methods:
   - `generate_signals`: Create trading signals based on market data
   - `check_exit_conditions`: Determine when to exit positions

Example:

```python
from core.trading_engine import Strategy

class MyCustomStrategy(Strategy):
    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        # Initialize strategy-specific parameters
        
    def generate_signals(self, current_date, daily_data):
        # Implement signal generation logic
        signals = []
        # ... your logic here
        return signals
        
    def check_exit_conditions(self, position, market_data):
        # Implement exit condition logic
        # Return (exit_flag, reason)
        return False, None
```

### Analyzing Results

The system generates HTML reports with interactive charts in the specified output directory. These reports include:

- Equity curve
- Drawdown analysis
- Performance metrics (Sharpe ratio, volatility, etc.)
- Position-level statistics
- Greek exposures over time

## Configuration

The system is configured using YAML files. Key parameters include:

```yaml
# Date range for backtesting
dates:
  start_date: "2024-01-01"
  end_date: "2024-12-31"

# Portfolio settings
portfolio:
  initial_capital: 100000
  max_leverage: 12
  max_nlv_percent: 1.0

# Strategy parameters
strategy:
  name: "ThetaEngine"
  enable_hedging: true
  hedge_mode: "ratio"
  delta_target: -0.05
  profit_target: 0.65
  stop_loss_threshold: 2.5
```

See `config/config.yaml` for a full list of configuration options.

## Running Tests

To run the test suite:

```bash
pytest tests/
```

To run a specific test:

```bash
pytest tests/test_portfolio.py::TestPortfolio::test_add_position
```

## Code Style

This project follows these coding conventions:

- **Imports**: Grouped by standard library, third-party, local modules
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Documentation**: Docstrings for all classes and functions
- **Type Hints**: Using the typing module for parameters and returns
- **Error Handling**: Try/except with specific exception types
- **Logging**: Structured logging with levels and context

## Acknowledgments

- Option pricing models based on Black-Scholes-Merton
- SPAN margin calculations inspired by CME Group methodology