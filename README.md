# Python Option Framework

A modular backtesting framework designed for options trading strategies—especially those focused on theta decay and delta hedging.

## Introduction

The Python Option Framework uses an object-oriented design to facilitate extensibility and maintainability while offering robust risk management and detailed performance reporting. It's specifically designed for options trading strategies with a focus on realistic simulations.

## Features

- **Accurate Backtesting**: Simulate realistic options trading scenarios with advanced risk management
- **Modular and Extensible Design**: Independent components that can be extended or replaced as needed
- **Robust Risk Management**: Delta hedging and SPAN margin calculations
- **Comprehensive Reporting**: Interactive HTML reports and structured output files
- **Business Logic Verification**: Detailed output files with predefined structure for easy review of trading metrics

## Installation

### Prerequisites

- Python 3.8+ 
- Required packages: pandas, numpy, matplotlib, pyyaml

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/option-framework.git
cd option-framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the system by editing `config/config.yaml`

## Usage

Run a backtest with the default configuration:

```bash
python Main.py
```

Run with a specific configuration file:

```bash
python Main.py -c config/my_custom_config.yaml
```

Override settings from command line:

```bash
python Main.py --strategy ThetaDecayStrategy --start-date 2024-01-01 --end-date 2024-12-31 --verbose
```

## Configuration

The system is configured using YAML files. Key configuration sections include:

- **Paths**: File locations for input/output data
- **Dates**: Backtesting period
- **Portfolio**: Capital and risk settings
- **Strategy**: Strategy selection and parameters
- **Risk**: Risk management settings
- **Margin Management**: Margin calculation parameters
- **Reporting**: Output and logging settings

Example configuration:

```yaml
# Strategy parameters
strategy:
  name: "ThetaDecayStrategy"
  enable_hedging: true
  hedge_mode: "ratio"
  delta_target: -0.05
  profit_target: 0.65
```

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