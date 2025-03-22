# Options Analysis Module

This directory contains modules for analyzing options data and identifying trading opportunities.

## Directory Structure

The analysis module is organized into the following subdirectories:

### Options

The `options` directory contains implementations of fundamental option pricing models and analytics, including:

- Black-Scholes Model (`black_scholes.py`): A classic option pricing model implementation with Greeks calculation
- Additional pricing models can be added here (Binomial, Monte Carlo, etc.)

### Relative Value

The `relative_value` directory contains models for analyzing relative value in options markets:

- SSVI Model (`ssvi_model.py`): Implementation of the Surface Stochastic Volatility Inspired parameterization for volatility surfaces
- This model helps identify mispriced options by comparing their implied volatility against a calibrated surface

## Usage

### Black-Scholes Model

```python
from analysis.options import BlackScholesModel

# Initialize model
bs_model = BlackScholesModel(risk_free_rate=0.02)

# Price an option
price = bs_model.price_option(
    underlying_price=100,
    strike_price=105,
    days_to_expiry=30,
    volatility=0.2,
    option_type='call'
)

# Calculate Greeks
greeks = bs_model.calculate_greeks(
    underlying_price=100,
    strike_price=105,
    days_to_expiry=30,
    volatility=0.2,
    option_type='call'
)

print(f"Option Price: {price}")
print(f"Delta: {greeks['delta']}")
print(f"Gamma: {greeks['gamma']}")
print(f"Theta: {greeks['theta']}")
print(f"Vega: {greeks['vega']}")
```

### SSVI Model for Relative Value

```python
from analysis.relative_value import SSVIModel
import pandas as pd

# Initialize model
ssvi_model = SSVIModel()

# Load option chain data
option_chain = pd.read_csv('option_data.csv')
underlying_price = 100.0

# Fit SSVI model to data
ssvi_model.fit(option_chain, underlying_price)

# Find relative value opportunities
rv_opportunities = ssvi_model.identify_rv_opportunities(
    option_chain, 
    underlying_price,
    zscore_threshold=1.5
)

# Print rich/cheap options
print(rv_opportunities[rv_opportunities['RVSignal'] == 'RICH'])
print(rv_opportunities[rv_opportunities['RVSignal'] == 'CHEAP'])
```

## Extending the Analysis Module

To add new analysis capabilities:

1. Create a new file in the appropriate subdirectory
2. Implement your model or analysis function
3. Update the corresponding `__init__.py` file to expose your new functionality
4. Add tests in the `tests/unit/analysis` directory
5. Update this README with usage examples 