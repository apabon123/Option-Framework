# Margin Tests

This directory contains tests for the margin calculation components of the Option Framework.

## Test Files

### `margin_calculator_test.py`
Comprehensive tests for all margin calculators:
- Basic `MarginCalculator` (simple leverage-based)
- `OptionMarginCalculator` (with OTM discounts)
- `SPANMarginCalculator` (full SPAN methodology implementation)

This script tests margin calculations across various option types, moneyness levels, and volatility scenarios.

### `run_hedge_test.py`
Tests specifically for the delta hedging functionality of the `SPANMarginCalculator`:
- Tests that positions with offsetting deltas receive hedging benefits
- Tests that positions with deltas in the same direction receive no hedging benefits

### `margin_test.py`
Simpler, more targeted tests for specific margin calculation scenarios.

### `run_margin_test.py`
Runner script to execute the comprehensive margin calculator tests.

## Running the Tests

### Running All Margin Tests
From the project root:
```
python tests/margin_tests/run_margin_test.py
```

### Running Delta Hedge Tests Specifically
From the project root:
```
python tests/margin_tests/run_hedge_test.py
```

### Running All Tests in the Repository
From the project root:
```
python tests/run_tests.py
```

## Test Output
Test output logs are stored in the `output` directory at the project root. 