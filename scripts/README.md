# Option Framework Strategy Scripts

This directory contains the runner scripts for the various trading strategies implemented in the Option Framework.

## Unified Strategy Runner

The `run_strategy.py` script provides a unified interface for running any strategy registered in the system. It reads from the central strategy registry to determine which strategy to run and with what configuration.

### Usage

```bash
python scripts/run_strategy.py [strategy_id] [options]
```

To list all available strategies:

```bash
python scripts/run_strategy.py --list
```

To list strategies in a specific category:

```bash
python scripts/run_strategy.py --list_category options_strategies
```

### Options

- `strategy_id`: ID of the strategy to run (from the strategy registry)
- `--config`: Path to the strategy configuration file (overrides registry config)
- `--start_date`: Start date for backtest (YYYY-MM-DD)
- `--end_date`: End date for backtest (YYYY-MM-DD)
- `--input_file`: Path to input data file
- `--output_dir`: Directory for output files
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Individual Strategy Runners

Each strategy also has its own dedicated runner script, which can be run directly:

### Theta Decay Strategy

```bash
python scripts/run_theta_decay.py --config config/strategy/theta_decay_config.yaml
```

### Call-Put Strategy

```bash
python scripts/run_call_put.py --config config/strategy/call_put_config.yaml
```

### Intraday Momentum Strategy

```bash
python scripts/run_intraday_momentum.py --config config/strategy/intraday_momentum_config.yaml
```

### Volatility Breakout Strategy

```bash
python scripts/run_volatility_breakout.py --config config/strategy/volatility_breakout_config.yaml
```

### Put Selling Strategy

```bash
python scripts/run_put_sell_strat.py --config config/strategy/put_sell_config.yaml
```

## Common Options for Individual Runners

All individual strategy runners accept the following common options:

- `--config`: Path to the strategy configuration file
- `--start_date`: Start date for backtest (YYYY-MM-DD), overrides config file
- `--end_date`: End date for backtest (YYYY-MM-DD), overrides config file
- `--input_file`: Path to input data file, overrides config file
- `--output_dir`: Directory for output files, overrides config file
- `--log_level`: Logging level, overrides config file

## Configuration Files

Each strategy has its own configuration file located in the `config/strategy/` directory. These files define all the parameters needed for the strategy to run, including:

- Strategy-specific parameters
- Data input settings
- Backtest date range
- Portfolio settings
- Logging configuration
- Output settings

## Examples

Run the Theta Decay Strategy with default configuration:

```bash
python scripts/run_strategy.py theta_decay
```

Run the Volatility Breakout Strategy with custom date range:

```bash
python scripts/run_strategy.py volatility_breakout --start_date 2024-01-01 --end_date 2024-02-01
```

Run the Put Selling Strategy with a custom configuration file:

```bash
python scripts/run_strategy.py put_sell --config custom_configs/my_put_sell_config.yaml
```

Run the Intraday Momentum Strategy with debug logging:

```bash
python scripts/run_strategy.py intraday_momentum --log_level DEBUG
``` 