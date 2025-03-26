# Enhanced Logging System

The Option-Framework includes a flexible logging system that allows you to control the verbosity of logs for both the overall application and for specific components like margin calculations and portfolio operations.

## Log Levels

### System-wide Log Levels

These control the overall logging verbosity:

- `DEBUG`: Show all logs including detailed debugging information
- `INFO`: Show information messages, warnings, and errors (default)
- `WARNING`: Only show warnings and errors
- `ERROR`: Only show errors
- `CRITICAL`: Only show critical errors

### Component-specific Log Levels

Components like margin calculations, portfolio operations, and trading have their own verbosity settings in the configuration file:

- `INFO`: Normal operational details (default)
- `DEBUG`: Detailed calculation steps and decision points
- `WARNING`: Only show warnings and errors
- `ERROR`: Only show errors

## Configuration Options

### In Configuration File (config.yaml)

You can configure logging preferences in the YAML configuration file:

```yaml
# Global logging settings
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_to_file: true  # Enable logging to file
  
  # Component-specific logging levels
  component_levels:
    margin: "INFO"  # INFO, DEBUG, WARNING, ERROR
    portfolio: "INFO"  # INFO, DEBUG, WARNING, ERROR
    trading: "INFO"  # INFO, DEBUG, WARNING, ERROR
```

### Command-line Options

All runner scripts support consistent command-line options:

```bash
# Use the configuration file but enable debug mode
python runners/run_put_sell_strat.py --debug

# Specify a different configuration file
python runners/run_put_sell_strat.py -c config/strategy/custom_config.yaml

# Skip data analysis for faster repeated runs
python runners/run_put_sell_strat.py --skip-analysis

# Override start and end dates
python runners/run_put_sell_strat.py --start-date 2024-01-01 --end-date 2024-01-31
```

## Standardized Logging Behavior

All runner scripts (`run_*.py`) now follow a consistent pattern:

1. Use `INFO` level logging by default
2. Support the `--debug` flag to enable `DEBUG` level logging
3. Respect logging settings from YAML configuration files
4. Standardized component-level logging configuration

## Examples

### Default INFO Level Logging

```bash
# Run with standard INFO level logging
python runners/run_put_sell_strat.py
```

### Detailed DEBUG Logging

```bash
# Run with detailed DEBUG logging
python runners/run_put_sell_strat.py --debug
```

### Using Custom Configuration

```bash
# Use a custom configuration with its own logging settings
python runners/run_put_sell_strat.py -c config/strategy/custom_config.yaml
```

## Log Sections

The logs are organized into sections for easier reading:

- `[Margin]`: Margin calculation logs
- `[Portfolio]`: Portfolio management logs
- `[Trading]`: Trading engine logs
- `[INIT]`: Initialization logs
- `[STATUS]`: Status updates

## Log File Output

All runner scripts automatically log to files in addition to the console output. The log files are stored in the directory specified in the configuration:

```yaml
paths:
  output_dir: "output/logs"
```

## Summary Sections

The system automatically generates summary sections that are always displayed regardless of log level. For example:

```
======= MARGIN CALCULATION SUMMARY =======
Total positions: 5
Total margin: $13,245.67
Hedging benefits: $2,458.12
Position value: $100,564.87
Margin utilization: 13.17%
=========================================
``` 