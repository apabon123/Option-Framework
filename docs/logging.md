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

Components like margin calculations and portfolio operations have their own verbosity settings:

- `minimal`: Only display essential information (summary results)
- `standard`: Normal operational details (default)
- `verbose`: Detailed calculation steps and decision points
- `debug`: Full internal details for troubleshooting

## Configuration Options

### In Configuration File (config.yaml)

You can configure logging preferences in the YAML configuration file:

```yaml
# Global logging settings
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Margin component logging
margin:
  logging:
    level: "standard"  # minimal, standard, verbose, debug

# Portfolio component logging
portfolio:
  logging:
    level: "standard"  # minimal, standard, verbose, debug
```

### Command-line Options

You can override logging settings from the command line:

```bash
# Override global logging level
python Main.py --log-level INFO

# Override margin calculation verbosity
python Main.py --margin-log-level minimal

# Override portfolio operations verbosity
python Main.py --portfolio-log-level verbose

# Enable verbose console output
python Main.py -v

# Enable debug mode (equivalent to --log-level DEBUG)
python Main.py -d
```

## Examples

### Minimal Logging for Daily Use

```bash
# Run with minimal margin logs but standard portfolio logs
python Main.py --config config/config.yaml --margin-log-level minimal
```

### Detailed Logging for Debugging

```bash
# Run in debug mode with verbose margin logs
python Main.py --config config/config.yaml -d --margin-log-level verbose
```

### Quiet Mode

```bash
# Run with minimal logs for all components
python Main.py --log-level WARNING --margin-log-level minimal --portfolio-log-level minimal
```

## Log Sections

The logs are organized into sections for easier reading:

- `[Margin]`: Margin calculation logs
- `[Portfolio]`: Portfolio management logs
- `[TradeManager]`: Trade execution logs
- `[INIT]`: Initialization logs
- `[STATUS]`: Status updates

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