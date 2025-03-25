# Utilities

This directory contains utility classes and functions used throughout the Options Framework.

## SimpleLoggingManager

The `SimpleLoggingManager` provides a simplified logging interface for the Options Framework. It handles setup of logging to both console and file outputs.

### Features

- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Console and file output
- Component-specific logging levels
- Automatic log file naming based on strategy name and timestamp
- Output directory configuration based on project settings

### Usage

```python
from utils.simple_logging import SimpleLoggingManager

# Create the logging manager
logging_manager = SimpleLoggingManager()

# Set up logging with a configuration dictionary
logger = logging_manager.setup_logging(
    config_dict=config,
    verbose_console=True,  # Show detailed logs in console 
    debug_mode=True,       # Enable debug level logging
    clean_format=False     # Include timestamp and level in logs
)

# Get the log file path for reference
log_file_path = logging_manager.get_log_file_path()
```

### Configuration

The logging manager will read the output directory from the config file in the following order:
1. `config['paths']['output_dir']`
2. `config['output']['directory']`
3. `config['paths']['output_directory']`

If none of these are found, it will use a default 'logs' directory.

## Other Utilities

### data_processor.py

Provides functions for processing and manipulating option data.

### theta_engine_config.py

Configuration management for the Theta engine. 