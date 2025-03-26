"""
Simplified Logging Manager for Option Framework

This module provides a stripped-down version of the LoggingManager class
that removes complex functionality causing issues.
"""

import os
import logging
import sys
import io
from datetime import datetime
from typing import Dict, Any, Optional


class UnicodeStreamHandler(logging.StreamHandler):
    """A StreamHandler that can handle Unicode characters properly."""
    
    def __init__(self, stream=None):
        """Initialize the handler with a stream that can handle Unicode."""
        super().__init__(stream)
        
    def emit(self, record):
        """
        Emit a record with proper Unicode handling.
        
        Args:
            record: LogRecord to emit
        """
        try:
            msg = self.format(record)
            stream = self.stream
            # Replace problematic characters with their ASCII equivalents
            msg = msg.replace("✓", "VERIFIED")
            msg = msg.replace("✗", "FAILED")
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


class SimpleLoggingManager:
    """
    Simplified logging manager that avoids complex filtering logic.
    """

    def __init__(self):
        """Initialize the LoggingManager"""
        self.logger = None
        self.log_file = None
        self.original_stdout = None
        self.original_stderr = None
        self.component_log_levels = {}
        
    def setup_logging(self, config_dict: Dict[str, Any], verbose_console: bool = True, 
                     debug_mode: bool = False, clean_format: bool = False) -> logging.Logger:
        """
        Set up logging configuration.
        
        Args:
            config_dict: Configuration dictionary
            verbose_console: Whether to output verbose logs to console
            debug_mode: Enable debug level logging
            clean_format: Use clean logging format without timestamp/level
            
        Returns:
            logging.Logger: Configured logger
        """
        # Get log level from config
        log_level_str = config_dict.get('logging', {}).get('level', 'INFO')
        log_levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        # Use DEBUG if debug_mode is True, otherwise use the configured level
        log_level = logging.DEBUG if debug_mode else log_levels.get(log_level_str, logging.INFO)
        
        # Set up component-specific log levels (always as strings)
        self.component_log_levels = {}
        
        # Check for component-specific logging levels in component_levels (new preferred structure)
        if 'logging' in config_dict and 'component_levels' in config_dict['logging']:
            for component, level in config_dict['logging']['component_levels'].items():
                self.component_log_levels[component] = level
        
        # Check for component-specific logging settings in the components structure (backward compatibility)
        if 'logging' in config_dict and 'components' in config_dict['logging']:
            # Check for margin logging settings
            if 'margin' in config_dict['logging']['components']:
                level = config_dict['logging']['components']['margin'].get('level', 'standard')
                self.component_log_levels['margin'] = str(level) if level is not None else 'standard'
                
            # Check for portfolio logging settings
            if 'portfolio' in config_dict['logging']['components']:
                level = config_dict['logging']['components']['portfolio'].get('level', 'standard')
                self.component_log_levels['portfolio'] = str(level) if level is not None else 'standard'
        else:
            # Fallback to the old structure for backward compatibility
            # Check for margin logging settings
            if 'margin' in config_dict and 'logging' in config_dict['margin']:
                level = config_dict['margin']['logging'].get('level', 'standard')
                self.component_log_levels['margin'] = str(level) if level is not None else 'standard'
                
            # Check for portfolio logging settings
            if 'portfolio' in config_dict and 'logging' in config_dict['portfolio']:
                level = config_dict['portfolio']['logging'].get('level', 'standard')
                self.component_log_levels['portfolio'] = str(level) if level is not None else 'standard'
        
        # Print the component log levels for debugging
        print(f"Component log levels after setup: {self.component_log_levels}")
        
        # Create logger
        logger = logging.getLogger('trading_engine')
        logger.setLevel(log_level)
        
        # Clear any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create console handler with Unicode support
        ch = UnicodeStreamHandler()
        ch.setLevel(log_level if verbose_console else logging.WARNING)
        
        # Create formatter
        if clean_format:
            formatter = logging.Formatter('%(message)s')
        else:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # Determine output directory for log files
        # Check various config locations where the output directory might be specified
        output_dir = 'logs'  # Default fallback
        
        # Option 1: Check paths.output_dir
        if 'paths' in config_dict and 'output_dir' in config_dict['paths']:
            output_dir = config_dict['paths']['output_dir']
        
        # Option 2: Check output.directory
        elif 'output' in config_dict and 'directory' in config_dict['output']:
            output_dir = config_dict['output']['directory']
        
        # Option 3: Check the old paths.output_directory format
        elif 'paths' in config_dict and 'output_directory' in config_dict['paths']:
            output_dir = config_dict['paths']['output_directory']
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = config_dict.get('strategy', {}).get('name', 'strategy')
        log_filename = f"{strategy_name}_{timestamp}.log"
        
        # Check if file logging is enabled in config
        file_logging_enabled = config_dict.get('logging', {}).get('log_to_file', 
                                              config_dict.get('logging', {}).get('file', True))
        
        if file_logging_enabled:
            self.log_file = os.path.join(output_dir, log_filename)
            fh = logging.FileHandler(self.log_file, encoding='utf-8')
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            print(f"Logging to file: {self.log_file}")
        
        self.logger = logger
        return logger
    
    def get_log_file_path(self) -> Optional[str]:
        """Get the path to the current log file."""
        return self.log_file
    
    def should_log(self, component_name: str, level_name: str) -> bool:
        """
        Determine if a log message should be logged based on component and level.
        
        Args:
            component_name: Name of the component (e.g., 'margin', 'portfolio')
            level_name: Level to log at (e.g., 'minimal', 'standard', 'verbose')
            
        Returns:
            bool: True if the message should be logged, False otherwise
        """
        # If the logger is None or disabled, don't log
        if not self.logger or getattr(self.logger, 'disabled', False):
            return False
            
        # Map level names to numeric priorities (higher = more verbose)
        level_priorities = {
            'minimal': 0,    # Always show these logs (WARNING+)
            'standard': 1,   # Show in normal operation (INFO+)
            'verbose': 2,    # Show in verbose mode (DEBUG+)
            'debug': 3       # Show only in debug mode
        }
        
        # Get the configured level for this component
        component_level = self.get_component_log_level(component_name)
        
        # Map component_level string to logger levels
        component_level_map = {
            'minimal': logging.WARNING,
            'standard': logging.INFO,
            'verbose': logging.DEBUG,
            'debug': logging.DEBUG
        }
        
        # Map string levels to logging levels
        string_to_log_level = {
            'WARNING': logging.WARNING,
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG
        }
        
        # If component_level is a string level (WARNING, INFO, DEBUG), use direct mapping
        current_level = string_to_log_level.get(component_level, 
                                               component_level_map.get(component_level, logging.INFO))
        
        # Compare with the message's level
        message_level = component_level_map.get(level_name, logging.INFO)
        
        # Allow the message if its level is at least as important as the current level
        return message_level >= current_level
        
    def get_component_log_level(self, component_name: str) -> str:
        """Get the component log level as a string."""
        return self.component_log_levels.get(component_name, "standard")
        
    def disable(self):
        """Disable logging."""
        if self.logger:
            self.logger.disabled = True
            
    def _clear_cache(self):
        """No cache to clear in this simple implementation."""
        pass 