"""
Strategies package for trading system.

This package contains various trading strategy implementations
that can be used with the trading engine.
"""

# Import strategies so they're available directly from the package
from .example_strategy import SimpleOptionStrategy

# Export the strategy classes
__all__ = ['SimpleOptionStrategy']