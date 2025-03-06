"""
Trading System Data Analysis and Backtesting Framework

This package provides tools for analyzing financial market data,
backtesting trading strategies, and managing position risk.
"""

# Import key components to make them available at package level
from .core import (
    TradingEngine,
    Strategy,
    Position,
    OptionPosition,
    Portfolio
)

__version__ = '0.1.0'