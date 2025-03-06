"""
Core package for trading engine components.

This package contains all the core components needed for the trading system,
including position management, portfolio management, margin calculation,
and the main trading engine.
"""

# Make key classes directly importable from the core package
from .trading_engine import TradingEngine, Strategy, LoggingManager
from .position import Position, OptionPosition
from .portfolio import Portfolio
from .data_manager import DataManager
from .margin import MarginCalculator, OptionMarginCalculator, SPANMarginCalculator
from .hedging import HedgingManager
from .reporting import ReportingSystem

__all__ = [
    'TradingEngine',
    'Strategy',
    'LoggingManager',
    'Position',
    'OptionPosition',
    'Portfolio',
    'DataManager',
    'MarginCalculator',
    'OptionMarginCalculator',
    'SPANMarginCalculator',
    'HedgingManager',
    'ReportingSystem'
]