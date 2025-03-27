# Option Trading Framework Documentation

Welcome to the Option Trading Framework documentation. This framework provides a comprehensive system for developing, testing, and deploying options trading strategies with advanced risk management capabilities.

## Documentation Structure

The documentation is organized into several sections:

### Core Components

- [Risk Management Architecture](risk_management_architecture.md) - Overview of the risk management system
- [Position Sizer](position_sizer.md) - Documentation for the Position Sizer component
- [Risk Scaler](risk_scaler.md) - Documentation for the Risk Scaler component
- [Component Integration](component_integration.md) - How components work together

### Configuration

- [Strategy Configuration](strategy_configuration.md) - How to configure trading strategies
- [Position Sizing Configuration](position_sizing_configuration.md) - Position sizing parameters
- [Risk Scaling Configuration](risk_scaling_configuration.md) - Risk scaling parameters
- [Hedging Configuration](hedging_configuration.md) - Hedging parameters

### Guides

- [Getting Started](getting_started.md) - Quick start guide
- [Creating Strategies](creating_strategies.md) - Guide to creating new strategies
- [Backtesting](backtesting.md) - How to backtest strategies
- [Live Trading](live_trading.md) - How to deploy strategies for live trading

### Reference

- [API Reference](api_reference.md) - Complete API documentation
- [Configuration Reference](configuration_reference.md) - All configuration options
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

## Quick Start

1. [Install the framework](getting_started.md#installation)
2. [Configure your first strategy](getting_started.md#configuration)
3. [Run a backtest](backtesting.md)
4. [Analyze the results](backtesting.md#analysis)
5. [Deploy for live trading](live_trading.md)

## Key Concepts

### Risk Management Architecture

The framework uses a modular risk management architecture with several key components:

- **Position Sizer**: Determines appropriate position sizes based on account value, margin requirements, and risk parameters
- **Risk Scaler**: Dynamically adjusts position sizes based on performance metrics
- **Margin Manager**: Calculates margin requirements for positions
- **Hedging Manager**: Manages delta hedging relationships
- **Trading Engine**: Coordinates all components during strategy execution

For more details, see the [Risk Management Architecture](risk_management_architecture.md) documentation.

### Strategy Development

The framework supports both rule-based and model-based strategies:

- **Rule-based strategies**: Defined through YAML configuration files
- **Model-based strategies**: Implemented using Python classes
- **Hybrid strategies**: Combining rules and models

For more information, see the [Creating Strategies](creating_strategies.md) documentation.

### Data Sources

The framework supports multiple data sources for options and underlying assets:

- **Historical data**: CSV files, databases, and APIs
- **Real-time data**: Market data providers and broker APIs
- **Custom data**: User-defined data sources

### Execution

Strategies can be executed in different modes:

- **Backtesting**: Test strategies against historical data
- **Paper trading**: Simulate trading with real-time data
- **Live trading**: Execute trades through a broker API

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Trading Engine                           │
└───────────────┬─────────────────┬────────────────┬─────────────┘
                │                 │                │
    ┌───────────▼──────┐ ┌───────▼────────┐ ┌─────▼───────┐
    │  Position Sizer  │ │  Risk Scaler   │ │ Data Engine │
    └───────────┬──────┘ └────────────────┘ └─────────────┘
                │
    ┌───────────▼──────┐ ┌──────────────────┐
    │  Margin Manager  │ │ Hedging Manager  │
    └──────────────────┘ └──────────────────┘
```

## Getting Started

See the [Getting Started](getting_started.md) guide for installation instructions and a quick tutorial on setting up your first strategy.

## Support and Community

- [GitHub Issues](https://github.com/apabon123/Option-Framework/issues) - Report bugs or request features
- [Discussion Forum](https://github.com/apabon123/Option-Framework/discussions) - Ask questions and share ideas

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 