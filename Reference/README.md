# Reference Implementations

This directory contains reference implementations and legacy code that served as the foundation for the Option-Framework. These files are kept for historical reference, research purposes, and to compare implementation approaches.

## Contents

- **IntraDayMom2_original.py**: Original implementation of the intraday momentum strategy based on the paper "Beat the Market - An Effective Intraday Momentum Strategy". This implementation has been refactored into the modular architecture of the Option-Framework.

- **RunStrategies.py**: Legacy script for running various trading strategies. The functionality has been enhanced and incorporated into the current trading engine.

- **theta_engine_2.py**: Legacy implementation of the theta decay strategy engine. The core concepts have been integrated into the current framework with improved risk management and position tracking.

## Note on Usage

These files are not actively maintained and are provided for reference only. It is recommended to use the current modular implementation in the Option-Framework instead of these legacy versions. The current implementation benefits from:

- Improved code organization and modularity
- Enhanced risk management capabilities
- Better data management with specialized data managers
- Comprehensive testing and validation
- Consistent error handling and logging

## Migration

If you have code that depends on these legacy implementations, consider migrating to the current framework structure. The `examples/` directory provides demonstrations of how to use the current framework components.

For questions about migration or understanding the relationship between legacy and current implementations, please refer to the documentation or open an issue on the project repository. 