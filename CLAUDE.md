# Python Quantitative Finance Trading System Guidelines

## Running Code
- Main entry point: `python main.py`
- Option data extraction: `python OptionDataExtraction.py`
- Strategy backtesting: `python RunStrategies.py`
- Theta engine: `python theta_engine_2.py`
- Market data: `python get_bars.py`
- Type checking: `mypy .`
- Tests: `pytest tests/` (individual test: `pytest tests/test_file.py::test_function`)

## Code Style Guidelines
- **Imports**: Group by: standard library, third-party, local modules
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Documentation**: Docstrings for all classes and functions
- **Type Hints**: Use typing module for parameters and returns
- **Error Handling**: Try/except with specific exception types
- **Logging**: Use LoggingManager for structured logging
- **Data Processing**: Prefer pandas vectorized operations

## Architecture
- Object-oriented design with separation of concerns
- Strategy pattern for implementing different trading algorithms
- Position tracking with unified PnL calculation
- Margin calculation with SPAN margin simulation
- Data flow: extraction → processing → analysis → visualization

## Project Organization
- `core/` - Core trading components (data, position, portfolio, margin)
- `strategies/` - Trading strategy implementations
- `utils/` - Shared utility functions
- `config/` - Configuration files
- `scenario_results/` - Backtest results storage