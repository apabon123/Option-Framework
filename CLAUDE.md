# Python Quantitative Finance Trading System Guidelines

## Running Code & Testing
- Main entry point: `python Main.py`
- Run strategy: `python run_simple_strategy.py`
- Reference scripts: `python Reference/RunStrategies.py`, `python Reference/theta_engine_2.py`
- Type checking: `mypy .` (run before committing changes)
- Run all tests: `pytest tests/`
- Run single test file: `pytest tests/test_portfolio.py`
- Run specific test: `pytest tests/test_portfolio.py::test_function_name -v`
- Debug test: `pytest tests/test_file.py -v --pdb`

## Code Style Guidelines
- **Imports**: Group by: 1) standard library, 2) third-party, 3) local modules
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Documentation**: Docstrings with parameters and return descriptions for all functions/classes
- **Type Hints**: Required for all function parameters, return values and variables
- **Error Handling**: Use specific exception types, log all errors appropriately
- **Logging**: Use LoggingManager for consistent structured logging
- **Data Processing**: Prefer pandas vectorized operations over loops

## Project Organization
- `core/` - Core trading components (position, portfolio, margin, data management)
- `strategies/` - Trading strategy implementations
- `utils/` - Shared utility functions
- `config/` - Configuration files (YAML format)
- `tests/` - Unit and integration tests
- `scenario_results/` - Backtest results storage