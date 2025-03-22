# Data Managers

This directory contains data management utilities for the Option-Framework, designed to handle various types of financial data with specialized preprocessing and analysis capabilities.

## Overview

The data managers are responsible for:
1. Loading financial data from files
2. Validating and preprocessing data
3. Handling data-specific concerns (like timezone issues for intraday data)
4. Providing filtered views of data for analysis
5. Converting between different data formats

## Transition from Legacy Data Manager

This package replaces the single `core/data_manager.py` with a more modular and extensible approach. The advantages include:

- **Specialized Processing**: Each data type has its own manager with custom logic
- **Better Timezone Support**: Proper handling of timezones for intraday data
- **Enhanced Validation**: More robust data validation and cleaning
- **More Features**: Each data type gets specialized filtering and analysis methods
- **Consistent Interface**: All data managers share common methods, making it easy to switch between them

## Data Manager Types

### BaseDataManager

The `BaseDataManager` is an abstract base class that defines the common interface and functionality for all data managers. It provides:

- Basic file validation
- Date-based filtering
- Data summary logging
- Common utility methods

### OptionDataManager

The `OptionDataManager` specializes in handling options data with features for:

- Loading and validating options chains
- Calculating missing Greeks
- Filtering by strike, expiration, moneyness, etc.
- Creating option chains for specific expiration dates
- Visualizing volatility surfaces

### IntradayDataManager

The `IntradayDataManager` handles minute-level price data with specialized features for:

- Timezone handling and normalization
- Market hours filtering
- OHLC data validation and correction
- Intraday metrics calculation (VWAP, minute-of-day)
- Session markers (pre-market, regular hours, post-market)

### DailyDataManager

The `DailyDataManager` manages daily OHLC data with features for:

- Calendar date handling
- Daily return calculation
- Moving average generation
- Gap analysis
- Volume profile analysis

## Utilities

The `utils.py` module provides helper functions for:

- Detecting data frequency (daily vs intraday)
- Converting between different data formats
- Standardizing column names and data types
- Automatic data preparation based on detected type

## Example Usage

```python
# Working with option data
from data_managers import OptionDataManager

option_dm = OptionDataManager()
options_data = option_dm.prepare_data_for_analysis(
    "data/spy_options.csv", 
    option_type="C",
    min_dte=5, 
    max_dte=30,
    min_moneyness=0.9,
    max_moneyness=1.1
)

# Working with intraday data
from data_managers import IntradayDataManager

intraday_dm = IntradayDataManager({"timezone": "America/New_York"})
minute_data = intraday_dm.prepare_data_for_analysis(
    "data/spy_minute.csv",
    days_to_analyze=20
)
market_hours_data = intraday_dm.filter_market_hours(minute_data)

# Working with daily data
from data_managers import DailyDataManager

daily_dm = DailyDataManager()
daily_data = daily_dm.prepare_data_for_analysis(
    "data/spy_daily.csv",
    lookback_days=252  # 1 year of trading days
)

# Converting data formats
from data_managers.utils import convert_to_appropriate_format

standardized_file = convert_to_appropriate_format(
    "raw_data/market_data.csv",
    verbose=True
)
```

## Extending with New Data Types

To add a new data type:

1. Create a new class that inherits from `BaseDataManager`
2. Implement the required methods (`load_data`, `preprocess_data`)
3. Add specialized methods for your data type
4. Update the `__init__.py` file to expose your new class
5. Consider adding utilities to convert to/from your format in `utils.py`

## Migration Guide

If you're currently using the legacy `core/data_manager.py`:

1. Replace `from core.data_manager import DataManager` with the appropriate import
   from the data_managers package based on your data type
2. Update method calls to match the new interface (most common methods have the
   same signatures, but specialized methods may differ)
3. Consider using the `prepare_data_for_analysis` method for the full processing
   pipeline instead of calling individual methods

Legacy code will continue to work, but new development should use these data managers. 