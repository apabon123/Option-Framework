# Data Directory

This directory is where you should place your market data files for use with the Option-Framework strategies.

## Required Data Files for Intraday Momentum Strategy

For the Intraday Momentum Strategy, you'll need intraday data (preferably minute bars) in CSV format with the following structure:

### spy_minute_data.csv Example Format

```
TimeStamp,Open,High,Low,Close,Volume,UpVolume,DownVolume
2024-01-02 09:30:00,469.23,469.56,469.15,469.45,1234567,987654,246913
2024-01-02 09:31:00,469.46,469.58,469.32,469.38,987654,456789,530865
...
```

### Required Columns

- `TimeStamp`: Datetime in YYYY-MM-DD HH:MM:SS format
- `Open`: Opening price for the bar
- `High`: Highest price for the bar
- `Low`: Lowest price for the bar
- `Close`: Closing price for the bar
- `Volume`: Total volume for the bar
- `UpVolume`: Volume on up ticks (optional)
- `DownVolume`: Volume on down ticks (optional)

### Data Sources

You can obtain intraday data from various sources:
- Interactive Brokers
- Alpha Vantage
- Yahoo Finance (limited historical intraday data)
- IEX Cloud
- Polygon.io

### Processing Data

If your data source doesn't provide data in the exact format required, you may need to preprocess it. Python's pandas library is excellent for this purpose:

```python
import pandas as pd

# Load data
df = pd.read_csv('raw_data.csv')

# Process data
df = df.rename(columns={'timestamp': 'TimeStamp', 'open': 'Open', ...})
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

# Sort by timestamp
df = df.sort_values('TimeStamp')

# Save in required format
df.to_csv('spy_minute_data.csv', index=False)
```

### Permissions

Please ensure that this directory and its files have the appropriate permissions for the framework to read them. 