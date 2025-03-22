"""
Daily Data Manager Module

This module provides data management for daily market data,
focusing on calendar date handling and daily metrics calculation.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path

from data_managers.base_data_manager import BaseDataManager


class DataLoadError(Exception):
    """Exception raised for errors during data loading."""
    pass


class DataValidationError(Exception):
    """Exception raised for data validation errors."""
    pass


class DailyDataManager(BaseDataManager):
    """
    Manages daily market data.
    
    This class extends BaseDataManager with functionality specifically for
    dealing with daily data, including calendar date handling and calculation
    of daily metrics.
    """
    
    # Class-level constants for validation
    REQUIRED_COLUMNS = {
        'Open', 'High', 'Low', 'Close'
    }
    
    # Optional columns that may be present
    OPTIONAL_COLUMNS = {
        'Volume', 'Adjusted Close', 'Adj Close', 'Dividend', 'Split'
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the daily data manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)
        self._data_cache = {}
        
    def load_data(
        self, 
        file_path: str, 
        date_col: str = 'Date',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load daily data from a file.
        
        Args:
            file_path: Path to the data file
            date_col: Name of the date column
            **kwargs: Additional arguments for data loading
            
        Returns:
            pd.DataFrame: Loaded data
        """
        if not self.validate_file(file_path):
            raise DataLoadError(f"Invalid file: {file_path}")
            
        try:
            # Load data from CSV
            self.logger.info(f"Loading daily data from {file_path}")
            df = pd.read_csv(
                file_path,
                parse_dates=[date_col] if date_col in kwargs.get('parse_dates', [date_col]) else [date_col],
                **{k: v for k, v in kwargs.items() if k != 'parse_dates'}
            )
            
            if df.empty:
                self.logger.warning(f"File {file_path} contains no data")
                return pd.DataFrame()
            
            # Make the date column the index if it's not already
            if date_col in df.columns:
                df = df.set_index(date_col)
            
            # Validate required columns
            missing_columns = self.REQUIRED_COLUMNS - set(df.columns)
            if missing_columns:
                self.logger.warning(f"Missing required columns: {missing_columns}")
                
            # Store data
            self.data = df
            
            # Log data info
            self.logger.info(f"Loaded {len(df):,} rows from {file_path}")
            self.log_data_summary(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading daily data: {e}", exc_info=True)
            raise DataLoadError(f"Failed to load daily data: {str(e)}")

    def preprocess_data(
        self, 
        data: pd.DataFrame, 
        handle_duplicates: bool = True,
        validate_ohlc: bool = True,
        fill_missing: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Preprocess daily data.
        
        Args:
            data: Data to preprocess
            handle_duplicates: Whether to handle duplicate dates
            validate_ohlc: Whether to validate OHLC data
            fill_missing: Whether to fill missing values
            **kwargs: Additional arguments for preprocessing
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if data is None or data.empty:
            self.logger.warning("No data to preprocess")
            return pd.DataFrame()
            
        try:
            df = data.copy()
            
            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.warning("Index is not DatetimeIndex, attempting to convert")
                df = self.ensure_datetime_index(df)
                
            # Handle duplicates if needed
            if handle_duplicates:
                duplicates = df.index.duplicated()
                if duplicates.any():
                    self.logger.warning(f"Removing {duplicates.sum()} duplicate dates")
                    df = df[~duplicates]
            
            # Sort by index if needed
            if not df.index.is_monotonic_increasing:
                self.logger.info("Sorting data by date")
                df = df.sort_index()
                
            # Validate OHLC data if requested
            if validate_ohlc:
                df = self._validate_ohlc_data(df)
                
            # Fill missing values if requested
            if fill_missing:
                self.logger.info("Filling missing values")
                df = self._fill_missing_values(df)
                
            # Standardize column names
            df = self._standardize_column_names(df)
                
            # Calculate basic metrics
            df = self._calculate_basic_metrics(df)
            
            self.logger.info(f"Preprocessing complete: {len(df):,} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}", exc_info=True)
            raise

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for consistency.
        
        Args:
            df: DataFrame to standardize
            
        Returns:
            pd.DataFrame: DataFrame with standardized column names
        """
        df = df.copy()
        
        # Standardize adjusted close column
        if 'Adj Close' in df.columns and 'Adjusted Close' not in df.columns:
            df['Adjusted Close'] = df['Adj Close']
        elif 'AdjClose' in df.columns and 'Adjusted Close' not in df.columns:
            df['Adjusted Close'] = df['AdjClose']
            
        # Ensure volume column is properly named
        if 'Vol' in df.columns and 'Volume' not in df.columns:
            df['Volume'] = df['Vol']
            
        return df

    def _validate_ohlc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix OHLC data inconsistencies.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            pd.DataFrame: Validated DataFrame
        """
        # Check if we have OHLC columns
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            self.logger.warning("Missing OHLC columns, skipping validation")
            return df
            
        df = df.copy()
        
        # Track invalid conditions
        invalid_rows = pd.DataFrame(index=df.index)
        invalid_rows['high_low'] = df['High'] < df['Low']
        invalid_rows['open_range'] = (df['Open'] > df['High']) | (df['Open'] < df['Low'])
        invalid_rows['close_range'] = (df['Close'] > df['High']) | (df['Close'] < df['Low'])
        
        # Log issues found
        total_issues = invalid_rows.sum().sum()
        if total_issues > 0:
            self.logger.warning(f"Found {total_issues} OHLC inconsistencies")
            
            # Fix High/Low inversions
            if invalid_rows['high_low'].any():
                mask = invalid_rows['high_low']
                self.logger.debug(f"Fixing {mask.sum()} High/Low inversions")
                df.loc[mask, ['High', 'Low']] = df.loc[mask, ['Low', 'High']].values
                
            # Fix Open outside range
            if invalid_rows['open_range'].any():
                mask = invalid_rows['open_range']
                self.logger.debug(f"Fixing {mask.sum()} Open values outside range")
                df.loc[mask, 'Open'] = df.loc[mask].apply(
                    lambda x: np.clip(x['Open'], x['Low'], x['High']), axis=1
                )
                
            # Fix Close outside range
            if invalid_rows['close_range'].any():
                mask = invalid_rows['close_range']
                self.logger.debug(f"Fixing {mask.sum()} Close values outside range")
                df.loc[mask, 'Close'] = df.loc[mask].apply(
                    lambda x: np.clip(x['Close'], x['Low'], x['High']), axis=1
                )
        
        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in DataFrame.
        
        Args:
            df: DataFrame with possibly missing values
            
        Returns:
            pd.DataFrame: DataFrame with filled values
        """
        df = df.copy()
        
        # Check if there are any missing values
        if not df.isnull().any().any():
            return df
            
        # Fill OHLC data using appropriate methods
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Forward fill close price
            df['Close'] = df['Close'].fillna(method='ffill')
            
            # Fill open price with close if missing
            df['Open'] = df['Open'].fillna(df['Close'])
            
            # Fill high and low with open/close
            df['High'] = df['High'].fillna(df[['Open', 'Close']].max(axis=1))
            df['Low'] = df['Low'].fillna(df[['Open', 'Close']].min(axis=1))
        
        # Fill volume data with zeros
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
                
        # Fill Adjusted Close if present
        if 'Adjusted Close' in df.columns:
            df['Adjusted Close'] = df['Adjusted Close'].fillna(df['Close'])
                
        return df

    def _calculate_basic_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic metrics from OHLC data.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            pd.DataFrame: DataFrame with additional metrics
        """
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            self.logger.warning("Missing OHLC columns, skipping metric calculation")
            return df
            
        df = df.copy()
        
        # Add date components
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
        df['trading_date'] = df.index.date  # For consistency with intraday data
        
        # Calculate returns
        df['daily_return'] = df['Close'].pct_change()
        df['daily_log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Calculate previous day's values
        df['prev_close'] = df['Close'].shift(1)
        df['prev_high'] = df['High'].shift(1)
        df['prev_low'] = df['Low'].shift(1)
        
        # Calculate true range
        df['true_range'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['prev_close']),
                abs(df['Low'] - df['prev_close'])
            )
        )
        
        # Calculate gap metrics
        df['gap_open'] = (df['Open'] - df['prev_close']) / df['prev_close']
        
        # Calculate intraday volatility
        df['intraday_volatility'] = (df['High'] - df['Low']) / df['Open']
        
        # Calculate volume metrics if volume is available
        if 'Volume' in df.columns:
            df['vol_change'] = df['Volume'].pct_change()
            df['volume_ma20'] = df['Volume'].rolling(window=20).mean()
            df['relative_volume'] = df['Volume'] / df['volume_ma20']
        
        # Calculate moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'ma{window}'] = df['Close'].rolling(window=window).mean()
            
        # Calculate price relative to moving averages
        for window in [5, 10, 20, 50, 200]:
            ma_col = f'ma{window}'
            if ma_col in df.columns:
                df[f'price_rel_ma{window}'] = df['Close'] / df[ma_col] - 1
        
        return df

    def filter_by_date_range(
        self,
        data: pd.DataFrame,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None
    ) -> pd.DataFrame:
        """
        Filter data by date range.
        
        Args:
            data: Data to filter
            start_date: Start date for filter
            end_date: End date for filter
            
        Returns:
            pd.DataFrame: Filtered data
        """
        if data is None or data.empty:
            return data
            
        return self.filter_by_date(data, start_date, end_date)
        
    def prepare_data_for_analysis(
        self, 
        file_path: str, 
        lookback_days: int = 252,  # Default to ~1 year of trading days
        **kwargs
    ) -> pd.DataFrame:
        """
        Prepare data for analysis with full processing pipeline.
        
        Args:
            file_path: Path to data file
            lookback_days: Number of days to include in analysis
            **kwargs: Additional arguments for processing
            
        Returns:
            pd.DataFrame: Fully processed data ready for analysis
        """
        try:
            # Create cache key
            cache_key = f"{file_path}_{lookback_days}"
            
            # Check if we have cached data
            if cache_key in self._data_cache:
                self.logger.info("Using cached data")
                return self._data_cache[cache_key].copy()
                
            # Load data
            df = self.load_data(file_path, **kwargs)
            
            # Preprocess data
            df = self.preprocess_data(df, **kwargs)
            
            # Filter to lookback period if specified
            if lookback_days > 0:
                end_date = df.index.max()
                if isinstance(end_date, pd.Timestamp):
                    start_date = end_date - pd.Timedelta(days=lookback_days)
                    df = self.filter_by_date_range(df, start_date, end_date)
                    self.logger.info(f"Filtered to date range: {start_date.date()} to {end_date.date()}")
            
            # Validate that we have enough data
            if len(df) < 20:  # Minimum data required for most analyses
                self.logger.warning(f"Limited data available: {len(df)} days")
            
            # Cache the results
            self._data_cache[cache_key] = df.copy()
            
            # Log data preparation summary
            self._log_data_preparation_summary(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}", exc_info=True)
            raise

    def _log_data_preparation_summary(self, df: pd.DataFrame) -> None:
        """
        Log summary of prepared data.
        
        Args:
            df: Prepared DataFrame
        """
        if df is None or df.empty:
            self.logger.warning("No data to summarize")
            return
            
        try:
            self.logger.info("\nPrepared Data Summary:")
            self.logger.info(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
            self.logger.info(f"Total trading days: {len(df):,}")
            
            # Calculate basic statistics if possible
            if 'daily_return' in df.columns:
                returns = df['daily_return'].dropna()
                self.logger.info(f"Average daily return: {returns.mean():.4%}")
                self.logger.info(f"Daily return volatility: {returns.std():.4%}")
                self.logger.info(f"Minimum daily return: {returns.min():.4%}")
                self.logger.info(f"Maximum daily return: {returns.max():.4%}")
                
            # Log volume info if available
            if 'Volume' in df.columns:
                self.logger.info(f"Average daily volume: {df['Volume'].mean():,.0f}")
                
            # Count years and months in the data
            years = df.index.year.unique()
            months = len(df.index.to_period('M').unique())
            self.logger.info(f"Years covered: {len(years)} ({min(years)} to {max(years)})")
            self.logger.info(f"Months covered: {months}")
            
        except Exception as e:
            self.logger.error(f"Error logging data summary: {e}", exc_info=True) 