"""
Intraday Data Manager Module

This module provides data management for intraday (minute-level) market data,
with particular focus on timezone handling and market hours filtering.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time, date
from pathlib import Path
import pytz

from data_managers.base_data_manager import BaseDataManager


class DataLoadError(Exception):
    """Exception raised for errors during data loading."""
    pass


class DataValidationError(Exception):
    """Exception raised for data validation errors."""
    pass


class TimeZoneError(Exception):
    """Exception raised for timezone-related errors."""
    pass


class IntradayDataManager(BaseDataManager):
    """
    Manages intraday (minute-level) market data with timezone support.
    
    This class extends BaseDataManager with functionality specifically for
    dealing with intraday data, including timezone handling, market hours
    filtering, and calculation of intraday metrics.
    """
    
    # Class-level constants for validation
    REQUIRED_COLUMNS = {
        'Open', 'High', 'Low', 'Close'
    }
    
    # Optional columns that may be present
    OPTIONAL_COLUMNS = {
        'Volume', 'UpVolume', 'DownVolume'
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the intraday data manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.timezone = self._get_config_timezone()
        self._data_cache = {}
        
    def _get_config_timezone(self) -> str:
        """Get timezone from configuration or use default."""
        if self.config and 'timezone' in self.config:
            return self.config['timezone']
        else:
            self.logger.warning("No timezone specified in configuration, using 'UTC'")
            return 'UTC'

    def load_data(
        self, 
        file_path: str, 
        timezone: Optional[str] = None,
        date_col: str = 'TimeStamp',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load intraday data from a file.
        
        Args:
            file_path: Path to the data file
            timezone: Timezone of the data (default from config or UTC)
            date_col: Name of the date/time column
            **kwargs: Additional arguments for data loading
            
        Returns:
            pd.DataFrame: Loaded data with proper timezone handling
        """
        if not self.validate_file(file_path):
            raise DataLoadError(f"Invalid file: {file_path}")
            
        try:
            # Get timezone
            tz = timezone or self.timezone
            
            # Load data from CSV
            self.logger.info(f"Loading intraday data from {file_path}")
            df = pd.read_csv(
                file_path,
                parse_dates=[date_col] if date_col in kwargs.get('parse_dates', [date_col]) else [date_col],
                **{k: v for k, v in kwargs.items() if k != 'parse_dates'}
            )
            
            if df.empty:
                self.logger.warning(f"File {file_path} contains no data")
                return pd.DataFrame()
            
            # Make the timestamp column the index if it's not already
            if date_col in df.columns:
                df = df.set_index(date_col)
            
            # Validate required columns
            missing_columns = self.REQUIRED_COLUMNS - set(df.columns)
            if missing_columns:
                self.logger.warning(f"Missing required columns: {missing_columns}")
                
            # Apply timezone handling
            df = self._apply_timezone(df, tz)
            
            # Store data
            self.data = df
            
            # Log data info
            self.logger.info(f"Loaded {len(df):,} rows from {file_path}")
            self.log_data_summary(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading intraday data: {e}", exc_info=True)
            raise DataLoadError(f"Failed to load intraday data: {str(e)}")

    def preprocess_data(
        self, 
        data: pd.DataFrame, 
        resample: Optional[str] = None,
        handle_duplicates: bool = True,
        validate_ohlc: bool = True,
        fill_missing: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Preprocess intraday data.
        
        Args:
            data: Data to preprocess
            resample: Resample frequency (e.g., '1T' for 1 minute)
            handle_duplicates: Whether to handle duplicate timestamps
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
                    self.logger.warning(f"Removing {duplicates.sum()} duplicate timestamps")
                    df = df[~duplicates]
            
            # Sort by index if needed
            if not df.index.is_monotonic_increasing:
                self.logger.info("Sorting data by timestamp")
                df = df.sort_index()
                
            # Validate OHLC data if requested
            if validate_ohlc:
                df = self._validate_ohlc_data(df)
                
            # Resample if requested
            if resample:
                self.logger.info(f"Resampling data to {resample} frequency")
                df = self._resample_data(df, resample)
                
            # Fill missing values if requested
            if fill_missing:
                self.logger.info("Filling missing values")
                df = self._fill_missing_values(df)
                
            # Calculate basic metrics
            df = self._calculate_basic_metrics(df)
            
            self.logger.info(f"Preprocessing complete: {len(df):,} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}", exc_info=True)
            raise

    def filter_market_hours(
        self, 
        data: pd.DataFrame, 
        market_open: Optional[time] = None, 
        market_close: Optional[time] = None
    ) -> pd.DataFrame:
        """
        Filter data to include only market hours.
        
        Args:
            data: Data to filter
            market_open: Market open time (default from config)
            market_close: Market close time (default from config)
            
        Returns:
            pd.DataFrame: Filtered data
        """
        if data is None or data.empty:
            return data
            
        # Get market hours from config if not provided
        if market_open is None or market_close is None:
            market_hours = self._get_market_hours()
            market_open = market_open or market_hours.get('market_open')
            market_close = market_close or market_hours.get('market_close')
            
        if market_open is None or market_close is None:
            self.logger.warning("Market hours not specified, cannot filter")
            return data
            
        # Create mask for market hours
        market_hours_mask = (
            (data.index.time >= market_open) &
            (data.index.time <= market_close)
        )
        
        # Apply filter
        filtered_data = data[market_hours_mask].copy()
        
        self.logger.info(f"Filtered to market hours ({market_open} - {market_close}): {len(filtered_data):,} rows")
        return filtered_data

    def _get_market_hours(self) -> Dict[str, time]:
        """Get market hours from configuration."""
        market_hours = {}
        
        if not self.config:
            return market_hours
            
        # First, check if contract_specs has market hours
        if 'contract_specs' in self.config and 'market_open' in self.config.get('contract_specs', {}):
            specs = self.config['contract_specs']
            market_hours['market_open'] = self._parse_time(specs.get('market_open', '09:30'))
            market_hours['market_close'] = self._parse_time(specs.get('market_close', '16:00'))
            market_hours['last_entry'] = self._parse_time(specs.get('last_entry', '15:30'))
        # Otherwise, check for market_hours directly in config
        elif 'market_hours' in self.config:
            hours = self.config['market_hours']
            market_hours['market_open'] = self._parse_time(hours.get('open', '09:30'))
            market_hours['market_close'] = self._parse_time(hours.get('close', '16:00'))
            market_hours['last_entry'] = self._parse_time(hours.get('last_entry', '15:30'))
        else:
            # Use defaults for US equity market
            self.logger.warning("No market hours specified in config, using defaults (9:30-16:00 ET)")
            market_hours['market_open'] = time(9, 30)
            market_hours['market_close'] = time(16, 0)
            market_hours['last_entry'] = time(15, 30)
            
        return market_hours

    @staticmethod
    def _parse_time(time_str: Union[str, time]) -> time:
        """Parse time string to time object."""
        if isinstance(time_str, time):
            return time_str
        elif isinstance(time_str, str):
            try:
                return datetime.strptime(time_str, '%H:%M').time()
            except ValueError:
                return datetime.strptime(time_str, '%H:%M:%S').time()
        else:
            raise ValueError(f"Invalid time format: {time_str}")

    def _apply_timezone(self, df: pd.DataFrame, timezone: str) -> pd.DataFrame:
        """
        Apply timezone handling to DataFrame index.
        
        Args:
            df: DataFrame with DatetimeIndex
            timezone: Target timezone
            
        Returns:
            pd.DataFrame: DataFrame with timezone-aware index
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("Index is not DatetimeIndex, timezone handling may not work correctly")
            return df
            
        try:
            # If index has no timezone, localize it
            if df.index.tz is None:
                self.logger.info(f"Localizing index to timezone: {timezone}")
                df.index = df.index.tz_localize(timezone)
            # If index has a different timezone, convert it
            elif str(df.index.tz) != timezone:
                self.logger.info(f"Converting index from {df.index.tz} to {timezone}")
                df.index = df.index.tz_convert(timezone)
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error applying timezone: {e}", exc_info=True)
            raise TimeZoneError(f"Failed to apply timezone: {str(e)}")

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

    def _resample_data(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resample data to a new frequency.
        
        Args:
            df: DataFrame to resample
            freq: Frequency string (e.g., '1T' for 1 minute)
            
        Returns:
            pd.DataFrame: Resampled DataFrame
        """
        # Check if we have OHLC columns
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            self.logger.warning("Missing OHLC columns, basic resampling will be performed")
            return df.resample(freq).last()
            
        # Define aggregation functions for OHLC data
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }
        
        # Add volume aggregation if available
        if 'Volume' in df.columns:
            agg_dict['Volume'] = 'sum'
        if 'UpVolume' in df.columns:
            agg_dict['UpVolume'] = 'sum'
        if 'DownVolume' in df.columns:
            agg_dict['DownVolume'] = 'sum'
            
        # Resample
        resampled = df.resample(freq).agg(agg_dict)
        
        # Drop rows where all OHLC values are NaN
        resampled = resampled.dropna(subset=['Open', 'High', 'Low', 'Close'], how='all')
        
        return resampled

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in DataFrame.
        
        Args:
            df: DataFrame with possibly missing values
            
        Returns:
            pd.DataFrame: DataFrame with filled values
        """
        df = df.copy()
        
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
        for col in ['Volume', 'UpVolume', 'DownVolume']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                
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
        
        # Add minute of day
        df['minute_of_day'] = df.index.hour * 60 + df.index.minute
        
        # Add trading date column
        df['trading_date'] = df.index.date
        
        # Get market hours
        market_hours = self._get_market_hours()
        market_open = market_hours.get('market_open')
        market_close = market_hours.get('market_close')
        
        if market_open and market_close:
            # Create market filters
            market_open_mask = (df.index.time == market_open)
            market_close_mask = (df.index.time == market_close)
            market_hours_mask = (
                (df.index.time >= market_open) &
                (df.index.time <= market_close)
            )
            
            # Add market session flags
            df['trading_hour'] = market_hours_mask
            df['pre_market'] = df.index.time < market_open
            df['post_market'] = df.index.time > market_close
            
            # Get market open prices for each day
            opens = df[market_open_mask].groupby('trading_date')['Open'].first()
            df['day_open'] = df['trading_date'].map(opens)
            
            # Get market close prices for each day
            closes = df[market_close_mask].groupby('trading_date')['Close'].last()
            
            # Calculate previous close
            df['prev_close'] = df['trading_date'].map(closes.shift(1))
            
            # Calculate move from open (only during market hours)
            df['move_from_open'] = np.nan
            if market_hours_mask.any():
                df.loc[market_hours_mask, 'move_from_open'] = (
                    abs(df.loc[market_hours_mask, 'Close'] - df.loc[market_hours_mask, 'day_open']) /
                    df.loc[market_hours_mask, 'day_open']
                )
                
            # Calculate VWAP if volume available
            if any(col in df.columns for col in ['Volume', 'UpVolume', 'DownVolume']):
                # Determine which volume column to use
                if 'Volume' in df.columns:
                    volume_col = 'Volume'
                elif 'UpVolume' in df.columns and 'DownVolume' in df.columns:
                    df['Volume'] = df['UpVolume'] + df['DownVolume']
                    volume_col = 'Volume'
                else:
                    volume_col = None
                    
                # Calculate VWAP
                if volume_col and market_hours_mask.any():
                    df['vwap'] = np.nan
                    market_data = df[market_hours_mask].copy()
                    if not market_data.empty:
                        volume = market_data[volume_col]
                        price_volume = market_data['Close'] * volume
                        df.loc[market_hours_mask, 'vwap'] = (
                            price_volume.groupby(market_data['trading_date']).cumsum() /
                            volume.groupby(market_data['trading_date']).cumsum()
                        )
            
        return df

    def prepare_data_for_analysis(
        self, 
        file_path: str, 
        days_to_analyze: int = 20,
        lookback_buffer: int = 10,
        timezone: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Prepare data for analysis with full processing pipeline.
        
        Args:
            file_path: Path to data file
            days_to_analyze: Number of days to include in analysis
            lookback_buffer: Additional days to include for lookback calculations
            timezone: Target timezone (default from config)
            **kwargs: Additional arguments for processing
            
        Returns:
            pd.DataFrame: Fully processed data ready for analysis
        """
        try:
            # Create cache key
            cache_key = f"{file_path}_{days_to_analyze}_{lookback_buffer}_{timezone or self.timezone}"
            
            # Check if we have cached data
            if cache_key in self._data_cache:
                self.logger.info("Using cached data")
                return self._data_cache[cache_key].copy()
                
            # Load data
            df = self.load_data(file_path, timezone=timezone, **kwargs)
            
            # Preprocess data
            df = self.preprocess_data(df, **kwargs)
            
            # Filter to analysis period
            df = self._filter_analysis_period(df, days_to_analyze, lookback_buffer)
            
            # Add trading markers
            df = self._add_trading_markers(df)
            
            # Validate prepared data
            self._validate_prepared_data(df)
            
            # Cache the results
            self._data_cache[cache_key] = df.copy()
            
            # Log data preparation summary
            self._log_data_preparation_summary(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}", exc_info=True)
            raise

    def _filter_analysis_period(self, df: pd.DataFrame, days_to_analyze: int,
                              lookback_buffer: int) -> pd.DataFrame:
        """
        Filter data to analysis period.
        
        Args:
            df: DataFrame to filter
            days_to_analyze: Number of days to include in analysis
            lookback_buffer: Additional days to include for lookback calculations
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if df is None or df.empty:
            return df
            
        try:
            # Get unique trading dates
            trading_dates = sorted(df.index.date.unique())
            
            # Calculate start indices
            start_idx = max(len(trading_dates) - (days_to_analyze + lookback_buffer), 0)
            trading_start_idx = min(start_idx + lookback_buffer, len(trading_dates) - 1)
            
            # Get start dates
            start_date = trading_dates[start_idx]
            trading_start = trading_dates[trading_start_idx]
            
            # Filter data
            filtered_df = df[df.index.date >= start_date].copy()
            filtered_df['is_training_period'] = (
                (filtered_df.index.date >= start_date) & 
                (filtered_df.index.date < trading_start)
            )
            filtered_df['is_trading_period'] = filtered_df.index.date >= trading_start
            
            self.logger.info(f"Analysis period: {start_date} to {df.index.date.max()}")
            self.logger.info(f"Training period: {start_date} to {trading_start}")
            self.logger.info(f"Trading period: {trading_start} to {df.index.date.max()}")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Error filtering analysis period: {e}", exc_info=True)
            raise

    def _add_trading_markers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trading session markers to DataFrame.
        
        Args:
            df: DataFrame to augment
            
        Returns:
            pd.DataFrame: DataFrame with trading markers
        """
        if df is None or df.empty:
            return df
            
        try:
            df = df.copy()
            
            # Get market hours
            market_hours = self._get_market_hours()
            market_open = market_hours.get('market_open')
            market_close = market_hours.get('market_close')
            last_entry = market_hours.get('last_entry')
            
            if market_open and market_close:
                # Add trading session markers if not already present
                if 'trading_hour' not in df.columns:
                    df['trading_hour'] = (
                        (df.index.time >= market_open) &
                        (df.index.time <= market_close)
                    )
                    
                if 'pre_market' not in df.columns:
                    df['pre_market'] = df.index.time < market_open
                    
                if 'post_market' not in df.columns:
                    df['post_market'] = df.index.time > market_close
                    
                if last_entry and 'can_enter' not in df.columns:
                    df['can_enter'] = (
                        (df.index.time >= market_open) &
                        (df.index.time <= last_entry)
                    )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding trading markers: {e}", exc_info=True)
            raise

    def _validate_prepared_data(self, df: pd.DataFrame) -> None:
        """
        Validate the prepared dataset.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            DataValidationError: If validation fails
        """
        if df is None or df.empty:
            raise DataValidationError("Dataset is empty")
            
        try:
            # Define required columns for analysis
            required_columns = {
                'Open', 'High', 'Low', 'Close',
                'trading_date', 'minute_of_day'
            }
            
            # Check for missing columns
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise DataValidationError(f"Missing required columns: {missing_columns}")
                
            # Validate data quality
            null_counts = df[list(required_columns)].isnull().sum()
            if null_counts.any():
                self.logger.warning("Found null values:")
                for col, count in null_counts[null_counts > 0].items():
                    self.logger.warning(f"  {col}: {count} null values")
                    
            # Validate numerical ranges
            if (df['minute_of_day'] < 0).any() or (df['minute_of_day'] >= 1440).any():
                raise DataValidationError("Invalid minute_of_day values detected")
                
            # Validate market hours if present
            if 'trading_hour' in df.columns and not df['trading_hour'].any():
                self.logger.warning("No data points during trading hours")
                
            self.logger.info("Data validation completed successfully")
            
        except Exception as e:
            if not isinstance(e, DataValidationError):
                e = DataValidationError(f"Data validation failed: {str(e)}")
            raise e

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
            self.logger.info(f"Total rows: {len(df):,}")
            
            # Log trading periods
            if 'is_training_period' in df.columns:
                self.logger.info(f"Training period rows: {df['is_training_period'].sum():,}")
            if 'is_trading_period' in df.columns:
                self.logger.info(f"Trading period rows: {df['is_trading_period'].sum():,}")
                
            # Log market hours
            if 'trading_hour' in df.columns:
                self.logger.info(f"Trading hours rows: {df['trading_hour'].sum():,}")
            if 'pre_market' in df.columns:
                self.logger.info(f"Pre-market rows: {df['pre_market'].sum():,}")
            if 'post_market' in df.columns:
                self.logger.info(f"Post-market rows: {df['post_market'].sum():,}")
                
            # Log trading dates
            unique_dates = len(df.index.date.unique())
            self.logger.info(f"Trading days: {unique_dates}")
            
        except Exception as e:
            self.logger.error(f"Error logging data summary: {e}", exc_info=True) 