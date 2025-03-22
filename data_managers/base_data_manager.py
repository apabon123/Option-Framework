"""
Base Data Manager Module

This module provides the abstract base class for all data managers,
defining the common interface and shared functionality.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from abc import ABC, abstractmethod


class BaseDataManager(ABC):
    """
    Abstract base class for data managers.
    
    This class defines the common interface and shared functionality
    for all data managers. Subclasses should implement the abstract
    methods to provide type-specific functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the base data manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or self._get_default_logger()
        self.data = None
        self.data_info = {}
        
    def _get_default_logger(self) -> logging.Logger:
        """Get a default logger if none is provided."""
        logger = logging.getLogger(self.__class__.__name__)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
        return logger
    
    @abstractmethod
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for data loading
            
        Returns:
            pd.DataFrame: Loaded data
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Preprocess data.
        
        Args:
            data: Data to preprocess
            **kwargs: Additional arguments for preprocessing
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        pass
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate if a file exists and is readable.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if file is valid
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                self.logger.error(f"File not found: {file_path}")
                return False
                
            if not path.is_file():
                self.logger.error(f"Not a file: {file_path}")
                return False
                
            if not os.access(path, os.R_OK):
                self.logger.error(f"File not readable: {file_path}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating file: {str(e)}")
            return False
    
    def get_dates(self, data: pd.DataFrame) -> List[date]:
        """
        Get unique dates from data.
        
        Args:
            data: DataFrame with datetime index or date column
            
        Returns:
            List[date]: List of unique dates
        """
        date_col = self._get_date_column(data)
        
        if date_col is None:
            self.logger.warning("No date column found in data")
            return []
            
        if date_col == 'index':
            dates = pd.Series([d.date() for d in data.index])
        else:
            if pd.api.types.is_datetime64_dtype(data[date_col]):
                dates = pd.Series([d.date() for d in data[date_col]])
            else:
                self.logger.warning(f"Column {date_col} is not datetime type")
                return []
                
        return sorted(dates.unique())
    
    def _get_date_column(self, data: pd.DataFrame) -> Optional[str]:
        """
        Find the date column in the data.
        
        Args:
            data: DataFrame to examine
            
        Returns:
            Optional[str]: Name of date column or 'index' if index is datetime
        """
        if isinstance(data.index, pd.DatetimeIndex):
            return 'index'
            
        # Look for datetime columns
        for col in data.columns:
            if pd.api.types.is_datetime64_dtype(data[col]):
                return col
                
        # Look for columns that might be date columns by name
        date_column_names = ['date', 'timestamp', 'time', 'datetime', 'expiration', 'quote_date']
        for name in date_column_names:
            for col in data.columns:
                if name.lower() in col.lower():
                    return col
                    
        return None
    
    def filter_by_date(
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
            
        if start_date is None and end_date is None:
            return data
            
        date_col = self._get_date_column(data)
        
        if date_col is None:
            self.logger.warning("Cannot filter by date: no date column found")
            return data
            
        # Convert dates to pandas Timestamp objects
        if start_date is not None:
            start_ts = pd.Timestamp(start_date).normalize()
        else:
            start_ts = pd.Timestamp.min
            
        if end_date is not None:
            end_ts = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        else:
            end_ts = pd.Timestamp.max
            
        # Apply filter based on date column
        if date_col == 'index':
            filtered = data[(data.index >= start_ts) & (data.index <= end_ts)].copy()
        else:
            # Convert column to datetime if needed
            if not pd.api.types.is_datetime64_dtype(data[date_col]):
                data = data.copy()
                data[date_col] = pd.to_datetime(data[date_col])
                
            filtered = data[(data[date_col] >= start_ts) & (data[date_col] <= end_ts)].copy()
            
        original_len = len(data)
        filtered_len = len(filtered)
        
        if filtered_len < original_len:
            self.logger.info(f"Filtered {original_len - filtered_len} rows ({100 * (original_len - filtered_len) / original_len:.1f}%) by date range")
            if filtered_len == 0:
                self.logger.warning("No data remains after date filtering")
                
        return filtered
        
    def log_data_summary(self, data: pd.DataFrame) -> None:
        """
        Log summary information about the data.
        
        Args:
            data: DataFrame to summarize
        """
        if data is None or data.empty:
            self.logger.warning("No data to summarize")
            return
            
        # Get shape
        rows, cols = data.shape
        self.logger.info(f"Data shape: {rows} rows x {cols} columns")
        
        # Get column types
        type_counts = data.dtypes.value_counts()
        for dtype, count in type_counts.items():
            self.logger.info(f"  {count} columns of type {dtype}")
            
        # Log date range
        date_col = self._get_date_column(data)
        if date_col:
            if date_col == 'index':
                min_date = data.index.min()
                max_date = data.index.max()
            else:
                min_date = data[date_col].min()
                max_date = data[date_col].max()
                
            self.logger.info(f"Date range: {min_date} to {max_date}")
            
            if hasattr(min_date, 'date') and hasattr(max_date, 'date'):
                days = (max_date.date() - min_date.date()).days + 1
                self.logger.info(f"Spanning {days} days")
    
    def ensure_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that the DataFrame has a DatetimeIndex.
        
        Args:
            data: DataFrame to process
            
        Returns:
            pd.DataFrame: DataFrame with DatetimeIndex
        """
        if data is None or data.empty:
            return data
            
        # If already has DatetimeIndex, return as is
        if isinstance(data.index, pd.DatetimeIndex):
            return data
            
        # Try to find a date column to use as index
        date_col = self._get_date_column(data)
        
        if date_col and date_col != 'index':
            # Convert column to datetime if needed
            if not pd.api.types.is_datetime64_dtype(data[date_col]):
                data = data.copy()
                data[date_col] = pd.to_datetime(data[date_col])
                
            # Set as index
            df = data.set_index(date_col)
            self.logger.info(f"Set index to column: {date_col}")
            return df
        else:
            self.logger.warning("No suitable date column found for index")
            return data 