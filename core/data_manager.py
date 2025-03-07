"""
Data Manager Module

This module provides tools for loading, preprocessing, and managing financial 
data, particularly options data, for use in trading strategies.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataManager:
    """
    Manages data loading and preprocessing for options and other financial data.
    
    This class handles loading data from files, preprocessing it for analysis and trading,
    and providing filtered views of the data based on date ranges, option characteristics, etc.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the DataManager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger('trading')
        self.config = config or {}
        self.data = None
        self.underlying_data = None  # Store underlying price data separately
        self.data_info = {}
    
    def load_option_data(
        self, 
        file_path: str, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load and preprocess option data from a CSV file with date filtering.
        
        Args:
            file_path: Path to the option data file
            start_date: Start date for filtering data (optional)
            end_date: End date for filtering data (optional)
            
        Returns:
            DataFrame: Processed option data
        """
        self.logger.info(f"Loading option data from {file_path}...")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Read the CSV file
            df = pd.read_csv(file_path, parse_dates=['DataDate', 'Expiration'])
            
            # Filter by date range if provided
            if start_date:
                df = df[df['DataDate'] >= start_date]
                self.logger.info(f"Filtered data from {start_date}")
                
            if end_date:
                df = df[df['DataDate'] <= end_date]
                self.logger.info(f"Filtered data to {end_date}")
            
            # Reset index
            df.reset_index(inplace=True, drop=True)
            
            # Calculate days to expiry if not present
            if 'DaysToExpiry' not in df.columns:
                df['DaysToExpiry'] = (df['Expiration'] - df['DataDate']).dt.days
                self.logger.info("Added DaysToExpiry column")
                
            # Calculate MidPrice if not present
            if 'MidPrice' not in df.columns:
                self.logger.info("Calculating MidPrice from Bid and Ask...")
                df = self.calculate_mid_prices(df)
            
            # Store and log data info
            self.data = df
            
            # Extract underlying data
            self._extract_underlying_data(df)
            
            self._log_data_info(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
            
    def load_from_file(self, file_path: str) -> bool:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Just delegate to load_option_data method
            self.load_option_data(file_path)
            return True
        except Exception as e:
            self.logger.error(f"Error loading data from file: {e}")
            return False
    
    def calculate_mid_prices(self, df: pd.DataFrame, normal_spread: float = 0.20) -> pd.DataFrame:
        """
        Calculate option mid prices with spread validation and filtering.
        
        Args:
            df: DataFrame of option data
            normal_spread: Maximum acceptable bid-ask spread percentage
            
        Returns:
            DataFrame: DataFrame with mid prices calculated and spread validation
        """
        self.logger.info("Calculating mid prices...")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if 'Last' column exists, if not, try to create it
        if 'Last' not in result_df.columns and 'Close' in result_df.columns:
            result_df['Last'] = result_df['Close']
        elif 'Last' not in result_df.columns:
            result_df['Last'] = 0.0
            
        # Calculate mid prices
        bid = result_df['Bid']
        ask = result_df['Ask']
        last = result_df['Last']
        
        # Create a mask for valid bid/ask
        valid_mask = (bid > 0) & (ask > 0)
        
        # Calculate mid price where valid
        mid = pd.Series(index=result_df.index)
        mid.loc[valid_mask] = (bid + ask) / 2
        
        # Calculate spread percentage where mid > 0
        spread_pct = pd.Series(index=result_df.index)
        spread_pct.loc[valid_mask & (mid > 0)] = (ask - bid) / mid
        
        # For non-valid bid/ask, use Last price if available
        last_price_mask = (~valid_mask) & (last > 0)
        mid.loc[last_price_mask] = last
        
        # Mark invalid bids
        result_df['MidPrice'] = mid
        result_df['SpreadPct'] = spread_pct
        result_df['ValidBidAsk'] = valid_mask
        result_df['UseLastPrice'] = last_price_mask
        result_df['AbnormalSpread'] = spread_pct > normal_spread
        
        count_valid = valid_mask.sum()
        count_last = last_price_mask.sum()
        count_abnormal = (result_df['AbnormalSpread'] == True).sum()
        
        self.logger.info(f"Mid price calculation: {count_valid} valid bid/ask pairs")
        self.logger.info(f"  {count_last} used last price")
        self.logger.info(f"  {count_abnormal} with abnormal spread (>{normal_spread:%})")
        
        return result_df
    
    def filter_chain(
        self, 
        df: Optional[pd.DataFrame] = None, 
        date: Optional[datetime] = None, 
        min_dte: int = 0, 
        max_dte: int = 365,
        min_delta: float = 0, 
        max_delta: float = 1.0,
        option_type: Optional[str] = None,
        min_bid: float = 0.05,
        only_standard: bool = True
    ) -> pd.DataFrame:
        """
        Filter option chain by various criteria.
        
        Args:
            df: DataFrame to filter (defaults to self.data)
            date: Date to filter for (defaults to first date)
            min_dte: Minimum days to expiry
            max_dte: Maximum days to expiry
            min_delta: Minimum absolute delta
            max_delta: Maximum absolute delta
            option_type: Option type to filter for (CALL, PUT, or None for both)
            min_bid: Minimum bid price
            only_standard: Only include standard expirations
            
        Returns:
            DataFrame: Filtered data
        """
        # Use provided DataFrame or default to stored data
        if df is None:
            if self.data is None:
                self.logger.error("No data loaded")
                return pd.DataFrame()
            df = self.data
        
        # Start with the original data
        filtered = df.copy()
        
        # Filter by date
        if date:
            filtered = filtered[filtered['DataDate'] == date]
            self.logger.info(f"Filtered to date: {date}")
        else:
            # Use the first date in the DataFrame
            first_date = filtered['DataDate'].min()
            filtered = filtered[filtered['DataDate'] == first_date]
            self.logger.info(f"Filtered to first date: {first_date}")
        
        # Filter by days to expiry
        filtered = filtered[(filtered['DaysToExpiry'] >= min_dte) & (filtered['DaysToExpiry'] <= max_dte)]
        
        # Filter by option type
        if option_type:
            filtered = filtered[filtered['Type'] == option_type.upper()]
            
        # Filter by bid price
        filtered = filtered[filtered['Bid'] >= min_bid]
        
        # Filter by delta
        if 'Delta' in filtered.columns:
            # Use absolute delta for comparison
            filtered['AbsDelta'] = filtered['Delta'].abs()
            filtered = filtered[(filtered['AbsDelta'] >= min_delta) & (filtered['AbsDelta'] <= max_delta)]
            
        # Filter for standard expirations if requested
        if only_standard and 'Standard' in filtered.columns:
            filtered = filtered[filtered['Standard'] == True]
            
        self.logger.info(f"Filtered to {len(filtered)} options")
        
        return filtered
    
    def get_data_for_date(self, date: datetime) -> pd.DataFrame:
        """
        Get all data for a specific date.
        
        Args:
            date: Date to get data for
            
        Returns:
            DataFrame: Data for the specified date
        """
        if self.data is None:
            self.logger.error("No data loaded")
            return pd.DataFrame()
            
        return self.data[self.data['DataDate'] == date]
    
    def get_option_data(
        self, 
        symbol: str, 
        date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get data for a specific option symbol, optionally filtered by date.
        
        Args:
            symbol: Option symbol to get data for
            date: Date to filter for (optional)
            
        Returns:
            DataFrame: Data for the specified option
        """
        if self.data is None:
            self.logger.error("No data loaded")
            return pd.DataFrame()
            
        filtered = self.data[self.data['OptionSymbol'] == symbol]
        
        if date:
            filtered = filtered[filtered['DataDate'] == date]
            
        return filtered
    
    def get_underlying_data(self, symbol: str) -> pd.DataFrame:
        """
        Get underlying price data for a specific symbol.
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            DataFrame: Underlying price data
        """
        if self.underlying_data is None:
            self.logger.error("No underlying data available")
            return pd.DataFrame()
            
        if symbol not in self.underlying_data:
            self.logger.error(f"No data for underlying {symbol}")
            return pd.DataFrame()
            
        return self.underlying_data[symbol]
    
    def _extract_underlying_data(self, df: pd.DataFrame) -> None:
        """
        Extract and store underlying price data from option data.
        
        Args:
            df: Option data DataFrame
        """
        if 'UnderlyingSymbol' not in df.columns or 'UnderlyingPrice' not in df.columns or 'DataDate' not in df.columns:
            self.logger.warning("Cannot extract underlying data - missing required columns")
            return
            
        self.logger.info("Extracting underlying price data...")
        
        # Group by underlying and date, taking the first price entry
        underlying_data = {}
        
        for symbol in df['UnderlyingSymbol'].unique():
            # Filter data for this symbol
            symbol_data = df[df['UnderlyingSymbol'] == symbol]
            
            # Group by date and get first price
            prices_by_date = symbol_data.groupby('DataDate')['UnderlyingPrice'].first()
            
            # Convert to DataFrame
            price_df = pd.DataFrame({
                'DataDate': prices_by_date.index,
                'Price': prices_by_date.values
            })
            
            underlying_data[symbol] = price_df
            
        self.underlying_data = underlying_data
    
    def _log_data_info(self, df: pd.DataFrame) -> None:
        """
        Log information about the data and store in data_info.
        
        Args:
            df: Data DataFrame
        """
        # Count unique values
        num_dates = df['DataDate'].nunique()
        start_date = df['DataDate'].min()
        end_date = df['DataDate'].max()
        
        # Get option types if available
        if 'Type' in df.columns:
            call_count = len(df[df['Type'] == 'CALL'])
            put_count = len(df[df['Type'] == 'PUT'])
        else:
            call_count = put_count = 0
            
        # Get underlying symbols if available
        if 'UnderlyingSymbol' in df.columns:
            unique_symbols = df['UnderlyingSymbol'].nunique()
            symbols = df['UnderlyingSymbol'].unique()
        else:
            unique_symbols = 0
            symbols = []
            
        # Store info
        self.data_info = {
            'rows': len(df),
            'dates': num_dates,
            'start_date': start_date,
            'end_date': end_date,
            'call_count': call_count,
            'put_count': put_count,
            'unique_symbols': unique_symbols,
            'symbols': symbols
        }
        
        # Log summary
        self.logger.info(f"Loaded {len(df)} rows for {num_dates} dates from {start_date} to {end_date}")
        if call_count + put_count > 0:
            self.logger.info(f"  {call_count} calls, {put_count} puts")
        if unique_symbols > 0:
            self.logger.info(f"  {unique_symbols} unique underlying symbols")
            
    def get_dates(self) -> List[datetime]:
        """
        Get a list of all dates in the data.
        
        Returns:
            list: List of dates
        """
        if self.data is None:
            return []
            
        return sorted(self.data['DataDate'].unique())
    
    def get_symbols(self) -> List[str]:
        """
        Get a list of all underlying symbols.
        
        Returns:
            list: List of symbols
        """
        if self.data is None or 'UnderlyingSymbol' not in self.data.columns:
            return []
            
        return sorted(self.data['UnderlyingSymbol'].unique())
    
    def generate_trading_dates(
        self, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None,
        skip_weekends: bool = True
    ) -> List[datetime]:
        """
        Generate a list of trading dates between start and end dates.
        
        Args:
            start_date: Start date (defaults to first date in data)
            end_date: End date (defaults to last date in data)
            skip_weekends: Whether to skip weekends
            
        Returns:
            list: List of dates
        """
        if self.data is None:
            self.logger.error("No data loaded")
            return []
            
        if not start_date:
            start_date = self.data['DataDate'].min()
            
        if not end_date:
            end_date = self.data['DataDate'].max()
            
        # Get a date range
        days = (end_date - start_date).days + 1
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Skip weekends if requested
        if skip_weekends:
            dates = [d for d in dates if d.weekday() < 5]  # 0-4 = Monday-Friday
            
        return dates