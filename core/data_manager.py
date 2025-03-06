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
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the DataManager.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger('trading')
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
        
        # Create MidPrice column with default from Last
        result_df['MidPrice'] = last
        
        # Update where spread is normal
        normal_spread_mask = valid_mask & (spread_pct <= normal_spread)
        result_df.loc[normal_spread_mask, 'MidPrice'] = mid.loc[normal_spread_mask]
        
        # Add spread percentage as a column for reference
        result_df['SpreadPct'] = spread_pct
        
        # Log statistics
        valid_prices = result_df[result_df['MidPrice'] > 0]
        self.logger.info(f"Mid prices calculated: {len(valid_prices)} valid prices")
        
        if not valid_prices.empty:
            self.logger.info(f"Price range: ${valid_prices['MidPrice'].min():.2f} to ${valid_prices['MidPrice'].max():.2f}")
            avg_spread = (valid_prices['Ask'] - valid_prices['Bid']).mean()
            self.logger.info(f"Average bid-ask spread: ${avg_spread:.4f}")
            
        return result_df


    def prepare_option_data(self, df, current_date, normal_spread):
        """
        Prepare option data with mid prices and days to expiry calculations.
        Also adds underlying data as separate rows for hedging.

        Args:
            df: DataFrame of option data
            current_date: Current simulation date
            normal_spread: Maximum acceptable bid-ask spread percentage

        Returns:
            DataFrame: Processed DataFrame with MidPrice, DaysToExpiry and underlying rows
        """
        self.logger.info(f"[Engine] Preparing option data for {current_date}")

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Calculate mid prices
        result_df = self.calculate_mid_prices(result_df, normal_spread)

        # Calculate days to expiry if needed
        if 'DaysToExpiry' not in result_df.columns:
            # Convert current_date to datetime if it's a string
            if isinstance(current_date, str):
                current_dt = pd.to_datetime(current_date)
            else:
                current_dt = current_date

            # Convert Expiration to datetime if needed
            if isinstance(result_df['Expiration'].iloc[0], str):
                result_df['ExpirationDate'] = pd.to_datetime(result_df['Expiration'])
            else:
                result_df['ExpirationDate'] = result_df['Expiration']

            # Calculate days to expiry
            result_df['DaysToExpiry'] = (result_df['ExpirationDate'] - current_dt).dt.days
            
        # Get underlying data and append it to our result
        underlying_data = self.get_underlying_data(current_date)
        
        if not underlying_data.empty:
            # Create DataFrame for underlying securities
            underlying_rows = []
            
            for _, row in underlying_data.iterrows():
                # Create a row for each underlying security
                underlying_row = {
                    'DataDate': row['DataDate'],
                    'Symbol': row['Symbol'],            # Use the pure symbol as the key
                    'OptionSymbol': row['Symbol'],      # For compatibility with option code
                    'UnderlyingSymbol': row['Symbol'],  # Same as symbol for the underlying
                    'UnderlyingPrice': row['UnderlyingPrice'],
                    'MidPrice': row['UnderlyingPrice'], # Use underlying price as mid price
                    'Type': 'underlying',               # Mark as underlying type
                    'Strike': row['UnderlyingPrice'],   # Set strike equal to price for delta calc
                    'DaysToExpiry': 0,                  # No expiry for underlying
                    'Delta': 1.0,                       # Delta is always 1.0 for underlying
                    'Gamma': 0.0,                       # No gamma for underlying
                    'Theta': 0.0,                       # No theta for underlying
                    'Vega': 0.0                         # No vega for underlying
                }
                underlying_rows.append(underlying_row)
                
            # Create DataFrame from the underlying rows
            if underlying_rows:
                # Only add if we have rows
                underlying_df = pd.DataFrame(underlying_rows)
                
                # Combine with the option data
                result_df = pd.concat([result_df, underlying_df], ignore_index=True)
                
                self.logger.info(f"[Engine] Added {len(underlying_rows)} underlying securities to daily data")
        
        self.logger.info(f"[Engine] Prepared data: {len(result_df)} rows with MidPrice and DaysToExpiry")

        return result_df

    def get_daily_data(self, date: datetime) -> pd.DataFrame:
        """
        Get data for a specific date from the loaded dataset.
        
        Args:
            date: Date to filter data for
            
        Returns:
            DataFrame: Option data for the specified date
        """
        if self.data is None:
            self.logger.warning("No data loaded. Call load_option_data first.")
            return pd.DataFrame()
            
        daily_data = self.data[self.data['DataDate'] == date].copy()
        self.logger.debug(f"Retrieved {len(daily_data)} records for {date}")
        return daily_data
        
    def get_underlying_data(self, date: datetime) -> pd.DataFrame:
        """
        Get underlying price data for a specific date.
        
        Args:
            date: Date to filter data for
            
        Returns:
            DataFrame: Underlying price data for the specified date
        """
        if self.underlying_data is None:
            self.logger.warning("No underlying data available.")
            return pd.DataFrame()
            
        daily_underlying = self.underlying_data[self.underlying_data['DataDate'] == date].copy()
        self.logger.debug(f"Retrieved {len(daily_underlying)} underlying records for {date}")
        return daily_underlying
    
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """
        Get the date range of the loaded data.
        
        Returns:
            tuple: (min_date, max_date)
        """
        if self.data is None or self.data.empty:
            return None, None
            
        min_date = self.data['DataDate'].min()
        max_date = self.data['DataDate'].max()
        
        return min_date, max_date
    
    def get_dates_list(self) -> List[datetime]:
        """
        Get a sorted list of all unique dates in the dataset.
        
        Returns:
            list: List of unique dates
        """
        if self.data is None or self.data.empty:
            return []
            
        return sorted(self.data['DataDate'].unique())
    
    def get_market_data_by_symbol(self, daily_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create a dictionary of market data by symbol (option or underlying).
        
        Args:
            daily_data: DataFrame of daily option data
            
        Returns:
            dict: Dictionary of {symbol: data_row}
        """
        market_data = {}
        
        # Process each row
        for _, row in daily_data.iterrows():
            # For option symbols, use OptionSymbol as key
            if 'OptionSymbol' in row:
                market_data[row['OptionSymbol']] = row
                
            # For underlying symbols (when present as separate entries)
            if 'Type' in row and row['Type'] == 'underlying' and 'Symbol' in row:
                # Use underlying symbol as key
                market_data[row['Symbol']] = row
                
        return market_data
    
    def filter_by_criteria(
        self, 
        df: pd.DataFrame, 
        option_type: Optional[str] = None,
        min_dte: Optional[int] = None,
        max_dte: Optional[int] = None,
        min_delta: Optional[float] = None,
        max_delta: Optional[float] = None,
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter options data by multiple criteria.
        
        Args:
            df: DataFrame of option data
            option_type: 'call', 'put', or None for all
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            min_delta: Minimum delta value
            max_delta: Maximum delta value
            min_strike: Minimum strike price
            max_strike: Maximum strike price
            
        Returns:
            DataFrame: Filtered option data
        """
        filtered_df = df.copy()
        
        # Apply filters if provided
        if option_type:
            filtered_df = filtered_df[filtered_df['Type'].str.lower() == option_type.lower()]
            
        if min_dte is not None:
            filtered_df = filtered_df[filtered_df['DaysToExpiry'] >= min_dte]
            
        if max_dte is not None:
            filtered_df = filtered_df[filtered_df['DaysToExpiry'] <= max_dte]
            
        if min_delta is not None:
            filtered_df = filtered_df[filtered_df['Delta'] >= min_delta]
            
        if max_delta is not None:
            filtered_df = filtered_df[filtered_df['Delta'] <= max_delta]
            
        if min_strike is not None:
            filtered_df = filtered_df[filtered_df['Strike'] >= min_strike]
            
        if max_strike is not None:
            filtered_df = filtered_df[filtered_df['Strike'] <= max_strike]
            
        self.logger.debug(f"Filtered from {len(df)} to {len(filtered_df)} options")
        return filtered_df
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality and return statistics about issues found.
        
        Args:
            df: DataFrame to check
            
        Returns:
            dict: Data quality metrics
        """
        stats = {
            'total_rows': len(df),
            'missing_values': {},
            'zero_values': {},
            'negative_values': {},
            'outliers': {}
        }
        
        # Check for missing values
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                stats['missing_values'][col] = missing
        
        # Check for zero values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['Strike', 'UnderlyingPrice', 'MidPrice', 'Bid', 'Ask']:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    stats['zero_values'][col] = zero_count
        
        # Check for negative values in price columns
        price_cols = ['Strike', 'UnderlyingPrice', 'MidPrice', 'Bid', 'Ask']
        for col in price_cols:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    stats['negative_values'][col] = neg_count
        
        # Add other checks as needed
        
        return stats
    
    def _extract_underlying_data(self, df: pd.DataFrame) -> None:
        """
        Extract and store underlying price data from option data.
        
        This function creates a DataFrame containing underlying price data
        for each date and underlying symbol, which can be used for hedging
        and other calculations.
        
        Args:
            df: Option data DataFrame
        """
        # Check if necessary columns exist
        if 'UnderlyingSymbol' not in df.columns or 'UnderlyingPrice' not in df.columns:
            self.logger.warning("Cannot extract underlying data: missing required columns")
            return
            
        try:
            # Group by date and underlying symbol, taking the first value for each group
            # (underlying prices should be the same for all options on same underlying and date)
            underlying_data = df.groupby(['DataDate', 'UnderlyingSymbol']).agg({
                'UnderlyingPrice': 'first'
            }).reset_index()
            
            # Create a symbol column for easy lookup
            underlying_data['Symbol'] = underlying_data['UnderlyingSymbol']
            
            # Store the underlying data
            self.underlying_data = underlying_data
            
            self.logger.info(f"Extracted underlying price data for {len(underlying_data)} date-symbol pairs")
            
            # Add some statistics
            unique_underlyings = underlying_data['UnderlyingSymbol'].nunique()
            unique_dates = underlying_data['DataDate'].nunique()
            self.logger.info(f"Underlying data covers {unique_underlyings} symbols across {unique_dates} dates")
            
        except Exception as e:
            self.logger.error(f"Error extracting underlying data: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _log_data_info(self, df: pd.DataFrame) -> None:
        """
        Log information about the loaded data.
        
        Args:
            df: DataFrame to analyze
        """
        # Calculate and store data info
        date_range = f"{df['DataDate'].min()} to {df['DataDate'].max()}"
        
        # Get unique values counts
        unique_dates = df['DataDate'].nunique()
        unique_symbols = df['OptionSymbol'].nunique() if 'OptionSymbol' in df.columns else 0
        unique_underlyings = df['UnderlyingSymbol'].nunique() if 'UnderlyingSymbol' in df.columns else 0
        unique_expiries = df['Expiration'].nunique()
        
        # Summarize price ranges
        underlying_range = f"${df['UnderlyingPrice'].min():.2f} to ${df['UnderlyingPrice'].max():.2f}"
        strike_range = f"${df['Strike'].min():.2f} to ${df['Strike'].max():.2f}"
        
        # Store in data_info
        self.data_info = {
            'rows': len(df),
            'date_range': date_range,
            'unique_dates': unique_dates,
            'unique_symbols': unique_symbols,
            'unique_underlyings': unique_underlyings,
            'unique_expiries': unique_expiries,
            'underlying_range': underlying_range,
            'strike_range': strike_range
        }
        
        # Log the information
        self.logger.info(f"Data loaded: {len(df)} rows")
        self.logger.info(f"Date range: {date_range}")
        self.logger.info(f"Unique trading days: {unique_dates}")
        self.logger.info(f"Unique option symbols: {unique_symbols}")
        self.logger.info(f"Unique underlying symbols: {unique_underlyings}")
        self.logger.info(f"Unique expiration dates: {unique_expiries}")
        self.logger.info(f"Underlying price range: {underlying_range}")
        self.logger.info(f"Strike price range: {strike_range}")
        
        # Check for key columns
        required_cols = ['OptionSymbol', 'Strike', 'Expiration', 'Type', 'Delta']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing important columns: {', '.join(missing_cols)}")