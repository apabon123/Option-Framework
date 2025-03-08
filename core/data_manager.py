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
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.data = None
        self.underlying_data = None  # Store underlying price data separately
        self.data_info = {}
        self.trading_dates = []
    
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
                return None
                
            # Load the CSV file
            self.logger.debug(f"Reading CSV file: {file_path}")
            print(f"Reading CSV file: {file_path}...")
            try:
                # Print the first few lines of the file to debug
                with open(file_path, 'r') as f:
                    first_lines = [next(f) for _ in range(5)]
                    self.logger.debug(f"First few lines of the file:\n{''.join(first_lines)}")
                
                df = pd.read_csv(file_path)
                print(f"CSV file loaded with {len(df)} rows")
                self.logger.debug(f"CSV file loaded successfully with {len(df)} rows")
            except Exception as e:
                self.logger.error(f"Error reading CSV file: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                print(f"ERROR reading CSV file: {e}")
                return None
                
            # Check if we have data
            if df.empty:
                self.logger.warning("CSV file is empty")
                print("WARNING: CSV file is empty")
                return None
                
            # Log the columns we have
            self.logger.debug(f"CSV columns: {df.columns.tolist()}")
            
            # Map column names to standard names
            column_mapping = {}
            
            # Check for date column
            if 'DataDate' in df.columns:
                date_col = 'DataDate'
            elif 'date' in df.columns:
                date_col = 'date'
                column_mapping['date'] = 'DataDate'
                print("Mapping 'date' column to 'DataDate'")
            else:
                self.logger.warning("No date column found in data")
                print("WARNING: No date column found in data")
                date_col = None
                
            # Check for expiry column
            if 'Expiration' in df.columns:
                expiry_col = 'Expiration'
            elif 'expiry' in df.columns:
                expiry_col = 'expiry'
                column_mapping['expiry'] = 'Expiration'
                print("Mapping 'expiry' column to 'Expiration'")
            else:
                self.logger.warning("No expiry column found in data")
                print("WARNING: No expiry column found in data")
                expiry_col = None
                
            # Rename columns if needed
            if column_mapping:
                self.logger.debug(f"Renaming columns: {column_mapping}")
                df = df.rename(columns=column_mapping)
                
            # Convert date columns to datetime if they exist
            date_columns = ['DataDate', 'Expiration']
            for col in date_columns:
                if col in df.columns:
                    self.logger.debug(f"Converting {col} column to datetime")
                    print(f"Converting {col} column to datetime...")
                    try:
                        df[col] = pd.to_datetime(df[col])
                        self.logger.debug(f"Successfully converted {col} to datetime")
                    except Exception as e:
                        self.logger.warning(f"Error converting {col} to datetime: {e}")
                        import traceback
                        self.logger.debug(traceback.format_exc())
                        print(f"WARNING: Error converting {col} to datetime")
            
            # Filter by date range if specified
            original_rows = len(df)
            if start_date is not None and 'DataDate' in df.columns:
                self.logger.debug(f"Filtering by start date: {start_date}")
                print(f"Filtering data from {start_date}...")
                df = df[df['DataDate'] >= start_date]
                    
            if end_date is not None and 'DataDate' in df.columns:
                self.logger.debug(f"Filtering by end date: {end_date}")
                print(f"Filtering data to {end_date}...")
                df = df[df['DataDate'] <= end_date]
                
            if original_rows != len(df):
                print(f"Date filtering: {original_rows} rows -> {len(df)} rows")
            
            # Calculate days to expiry if not already present
            if 'DaysToExpiry' not in df.columns and 'DataDate' in df.columns and 'Expiration' in df.columns:
                self.logger.debug("Calculating days to expiry")
                print("Calculating days to expiry...")
                try:
                    df['DaysToExpiry'] = (df['Expiration'] - df['DataDate']).dt.days
                    self.logger.info("Added DaysToExpiry column")
                except Exception as e:
                    self.logger.warning(f"Error calculating days to expiry: {e}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                    print(f"WARNING: Error calculating days to expiry: {e}")
            elif 'days_to_expiry' in df.columns:
                # Rename days_to_expiry to DaysToExpiry if it exists
                df = df.rename(columns={'days_to_expiry': 'DaysToExpiry'})
                print("Renamed 'days_to_expiry' to 'DaysToExpiry'")
                
            # Store the data
            self.data = df
            self.logger.debug(f"Option data loaded successfully with {len(df)} rows")
            print(f"Data preparation complete - {len(df)} rows ready")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading option data: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            print(f"ERROR loading option data: {e}")
            return None
    
    def load_from_file(self, file_path: str) -> bool:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.debug(f"Attempting to load data from file: {file_path}")
            # Check if file exists
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                print(f"ERROR: File not found: {file_path}")
                return False
                
            # Just delegate to load_option_data method
            self.data = self.load_option_data(file_path)
            
            if self.data is None or self.data.empty:
                self.logger.error("No data loaded from file")
                print("ERROR: No data loaded from file (empty dataset)")
                return False
                
            self.logger.debug(f"Successfully loaded data from file: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading data from file: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            print(f"ERROR loading file: {e}")
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

    def load_data(self) -> pd.DataFrame:
        """
        Load data from the configured data source.
        
        Returns:
            DataFrame containing the loaded data or None if loading fails
        """
        # Check if input file is specified in config
        self.logger.debug(f"Loading data with config: {self.config}")
        
        # Check for input file in different possible config structures
        input_file = None
        
        # Try direct paths.input_file
        if 'paths' in self.config and 'input_file' in self.config['paths']:
            input_file = self.config['paths']['input_file']
            self.logger.debug(f"Found input_file in paths: {input_file}")
        
        # Try data.sources[0].file_path
        elif 'data' in self.config and 'sources' in self.config['data'] and self.config['data']['sources']:
            data_sources = self.config['data']['sources']
            if isinstance(data_sources, list) and len(data_sources) > 0 and 'file_path' in data_sources[0]:
                input_file = data_sources[0]['file_path']
                self.logger.debug(f"Found input_file in data.sources: {input_file}")
        
        if not input_file:
            self.logger.error("No input file specified in configuration")
            self.logger.debug(f"Config keys: {list(self.config.keys())}")
            if 'paths' in self.config:
                self.logger.debug(f"Paths config: {self.config['paths']}")
            print("ERROR: No input file specified in configuration")
            return None
            
        try:
            # If load_from_file returns True, self.data should be populated
            self.logger.info(f"Loading data from file: {input_file}")
            print(f"Loading data from file: {input_file}")
            success = self.load_from_file(input_file)
            if not success:
                self.logger.error(f"Failed to load data from {input_file}")
                print(f"ERROR: Failed to load data from {input_file}")
                return None
                
            # Log data info
            if self.data is not None:
                self._log_data_info(self.data)
                print(f"Successfully loaded {len(self.data)} rows of data from {input_file}")
                self.logger.info(f"Successfully loaded {len(self.data)} rows of data")
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            print(f"ERROR: {e}")
            return None

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for trading.
        
        Args:
            data: Raw data loaded from file
            
        Returns:
            Preprocessed data ready for trading
        """
        if data is None or data.empty:
            self.logger.warning("No data to preprocess")
            print("WARNING: No data to preprocess")
            return data
            
        # Make a copy to avoid modifying the original data
        processed = data.copy()
        print(f"Preprocessing {len(data)} rows of data...")
        
        try:
            # Calculate mid prices if not already present
            if 'mid' not in processed.columns and 'ask' in processed.columns and 'bid' in processed.columns:
                print("Calculating mid prices from bid/ask...")
                normal_spread = self.config.get('trading', {}).get('normal_spread', 0.20)
                processed = self.calculate_mid_prices(processed, normal_spread)
                
            # Extract trading dates if not already set
            if not self.trading_dates:
                start_date = self.config.get('dates', {}).get('start_date')
                end_date = self.config.get('dates', {}).get('end_date')
                
                if start_date and isinstance(start_date, str):
                    start_date = datetime.strptime(start_date, "%Y-%m-%d")
                if end_date and isinstance(end_date, str):
                    end_date = datetime.strptime(end_date, "%Y-%m-%d")
                    
                # Get all dates in the data
                if 'DataDate' in processed.columns:
                    print("Extracting and sorting trading dates...")
                    all_dates = processed['DataDate'].unique()
                    # Filter to dates within the configured range
                    if start_date:
                        all_dates = [d for d in all_dates if d >= start_date]
                    if end_date:
                        all_dates = [d for d in all_dates if d <= end_date]
                    # Sort dates
                    self.trading_dates = sorted(all_dates)
                    print(f"Found {len(self.trading_dates)} trading dates in the dataset")
                    
            # Extract underlying data if needed
            if self.underlying_data is None:
                print("Extracting underlying price data...")
                self._extract_underlying_data(processed)
                
            self.logger.info(f"Data preprocessing completed: {len(processed)} rows")
            print(f"Data preprocessing completed: {len(processed)} rows")
            return processed
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            print(f"ERROR during preprocessing: {e}")
            return data  # Return original data if preprocessing fails