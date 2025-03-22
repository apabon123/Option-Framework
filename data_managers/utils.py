"""
Data Manager Utilities

This module provides utility functions for working with the data managers,
including data format conversion and adaptations for various data sources.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
import pytz
import csv


def convert_csv_to_intraday_format(
    input_file: str,
    output_file: str,
    date_format: str = '%Y-%m-%d',
    time_format: str = '%H:%M:%S',
    date_col: str = 'Date',
    time_col: Optional[str] = 'Time',
    timestamp_col: Optional[str] = None,
    ohlc_mapping: Optional[Dict[str, str]] = None,
    timezone: str = 'UTC',
    delimiter: str = ',',
    verbose: bool = False
) -> bool:
    """
    Convert a CSV file to a standardized intraday format that works with IntradayDataManager.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        date_format: Format of date column
        time_format: Format of time column
        date_col: Name of date column
        time_col: Name of time column (None if using timestamp_col)
        timestamp_col: Name of timestamp column (if date and time are combined)
        ohlc_mapping: Mapping of input columns to standard OHLC names
        timezone: Timezone to apply to timestamps
        delimiter: CSV delimiter
        verbose: Print detailed progress
        
    Returns:
        bool: True if conversion was successful
    """
    if verbose:
        print(f"Converting {input_file} to intraday format...")
    
    try:
        # Determine the file format by examining the header
        with open(input_file, 'r') as f:
            header_line = f.readline().strip()
            columns = header_line.split(delimiter)
            columns = [c.strip().strip('"\'') for c in columns]
        
        if verbose:
            print(f"Detected columns: {columns}")
        
        # Set default OHLC mapping if not provided
        if ohlc_mapping is None:
            # Try to guess column mappings
            ohlc_mapping = {}
            
            # Look for standard variations of OHLC column names
            open_candidates = [c for c in columns if c.lower() in ('open', 'o', 'open_price', 'opening')]
            high_candidates = [c for c in columns if c.lower() in ('high', 'h', 'high_price', 'highest')]
            low_candidates = [c for c in columns if c.lower() in ('low', 'l', 'low_price', 'lowest')]
            close_candidates = [c for c in columns if c.lower() in ('close', 'c', 'close_price', 'closing', 'last')]
            volume_candidates = [c for c in columns if c.lower() in ('volume', 'vol', 'v', 'volume_traded')]
            
            # Use the first match for each or default to standard names if present
            if 'Open' in columns:
                ohlc_mapping['Open'] = 'Open'
            elif open_candidates:
                ohlc_mapping['Open'] = open_candidates[0]
                
            if 'High' in columns:
                ohlc_mapping['High'] = 'High'
            elif high_candidates:
                ohlc_mapping['High'] = high_candidates[0]
                
            if 'Low' in columns:
                ohlc_mapping['Low'] = 'Low'
            elif low_candidates:
                ohlc_mapping['Low'] = low_candidates[0]
                
            if 'Close' in columns:
                ohlc_mapping['Close'] = 'Close'
            elif close_candidates:
                ohlc_mapping['Close'] = close_candidates[0]
                
            if 'Volume' in columns:
                ohlc_mapping['Volume'] = 'Volume'
            elif volume_candidates:
                ohlc_mapping['Volume'] = volume_candidates[0]
                
        if verbose:
            print(f"Using OHLC mapping: {ohlc_mapping}")
            
        # Identify timestamp column situation
        has_timestamp = timestamp_col is not None and timestamp_col in columns
        has_date_time = date_col in columns
        has_separate_time = time_col is not None and time_col in columns
        
        if not (has_timestamp or has_date_time):
            raise ValueError(f"Could not find necessary timestamp columns in input file. Available columns: {columns}")
        
        # Read the data
        dtype = {col: float for col in ohlc_mapping.values() if col in columns}
        
        if has_timestamp:
            parse_dates = [timestamp_col]
            df = pd.read_csv(input_file, dtype=dtype, parse_dates=parse_dates, delimiter=delimiter)
            df = df.rename(columns={timestamp_col: 'TimeStamp'})
        else:
            if has_separate_time:
                # Read with date and time as separate columns
                df = pd.read_csv(input_file, dtype=dtype, delimiter=delimiter)
                
                # Combine date and time columns
                if verbose:
                    print(f"Combining {date_col} and {time_col} columns...")
                
                # Convert to datetime
                if not pd.api.types.is_datetime64_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
                
                if time_col in df.columns:
                    # Combine date and time into timestamp
                    def combine_date_time(row):
                        d = row[date_col]
                        t = row[time_col]
                        
                        # Handle case where time might be just HH:MM
                        if len(str(t).split(':')) == 2:
                            t = f"{t}:00"
                            
                        try:
                            # Parse time component
                            t_obj = datetime.strptime(str(t), time_format).time()
                            # Combine with date
                            return datetime.combine(d.date(), t_obj)
                        except Exception as e:
                            if verbose:
                                print(f"Error combining date and time: {e} - Date: {d}, Time: {t}")
                            return pd.NaT
                    
                    df['TimeStamp'] = df.apply(combine_date_time, axis=1)
                    
                else:
                    # Just use date as timestamp (daily data with intraday format)
                    df['TimeStamp'] = df[date_col]
            else:
                # Only date column available (daily data)
                parse_dates = [date_col]
                df = pd.read_csv(input_file, dtype=dtype, parse_dates=parse_dates, delimiter=delimiter)
                df['TimeStamp'] = df[date_col]
                
        # Drop rows with invalid timestamps
        invalid_mask = df['TimeStamp'].isna()
        if invalid_mask.any():
            if verbose:
                print(f"Dropping {invalid_mask.sum()} rows with invalid timestamps")
            df = df[~invalid_mask].copy()
        
        # Apply timezone localization if needed
        if 'TimeStamp' in df.columns and df['TimeStamp'].dt.tz is None:
            if verbose:
                print(f"Localizing timestamps to {timezone} timezone")
            df['TimeStamp'] = df['TimeStamp'].dt.tz_localize(timezone)
        
        # Rename columns according to mapping
        for standard_name, file_name in ohlc_mapping.items():
            if file_name in df.columns and file_name != standard_name:
                df[standard_name] = df[file_name]
        
        # Keep only needed columns
        keep_cols = ['TimeStamp'] + list(ohlc_mapping.keys())
        keep_cols = [col for col in keep_cols if col in df.columns]
        df = df[keep_cols]
        
        # Sort by timestamp
        df = df.sort_values('TimeStamp')
        
        # Check for duplicates
        duplicate_mask = df['TimeStamp'].duplicated()
        if duplicate_mask.any():
            if verbose:
                print(f"Warning: {duplicate_mask.sum()} duplicate timestamps found")
                print("Keeping only the last entry for each timestamp")
            df = df.drop_duplicates(subset=['TimeStamp'], keep='last')
        
        # Save to output file
        df.to_csv(output_file, index=False)
        
        if verbose:
            print(f"Successfully converted file to {output_file}")
            print(f"Saved {len(df)} rows with columns: {df.columns.tolist()}")
            print(f"Date range: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")
            
        return True
        
    except Exception as e:
        print(f"Error converting file: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_csv_to_daily_format(
    input_file: str,
    output_file: str,
    date_format: str = '%Y-%m-%d',
    date_col: str = 'Date',
    ohlc_mapping: Optional[Dict[str, str]] = None,
    delimiter: str = ',',
    verbose: bool = False
) -> bool:
    """
    Convert a CSV file to a standardized daily format that works with DailyDataManager.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        date_format: Format of date column
        date_col: Name of date column
        ohlc_mapping: Mapping of input columns to standard OHLC names
        delimiter: CSV delimiter
        verbose: Print detailed progress
        
    Returns:
        bool: True if conversion was successful
    """
    if verbose:
        print(f"Converting {input_file} to daily format...")
    
    try:
        # Determine the file format by examining the header
        with open(input_file, 'r') as f:
            header_line = f.readline().strip()
            columns = header_line.split(delimiter)
            columns = [c.strip().strip('"\'') for c in columns]
        
        if verbose:
            print(f"Detected columns: {columns}")
        
        # Set default OHLC mapping if not provided
        if ohlc_mapping is None:
            # Try to guess column mappings
            ohlc_mapping = {}
            
            # Look for standard variations of OHLC column names
            open_candidates = [c for c in columns if c.lower() in ('open', 'o', 'open_price', 'opening')]
            high_candidates = [c for c in columns if c.lower() in ('high', 'h', 'high_price', 'highest')]
            low_candidates = [c for c in columns if c.lower() in ('low', 'l', 'low_price', 'lowest')]
            close_candidates = [c for c in columns if c.lower() in ('close', 'c', 'close_price', 'closing', 'last')]
            volume_candidates = [c for c in columns if c.lower() in ('volume', 'vol', 'v', 'volume_traded')]
            adjclose_candidates = [c for c in columns if c.lower() in ('adj close', 'adjusted close', 'adj_close', 'adjusted_close')]
            
            # Use the first match for each or default to standard names if present
            if 'Open' in columns:
                ohlc_mapping['Open'] = 'Open'
            elif open_candidates:
                ohlc_mapping['Open'] = open_candidates[0]
                
            if 'High' in columns:
                ohlc_mapping['High'] = 'High'
            elif high_candidates:
                ohlc_mapping['High'] = high_candidates[0]
                
            if 'Low' in columns:
                ohlc_mapping['Low'] = 'Low'
            elif low_candidates:
                ohlc_mapping['Low'] = low_candidates[0]
                
            if 'Close' in columns:
                ohlc_mapping['Close'] = 'Close'
            elif close_candidates:
                ohlc_mapping['Close'] = close_candidates[0]
                
            if 'Volume' in columns:
                ohlc_mapping['Volume'] = 'Volume'
            elif volume_candidates:
                ohlc_mapping['Volume'] = volume_candidates[0]
                
            if 'Adjusted Close' in columns:
                ohlc_mapping['Adjusted Close'] = 'Adjusted Close'
            elif adjclose_candidates:
                ohlc_mapping['Adjusted Close'] = adjclose_candidates[0]
                
        if verbose:
            print(f"Using OHLC mapping: {ohlc_mapping}")
            
        # Validate that date column exists
        if date_col not in columns:
            raise ValueError(f"Date column '{date_col}' not found in input file. Available columns: {columns}")
        
        # Read the data
        dtype = {col: float for col in ohlc_mapping.values() if col in columns}
        df = pd.read_csv(input_file, dtype=dtype, delimiter=delimiter)
        
        # Convert date column to datetime
        if not pd.api.types.is_datetime64_dtype(df[date_col]):
            df['Date'] = pd.to_datetime(df[date_col], format=date_format)
        else:
            df['Date'] = df[date_col]
            
        # Rename columns according to mapping
        for standard_name, file_name in ohlc_mapping.items():
            if file_name in df.columns and file_name != standard_name:
                df[standard_name] = df[file_name]
        
        # Keep only needed columns
        keep_cols = ['Date'] + list(ohlc_mapping.keys())
        keep_cols = [col for col in keep_cols if col in df.columns]
        df = df[keep_cols]
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Check for duplicates
        duplicate_mask = df['Date'].duplicated()
        if duplicate_mask.any():
            if verbose:
                print(f"Warning: {duplicate_mask.sum()} duplicate dates found")
                print("Keeping only the last entry for each date")
            df = df.drop_duplicates(subset=['Date'], keep='last')
        
        # Save to output file
        df.to_csv(output_file, index=False)
        
        if verbose:
            print(f"Successfully converted file to {output_file}")
            print(f"Saved {len(df)} rows with columns: {df.columns.tolist()}")
            print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            
        return True
        
    except Exception as e:
        print(f"Error converting file: {e}")
        import traceback
        traceback.print_exc()
        return False


def detect_data_frequency(filepath: str) -> str:
    """
    Detect the frequency of time series data in a file.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        str: Data frequency ('intraday', 'daily', or 'unknown')
    """
    try:
        # Read a sample of the data to detect frequency
        df = pd.read_csv(filepath, nrows=1000)
        
        # Look for timestamp or date columns
        date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'timestamp'])]
        
        if not date_cols:
            print(f"No date or time columns found in {filepath}")
            return 'unknown'
            
        # Try to parse the first date column
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Filter out invalid timestamps
        valid_dates = df[~df[date_col].isna()]
        
        if len(valid_dates) < 2:
            print(f"Not enough valid dates to determine frequency in {filepath}")
            return 'unknown'
            
        # Sort by date
        valid_dates = valid_dates.sort_values(date_col)
        
        # Calculate differences between consecutive timestamps
        diffs = valid_dates[date_col].diff().dropna()
        
        # Check if most differences are less than a day
        intraday_diffs = (diffs < pd.Timedelta('1 day')).sum()
        has_times = any(t.time() != time(0, 0) for t in valid_dates[date_col] if not pd.isna(t))
        
        if intraday_diffs > 0 or has_times:
            return 'intraday'
        else:
            return 'daily'
            
    except Exception as e:
        print(f"Error detecting data frequency: {e}")
        return 'unknown'


def convert_to_appropriate_format(
    input_file: str,
    output_dir: Optional[str] = None,
    force_type: Optional[str] = None,
    verbose: bool = False
) -> str:
    """
    Detects the data type and converts it to the appropriate format.
    
    Args:
        input_file: Path to input file
        output_dir: Directory to save output file (defaults to same directory)
        force_type: Force 'intraday' or 'daily' conversion
        verbose: Print detailed progress
        
    Returns:
        str: Path to converted file
    """
    if verbose:
        print(f"Processing file: {input_file}")
        
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
        if not output_dir:
            output_dir = '.'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename without extension
    filename = os.path.basename(input_file)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Detect data type if not forced
    data_type = force_type
    if data_type is None:
        data_type = detect_data_frequency(input_file)
        if verbose:
            print(f"Detected data type: {data_type}")
    
    # Define output file
    if data_type == 'intraday':
        output_file = os.path.join(output_dir, f"{name_without_ext}_intraday.csv")
        convert_csv_to_intraday_format(input_file, output_file, verbose=verbose)
    elif data_type == 'daily':
        output_file = os.path.join(output_dir, f"{name_without_ext}_daily.csv")
        convert_csv_to_daily_format(input_file, output_file, verbose=verbose)
    else:
        print(f"Unknown data type for {input_file}. Skipping conversion.")
        return input_file
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert market data files to standardized formats")
    parser.add_argument("input_file", help="Path to input file")
    parser.add_argument("output_file", nargs="?", help="Path to output file (optional)")
    parser.add_argument("--type", choices=["intraday", "daily"], help="Force data type")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    
    args = parser.parse_args()
    
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if args.type == "intraday":
            convert_csv_to_intraday_format(args.input_file, args.output_file, verbose=args.verbose)
        elif args.type == "daily":
            convert_csv_to_daily_format(args.input_file, args.output_file, verbose=args.verbose)
        else:
            # Auto-detect
            data_type = detect_data_frequency(args.input_file)
            if data_type == "intraday":
                convert_csv_to_intraday_format(args.input_file, args.output_file, verbose=args.verbose)
            elif data_type == "daily":
                convert_csv_to_daily_format(args.input_file, args.output_file, verbose=args.verbose)
            else:
                print(f"Could not determine data type for {args.input_file}")
    else:
        # Auto-convert with auto-generated output filename
        output_file = convert_to_appropriate_format(
            args.input_file, 
            force_type=args.type,
            verbose=args.verbose
        )
        print(f"Converted file saved to: {output_file}") 