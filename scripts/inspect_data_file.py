#!/usr/bin/env python
"""
Data File Inspector

This utility script inspects the content of a data file to help troubleshoot
issues like the "No trading dates available" error.
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path
import datetime

# Add the root directory to the Python path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inspect a data file for troubleshooting")
    
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the data file to inspect",
    )
    
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date to check (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date to check (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--date_column",
        type=str,
        default="date",
        help="Name of the date column (default: 'date')",
    )
    
    return parser.parse_args()

def inspect_file(file_path, start_date=None, end_date=None, date_column="date"):
    """Inspect the content of a data file."""
    print(f"\n=== Inspecting Data File: {file_path} ===\n")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist: {file_path}")
        return 1
    
    print(f"File exists: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    # Try to read the file
    try:
        # First, try to read the first few lines to determine the format
        with open(file_path, 'r') as f:
            print("\nFile Header (first 5 lines):")
            for i, line in enumerate(f):
                if i < 5:
                    print(line.strip())
                else:
                    break

        # Read the CSV file
        print(f"\nAttempting to read as CSV with pandas...")
        df = pd.read_csv(file_path)
        
        # Basic information
        print(f"\nDataframe Shape: {df.shape} (rows, columns)")
        print(f"Columns: {list(df.columns)}")
        
        # Check for the date column
        if date_column not in df.columns:
            print(f"\nWARNING: Date column '{date_column}' not found in data!")
            print(f"Available columns: {list(df.columns)}")
            
            # Specifically check for DataDate as a common alternative
            if 'DataDate' in df.columns:
                print(f"Found 'DataDate' column, using it as date column")
                date_column = 'DataDate'
            else:
                # Try to find a date-like column
                date_like_columns = [col for col in df.columns if any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'expiry', 'expiration'])]
                if date_like_columns:
                    print(f"Possible date columns: {date_like_columns}")
                    date_column = date_like_columns[0]
                    print(f"Using '{date_column}' as date column for further analysis")
                else:
                    print("No date-like columns found. Cannot perform date analysis.")
                    return 1
        
        # Analyze the date column
        print(f"\n=== Date Column Analysis ===")
        print(f"Date column type: {df[date_column].dtype}")
        print(f"First 5 dates: {list(df[date_column].head(5))}")
        
        # Convert to datetime if not already
        if df[date_column].dtype != 'datetime64[ns]':
            print("Converting date column to datetime format...")
            try:
                df[date_column] = pd.to_datetime(df[date_column])
                print(f"After conversion - First 5 dates: {list(df[date_column].head(5))}")
            except Exception as e:
                print(f"Error converting to datetime: {str(e)}")
                
        # Get unique dates
        if df[date_column].dtype == 'datetime64[ns]':
            unique_dates = df[date_column].dt.date.unique()
            print(f"\nUnique dates in dataset: {unique_dates}")
            print(f"Total unique dates: {len(unique_dates)}")
            
            if len(unique_dates) > 0:
                print(f"Date range in data: {min(unique_dates)} to {max(unique_dates)}")
            
            # Check if dates are within the specified range
            if start_date and end_date:
                start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
                end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
                print(f"\nChecking for dates between {start} and {end}...")
                
                dates_in_range = [d for d in unique_dates if start <= d <= end]
                if dates_in_range:
                    print(f"Found {len(dates_in_range)} dates in specified range:")
                    print(dates_in_range)
                else:
                    print(f"ERROR: No dates found in the specified range {start} to {end}!")
                    print("This is likely the cause of the 'No trading dates available' error.")
                    
                    # Suggest a valid date range
                    if len(unique_dates) > 0:
                        valid_start = max(min(unique_dates), (datetime.datetime.now() - datetime.timedelta(days=365)).date())
                        valid_end = min(max(unique_dates), datetime.datetime.now().date())
                        print(f"\nSuggested valid date range: {valid_start} to {valid_end}")
        
        # Option data specific checks
        if any(col in df.columns for col in ['option_type', 'call_put', 'strike', 'expiry', 'expiration']):
            print("\n=== Option Data Analysis ===")
            
            # Check option types if available
            option_type_col = next((col for col in df.columns if col in ['option_type', 'call_put', 'type']), None)
            if option_type_col:
                option_types = df[option_type_col].unique()
                print(f"Option types: {option_types}")
            
            # Check strikes if available
            strike_col = next((col for col in df.columns if col in ['strike', 'strike_price']), None)
            if strike_col:
                print(f"Strike price range: {df[strike_col].min()} to {df[strike_col].max()}")
                print(f"Unique strike prices: {len(df[strike_col].unique())}")
            
            # Check expiries if available
            expiry_col = next((col for col in df.columns if col in ['expiry', 'expiration', 'exp_date']), None)
            if expiry_col:
                if df[expiry_col].dtype != 'datetime64[ns]':
                    try:
                        df[expiry_col] = pd.to_datetime(df[expiry_col])
                    except:
                        pass
                
                if df[expiry_col].dtype == 'datetime64[ns]':
                    print(f"Expiration date range: {df[expiry_col].min()} to {df[expiry_col].max()}")
                    print(f"Unique expiration dates: {len(df[expiry_col].unique())}")
        
        # Check for any missing values in important columns
        missing_values = df.isnull().sum()
        if missing_values.any():
            print("\n=== Missing Values Analysis ===")
            print(missing_values[missing_values > 0])
        
        return 0
    
    except Exception as e:
        print(f"Error reading or processing file: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    return inspect_file(
        args.file_path, 
        start_date=args.start_date, 
        end_date=args.end_date,
        date_column=args.date_column
    )

if __name__ == "__main__":
    sys.exit(main()) 