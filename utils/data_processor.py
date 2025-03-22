"""
Data Processor Utility for Option-Framework

This utility helps format and prepare data files for use with 
the Option-Framework strategies, especially the Intraday Momentum Strategy.
"""
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import pytz


def process_csv_data(input_file, output_file, 
                     date_column='date', 
                     time_column=None,
                     datetime_column=None,
                     datetime_format=None,
                     timezone='America/New_York',
                     open_column='open',
                     high_column='high',
                     low_column='low',
                     close_column='close',
                     volume_column='volume',
                     up_volume_column=None,
                     down_volume_column=None,
                     add_timestamp_column=True,
                     drop_na=True,
                     sort_by_date=True,
                     verbose=False):
    """
    Process a CSV file to format it for use with the Option-Framework strategies.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the processed CSV file
        date_column (str): Name of the column containing the date
        time_column (str, optional): Name of the column containing the time
        datetime_column (str, optional): Name of the column containing datetime (if date and time are combined)
        datetime_format (str, optional): Format string for parsing datetime (e.g., '%Y-%m-%d %H:%M:%S')
        timezone (str): Timezone name for the data
        open_column (str): Name of the column containing opening prices
        high_column (str): Name of the column containing high prices
        low_column (str): Name of the column containing low prices
        close_column (str): Name of the column containing closing prices
        volume_column (str): Name of the column containing volume data
        up_volume_column (str, optional): Name of the column containing up volume
        down_volume_column (str, optional): Name of the column containing down volume
        add_timestamp_column (bool): Whether to add a TimeStamp column if not present
        drop_na (bool): Whether to drop rows with NA values
        sort_by_date (bool): Whether to sort the data by date
        verbose (bool): Whether to print verbose output
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        if verbose:
            print(f"Processing {input_file}...")
        
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        if verbose:
            print(f"Original columns: {df.columns.tolist()}")
            print(f"Original shape: {df.shape}")
        
        # Process datetime
        if datetime_column:
            # If a combined datetime column is provided
            if datetime_format:
                df['TimeStamp'] = pd.to_datetime(df[datetime_column], format=datetime_format)
            else:
                df['TimeStamp'] = pd.to_datetime(df[datetime_column])
        elif date_column and time_column:
            # If separate date and time columns are provided
            df['TimeStamp'] = pd.to_datetime(df[date_column].astype(str) + ' ' + df[time_column].astype(str))
        elif date_column and not time_column:
            # If only date column is provided (daily data)
            df['TimeStamp'] = pd.to_datetime(df[date_column])
        elif add_timestamp_column:
            # If no datetime columns are provided but we want to add a TimeStamp column
            raise ValueError("No datetime columns provided. Cannot add TimeStamp column.")
        
        # Set timezone
        if timezone:
            try:
                tz = pytz.timezone(timezone)
                df['TimeStamp'] = df['TimeStamp'].dt.tz_localize(tz, ambiguous='raise')
            except pytz.exceptions.AmbiguousTimeError:
                # Handle ambiguous times (like during DST transitions)
                df['TimeStamp'] = df['TimeStamp'].dt.tz_localize(tz, ambiguous='NaT')
                if drop_na:
                    df = df.dropna(subset=['TimeStamp'])
                if verbose:
                    print("Warning: Ambiguous times found during timezone localization")
            except pytz.exceptions.NonExistentTimeError:
                # Handle nonexistent times (like during DST transitions)
                df['TimeStamp'] = df['TimeStamp'].dt.tz_localize(tz, nonexistent='NaT')
                if drop_na:
                    df = df.dropna(subset=['TimeStamp'])
                if verbose:
                    print("Warning: Nonexistent times found during timezone localization")
        
        # Rename price columns
        column_mapping = {
            open_column: 'Open',
            high_column: 'High',
            low_column: 'Low',
            close_column: 'Close',
            volume_column: 'Volume'
        }
        
        # Add optional volume columns if provided
        if up_volume_column:
            column_mapping[up_volume_column] = 'UpVolume'
        if down_volume_column:
            column_mapping[down_volume_column] = 'DownVolume'
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['TimeStamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            if verbose:
                print(f"Warning: Missing required columns: {missing_columns}")
            
            # For missing volume columns, add with zeros
            if 'Volume' in missing_columns:
                df['Volume'] = 0
                missing_columns.remove('Volume')
            
            # If still missing columns, error out
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Add UpVolume and DownVolume if they don't exist
        if 'UpVolume' not in df.columns:
            if verbose:
                print("Adding estimated UpVolume based on price changes")
            df['UpVolume'] = np.where(df['Close'] >= df['Open'], df['Volume'] * 0.6, df['Volume'] * 0.4)
        
        if 'DownVolume' not in df.columns:
            if verbose:
                print("Adding estimated DownVolume based on price changes")
            df['DownVolume'] = np.where(df['Close'] < df['Open'], df['Volume'] * 0.6, df['Volume'] * 0.4)
        
        # Drop NaN values if specified
        if drop_na:
            original_rows = len(df)
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            if verbose and len(df) < original_rows:
                print(f"Dropped {original_rows - len(df)} rows with NaN values")
        
        # Sort by date if specified
        if sort_by_date:
            df = df.sort_values('TimeStamp')
        
        # Keep only necessary columns
        columns_to_keep = ['TimeStamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'UpVolume', 'DownVolume']
        df = df[[col for col in columns_to_keep if col in df.columns]]
        
        if verbose:
            print(f"Final columns: {df.columns.tolist()}")
            print(f"Final shape: {df.shape}")
            print(f"Date range: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")
        
        # Save the processed file
        df.to_csv(output_file, index=False)
        if verbose:
            print(f"Processed data saved to {output_file}")
        
        return True
    
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to process data files from command line."""
    parser = argparse.ArgumentParser(description='Process data files for Option-Framework')
    parser.add_argument('input_file', type=str, help='Input CSV file')
    parser.add_argument('output_file', type=str, help='Output CSV file')
    parser.add_argument('--date-column', type=str, default='date', help='Date column name')
    parser.add_argument('--time-column', type=str, help='Time column name')
    parser.add_argument('--datetime-column', type=str, help='Datetime column name')
    parser.add_argument('--datetime-format', type=str, help='Datetime format string')
    parser.add_argument('--timezone', type=str, default='America/New_York', help='Timezone name')
    parser.add_argument('--open-column', type=str, default='open', help='Open column name')
    parser.add_argument('--high-column', type=str, default='high', help='High column name')
    parser.add_argument('--low-column', type=str, default='low', help='Low column name')
    parser.add_argument('--close-column', type=str, default='close', help='Close column name')
    parser.add_argument('--volume-column', type=str, default='volume', help='Volume column name')
    parser.add_argument('--up-volume-column', type=str, help='Up volume column name')
    parser.add_argument('--down-volume-column', type=str, help='Down volume column name')
    parser.add_argument('--no-drop-na', action='store_false', dest='drop_na', help='Do not drop NA values')
    parser.add_argument('--no-sort', action='store_false', dest='sort_by_date', help='Do not sort by date')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    success = process_csv_data(
        input_file=args.input_file,
        output_file=args.output_file,
        date_column=args.date_column,
        time_column=args.time_column,
        datetime_column=args.datetime_column,
        datetime_format=args.datetime_format,
        timezone=args.timezone,
        open_column=args.open_column,
        high_column=args.high_column,
        low_column=args.low_column,
        close_column=args.close_column,
        volume_column=args.volume_column,
        up_volume_column=args.up_volume_column,
        down_volume_column=args.down_volume_column,
        drop_na=args.drop_na,
        sort_by_date=args.sort_by_date,
        verbose=args.verbose
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main()) 