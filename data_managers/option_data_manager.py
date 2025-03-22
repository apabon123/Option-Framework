"""
Option Data Manager Module

This module provides data management for options data,
including loading, filtering, and preprocessing options data for analysis.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_managers.base_data_manager import BaseDataManager


class DataLoadError(Exception):
    """Exception raised for errors during data loading."""
    pass


class DataValidationError(Exception):
    """Exception raised for data validation errors."""
    pass


class OptionDataManager(BaseDataManager):
    """
    Manages options data.
    
    This class extends BaseDataManager with functionality specifically for
    dealing with options data, including filtering by strike, expiry, etc.
    """
    
    # Class-level constants for validation
    REQUIRED_OPTION_COLUMNS = {
        'underlying_price', 'strike', 'expiration', 'option_type', 
        'bid', 'ask', 'bid_size', 'ask_size', 'iv'
    }
    
    # Optional columns that may be present
    OPTIONAL_OPTION_COLUMNS = {
        'delta', 'gamma', 'theta', 'vega', 'volume', 'open_interest',
        'underlying_symbol', 'exchange', 'last_trade_date'
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the option data manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)
        self._data_cache = {}
        
    def load_data(
        self, 
        file_path: str, 
        date_col: str = 'expiration',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load options data from a file.
        
        Args:
            file_path: Path to the data file
            date_col: Name of the date column
            **kwargs: Additional arguments for data loading
            
        Returns:
            pd.DataFrame: Loaded options data
        """
        if not self.validate_file(file_path):
            raise DataLoadError(f"Invalid file: {file_path}")
            
        try:
            # Load data from CSV
            self.logger.info(f"Loading options data from {file_path}")
            df = pd.read_csv(
                file_path,
                parse_dates=[date_col] if date_col in kwargs.get('parse_dates', [date_col]) else [date_col],
                **{k: v for k, v in kwargs.items() if k != 'parse_dates'}
            )
            
            if df.empty:
                self.logger.warning(f"File {file_path} contains no data")
                return pd.DataFrame()
            
            # Validate options data columns
            missing_columns = self.REQUIRED_OPTION_COLUMNS - set(df.columns)
            if missing_columns:
                self.logger.warning(f"Missing required option columns: {missing_columns}")
                
            # Store data
            self.data = df
            
            # Log data info
            self.logger.info(f"Loaded {len(df):,} option contracts from {file_path}")
            self.log_data_summary(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading options data: {e}", exc_info=True)
            raise DataLoadError(f"Failed to load options data: {str(e)}")

    def preprocess_data(
        self, 
        data: pd.DataFrame,
        calculate_greeks: bool = True,
        clean_prices: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Preprocess options data.
        
        Args:
            data: Data to preprocess
            calculate_greeks: Whether to calculate missing greeks
            clean_prices: Whether to clean option prices
            **kwargs: Additional arguments for preprocessing
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if data is None or data.empty:
            self.logger.warning("No data to preprocess")
            return pd.DataFrame()
            
        try:
            df = data.copy()
            
            # Convert option_type to uppercase if it exists
            if 'option_type' in df.columns:
                df['option_type'] = df['option_type'].str.upper()
                
                # Ensure C/P format
                df['option_type'] = df['option_type'].replace({
                    'CALL': 'C',
                    'PUT': 'P',
                    'Call': 'C',
                    'Put': 'P'
                })
            
            # Clean option prices if requested
            if clean_prices:
                df = self._clean_option_prices(df)
                
            # Calculate mid price
            if 'bid' in df.columns and 'ask' in df.columns:
                df['mid'] = (df['bid'] + df['ask']) / 2
                
            # Calculate greeks if requested and not all present
            missing_greeks = {'delta', 'gamma', 'theta', 'vega'} - set(df.columns)
            if calculate_greeks and missing_greeks:
                self.logger.info(f"Calculating missing greeks: {missing_greeks}")
                df = self._calculate_missing_greeks(df)
                
            # Calculate days to expiration if not present
            if 'dte' not in df.columns and 'expiration' in df.columns:
                current_date = df['quote_date'].iloc[0] if 'quote_date' in df.columns else pd.Timestamp.now().normalize()
                df['dte'] = (df['expiration'] - current_date).dt.days
                
            # Calculate moneyness if not present
            if 'moneyness' not in df.columns and 'underlying_price' in df.columns and 'strike' in df.columns:
                df['moneyness'] = df['strike'] / df['underlying_price']
                
                # Add moneyness categories
                if 'option_type' in df.columns:
                    moneyness_cats = []
                    for _, row in df.iterrows():
                        m = row['moneyness']
                        opt_type = row['option_type']
                        
                        if opt_type == 'C':
                            if m < 0.95:
                                cat = 'ITM'
                            elif m < 0.98:
                                cat = 'NITM'
                            elif m < 1.02:
                                cat = 'ATM'
                            elif m < 1.05:
                                cat = 'NOTM'
                            else:
                                cat = 'OTM'
                        else:  # Put option
                            if m > 1.05:
                                cat = 'ITM'
                            elif m > 1.02:
                                cat = 'NITM'
                            elif m > 0.98:
                                cat = 'ATM'
                            elif m > 0.95:
                                cat = 'NOTM'
                            else:
                                cat = 'OTM'
                        moneyness_cats.append(cat)
                    
                    df['moneyness_category'] = moneyness_cats
            
            self.logger.info(f"Preprocessing complete: {len(df):,} option contracts")
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing options data: {e}", exc_info=True)
            raise

    def _clean_option_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean option prices by removing invalid entries.
        
        Args:
            df: DataFrame with option data
            
        Returns:
            pd.DataFrame: DataFrame with cleaned prices
        """
        if 'bid' not in df.columns or 'ask' not in df.columns:
            return df
            
        original_len = len(df)
        
        # Filter out options with invalid prices
        df = df[(df['bid'] >= 0) & (df['ask'] >= 0) & (df['ask'] >= df['bid'])].copy()
        
        # Filter out options with unreasonably large bid-ask spreads
        if 'underlying_price' in df.columns:
            max_spread_pct = 1.0  # 100% of underlying price as max spread
            max_spread = df['underlying_price'] * max_spread_pct
            df = df[((df['ask'] - df['bid']) <= max_spread) | (df['bid'] <= 0.05)].copy()
        
        if len(df) < original_len:
            self.logger.info(f"Removed {original_len - len(df)} options with invalid prices")
            
        return df

    def _calculate_missing_greeks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate missing greeks using Black-Scholes model.
        
        Args:
            df: DataFrame with option data
            
        Returns:
            pd.DataFrame: DataFrame with calculated greeks
        """
        # Check if we have required data for greek calculations
        required_cols = {'underlying_price', 'strike', 'expiration', 'option_type', 'iv'}
        if not all(col in df.columns for col in required_cols):
            self.logger.warning("Missing required columns for greek calculations")
            return df
            
        try:
            from scipy.stats import norm
            
            df = df.copy()
            
            # Determine current date
            if 'quote_date' in df.columns:
                current_date = pd.to_datetime(df['quote_date'])
            else:
                current_date = pd.Series([pd.Timestamp.now().normalize()] * len(df))
            
            # Calculate time to expiration in years
            df['T'] = (pd.to_datetime(df['expiration']) - current_date).dt.days / 365.25
            
            # Set risk-free rate (ideally this would come from external source)
            r = 0.03  # 3% risk-free rate as a default
            
            # Calculate d1 and d2
            df['d1'] = (np.log(df['underlying_price'] / df['strike']) + 
                         (r + 0.5 * df['iv']**2) * df['T']) / (df['iv'] * np.sqrt(df['T']))
            df['d2'] = df['d1'] - df['iv'] * np.sqrt(df['T'])
            
            # Calculate greeks
            if 'delta' not in df.columns:
                df['delta'] = np.where(
                    df['option_type'] == 'C',
                    norm.cdf(df['d1']),
                    -norm.cdf(-df['d1'])
                )
                
            if 'gamma' not in df.columns:
                df['gamma'] = norm.pdf(df['d1']) / (df['underlying_price'] * df['iv'] * np.sqrt(df['T']))
                
            if 'theta' not in df.columns:
                df['theta'] = np.where(
                    df['option_type'] == 'C',
                    -df['underlying_price'] * norm.pdf(df['d1']) * df['iv'] / (2 * np.sqrt(df['T'])) - 
                    r * df['strike'] * np.exp(-r * df['T']) * norm.cdf(df['d2']),
                    -df['underlying_price'] * norm.pdf(df['d1']) * df['iv'] / (2 * np.sqrt(df['T'])) + 
                    r * df['strike'] * np.exp(-r * df['T']) * norm.cdf(-df['d2'])
                ) / 365  # Convert to daily theta
                
            if 'vega' not in df.columns:
                df['vega'] = df['underlying_price'] * np.sqrt(df['T']) * norm.pdf(df['d1']) / 100  # Vega per 1% change in IV
                
            # Clean up temporary columns
            df = df.drop(columns=['d1', 'd2', 'T'], errors='ignore')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating greeks: {e}", exc_info=True)
            return df

    def filter_by_moneyness(
        self,
        data: pd.DataFrame,
        min_moneyness: float = 0.8,
        max_moneyness: float = 1.2
    ) -> pd.DataFrame:
        """
        Filter options by moneyness (strike/underlying).
        
        Args:
            data: Options data
            min_moneyness: Minimum moneyness
            max_moneyness: Maximum moneyness
            
        Returns:
            pd.DataFrame: Filtered options data
        """
        if data is None or data.empty:
            return data
            
        if 'moneyness' not in data.columns:
            if 'underlying_price' in data.columns and 'strike' in data.columns:
                data = data.copy()
                data['moneyness'] = data['strike'] / data['underlying_price']
            else:
                self.logger.warning("Cannot filter by moneyness: missing required columns")
                return data
                
        filtered = data[(data['moneyness'] >= min_moneyness) & (data['moneyness'] <= max_moneyness)].copy()
        
        self.logger.info(f"Filtered by moneyness range {min_moneyness:.2f}-{max_moneyness:.2f}: {len(filtered)} contracts")
        
        return filtered

    def filter_by_dte(
        self,
        data: pd.DataFrame,
        min_dte: int = 0,
        max_dte: int = 365
    ) -> pd.DataFrame:
        """
        Filter options by days to expiration.
        
        Args:
            data: Options data
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            
        Returns:
            pd.DataFrame: Filtered options data
        """
        if data is None or data.empty:
            return data
            
        if 'dte' not in data.columns:
            if 'expiration' in data.columns:
                data = data.copy()
                current_date = (
                    data['quote_date'].iloc[0]
                    if 'quote_date' in data.columns
                    else pd.Timestamp.now().normalize()
                )
                data['dte'] = (data['expiration'] - current_date).dt.days
            else:
                self.logger.warning("Cannot filter by DTE: missing required columns")
                return data
                
        filtered = data[(data['dte'] >= min_dte) & (data['dte'] <= max_dte)].copy()
        
        self.logger.info(f"Filtered by DTE range {min_dte}-{max_dte}: {len(filtered)} contracts")
        
        return filtered

    def filter_by_option_type(
        self,
        data: pd.DataFrame,
        option_type: str = 'BOTH'
    ) -> pd.DataFrame:
        """
        Filter options by option type.
        
        Args:
            data: Options data
            option_type: Option type ('C', 'P', or 'BOTH')
            
        Returns:
            pd.DataFrame: Filtered options data
        """
        if data is None or data.empty or option_type.upper() == 'BOTH':
            return data
            
        if 'option_type' not in data.columns:
            self.logger.warning("Cannot filter by option type: missing option_type column")
            return data
                
        option_type = option_type.upper()[0]  # Take first character (C or P)
        
        filtered = data[data['option_type'] == option_type].copy()
        
        self.logger.info(f"Filtered by option type {option_type}: {len(filtered)} contracts")
        
        return filtered

    def filter_by_expiration(
        self,
        data: pd.DataFrame,
        expiration_dates: Optional[List[Union[str, date, datetime]]] = None
    ) -> pd.DataFrame:
        """
        Filter options by expiration date.
        
        Args:
            data: Options data
            expiration_dates: List of expiration dates to include
            
        Returns:
            pd.DataFrame: Filtered options data
        """
        if data is None or data.empty or not expiration_dates:
            return data
            
        if 'expiration' not in data.columns:
            self.logger.warning("Cannot filter by expiration: missing expiration column")
            return data
                
        # Convert expiration_dates to datetime
        exp_dates = [pd.Timestamp(d).normalize() for d in expiration_dates]
        
        filtered = data[data['expiration'].dt.normalize().isin(exp_dates)].copy()
        
        self.logger.info(f"Filtered by expiration dates: {len(filtered)} contracts")
        
        return filtered

    def filter_by_min_volume(
        self,
        data: pd.DataFrame,
        min_volume: int = 10
    ) -> pd.DataFrame:
        """
        Filter options by minimum volume.
        
        Args:
            data: Options data
            min_volume: Minimum volume
            
        Returns:
            pd.DataFrame: Filtered options data
        """
        if data is None or data.empty or min_volume <= 0:
            return data
            
        if 'volume' not in data.columns:
            self.logger.warning("Cannot filter by volume: missing volume column")
            return data
                
        filtered = data[data['volume'] >= min_volume].copy()
        
        self.logger.info(f"Filtered by min volume {min_volume}: {len(filtered)} contracts")
        
        return filtered

    def filter_by_min_open_interest(
        self,
        data: pd.DataFrame,
        min_oi: int = 10
    ) -> pd.DataFrame:
        """
        Filter options by minimum open interest.
        
        Args:
            data: Options data
            min_oi: Minimum open interest
            
        Returns:
            pd.DataFrame: Filtered options data
        """
        if data is None or data.empty or min_oi <= 0:
            return data
            
        if 'open_interest' not in data.columns:
            self.logger.warning("Cannot filter by open interest: missing open_interest column")
            return data
                
        filtered = data[data['open_interest'] >= min_oi].copy()
        
        self.logger.info(f"Filtered by min open interest {min_oi}: {len(filtered)} contracts")
        
        return filtered

    def get_options_chain(
        self, 
        data: pd.DataFrame, 
        expiration_date: Optional[Union[str, date, datetime]] = None,
        strikes: Optional[List[float]] = None,
        min_moneyness: float = 0.8,
        max_moneyness: float = 1.2
    ) -> pd.DataFrame:
        """
        Get an options chain for a specific expiration date.
        
        Args:
            data: Options data
            expiration_date: Target expiration date (default: nearest expiration)
            strikes: List of strike prices to include (default: all)
            min_moneyness: Minimum moneyness if strikes not specified
            max_moneyness: Maximum moneyness if strikes not specified
            
        Returns:
            pd.DataFrame: Options chain
        """
        if data is None or data.empty:
            return pd.DataFrame()
            
        df = data.copy()
        
        # Get expiration date if not specified
        if expiration_date is None:
            # Find nearest expiration date
            if 'dte' in df.columns:
                future_exp = df[df['dte'] >= 0]
                if not future_exp.empty:
                    min_dte = future_exp['dte'].min()
                    nearest_exp = future_exp[future_exp['dte'] == min_dte]['expiration'].iloc[0]
                    expiration_date = nearest_exp
            
            # If still not set, use the closest expiration date
            if expiration_date is None and 'expiration' in df.columns:
                expiration_date = df['expiration'].min()
        
        # Filter by expiration
        if expiration_date is not None:
            exp_date = pd.Timestamp(expiration_date).normalize()
            df = df[df['expiration'].dt.normalize() == exp_date]
            
        # Filter by strikes if specified
        if strikes:
            df = df[df['strike'].isin(strikes)]
        elif 'moneyness' in df.columns:
            # Otherwise filter by moneyness
            df = df[(df['moneyness'] >= min_moneyness) & (df['moneyness'] <= max_moneyness)]
            
        # Sort by strike and option_type
        if not df.empty:
            df = df.sort_values(['strike', 'option_type'])
            
        self.logger.info(f"Options chain for {expiration_date}: {len(df)} contracts")
        
        return df

    def plot_volatility_surface(
        self,
        data: pd.DataFrame,
        z_column: str = 'iv',
        ax=None,
        title: Optional[str] = None,
        cmap: str = 'viridis',
        **kwargs
    ) -> Any:
        """
        Plot a volatility surface or other 3D surface from option data.
        
        Args:
            data: Options data
            z_column: Column to use for z-axis (default: 'iv')
            ax: Matplotlib axis to plot on
            title: Plot title
            cmap: Colormap name
            **kwargs: Additional arguments for plotting
            
        Returns:
            Matplotlib axis object
        """
        if data is None or data.empty:
            self.logger.warning("No data to plot")
            return None
            
        if 'moneyness' not in data.columns or 'dte' not in data.columns:
            self.logger.warning("Missing required columns for surface plot")
            return None
            
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Create axis if not provided
            if ax is None:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
            # Get unique values for grid
            moneyness_vals = np.sort(data['moneyness'].unique())
            dte_vals = np.sort(data['dte'].unique())
            
            # Create grid for surface
            X, Y = np.meshgrid(moneyness_vals, dte_vals)
            Z = np.zeros_like(X)
            
            # Fill Z values
            for i, dte in enumerate(dte_vals):
                for j, m in enumerate(moneyness_vals):
                    mask = (data['dte'] == dte) & (data['moneyness'] == m)
                    if mask.any():
                        Z[i, j] = data.loc[mask, z_column].mean()
            
            # Plot surface
            surf = ax.plot_surface(X, Y, Z, cmap=cmap, **kwargs)
            
            # Set labels
            ax.set_xlabel('Moneyness (Strike/Underlying)')
            ax.set_ylabel('Days to Expiration')
            ax.set_zlabel(z_column.upper())
            
            if title:
                ax.set_title(title)
                
            # Add colorbar
            plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            return ax
            
        except Exception as e:
            self.logger.error(f"Error plotting surface: {e}", exc_info=True)
            return None

    def prepare_data_for_analysis(
        self, 
        file_path: str, 
        option_type: str = 'BOTH',
        min_dte: int = 0,
        max_dte: int = 120,
        min_moneyness: float = 0.8,
        max_moneyness: float = 1.2,
        **kwargs
    ) -> pd.DataFrame:
        """
        Prepare options data for analysis with full processing pipeline.
        
        Args:
            file_path: Path to data file
            option_type: Option type to filter ('C', 'P', 'BOTH')
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            min_moneyness: Minimum moneyness
            max_moneyness: Maximum moneyness
            **kwargs: Additional arguments for processing
            
        Returns:
            pd.DataFrame: Fully processed data ready for analysis
        """
        try:
            # Create cache key
            cache_key = f"{file_path}_{option_type}_{min_dte}_{max_dte}_{min_moneyness}_{max_moneyness}"
            
            # Check if we have cached data
            if cache_key in self._data_cache:
                self.logger.info("Using cached data")
                return self._data_cache[cache_key].copy()
                
            # Load data
            df = self.load_data(file_path, **kwargs)
            
            # Skip processing if data is empty
            if df.empty:
                return df
                
            # Preprocess data
            df = self.preprocess_data(df, **kwargs)
            
            # Apply filters
            df = self.filter_by_option_type(df, option_type)
            df = self.filter_by_dte(df, min_dte, max_dte)
            df = self.filter_by_moneyness(df, min_moneyness, max_moneyness)
            
            # Additional filters if specified in kwargs
            if kwargs.get('min_volume', 0) > 0:
                df = self.filter_by_min_volume(df, kwargs['min_volume'])
                
            if kwargs.get('min_open_interest', 0) > 0:
                df = self.filter_by_min_open_interest(df, kwargs['min_open_interest'])
            
            # Cache the results
            self._data_cache[cache_key] = df.copy()
            
            # Log summary
            self.logger.info(f"Prepared {len(df)} option contracts for analysis")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}", exc_info=True)
            raise 