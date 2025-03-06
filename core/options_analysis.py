"""
Options Analysis Module

This module provides tools for analyzing options data, including:
- Rich/cheap detection through volatility surface analysis
- Greeks calculation and visualization
- Z-score analysis for option valuation
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.interpolate import griddata


class VolatilitySurface:
    """
    Represents and manages a volatility surface for options analysis.
    
    Provides methods for constructing, visualizing, and querying the volatility
    surface, as well as calculating z-scores for options based on their position
    relative to the surface.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize a volatility surface object.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger('trading')
        self.surface_data = None
        self.surface_model = None
        self.last_update_time = None
        self.strikes = []
        self.expiries = []
        self.historical_surfaces = {}
        
    def update_surface(self, options_data: pd.DataFrame) -> bool:
        """
        Update the volatility surface with new options data.
        
        Args:
            options_data: DataFrame containing options data with at least:
                          Strike, DaysToExpiry, IV, Type, etc.
                          
        Returns:
            bool: True if surface was successfully updated, False otherwise
        """
        if options_data.empty:
            self.logger.warning("Cannot update volatility surface: Empty options data")
            return False
            
        try:
            # Check for required columns
            required_cols = ['Strike', 'DaysToExpiry', 'IV', 'Type']
            missing_cols = [col for col in required_cols if col not in options_data.columns]
            if missing_cols:
                self.logger.warning(f"Missing required columns for volatility surface: {missing_cols}")
                return False
                
            # Filter data and prepare for surface creation
            cleaned_data = options_data.dropna(subset=['IV', 'Strike', 'DaysToExpiry'])
            
            if cleaned_data.empty:
                self.logger.warning("No valid data points for volatility surface after cleaning")
                return False
                
            # Store the surface data
            self.surface_data = cleaned_data
            
            # Extract unique strikes and expiries for the grid
            self.strikes = sorted(cleaned_data['Strike'].unique())
            self.expiries = sorted(cleaned_data['DaysToExpiry'].unique())
            
            # Create the volatility surface model
            self._create_surface_model()
            
            # Store timestamp
            self.last_update_time = datetime.now()
            
            # Store this surface in historical collection if it's a new date
            if 'DataDate' in cleaned_data.columns:
                data_date = cleaned_data['DataDate'].iloc[0]
                if isinstance(data_date, pd.Timestamp) and data_date not in self.historical_surfaces:
                    # Store a compact representation for historical tracking
                    self.historical_surfaces[data_date] = {
                        'avg_iv': cleaned_data['IV'].mean(),
                        'surface_params': self._get_surface_params()
                    }
            
            self.logger.info(f"Volatility surface updated with {len(cleaned_data)} data points")
            self.logger.info(f"  Strikes: {len(self.strikes)} unique values")
            self.logger.info(f"  Expiries: {len(self.expiries)} unique values")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating volatility surface: {e}")
            return False
            
    def _create_surface_model(self):
        """
        Create the volatility surface interpolation model from the data.
        """
        if self.surface_data is None or self.surface_data.empty:
            return
            
        try:
            # Prepare data for interpolation
            points = self.surface_data[['Strike', 'DaysToExpiry']].values
            values = self.surface_data['IV'].values
            
            # Create grid for the model
            grid_x, grid_y = np.meshgrid(self.strikes, self.expiries)
            
            # Create interpolation model
            self.surface_model = {
                'points': points,
                'values': values,
                'grid_x': grid_x,
                'grid_y': grid_y
            }
            
            self.logger.debug("Volatility surface model created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating surface model: {e}")
            self.surface_model = None
            
    def _get_surface_params(self) -> Dict[str, Any]:
        """
        Get compact parameters representing the current surface.
        
        Returns:
            dict: Key parameters that characterize the surface
        """
        # This is a simplified implementation - in practice you might
        # use polynomial coefficients or other sparse representations
        return {
            'avg_iv': self.surface_data['IV'].mean() if self.surface_data is not None else 0,
            'min_iv': self.surface_data['IV'].min() if self.surface_data is not None else 0,
            'max_iv': self.surface_data['IV'].max() if self.surface_data is not None else 0,
            'timestamp': self.last_update_time
        }
        
    def get_iv_at_point(self, strike: float, days_to_expiry: float) -> Optional[float]:
        """
        Get the interpolated implied volatility at a specific strike/expiry point.
        
        Args:
            strike: Option strike price
            days_to_expiry: Days to expiration
            
        Returns:
            float: Interpolated IV value or None if interpolation fails
        """
        if self.surface_model is None:
            self.logger.warning("Cannot get IV: No surface model available")
            return None
            
        try:
            # Use the model to interpolate IV
            points = self.surface_model['points']
            values = self.surface_model['values']
            
            # Perform interpolation
            interpolated_iv = griddata(
                points, 
                values, 
                np.array([[strike, days_to_expiry]]), 
                method='linear'
            )[0]
            
            # Check if interpolation succeeded
            if np.isnan(interpolated_iv):
                # Try nearest method if linear interpolation fails
                interpolated_iv = griddata(
                    points, 
                    values, 
                    np.array([[strike, days_to_expiry]]), 
                    method='nearest'
                )[0]
            
            return float(interpolated_iv)
            
        except Exception as e:
            self.logger.error(f"Error interpolating IV at strike={strike}, dte={days_to_expiry}: {e}")
            return None
            
    def calculate_iv_zscore(self, strike: float, days_to_expiry: float, observed_iv: float) -> Optional[float]:
        """
        Calculate the z-score of an option's IV relative to the surface.
        
        Args:
            strike: Option strike price
            days_to_expiry: Days to expiration
            observed_iv: Observed implied volatility of the option
            
        Returns:
            float: Z-score of the option's IV or None if calculation fails
        """
        if self.surface_data is None:
            return None
            
        try:
            # Get the expected IV from the surface
            expected_iv = self.get_iv_at_point(strike, days_to_expiry)
            if expected_iv is None:
                return None
                
            # Get nearby points to calculate local standard deviation
            nearby_strikes = self._get_nearby_values(self.strikes, strike, 5)
            nearby_expiries = self._get_nearby_values(self.expiries, days_to_expiry, 3)
            
            # Filter surface data for nearby points
            nearby_data = self.surface_data[
                (self.surface_data['Strike'].isin(nearby_strikes)) &
                (self.surface_data['DaysToExpiry'].isin(nearby_expiries))
            ]
            
            if len(nearby_data) < 3:
                # Not enough nearby points for a reliable std calculation
                # Fall back to global std
                std_iv = self.surface_data['IV'].std()
            else:
                std_iv = nearby_data['IV'].std()
                
            if std_iv == 0:
                # Avoid division by zero
                self.logger.warning("Zero standard deviation in IV values, using default")
                std_iv = 0.01  # Default small value
                
            # Calculate z-score
            z_score = (observed_iv - expected_iv) / std_iv
            
            return z_score
            
        except Exception as e:
            self.logger.error(f"Error calculating IV z-score: {e}")
            return None
            
    def _get_nearby_values(self, values: List[float], target: float, count: int) -> List[float]:
        """
        Get the closest values to a target from a list.
        
        Args:
            values: List of values to search
            target: Target value
            count: Number of nearby values to return
            
        Returns:
            list: Nearby values
        """
        if not values:
            return []
            
        # Calculate distances
        distances = [(abs(v - target), v) for v in values]
        
        # Sort by distance and take the closest 'count' items
        sorted_values = [v for _, v in sorted(distances)]
        
        return sorted_values[:min(count, len(values))]


class OptionsAnalyzer:
    """
    Analyzes options for various metrics including richness/cheapness,
    z-scores, and curve-based metrics.
    """
    
    def __init__(
        self, 
        vol_surface: Optional[VolatilitySurface] = None,
        z_score_threshold: float = 1.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the options analyzer.
        
        Args:
            vol_surface: Volatility surface for IV analysis
            z_score_threshold: Threshold for classifying options as rich/cheap
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger('trading')
        self.vol_surface = vol_surface or VolatilitySurface(logger)
        self.z_score_threshold = z_score_threshold
        
    def analyze_option(
        self, 
        option_data: Union[pd.Series, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze a single option and return its metrics.
        
        Args:
            option_data: Option data (Series or dict)
            
        Returns:
            dict: Analysis results
        """
        try:
            # Extract data with compatible handling for both dict and Series
            if hasattr(option_data, 'get') and not hasattr(option_data, 'iloc'):
                # Dictionary style access
                symbol = option_data.get('OptionSymbol', 'Unknown')
                strike = option_data.get('Strike', 0)
                days_to_expiry = option_data.get('DaysToExpiry', 0)
                option_type = option_data.get('Type', '').lower()
                iv = option_data.get('IV', 0)
                price = option_data.get('MidPrice', 0)
                delta = option_data.get('Delta', 0)
                underlying_price = option_data.get('UnderlyingPrice', 0)
            else:
                # Series style access
                symbol = option_data['OptionSymbol'] if 'OptionSymbol' in option_data else 'Unknown'
                strike = option_data['Strike'] if 'Strike' in option_data else 0
                days_to_expiry = option_data['DaysToExpiry'] if 'DaysToExpiry' in option_data else 0
                option_type = option_data['Type'].lower() if 'Type' in option_data else ''
                iv = option_data['IV'] if 'IV' in option_data else 0
                price = option_data['MidPrice'] if 'MidPrice' in option_data else 0
                delta = option_data['Delta'] if 'Delta' in option_data else 0
                underlying_price = option_data['UnderlyingPrice'] if 'UnderlyingPrice' in option_data else 0
            
            # Calculate moneyness
            moneyness = underlying_price / strike if strike > 0 else 0
            
            # Calculate risk metrics
            if underlying_price > 0 and price > 0:
                # Calculate implied leverage
                implied_leverage = underlying_price / (price * 100)
                
                # Calculate theta/vega ratio (if available)
                theta = option_data.get('Theta', 0) if hasattr(option_data, 'get') else (
                    option_data['Theta'] if 'Theta' in option_data else 0
                )
                vega = option_data.get('Vega', 0) if hasattr(option_data, 'get') else (
                    option_data['Vega'] if 'Vega' in option_data else 0
                )
                
                theta_vega_ratio = theta / vega if vega != 0 else 0
            else:
                implied_leverage = 0
                theta_vega_ratio = 0
            
            # Analyze IV relative to the volatility surface
            iv_zscore = None
            if self.vol_surface and iv > 0 and strike > 0 and days_to_expiry > 0:
                iv_zscore = self.vol_surface.calculate_iv_zscore(strike, days_to_expiry, iv)
            
            # Determine if option is rich or cheap based on z-score
            is_rich = iv_zscore > self.z_score_threshold if iv_zscore is not None else None
            is_cheap = iv_zscore < -self.z_score_threshold if iv_zscore is not None else None
            
            return {
                'symbol': symbol,
                'strike': strike,
                'days_to_expiry': days_to_expiry,
                'option_type': option_type,
                'moneyness': moneyness,
                'iv': iv,
                'iv_zscore': iv_zscore,
                'is_rich': is_rich,
                'is_cheap': is_cheap,
                'implied_leverage': implied_leverage,
                'theta_vega_ratio': theta_vega_ratio,
                'risk_assessment': self._assess_risk(delta, days_to_expiry, iv_zscore)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing option: {e}")
            return {'error': str(e), 'symbol': option_data.get('OptionSymbol', 'Unknown')}
            
    def analyze_options_chain(self, options_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze an entire options chain.
        
        Args:
            options_chain: DataFrame of options data
            
        Returns:
            DataFrame: Original data with analysis columns added
        """
        if options_chain.empty:
            return options_chain
        
        try:
            # Update volatility surface with this data
            self.vol_surface.update_surface(options_chain)
            
            # Create output DataFrame
            results = options_chain.copy()
            
            # Initialize analysis columns
            results['IV_ZScore'] = np.nan
            results['IsRich'] = False
            results['IsCheap'] = False
            results['ImpliedLeverage'] = np.nan
            results['RiskAssessment'] = 'Unknown'
            
            # Process each option
            for idx, row in options_chain.iterrows():
                analysis = self.analyze_option(row)
                
                # Update analysis columns
                results.loc[idx, 'IV_ZScore'] = analysis.get('iv_zscore')
                results.loc[idx, 'IsRich'] = analysis.get('is_rich', False)
                results.loc[idx, 'IsCheap'] = analysis.get('is_cheap', False)
                results.loc[idx, 'ImpliedLeverage'] = analysis.get('implied_leverage', 0)
                results.loc[idx, 'RiskAssessment'] = analysis.get('risk_assessment', 'Unknown')
            
            self.logger.info(f"Analyzed {len(options_chain)} options in chain")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing options chain: {e}")
            # Return original data unchanged on error
            return options_chain
            
    def _assess_risk(
        self, 
        delta: float, 
        days_to_expiry: float, 
        iv_zscore: Optional[float]
    ) -> str:
        """
        Assess the risk of an option based on key metrics.
        
        Args:
            delta: Option delta
            days_to_expiry: Days to expiration
            iv_zscore: IV z-score if available
            
        Returns:
            str: Risk assessment ('Low', 'Medium', 'High', 'Very High')
        """
        # Start with medium risk
        risk_level = "Medium"
        
        # Short-dated options are higher risk
        if days_to_expiry < 14:
            risk_level = "Very High"
        elif days_to_expiry < 30:
            risk_level = "High"
        
        # Far OTM options (small absolute delta) may be lower risk for buyers
        # but can still be risky for sellers
        abs_delta = abs(delta)
        if abs_delta < 0.1:
            # Very far OTM
            if risk_level in ["Medium", "Low"]:
                risk_level = "Medium"  # Far OTM is still medium risk minimum
        elif abs_delta < 0.25:
            # Moderately OTM
            if risk_level == "Low":
                risk_level = "Medium"
                
        # If we have IV z-score information, consider that too
        if iv_zscore is not None:
            # Buying expensive options (high z-score) is risky
            if iv_zscore > 1.0 and risk_level not in ["Very High"]:
                # Bump up risk one level
                risk_levels = ["Low", "Medium", "High", "Very High"]
                current_idx = risk_levels.index(risk_level)
                risk_level = risk_levels[min(current_idx + 1, len(risk_levels) - 1)]
                
        return risk_level
        
    def find_arbitrage_opportunities(self, options_chain: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find potential arbitrage opportunities in the options chain.
        
        Args:
            options_chain: DataFrame of options data
            
        Returns:
            list: List of potential arbitrage trades
        """
        opportunities = []
        
        if options_chain.empty:
            return opportunities
            
        try:
            # Update volatility surface
            self.vol_surface.update_surface(options_chain)
            
            # Find potential vertical spreads with mispricing
            calls = options_chain[options_chain['Type'].str.lower() == 'call'].sort_values('Strike')
            puts = options_chain[options_chain['Type'].str.lower() == 'put'].sort_values('Strike')
            
            # Analyze calls for vertical spread opportunities (call debit spreads)
            for expiry in calls['DaysToExpiry'].unique():
                expiry_calls = calls[calls['DaysToExpiry'] == expiry]
                
                # We need at least 2 strikes to form a spread
                if len(expiry_calls) < 2:
                    continue
                    
                # Check consecutive strikes
                for i in range(len(expiry_calls) - 1):
                    lower_call = expiry_calls.iloc[i]
                    upper_call = expiry_calls.iloc[i + 1]
                    
                    # Analyze both legs
                    lower_analysis = self.analyze_option(lower_call)
                    upper_analysis = self.analyze_option(upper_call)
                    
                    # Look for rich lower and cheap upper
                    if lower_analysis.get('is_rich') and upper_analysis.get('is_cheap'):
                        opportunities.append({
                            'type': 'Call Credit Spread',
                            'expiry': expiry,
                            'lower_strike': lower_call['Strike'],
                            'upper_strike': upper_call['Strike'],
                            'lower_premium': lower_call['MidPrice'],
                            'upper_premium': upper_call['MidPrice'],
                            'net_premium': lower_call['MidPrice'] - upper_call['MidPrice'],
                            'lower_z': lower_analysis.get('iv_zscore'),
                            'upper_z': upper_analysis.get('iv_zscore')
                        })
            
            # Similarly analyze puts for vertical spread opportunities
            for expiry in puts['DaysToExpiry'].unique():
                expiry_puts = puts[puts['DaysToExpiry'] == expiry]
                
                if len(expiry_puts) < 2:
                    continue
                    
                for i in range(len(expiry_puts) - 1):
                    lower_put = expiry_puts.iloc[i]
                    upper_put = expiry_puts.iloc[i + 1]
                    
                    lower_analysis = self.analyze_option(lower_put)
                    upper_analysis = self.analyze_option(upper_put)
                    
                    # Look for cheap lower and rich upper
                    if lower_analysis.get('is_cheap') and upper_analysis.get('is_rich'):
                        opportunities.append({
                            'type': 'Put Debit Spread',
                            'expiry': expiry,
                            'lower_strike': lower_put['Strike'],
                            'upper_strike': upper_put['Strike'],
                            'lower_premium': lower_put['MidPrice'],
                            'upper_premium': upper_put['MidPrice'],
                            'net_premium': upper_put['MidPrice'] - lower_put['MidPrice'],
                            'lower_z': lower_analysis.get('iv_zscore'),
                            'upper_z': upper_analysis.get('iv_zscore')
                        })
            
            self.logger.info(f"Found {len(opportunities)} potential arbitrage opportunities")
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding arbitrage opportunities: {e}")
            return []
            
    def filter_by_criteria(
        self, 
        options_chain: pd.DataFrame, 
        criteria: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Filter options chain by custom criteria.
        
        Args:
            options_chain: DataFrame of options data
            criteria: Dictionary of criteria to filter by
            
        Returns:
            DataFrame: Filtered options
        """
        if options_chain.empty:
            return options_chain
            
        filtered = options_chain.copy()
        
        try:
            # Apply common filters
            if 'min_dte' in criteria:
                filtered = filtered[filtered['DaysToExpiry'] >= criteria['min_dte']]
                
            if 'max_dte' in criteria:
                filtered = filtered[filtered['DaysToExpiry'] <= criteria['max_dte']]
                
            if 'option_type' in criteria:
                filtered = filtered[filtered['Type'].str.lower() == criteria['option_type'].lower()]
                
            if 'min_delta' in criteria:
                filtered = filtered[filtered['Delta'] >= criteria['min_delta']]
                
            if 'max_delta' in criteria:
                filtered = filtered[filtered['Delta'] <= criteria['max_delta']]
                
            # Apply richness/cheapness criteria if requested
            if 'iv_analysis' in criteria and criteria['iv_analysis']:
                # Update volatility surface
                self.vol_surface.update_surface(options_chain)
                
                # Analyze all options and add metrics
                analyzed = self.analyze_options_chain(filtered)
                
                # Apply rich/cheap filters
                if 'find_rich' in criteria and criteria['find_rich']:
                    analyzed = analyzed[analyzed['IsRich'] == True]
                    
                if 'find_cheap' in criteria and criteria['find_cheap']:
                    analyzed = analyzed[analyzed['IsCheap'] == True]
                    
                filtered = analyzed
                
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error filtering options: {e}")
            return options_chain