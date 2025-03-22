"""
SSVI (Surface Stochastic Volatility Inspired) Model

This module implements the SSVI parameterization for volatility surfaces, providing
tools for fitting, analyzing and identifying relative value opportunities in options markets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.optimize import minimize, differential_evolution
from datetime import datetime, timedelta
import logging

class SSVIModel:
    """
    Surface Stochastic Volatility Inspired (SSVI) model implementation.
    
    This class implements the SSVI parameterization for volatility surfaces, which provides
    a parsimonious and arbitrage-free representation of the implied volatility surface.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the SSVI model.
        
        Args:
            logger: Optional logger for tracking fitting and analysis
        """
        self.logger = logger or logging.getLogger(__name__)
        self.params = None
        self.param_history = []
        self.last_fit_date = None
        self.fit_quality = None
        
    def fit(self, 
            option_chain: pd.DataFrame, 
            underlying_price: float,
            method: str = 'global_then_local',
            max_iterations: int = 1000) -> Dict[str, float]:
        """
        Fit SSVI parameters to market data.
        
        Args:
            option_chain: DataFrame containing option data with at least:
                          Strike, IV (implied volatility), DTE (days to expiration), Type
            underlying_price: Price of the underlying asset
            method: Fitting method ('global_then_local', 'global', or 'local')
            max_iterations: Maximum number of iterations for the optimization
            
        Returns:
            dict: Fitted SSVI parameters
        """
        try:
            # Pre-process the data
            if option_chain.empty:
                self.logger.warning("Cannot fit SSVI: Empty option chain")
                return {}
                
            # Check for required columns
            required_cols = ['Strike', 'IV', 'DTE', 'Type']
            missing_cols = [col for col in required_cols if col not in option_chain.columns]
            if missing_cols:
                self.logger.warning(f"Missing required columns for SSVI fit: {missing_cols}")
                return {}
                
            # Prepare the data for fitting
            data = option_chain.copy()
            data['Moneyness'] = np.log(data['Strike'] / underlying_price)
            data['TimeToExpiry'] = data['DTE'] / 365.0  # Convert to years
            
            # Ensure we have positive values for IV
            data = data[data['IV'] > 0]
            
            # Group by expiration (we fit separate SSVI parameters for each expiry)
            param_sets = {}
            fit_quality = {}
            
            for dte, group in data.groupby('DTE'):
                if len(group) < 5:  # Need enough data points for a reliable fit
                    continue
                    
                # Fit parameters for this expiry
                time_to_expiry = dte / 365.0
                params, quality = self._fit_single_expiry(group, time_to_expiry, method, max_iterations)
                
                if params:
                    param_sets[dte] = params
                    fit_quality[dte] = quality
                    
            if not param_sets:
                self.logger.warning("SSVI fitting failed: No valid parameter sets found")
                return {}
                
            # Store the fit results
            self.params = param_sets
            self.fit_quality = fit_quality
            self.last_fit_date = datetime.now()
            
            # Add to parameter history
            self.param_history.append({
                'date': self.last_fit_date,
                'params': param_sets,
                'fit_quality': fit_quality
            })
            
            # Keep history to a reasonable size
            if len(self.param_history) > 100:
                self.param_history = self.param_history[-100:]
                
            # Return the overall parameters
            avg_params = self._average_params(param_sets)
            self.logger.info(f"SSVI fit completed with average parameters: {avg_params}")
            return avg_params
            
        except Exception as e:
            self.logger.error(f"Error fitting SSVI model: {e}")
            return {}
    
    def _fit_single_expiry(self, 
                          data: pd.DataFrame, 
                          time_to_expiry: float,
                          method: str,
                          max_iterations: int) -> Tuple[Dict[str, float], float]:
        """
        Fit SSVI parameters for a single expiration.
        
        Args:
            data: DataFrame filtered to a single expiration
            time_to_expiry: Time to expiry in years
            method: Fitting method
            max_iterations: Maximum iterations
            
        Returns:
            tuple: (params dict, fit quality)
        """
        # Initial parameter guesses
        # SSVI parameters: a (ATM vol), b (skew), rho (correlation), m (shift), sigma (curvature)
        initial_guess = {
            'a': np.median(data['IV']),  # ATM volatility level
            'b': 0.1,                    # Controls the skew
            'rho': -0.7,                 # Controls the correlation (-1 to 1)
            'm': 0.0,                    # Horizontal shift
            'sigma': 0.1                 # Controls the smile curvature
        }
        
        # Parameter bounds
        bounds = [
            (0.01, 1.0),    # a: ATM vol (1% to 100%)
            (0.001, 0.5),   # b: skew (constrained for no-arbitrage)
            (-0.99, 0.99),  # rho: correlation (-1 to 1, exclusive)
            (-0.5, 0.5),    # m: shift
            (0.01, 1.0)     # sigma: curvature
        ]
        
        # Setup the objective function for minimization
        def objective(params):
            a, b, rho, m, sigma = params
            
            # Calculate the w(k) function and implied variance v(k,t)
            total_squared_error = 0
            for _, row in data.iterrows():
                k = row['Moneyness']
                market_iv = row['IV']
                
                # Calculate SSVI implied volatility
                model_iv = self._ssvi_iv(k, time_to_expiry, a, b, rho, m, sigma)
                
                # Calculate squared error
                if not np.isnan(model_iv) and model_iv > 0:
                    squared_error = (model_iv - market_iv) ** 2
                    total_squared_error += squared_error
                else:
                    # Penalize invalid outputs heavily
                    total_squared_error += 100
                    
            return total_squared_error
            
        # Perform the optimization based on the method
        if method in ['global', 'global_then_local']:
            # Global optimization first
            result_global = differential_evolution(
                lambda x: objective([x[0], x[1], x[2], x[3], x[4]]),
                bounds,
                maxiter=max_iterations // 2,
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            
            if method == 'global':
                a, b, rho, m, sigma = result_global.x
                quality = 1.0 / (1.0 + result_global.fun)  # Convert error to quality metric
            else:
                # Continue with local optimization
                result_local = minimize(
                    lambda x: objective([x[0], x[1], x[2], x[3], x[4]]),
                    result_global.x,
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': max_iterations // 2}
                )
                
                a, b, rho, m, sigma = result_local.x
                quality = 1.0 / (1.0 + result_local.fun)  # Convert error to quality metric
        else:
            # Local optimization only
            result_local = minimize(
                lambda x: objective([x[0], x[1], x[2], x[3], x[4]]),
                [initial_guess['a'], initial_guess['b'], initial_guess['rho'], 
                 initial_guess['m'], initial_guess['sigma']],
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': max_iterations}
            )
            
            a, b, rho, m, sigma = result_local.x
            quality = 1.0 / (1.0 + result_local.fun)  # Convert error to quality metric
            
        # Validate the parameters to avoid arbitrage
        if not self._is_valid_parametrization(a, b, rho, time_to_expiry):
            self.logger.warning(f"Invalid SSVI parameters for DTE={time_to_expiry*365}")
            return {}, 0.0
            
        # Return the fitted parameters
        params = {
            'a': float(a),
            'b': float(b),
            'rho': float(rho),
            'm': float(m),
            'sigma': float(sigma),
            'time_to_expiry': time_to_expiry
        }
        
        return params, float(quality)
    
    def _ssvi_iv(self, 
                k: float, 
                t: float, 
                a: float, 
                b: float, 
                rho: float, 
                m: float = 0.0, 
                sigma: float = 0.1) -> float:
        """
        Calculate implied volatility using the SSVI parameterization.
        
        Args:
            k: Log-moneyness (log(K/S))
            t: Time to expiry in years
            a: ATM volatility parameter
            b: Skew parameter
            rho: Correlation parameter
            m: Shift parameter
            sigma: Curvature parameter
            
        Returns:
            float: SSVI implied volatility
        """
        # Adjust log-moneyness for shift
        k_adj = k - m
        
        # Total variance
        theta = a * t
        
        # SSVI variance parameterization
        w_k = theta * (1 + b * rho * k_adj + b * np.sqrt((k_adj + rho)**2 + (1 - rho**2)))
        
        # Convert variance to volatility
        if w_k <= 0:
            return np.nan
            
        # Add the curvature adjustment
        w_k *= (1 + sigma * k_adj**2)
        
        iv = np.sqrt(w_k / t)
        return iv
    
    def _is_valid_parametrization(self, a: float, b: float, rho: float, t: float) -> bool:
        """
        Check if SSVI parameters satisfy no-arbitrage conditions.
        
        Args:
            a: ATM volatility parameter
            b: Skew parameter
            rho: Correlation parameter
            t: Time to expiry
            
        Returns:
            bool: True if parameterization is valid
        """
        # Conditions from Gatheral & Jacquier (2014)
        if abs(rho) >= 1.0:
            return False
            
        if b < 0:
            return False
            
        # Check the key no-arbitrage condition
        if b * (1 + abs(rho)) >= 1.0:
            return False
            
        return True
    
    def _average_params(self, param_sets: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate average parameters across expirations.
        
        Args:
            param_sets: Dictionary of parameters by DTE
            
        Returns:
            dict: Average parameters
        """
        if not param_sets:
            return {}
            
        # Initialize aggregates
        param_sums = {'a': 0, 'b': 0, 'rho': 0, 'm': 0, 'sigma': 0}
        count = 0
        
        # Sum up parameters
        for dte, params in param_sets.items():
            for key in param_sums:
                if key in params:
                    param_sums[key] += params[key]
            count += 1
            
        # Calculate averages
        avg_params = {k: v / count for k, v in param_sums.items()}
        
        return avg_params
    
    def calculate_zscore(self, 
                        k: float, 
                        t: float, 
                        market_iv: float, 
                        history_window: int = 30) -> Optional[float]:
        """
        Calculate z-score of an option's IV relative to SSVI model.
        
        Args:
            k: Log-moneyness (log(K/S))
            t: Time to expiry in years
            market_iv: Market-observed implied volatility
            history_window: Number of historical fits to use
            
        Returns:
            float: Z-score or None if calculation fails
        """
        if not self.params:
            return None
            
        # Find the closest expiry
        closest_dte = min(self.params.keys(), key=lambda x: abs(x - t * 365))
        if abs(closest_dte - t * 365) > 7:  # More than a week difference
            self.logger.warning(f"No close expiry match for t={t} (closest: {closest_dte/365})")
            return None
            
        params = self.params[closest_dte]
        
        # Calculate model IV
        model_iv = self._ssvi_iv(k, t, params['a'], params['b'], params['rho'], 
                                params.get('m', 0), params.get('sigma', 0.1))
        
        if np.isnan(model_iv) or model_iv <= 0:
            return None
            
        # Get historical fits for this expiry
        historical_params = []
        for hist in self.param_history[-history_window:]:
            if closest_dte in hist['params']:
                historical_params.append(hist['params'][closest_dte])
                
        if len(historical_params) < 5:  # Need enough history
            self.logger.warning(f"Insufficient parameter history for DTE={closest_dte}")
            return None
            
        # Calculate historical IVs at this point
        historical_ivs = []
        for p in historical_params:
            iv = self._ssvi_iv(k, t, p['a'], p['b'], p['rho'], 
                              p.get('m', 0), p.get('sigma', 0.1))
            if not np.isnan(iv) and iv > 0:
                historical_ivs.append(iv)
                
        if not historical_ivs:
            return None
            
        # Calculate z-score
        mean_iv = np.mean(historical_ivs)
        std_iv = np.std(historical_ivs)
        
        if std_iv == 0:
            return None
            
        zscore = (market_iv - mean_iv) / std_iv
        return zscore
    
    def identify_rv_opportunities(self, 
                                option_chain: pd.DataFrame, 
                                underlying_price: float,
                                zscore_threshold: float = 1.5) -> pd.DataFrame:
        """
        Identify relative value opportunities based on SSVI model.
        
        Args:
            option_chain: DataFrame with option data
            underlying_price: Current price of underlying asset
            zscore_threshold: Threshold for considering an option rich/cheap
            
        Returns:
            DataFrame: Filtered options with z-scores and RV classifications
        """
        if not self.params or option_chain.empty:
            return pd.DataFrame()
            
        # Prepare data
        data = option_chain.copy()
        data['Moneyness'] = np.log(data['Strike'] / underlying_price)
        data['TimeToExpiry'] = data['DTE'] / 365.0
        data['ModelIV'] = np.nan
        data['ZScore'] = np.nan
        data['RVSignal'] = ''
        
        # Calculate model IVs and z-scores
        for idx, row in data.iterrows():
            k = row['Moneyness']
            t = row['TimeToExpiry']
            market_iv = row['IV']
            
            # Get the closest expiry
            closest_dte = min(self.params.keys(), key=lambda x: abs(x - row['DTE']))
            params = self.params[closest_dte]
            
            # Calculate model IV
            model_iv = self._ssvi_iv(k, t, params['a'], params['b'], params['rho'], 
                                   params.get('m', 0), params.get('sigma', 0.1))
            
            if not np.isnan(model_iv) and model_iv > 0:
                data.at[idx, 'ModelIV'] = model_iv
                
                # Calculate z-score
                zscore = self.calculate_zscore(k, t, market_iv)
                if zscore is not None:
                    data.at[idx, 'ZScore'] = zscore
                    
                    # Classify as rich/cheap
                    if zscore > zscore_threshold:
                        data.at[idx, 'RVSignal'] = 'RICH'
                    elif zscore < -zscore_threshold:
                        data.at[idx, 'RVSignal'] = 'CHEAP'
                    else:
                        data.at[idx, 'RVSignal'] = 'FAIR'
        
        # Filter for valid signals
        valid_signals = data[data['RVSignal'] != ''].copy()
        
        # Add IV difference
        valid_signals['IVDiff'] = valid_signals['IV'] - valid_signals['ModelIV']
        
        return valid_signals
    
    def get_param_time_series(self, param_name: str) -> pd.DataFrame:
        """
        Get time series of a specific SSVI parameter.
        
        Args:
            param_name: Parameter name ('a', 'b', 'rho', etc.)
            
        Returns:
            DataFrame: Time series of the parameter values
        """
        if not self.param_history:
            return pd.DataFrame()
            
        # Extract parameter across time
        data = []
        for entry in self.param_history:
            date = entry['date']
            
            # For each expiry in this historical entry
            for dte, params in entry['params'].items():
                if param_name in params:
                    data.append({
                        'Date': date,
                        'DTE': dte,
                        'Parameter': param_name,
                        'Value': params[param_name]
                    })
        
        if not data:
            return pd.DataFrame()
            
        return pd.DataFrame(data)
    
    def calculate_param_zscores(self, window: int = 30) -> Dict[str, Dict[int, float]]:
        """
        Calculate z-scores for current parameters relative to history.
        
        Args:
            window: Historical window for z-score calculation
            
        Returns:
            dict: Z-scores by parameter and expiry
        """
        if not self.params or len(self.param_history) < window:
            return {}
            
        # Parameter names to analyze
        param_names = ['a', 'b', 'rho', 'm', 'sigma']
        
        # Calculate z-scores for each parameter and expiry
        zscores = {param: {} for param in param_names}
        
        for param_name in param_names:
            for dte, current_params in self.params.items():
                if param_name not in current_params:
                    continue
                    
                # Get historical values
                historical_values = []
                for entry in self.param_history[-window:]:
                    if dte in entry['params'] and param_name in entry['params'][dte]:
                        historical_values.append(entry['params'][dte][param_name])
                
                if len(historical_values) < 5:  # Need enough history
                    continue
                    
                # Calculate z-score
                mean_value = np.mean(historical_values)
                std_value = np.std(historical_values)
                
                if std_value == 0:
                    continue
                    
                current_value = current_params[param_name]
                zscore = (current_value - mean_value) / std_value
                
                zscores[param_name][dte] = zscore
        
        return zscores 