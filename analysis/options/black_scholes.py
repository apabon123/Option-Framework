"""
Black-Scholes Option Pricing Model

This module provides a Black-Scholes option pricing model implementation for options valuation.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Union, Optional, Literal
from datetime import datetime, timedelta

class BlackScholesModel:
    """
    Black-Scholes option pricing model for European options.
    
    This class implements the Black-Scholes model for pricing European options and
    calculating option Greeks.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the Black-Scholes model.
        
        Args:
            risk_free_rate: Risk-free interest rate (default: 0.02 or 2%)
        """
        self.risk_free_rate = risk_free_rate
        
    def price_option(
        self, 
        option_type: Literal['CALL', 'PUT'], 
        underlying_price: float, 
        strike_price: float, 
        days_to_expiry: float, 
        volatility: float,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate the option price using Black-Scholes model.
        
        Args:
            option_type: 'CALL' or 'PUT'
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            days_to_expiry: Number of days until expiration
            volatility: Implied volatility (as a decimal, e.g., 0.20 for 20%)
            risk_free_rate: Risk-free interest rate (as a decimal)
            
        Returns:
            float: Option price according to Black-Scholes
        """
        # Use instance rate if not provided
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        # Convert days to years
        time_to_expiry = days_to_expiry / 365.0
        
        if time_to_expiry <= 0:
            # Handle expired options
            if option_type == 'CALL':
                return max(0, underlying_price - strike_price)
            else:
                return max(0, strike_price - underlying_price)
                
        # Black-Scholes formula components
        d1 = (np.log(underlying_price / strike_price) + 
              (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
             
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        if option_type == 'CALL':
            price = underlying_price * norm.cdf(d1) - \
                    strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:  # PUT
            price = strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - \
                    underlying_price * norm.cdf(-d1)
                    
        return price
        
    def calculate_greeks(
        self, 
        option_type: Literal['CALL', 'PUT'], 
        underlying_price: float, 
        strike_price: float, 
        days_to_expiry: float, 
        volatility: float,
        risk_free_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate option Greeks using Black-Scholes model.
        
        Args:
            option_type: 'CALL' or 'PUT'
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            days_to_expiry: Number of days until expiration
            volatility: Implied volatility (as a decimal)
            risk_free_rate: Risk-free interest rate (as a decimal)
            
        Returns:
            dict: Dictionary containing option Greeks (delta, gamma, theta, vega, rho)
        """
        # Use instance rate if not provided
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        # Convert days to years
        time_to_expiry = days_to_expiry / 365.0
        
        # Handle expired or nearly expired options
        if time_to_expiry <= 0.0001:  # Small positive value to avoid numerical issues
            if option_type == 'CALL':
                intrinsic = max(0, underlying_price - strike_price)
                return {
                    'delta': 1.0 if intrinsic > 0 else 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'rho': 0.0
                }
            else:
                intrinsic = max(0, strike_price - underlying_price)
                return {
                    'delta': -1.0 if intrinsic > 0 else 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'rho': 0.0
                }
                
        # Black-Scholes formula components
        d1 = (np.log(underlying_price / strike_price) + 
              (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
             
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        # Common calculations
        sqrt_time = np.sqrt(time_to_expiry)
        nd1 = norm.pdf(d1)
        exp_factor = np.exp(-risk_free_rate * time_to_expiry)
        
        # Calculate Greeks
        if option_type == 'CALL':
            delta = norm.cdf(d1)
            rho = strike_price * time_to_expiry * exp_factor * norm.cdf(d2) / 100  # Per 1% change
        else:  # PUT
            delta = norm.cdf(d1) - 1
            rho = -strike_price * time_to_expiry * exp_factor * norm.cdf(-d2) / 100  # Per 1% change
        
        gamma = nd1 / (underlying_price * volatility * sqrt_time)
        vega = underlying_price * sqrt_time * nd1 / 100  # Per 1% change
        
        # Theta (per calendar day)
        theta_part1 = -(underlying_price * volatility * nd1) / (2 * sqrt_time)
        if option_type == 'CALL':
            theta_part2 = -risk_free_rate * strike_price * exp_factor * norm.cdf(d2)
        else:  # PUT
            theta_part2 = risk_free_rate * strike_price * exp_factor * norm.cdf(-d2)
            
        theta = (theta_part1 + theta_part2) / 365  # Per calendar day
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def calculate_implied_volatility(
        self,
        option_type: Literal['CALL', 'PUT'],
        underlying_price: float,
        strike_price: float,
        days_to_expiry: float,
        option_price: float,
        risk_free_rate: Optional[float] = None,
        precision: float = 0.0001,
        max_iterations: int = 100
    ) -> Optional[float]:
        """
        Calculate implied volatility using iterative approach.
        
        Args:
            option_type: 'CALL' or 'PUT'
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            days_to_expiry: Number of days until expiration
            option_price: Market price of the option
            risk_free_rate: Risk-free interest rate
            precision: Desired precision of the result
            max_iterations: Maximum number of iterations
            
        Returns:
            float: Implied volatility or None if it couldn't be calculated
        """
        # Use instance rate if not provided
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        # Convert days to years
        time_to_expiry = days_to_expiry / 365.0
        
        if time_to_expiry <= 0:
            return None
            
        # Check for intrinsic value cases
        if option_type == 'CALL':
            intrinsic = max(0, underlying_price - strike_price)
            if option_price <= intrinsic:
                return 0.0001  # Minimum volatility
                
        else:  # PUT
            intrinsic = max(0, strike_price - underlying_price)
            if option_price <= intrinsic:
                return 0.0001  # Minimum volatility
                
        # Initial volatility guess based on price level
        vol_guess = 0.3  # Start with 30% as an initial guess
        
        # Binary search approach
        vol_min = 0.0001
        vol_max = 5.0  # Max 500% volatility
        
        for _ in range(max_iterations):
            price = self.price_option(
                option_type, 
                underlying_price, 
                strike_price, 
                days_to_expiry, 
                vol_guess, 
                risk_free_rate
            )
            
            price_diff = price - option_price
            
            if abs(price_diff) < precision:
                return vol_guess
                
            if price_diff > 0:
                vol_max = vol_guess
            else:
                vol_min = vol_guess
                
            vol_guess = (vol_min + vol_max) / 2
            
        # Return best guess if we reached max iterations
        return vol_guess 