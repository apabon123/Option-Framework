#!/usr/bin/env python
"""
Margin Calculator Test Script

This script systematically tests all margin calculators in the system with various
option scenarios to identify potential issues with margin calculations.

The results are logged to a file in the output directory specified in the config.
"""

from typing import Dict, List, Tuple, Any, Union, Optional
import logging
import sys
import os
import time
import yaml
from datetime import datetime
from pathlib import Path
import pandas as pd

# Set up import path
script_path = Path(os.path.abspath(__file__))
project_root = script_path.parent.parent.parent  # Go up three levels to get to the project root
sys.path.insert(0, str(project_root))

# Now import the project modules
from core.margin import MarginCalculator, OptionMarginCalculator, SPANMarginCalculator
from core.position import Position, OptionPosition

# Load configuration
def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Load the configuration file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Setup logging
def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to file and console"""
    # Make sure we have a valid output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except (PermissionError, OSError) as e:
        # Fall back to local output directory
        print(f"Warning: Could not access configured output directory: {e}")
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Using fallback output directory: {output_dir}")
    
    # Create logger
    logger = logging.getLogger('margin_test')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Log file path
    log_file = os.path.join(output_dir, f'margin_calculator_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Console handler with simplified formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging configured. Log file: {log_file}")
    return logger

def create_test_options() -> List[Dict[str, Any]]:
    """
    Create a diverse set of test option scenarios.
    Returns a list of dictionaries with option parameters.
    """
    spy_price = 480.0  # Current SPY price (approximate)
    
    test_options = [
        # Call options at various strikes and expirations
        {
            'name': "ATM Call Short Term",
            'symbol': "SPY240510C00480000",
            'option_type': 'C',
            'strike': 480.0,
            'underlying_price': spy_price,
            'price': 12.50,
            'delta': 0.52,
            'gamma': 0.05,
            'theta': -0.4,
            'vega': 0.3,
            'dte': 14,
            'is_short': True,
        },
        {
            'name': "OTM Call Short Term",
            'symbol': "SPY240510C00490000",
            'option_type': 'C',
            'strike': 490.0,
            'underlying_price': spy_price,
            'price': 6.50,
            'delta': 0.30,
            'gamma': 0.04,
            'theta': -0.35,
            'vega': 0.25,
            'dte': 14,
            'is_short': True,
        },
        {
            'name': "ITM Call Short Term",
            'symbol': "SPY240510C00470000",
            'option_type': 'C',
            'strike': 470.0,
            'underlying_price': spy_price,
            'price': 18.50,
            'delta': 0.75,
            'gamma': 0.03,
            'theta': -0.3,
            'vega': 0.2,
            'dte': 14,
            'is_short': True,
        },
        # Put options at various strikes and expirations
        {
            'name': "ATM Put Short Term",
            'symbol': "SPY240510P00480000",
            'option_type': 'P',
            'strike': 480.0,
            'underlying_price': spy_price,
            'price': 11.50,
            'delta': -0.48,
            'gamma': 0.04,
            'theta': -0.38,
            'vega': 0.28,
            'dte': 14,
            'is_short': True,
        },
        {
            'name': "OTM Put Short Term",
            'symbol': "SPY240510P00470000",
            'option_type': 'P',
            'strike': 470.0,
            'underlying_price': spy_price,
            'price': 7.20,
            'delta': -0.28,
            'gamma': 0.03,
            'theta': -0.32,
            'vega': 0.22,
            'dte': 14,
            'is_short': True,
        },
        # Medium term options
        {
            'name': "OTM Call Medium Term",
            'symbol': "SPY240621C00490000",
            'option_type': 'C',
            'strike': 490.0,
            'underlying_price': spy_price,
            'price': 12.80,
            'delta': 0.42,
            'gamma': 0.03,
            'theta': -0.22,
            'vega': 0.40,
            'dte': 60,
            'is_short': True,
        },
        {
            'name': "OTM Put Medium Term",
            'symbol': "SPY240621P00460000",
            'option_type': 'P',
            'strike': 460.0,
            'underlying_price': spy_price,
            'price': 12.20,
            'delta': -0.32,
            'gamma': 0.02,
            'theta': -0.18,
            'vega': 0.35,
            'dte': 60,
            'is_short': True,
        },
        # Long term options
        {
            'name': "OTM Call Long Term",
            'symbol': "SPY240920C00500000",
            'option_type': 'C',
            'strike': 500.0,
            'underlying_price': spy_price,
            'price': 22.40,
            'delta': 0.45,
            'gamma': 0.015,
            'theta': -0.12,
            'vega': 0.60,
            'dte': 180,
            'is_short': True,
        },
        {
            'name': "OTM Put Long Term",
            'symbol': "SPY240920P00450000",
            'option_type': 'P',
            'strike': 450.0,
            'underlying_price': spy_price,
            'price': 18.80,
            'delta': -0.30,
            'gamma': 0.01,
            'theta': -0.10,
            'vega': 0.55,
            'dte': 180,
            'is_short': True,
        },
        # Low-price options (where the price*100 vs margin issue is most likely)
        {
            'name': "Far OTM Call",
            'symbol': "SPY240510C00530000",
            'option_type': 'C',
            'strike': 530.0,
            'underlying_price': spy_price,
            'price': 0.55,
            'delta': 0.06,
            'gamma': 0.01,
            'theta': -0.08,
            'vega': 0.08,
            'dte': 14,
            'is_short': True,
        },
        {
            'name': "Far OTM Put",
            'symbol': "SPY240510P00430000",
            'option_type': 'P',
            'strike': 430.0,
            'underlying_price': spy_price,
            'price': 0.35,
            'delta': -0.04,
            'gamma': 0.005,
            'theta': -0.06,
            'vega': 0.05,
            'dte': 14,
            'is_short': True,
        },
        # Weekly options
        {
            'name': "Weekly ATM Call",
            'symbol': "SPY240412C00480000",
            'option_type': 'C',
            'strike': 480.0,
            'underlying_price': spy_price,
            'price': 6.20,
            'delta': 0.51,
            'gamma': 0.07,
            'theta': -0.65,
            'vega': 0.18,
            'dte': 7,
            'is_short': True,
        }
    ]
    
    return test_options

def create_option_position(option_data: Dict[str, Any], contracts: int = 1) -> OptionPosition:
    """Create an OptionPosition object from option data"""
    # Extract necessary data
    symbol = option_data['symbol']
    price = option_data['price']
    is_short = option_data.get('is_short', True)
    
    # Create option position
    position = OptionPosition(
        symbol=symbol,
        contracts=contracts,
        entry_price=price,
        current_price=price,
        is_short=is_short,
        option_data={
            'UnderlyingSymbol': 'SPY',
            'UnderlyingPrice': option_data['underlying_price'],
            'Strike': option_data['strike'],
            'Type': option_data['option_type'],
            'Expiration': None  # We don't need the actual date for this test
        }
    )
    
    # Set option-specific data
    position.option_type = option_data['option_type']
    position.strike = option_data['strike']
    position.underlying_price = option_data['underlying_price']
    
    # Set greeks
    position.current_delta = option_data['delta']
    position.current_gamma = option_data['gamma']
    position.current_theta = option_data['theta']
    position.current_vega = option_data['vega']
    
    return position

def test_margin_calculator(calculator: MarginCalculator, options: List[Dict[str, Any]], logger: logging.Logger) -> None:
    """
    Test a margin calculator with various option scenarios.
    
    Args:
        calculator: The margin calculator to test
        options: List of option data dictionaries
        logger: Logger instance
    """
    calculator_name = type(calculator).__name__
    logger.info(f"===== Testing {calculator_name} =====")
    
    # Create a DataFrame to hold results
    results = []
    
    # Test each option
    for option in options:
        # Log option details
        logger.info(f"\nTesting {option['name']} ({option['symbol']})")
        logger.info(f"  Price: ${option['price']:.2f}, Delta: {option['delta']:.2f}")
        logger.info(f"  Strike: ${option['strike']:.2f}, Underlying: ${option['underlying_price']:.2f}")
        
        # Create position
        position = create_option_position(option)
        
        # Calculate margin
        margin = calculator.calculate_position_margin(position)
        
        # Log results
        logger.info(f"  Calculated margin: ${margin:.2f}")
        
        # Check if margin is suspiciously low
        option_premium = option['price'] * 100
        
        if margin < option_premium:
            logger.warning(f"  WARNING: Calculated margin (${margin:.2f}) is LESS THAN option premium (${option_premium:.2f})")
            logger.warning(f"  This indicates a potential issue with the margin calculation!")
        else:
            logger.info(f"  Margin to option premium ratio: {margin / option_premium:.2f}x")
            
        # For low-priced options, check if missing multiplier is likely
        if option['price'] < 5.0 and margin < option_premium:
            potential_fixed_margin = margin * 100
            logger.warning(f"  Low-priced option: If missing multiplier, corrected margin would be: ${potential_fixed_margin:.2f}")
        
        # Calculate expected margin range based on typical requirements
        min_expected = max(option_premium, option['underlying_price'] * 0.1)  # Premium or 10% of underlying
        max_expected = option['underlying_price'] * 0.2  # 20% of underlying
        
        if margin < min_expected:
            logger.warning(f"  Margin below minimum expected (${min_expected:.2f})")
        elif margin > max_expected:
            logger.warning(f"  Margin above maximum expected (${max_expected:.2f})")
        else:
            logger.info(f"  Margin within expected range: ${min_expected:.2f} - ${max_expected:.2f}")
        
        # Store results
        results.append({
            'Option': option['name'],
            'Symbol': option['symbol'],
            'Price': option['price'],
            'Premium': option_premium,
            'Margin': margin,
            'MarginToPremiumRatio': margin / option_premium,
            'BelowPremium': margin < option_premium,
            'Calculator': calculator_name
        })
    
    # Convert to DataFrame and log summary
    df = pd.DataFrame(results)
    
    # Log summary statistics
    logger.info(f"\n===== {calculator_name} Summary =====")
    logger.info(f"Average margin to premium ratio: {df['MarginToPremiumRatio'].mean():.2f}x")
    logger.info(f"Minimum margin to premium ratio: {df['MarginToPremiumRatio'].min():.2f}x")
    logger.info(f"Maximum margin to premium ratio: {df['MarginToPremiumRatio'].max():.2f}x")
    logger.info(f"Options with margin below premium: {df['BelowPremium'].sum()} out of {len(df)}")
    
    # Create and log a summary table
    logger.info("\nOptions with Potential Margin Issues:")
    problem_df = df[df['BelowPremium']]
    if not problem_df.empty:
        for _, row in problem_df.iterrows():
            logger.info(f"  {row['Symbol']} (${row['Price']:.2f}): Margin=${row['Margin']:.2f}, "
                       f"Premium=${row['Premium']:.2f}, Ratio={row['MarginToPremiumRatio']:.2f}x")
    else:
        logger.info("  None - all margins are above option premiums")

def test_hedged_margins(logger: logging.Logger = None):
    """Test calculation of portfolio margin for hedged positions.
    
    This test verifies that the SPANMarginCalculator correctly applies hedging
    benefits when positions have delta values in opposite directions.
    """
    # Use default logger if none provided
    if logger is None:
        logger = logging.getLogger("margin_test")
    
    logger.info("="*80)
    logger.info("TESTING HEDGED PORTFOLIO MARGIN CALCULATION")
    logger.info("="*80)
    
    # Create test option data directly
    option_name = "SPY240510P00470000"  # OTM Put
    option_details = {
        "UnderlyingSymbol": "SPY",
        "Strike": 470.0, 
        "Type": "P",
        "Delta": -0.28,
        "Gamma": 0.03,
        "Theta": -0.25,
        "Vega": 0.15,
        "UnderlyingPrice": 480.0,
        "LastPrice": 7.20
    }
    option_price = option_details["LastPrice"]
    option_delta = option_details["Delta"]
    option_is_put = option_details["Type"] == "P"
    
    logger.info(f"Testing with option: {option_name}")
    logger.info(f"Option price: ${option_price}")
    logger.info(f"Option delta: {option_delta}")
    logger.info(f"Option type: {'PUT' if option_is_put else 'CALL'}")
    logger.info("")
    
    # Create an option position
    option_position = OptionPosition(
        symbol=option_name,
        option_data=option_details,
        contracts=1,
        entry_price=option_price,
        current_price=option_price,
        is_short=False  # Long option
    )
    
    # Make sure underlying price is properly set directly on the position
    option_position.underlying_price = option_details["UnderlyingPrice"]
    
    # Get the actual delta calculated by the position
    actual_delta = option_position.current_delta
    logger.info(f"Position delta after creation: {actual_delta}")
    
    # Calculate delta-neutral hedge (opposite sign)
    # For proper hedging, hedge delta should have OPPOSITE sign of option delta
    hedge_delta = -actual_delta * 100  # Opposite sign, 100 shares per contract
    hedge_is_short = hedge_delta < 0
    
    # Create a stock position to hedge the option
    hedge_position = Position(
        symbol="SPY",
        contracts=abs(hedge_delta),  # Number of shares
        entry_price=option_details["UnderlyingPrice"],
        current_price=option_details["UnderlyingPrice"],
        is_short=hedge_is_short
    )
    
    # Calculate stock position delta
    stock_delta = hedge_delta  # Already calculated above
    
    logger.info(f"Option position delta: {actual_delta * 100}")
    logger.info(f"Stock hedge delta: {stock_delta}")
    logger.info(f"Net delta: {actual_delta * 100 + stock_delta}")
    logger.info(f"Stock position is {'SHORT' if hedge_is_short else 'LONG'}")
    logger.info("")
    
    # Create the margin calculator
    margin_calculator = SPANMarginCalculator(
        max_leverage=12.0,
        volatility_multiplier=1.0,
        initial_margin_percentage=0.1,
        maintenance_margin_percentage=0.07,
        hedge_credit_rate=0.8,  # 80% credit for hedged positions
        logger=logger
    )
    
    # Set logger's level to DEBUG to capture detailed margin calculation steps
    original_level = logger.level
    logger.setLevel(logging.DEBUG)
    
    # Calculate standalone margins
    option_margin = margin_calculator.calculate_position_margin(option_position)
    stock_margin = margin_calculator.calculate_position_margin(hedge_position)
    total_standalone_margin = option_margin + stock_margin
    
    logger.info(f"Option position margin: ${option_margin:.2f}")
    logger.info(f"Stock position margin: ${stock_margin:.2f}")
    logger.info(f"Total standalone margin: ${total_standalone_margin:.2f}")
    logger.info("")
    
    # Calculate portfolio margin with positions as a dictionary
    positions = {
        option_name: option_position,
        "SPY": hedge_position
    }
    
    # Display the hedge_credit_rate setting
    logger.info(f"Hedge credit rate: {margin_calculator.hedge_credit_rate:.2f}")
    logger.info(f"Initial margin percentage: {margin_calculator.initial_margin_percentage:.4f}")
    logger.info("")
    
    # Add attribute to ensure SPY is recognized as the underlying
    option_position.underlying = "SPY"
    
    # Calculate portfolio margin
    portfolio_margin_result = margin_calculator.calculate_portfolio_margin(positions)
    
    # Reset logger level
    logger.setLevel(original_level)
    
    # Extract margin results
    if isinstance(portfolio_margin_result, dict):
        total_margin = portfolio_margin_result.get("total_margin", 0)
        hedging_benefits = portfolio_margin_result.get("hedging_benefits", 0)
    else:
        total_margin = portfolio_margin_result
        hedging_benefits = 0
    
    logger.info(f"Portfolio margin: ${total_margin:.2f}")
    logger.info(f"Hedging benefits: ${hedging_benefits:.2f}")
    logger.info(f"Reduction from standalone: ${total_standalone_margin - total_margin:.2f}")
    logger.info(f"Percentage reduction: {(hedging_benefits / total_standalone_margin) * 100:.2f}%")
    logger.info("")
    
    # Calculate what we'd expect from a perfect hedge
    # With delta_hedge_quality = 1.0 and hedge_credit_rate = 0.8,
    # the expected offset factor should be 0.8
    # The hedging benefit should be applied to the smaller of the two positions
    min_position_margin = min(option_margin, stock_margin)
    max_expected_benefit = min_position_margin * margin_calculator.hedge_credit_rate
    
    logger.info(f"Expected hedging benefit analysis:")
    logger.info(f"  Perfect hedge quality (net delta = 0): 1.0")
    logger.info(f"  Hedge credit rate: {margin_calculator.hedge_credit_rate}")
    logger.info(f"  Expected offset factor: {margin_calculator.hedge_credit_rate * 1.0}")
    logger.info(f"  Smaller position margin: ${min_position_margin:.2f}")
    logger.info(f"  Maximum expected benefit: ${max_expected_benefit:.2f}")
    logger.info(f"  Actual benefit: ${hedging_benefits:.2f}")
    logger.info(f"  Pct of max benefit realized: {(hedging_benefits / max_expected_benefit) * 100 if max_expected_benefit > 0 else 0:.2f}%")
    logger.info("")
    
    # Verify hedging benefits
    if hedging_benefits > 0:
        logger.info("✅ HEDGING BENEFITS DETECTED")
    else:
        logger.info("❌ NO HEDGING BENEFITS DETECTED")
    
    # Check that portfolio margin is less than standalone margin
    if total_margin < total_standalone_margin:
        logger.info("✅ PORTFOLIO MARGIN IS LESS THAN STANDALONE MARGIN")
    else:
        logger.info("❌ PORTFOLIO MARGIN IS NOT LESS THAN STANDALONE MARGIN")
    
    # Run a second test with a modified option position that uses a higher margin
    # This will help identify if there's an issue with the specific position values
    logger.info("="*80)
    logger.info("SECOND TEST: INCREASED OPTION MARGIN")
    logger.info("="*80)
    
    # Modify option price to increase its margin
    option_position.current_price = 20.0  # Much higher price
    option_position.avg_entry_price = 20.0
    
    # Recalculate margins
    option_margin = margin_calculator.calculate_position_margin(option_position)
    total_standalone_margin = option_margin + stock_margin
    
    logger.info(f"Option position margin (increased): ${option_margin:.2f}")
    logger.info(f"Stock position margin: ${stock_margin:.2f}")
    logger.info(f"Total standalone margin: ${total_standalone_margin:.2f}")
    
    # Reset logger level to DEBUG for detailed output
    logger.setLevel(logging.DEBUG)
    
    # Recalculate portfolio margin
    portfolio_margin_result = margin_calculator.calculate_portfolio_margin(positions)
    
    # Reset logger level
    logger.setLevel(original_level)
    
    # Extract margin results
    if isinstance(portfolio_margin_result, dict):
        total_margin = portfolio_margin_result.get("total_margin", 0)
        hedging_benefits = portfolio_margin_result.get("hedging_benefits", 0)
    else:
        total_margin = portfolio_margin_result
        hedging_benefits = 0
    
    logger.info(f"Portfolio margin: ${total_margin:.2f}")
    logger.info(f"Hedging benefits: ${hedging_benefits:.2f}")
    logger.info(f"Reduction from standalone: ${total_standalone_margin - total_margin:.2f}")
    logger.info(f"Percentage reduction: {(hedging_benefits / total_standalone_margin) * 100:.2f}%")
    
    # Calculate what we'd expect with the new margins
    min_position_margin = min(option_margin, stock_margin)
    max_expected_benefit = min_position_margin * margin_calculator.hedge_credit_rate
    
    logger.info(f"Expected hedging benefit (second test):")
    logger.info(f"  Smaller position margin: ${min_position_margin:.2f}")
    logger.info(f"  Maximum expected benefit: ${max_expected_benefit:.2f}")
    logger.info(f"  Actual benefit: ${hedging_benefits:.2f}")
    logger.info(f"  Pct of max benefit realized: {(hedging_benefits / max_expected_benefit) * 100 if max_expected_benefit > 0 else 0:.2f}%")
    
    logger.info("="*80)

def test_initial_margin_percentage_impact(options: List[Dict[str, Any]], logger: logging.Logger) -> None:
    """
    Test how different initial_margin_percentage values affect margin calculations
    
    Args:
        options: List of option data dictionaries
        logger: Logger instance
    """
    logger.info("\n===== Testing Impact of initial_margin_percentage on SPAN Calculator =====")
    
    # Select a subset of options to test
    test_options = [
        options[0],  # ATM Call
        options[3],  # ATM Put
        options[9]   # Far OTM Put (low price)
    ]
    
    # Test values between 0.005 (0.5%) and 0.2 (20%)
    test_percentages = [0.005, 0.01, 0.05, 0.1, 0.2]
    
    for option in test_options:
        logger.info(f"\nTesting {option['name']} ({option['symbol']}, Price: ${option['price']:.2f})")
        
        # Create the position once
        position = create_option_position(option)
        
        # Test with different percentage values
        for pct in test_percentages:
            # Create calculator with specific percentage
            calculator = SPANMarginCalculator(initial_margin_percentage=pct)
            
            # Calculate margin
            margin = calculator.calculate_position_margin(position)
            
            # Calculate premium
            premium = option['price'] * 100
            
            # Log results
            logger.info(f"  initial_margin_percentage={pct:.3f} ({pct*100:.1f}%): "
                       f"Margin=${margin:.2f}, Premium=${premium:.2f}, "
                       f"Ratio to Premium={margin/premium:.2f}x")
            
            # Check if margin is below premium
            if margin < premium:
                logger.warning(f"    WARNING: Margin is below option premium!")

def test_margin_leverage_impact(options: List[Dict[str, Any]], logger: logging.Logger) -> None:
    """
    Test how different max_leverage values affect margin calculations
    
    Args:
        options: List of option data dictionaries
        logger: Logger instance
    """
    logger.info("\n===== Testing Impact of max_leverage on Margin Calculators =====")
    
    # Select a subset of options to test
    test_options = [
        options[0],  # ATM Call
        options[3],  # ATM Put
        options[9]   # Far OTM Put (low price)
    ]
    
    # Test different leverage values
    test_leverage = [1.0, 2.0, 4.0, 8.0, 12.0]
    
    for option in test_options:
        logger.info(f"\nTesting {option['name']} ({option['symbol']}, Price: ${option['price']:.2f})")
        position = create_option_position(option)
        premium = option['price'] * 100
        
        # Test both calculator types
        for calculator_class in [OptionMarginCalculator, SPANMarginCalculator]:
            calculator_name = calculator_class.__name__
            logger.info(f"  Using {calculator_name}:")
            
            for leverage in test_leverage:
                # For SPAN calculator, we need to keep initial_margin_percentage fixed
                if calculator_class == SPANMarginCalculator:
                    calculator = SPANMarginCalculator(max_leverage=leverage, initial_margin_percentage=0.1)
                else:
                    calculator = calculator_class(max_leverage=leverage)
                
                # Calculate margin
                margin = calculator.calculate_position_margin(position)
                
                # Log results
                logger.info(f"    max_leverage={leverage:.1f}: Margin=${margin:.2f}, "
                           f"Ratio to Premium={margin/premium:.2f}x")
                
                # Check if margin is below premium
                if margin < premium:
                    logger.warning(f"      WARNING: Margin is below option premium!")

def test_partially_hedged_margins(logger: logging.Logger = None):
    """Test calculation of portfolio margin for partially hedged positions.
    
    This test verifies that the SPANMarginCalculator correctly applies partial hedging
    benefits when positions have delta values in opposite directions, but don't perfectly offset.
    """
    # Use default logger if none provided
    if logger is None:
        logger = logging.getLogger("margin_test")
    
    logger.info("="*80)
    logger.info("TESTING PARTIALLY HEDGED PORTFOLIO MARGIN CALCULATION")
    logger.info("="*80)
    
    # Create test option data directly
    option_name = "SPY240510P00470000"  # OTM Put
    option_details = {
        "UnderlyingSymbol": "SPY",
        "Strike": 470.0, 
        "Type": "P",
        "Delta": -0.28,
        "Gamma": 0.03,
        "Theta": -0.25,
        "Vega": 0.15,
        "UnderlyingPrice": 480.0,
        "LastPrice": 7.20
    }
    option_price = option_details["LastPrice"]
    option_delta = option_details["Delta"]
    
    logger.info(f"Testing with option: {option_name}")
    logger.info(f"Option price: ${option_price}")
    logger.info(f"Option delta: {option_delta}")
    logger.info("")
    
    # Create an option position
    option_position = OptionPosition(
        symbol=option_name,
        option_data=option_details,
        contracts=1,
        entry_price=option_price,
        current_price=option_price,
        is_short=False  # Long option
    )
    
    # Make sure underlying price is properly set directly on the position
    option_position.underlying_price = option_details["UnderlyingPrice"]
    
    # Get the actual delta calculated by the position
    actual_delta = option_position.current_delta
    logger.info(f"Position delta after creation: {actual_delta}")
    
    # Create a partial hedge (70% of the delta)
    hedge_percent = 0.7  # 70% hedge
    
    # Calculate delta-neutral hedge (opposite sign)
    hedge_delta = -actual_delta * 100 * hedge_percent  # Opposite sign, 70% of full hedge
    hedge_is_short = hedge_delta < 0
    
    # Create a stock position to hedge the option
    hedge_position = Position(
        symbol="SPY",
        contracts=abs(hedge_delta),  # Number of shares
        entry_price=option_details["UnderlyingPrice"],
        current_price=option_details["UnderlyingPrice"],
        is_short=hedge_is_short
    )
    
    # Calculate expected net delta after partial hedge
    option_delta_total = actual_delta * 100  # Scale to match stock
    stock_delta = hedge_delta  # Already calculated above
    net_delta = option_delta_total + stock_delta
    
    # Calculate expected hedge quality
    total_absolute_delta = abs(option_delta_total) + abs(stock_delta)
    expected_hedge_quality = 1.0 - (abs(net_delta) / total_absolute_delta)
    
    logger.info(f"Partial hedge: {hedge_percent*100:.0f}% of full delta")
    logger.info(f"Option position delta: {option_delta_total}")
    logger.info(f"Stock hedge delta: {stock_delta}")
    logger.info(f"Net delta after partial hedge: {net_delta}")
    logger.info(f"Expected hedge quality: {expected_hedge_quality:.2f} (should be less than 1.0)")
    logger.info(f"Stock position is {'SHORT' if hedge_is_short else 'LONG'}")
    logger.info("")
    
    # Create the margin calculator
    margin_calculator = SPANMarginCalculator(
        max_leverage=12.0,
        volatility_multiplier=1.0,
        initial_margin_percentage=0.1,
        maintenance_margin_percentage=0.07,
        hedge_credit_rate=0.8,  # 80% credit for hedged positions
        logger=logger
    )
    
    # Set logger's level to DEBUG to capture detailed margin calculation steps
    original_level = logger.level
    logger.setLevel(logging.DEBUG)
    
    # Calculate standalone margins
    option_margin = margin_calculator.calculate_position_margin(option_position)
    stock_margin = margin_calculator.calculate_position_margin(hedge_position)
    total_standalone_margin = option_margin + stock_margin
    
    logger.info(f"Option position margin: ${option_margin:.2f}")
    logger.info(f"Stock position margin: ${stock_margin:.2f}")
    logger.info(f"Total standalone margin: ${total_standalone_margin:.2f}")
    logger.info("")
    
    # Calculate portfolio margin with positions as a dictionary
    positions = {
        option_name: option_position,
        "SPY": hedge_position
    }
    
    # Display the hedge_credit_rate setting
    logger.info(f"Hedge credit rate: {margin_calculator.hedge_credit_rate:.2f}")
    logger.info("")
    
    # Add attribute to ensure SPY is recognized as the underlying
    option_position.underlying = "SPY"
    
    # Calculate portfolio margin
    portfolio_margin_result = margin_calculator.calculate_portfolio_margin(positions)
    
    # Reset logger level
    logger.setLevel(original_level)
    
    # Extract margin results
    if isinstance(portfolio_margin_result, dict):
        total_margin = portfolio_margin_result.get("total_margin", 0)
        hedging_benefits = portfolio_margin_result.get("hedging_benefits", 0)
    else:
        total_margin = portfolio_margin_result
        hedging_benefits = 0
    
    logger.info(f"Portfolio margin: ${total_margin:.2f}")
    logger.info(f"Hedging benefits: ${hedging_benefits:.2f}")
    logger.info(f"Reduction from standalone: ${total_standalone_margin - total_margin:.2f}")
    logger.info(f"Percentage reduction: {(hedging_benefits / total_standalone_margin) * 100:.2f}%")
    logger.info("")
    
    # Calculate expected offset factor and benefits
    expected_offset_factor = margin_calculator.hedge_credit_rate * expected_hedge_quality
    expected_reduction = total_standalone_margin * expected_offset_factor
    
    logger.info(f"Analysis of partial hedging effectiveness:")
    logger.info(f"  Hedge percentage: {hedge_percent*100:.0f}%")
    logger.info(f"  Expected hedge quality: {expected_hedge_quality:.4f}")
    logger.info(f"  Hedge credit rate: {margin_calculator.hedge_credit_rate:.2f}")
    logger.info(f"  Expected offset factor: {expected_offset_factor:.4f}")
    logger.info(f"  Expected hedging benefit: ${expected_reduction:.2f}")
    logger.info(f"  Actual hedging benefit: ${hedging_benefits:.2f}")
    
    # Verify benefit scales appropriately with hedge quality
    benefit_ratio = hedging_benefits / expected_reduction if expected_reduction > 0 else 0
    logger.info(f"  Benefit ratio (actual/expected): {benefit_ratio:.2f}")
    
    if 0.9 <= benefit_ratio <= 1.1:
        logger.info("✅ PARTIAL HEDGING BENEFITS CORRECTLY SCALED")
    else:
        logger.info("❌ PARTIAL HEDGING BENEFITS NOT CORRECTLY SCALED")
    
    # Check that portfolio margin is less than standalone margin
    if total_margin < total_standalone_margin:
        logger.info("✅ PORTFOLIO MARGIN IS LESS THAN STANDALONE MARGIN")
    else:
        logger.info("❌ PORTFOLIO MARGIN IS NOT LESS THAN STANDALONE MARGIN")
    
    logger.info("="*80)

def run_margin_tests() -> None:
    """Run all margin calculator tests"""
    # Load configuration
    config = load_config()
    
    # Setup logging
    output_dir = config.get('paths', {}).get('output_dir', 'output')
    logger = setup_logging(output_dir)
    
    logger.info("Starting margin calculator tests")
    logger.info(f"Using output directory: {output_dir}")
    
    # Create test options
    test_options = create_test_options()
    logger.info(f"Created {len(test_options)} test option scenarios")
    
    # Initialize margin calculators
    base_calculator = MarginCalculator(logger=logger)
    option_calculator = OptionMarginCalculator(logger=logger)
    
    # Create SPAN calculator from configuration
    logger.info("Creating SPAN margin calculator from configuration")
    span_calculator = SPANMarginCalculator.from_config(logger=logger)
    
    # Run basic margin calculation tests
    logger.info("\n" + "="*80)
    logger.info("BASIC MARGIN CALCULATION TESTS")
    logger.info("="*80)
    
    test_margin_calculator(base_calculator, test_options, logger)
    test_margin_calculator(option_calculator, test_options, logger)
    test_margin_calculator(span_calculator, test_options, logger)
    
    # Run hedged margin tests
    logger.info("\n" + "="*80)
    logger.info("TESTING HEDGED POSITIONS")
    logger.info("="*80)
    
    test_hedged_margins(logger)
    test_partially_hedged_margins(logger)
    
    # Run parameter impact tests
    logger.info("\n" + "="*80)
    logger.info("TESTING PARAMETER IMPACTS")
    logger.info("="*80)
    
    test_initial_margin_percentage_impact(test_options, logger)
    test_margin_leverage_impact(test_options, logger)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    logger.info("All margin calculator tests completed.")
    logger.info(f"Log file has been saved to: {output_dir}")

if __name__ == "__main__":
    # Set up logging to see detailed test output
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("margin_test")
    
    # Create test options
    test_options = create_test_options()
    
    # Create SPAN margin calculator from configuration
    span_calculator = SPANMarginCalculator.from_config(logger=logger)
    
    print("\n===== RUNNING MARGIN CALCULATOR TESTS =====\n")
    
    # Run basic tests
    test_margin_calculator(span_calculator, test_options, logger)
    
    # Run test for hedged positions
    print("\n===== RUNNING HEDGED POSITION TESTS =====\n")
    test_hedged_margins(logger)
    test_partially_hedged_margins(logger)
    
    print("\n===== ALL TESTS COMPLETED =====\n") 