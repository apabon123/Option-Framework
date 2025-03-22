"""
Integration tests for Hedging and Option Data interactions.
These tests validate how the hedging system works with option data to select and execute hedges.
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the required modules
try:
    from core.portfolio import Portfolio, Position
    from core.hedging import DeltaHedger
    from data_managers.option_data_manager import OptionDataManager
except ImportError:
    pytest.skip("Core modules not available for testing", allow_module_level=True)


@pytest.fixture
def sample_option_data():
    """Create sample option chain data for testing."""
    # Create a DataFrame with option data
    expiry_date = datetime.now() + timedelta(days=30)
    expiry_str = expiry_date.strftime('%Y-%m-%d')
    
    # Create option data for SPY (simulating an index we might use for hedging)
    data = []
    
    underlying_price = 450.0
    
    # Generate a range of strikes
    for strike in range(400, 500, 5):
        # Call option
        call_symbol = f"SPY{expiry_date.strftime('%y%m%d')}C{strike:08d}"
        
        # Calculate approximate Black-Scholes values for the options
        call_moneyness = underlying_price / strike
        call_itm = underlying_price > strike
        
        # Calculate delta based on moneyness
        call_delta = 0.5 + 0.5 * (call_moneyness - 1) * 10
        call_delta = max(0.01, min(0.99, call_delta))
        
        call_price = max(0.1, underlying_price - strike) if call_itm else max(0.1, 5 * (1 - abs(call_moneyness - 1) * 5))
        call_gamma = max(0.001, 0.05 * (1 - abs(call_delta - 0.5) * 1.5))
        call_vega = max(0.01, 0.5 * (1 - abs(call_delta - 0.5) * 1.5))
        call_theta = -max(0.01, 0.1 * call_price)
        
        # Add call option
        data.append({
            'symbol': call_symbol,
            'underlying': 'SPY',
            'expiration': expiry_str,
            'strike': strike,
            'option_type': 'CALL',
            'bid': call_price * 0.95,
            'ask': call_price * 1.05,
            'last': call_price,
            'volume': int(1000 * (1 - abs(call_delta - 0.5))),
            'open_interest': int(5000 * (1 - abs(call_delta - 0.5))),
            'implied_volatility': 0.2 + 0.1 * abs(call_delta - 0.5),
            'delta': call_delta,
            'gamma': call_gamma,
            'vega': call_vega,
            'theta': call_theta,
            'underlying_price': underlying_price
        })
        
        # Put option
        put_symbol = f"SPY{expiry_date.strftime('%y%m%d')}P{strike:08d}"
        
        # Calculate delta based on moneyness (put delta is negative)
        put_delta = -0.5 - 0.5 * (call_moneyness - 1) * 10
        put_delta = min(-0.01, max(-0.99, put_delta))
        
        put_itm = underlying_price < strike
        put_price = max(0.1, strike - underlying_price) if put_itm else max(0.1, 5 * (1 - abs(call_moneyness - 1) * 5))
        put_gamma = max(0.001, 0.05 * (1 - abs(put_delta + 0.5) * 1.5))
        put_vega = max(0.01, 0.5 * (1 - abs(put_delta + 0.5) * 1.5))
        put_theta = -max(0.01, 0.1 * put_price)
        
        # Add put option
        data.append({
            'symbol': put_symbol,
            'underlying': 'SPY',
            'expiration': expiry_str,
            'strike': strike,
            'option_type': 'PUT',
            'bid': put_price * 0.95,
            'ask': put_price * 1.05,
            'last': put_price,
            'volume': int(1000 * (1 - abs(put_delta + 0.5))),
            'open_interest': int(5000 * (1 - abs(put_delta + 0.5))),
            'implied_volatility': 0.2 + 0.1 * abs(put_delta + 0.5),
            'delta': put_delta,
            'gamma': put_gamma,
            'vega': put_vega,
            'theta': put_theta,
            'underlying_price': underlying_price
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as temp_file:
        df.to_csv(temp_file.name, index=False)
        return temp_file.name


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio with delta exposure for testing hedging."""
    portfolio = Portfolio("Hedging Test Portfolio")
    
    # Add long stock position
    portfolio.add_position(
        Position(
            symbol="AAPL",
            position_type="STOCK",
            quantity=500,
            entry_price=150.0,
            entry_date=datetime.now() - timedelta(days=30)
        )
    )
    
    # Add some option positions
    portfolio.add_position(
        Position(
            symbol="AAPL220121C00160000",
            position_type="OPTION",
            quantity=30,
            entry_price=5.0,
            entry_date=datetime.now() - timedelta(days=10),
            expiration=datetime.now() + timedelta(days=30),
            strike=160.0,
            option_type="CALL",
            delta=0.45,
            gamma=0.05,
            vega=0.2,
            theta=-0.1
        )
    )
    
    return portfolio


class TestHedgingOptionDataIntegration:
    """Integration tests for hedging and option data components."""
    
    def test_option_data_loading(self, sample_option_data):
        """Test loading option data."""
        # Create option data manager
        option_manager = OptionDataManager()
        
        # Load data from the sample file
        option_data = option_manager.load_option_data(sample_option_data)
        
        # Verify data loaded correctly
        assert not option_data.empty
        assert 'symbol' in option_data.columns
        assert 'delta' in option_data.columns
        assert 'gamma' in option_data.columns
        assert len(option_data) > 10  # Should have multiple options
        
        # Check that we have both call and put options
        option_types = option_data['option_type'].unique()
        assert 'CALL' in option_types
        assert 'PUT' in option_types
    
    def test_option_filtering(self, sample_option_data):
        """Test filtering options based on criteria."""
        option_manager = OptionDataManager()
        option_data = option_manager.load_option_data(sample_option_data)
        
        # Filter for calls only
        calls = option_manager.filter_options(option_data, option_type='CALL')
        assert all(calls['option_type'] == 'CALL')
        
        # Filter for puts only
        puts = option_manager.filter_options(option_data, option_type='PUT')
        assert all(puts['option_type'] == 'PUT')
        
        # Filter by delta range (for calls)
        atm_calls = option_manager.filter_options(
            option_data, 
            option_type='CALL',
            min_delta=0.4,
            max_delta=0.6
        )
        assert all(atm_calls['delta'] >= 0.4)
        assert all(atm_calls['delta'] <= 0.6)
        
        # Filter by delta range (for puts)
        atm_puts = option_manager.filter_options(
            option_data, 
            option_type='PUT',
            min_delta=-0.6,
            max_delta=-0.4
        )
        assert all(atm_puts['delta'] >= -0.6)
        assert all(atm_puts['delta'] <= -0.4)
    
    def test_delta_hedger_with_stock(self, sample_portfolio):
        """Test delta hedging with the underlying stock."""
        # Create delta hedger in ratio mode
        hedger = DeltaHedger(
            mode='ratio',
            target_delta_ratio=0.5,  # Aim to hedge 50% of delta
            tolerance=0.1
        )
        
        # Get portfolio delta
        portfolio_delta = sample_portfolio.calculate_portfolio_delta()
        assert portfolio_delta > 0  # Ensure portfolio has positive delta
        
        # Calculate hedge adjustment
        hedge_adjustment = hedger.calculate_hedge_adjustment(portfolio_delta)
        
        # Should recommend a short position to offset positive delta
        assert hedge_adjustment < 0
        
        # Should hedge approximately 50% of portfolio delta (within tolerance)
        assert abs(hedge_adjustment) >= (portfolio_delta * 0.4)  # Lower bound with tolerance
        assert abs(hedge_adjustment) <= (portfolio_delta * 0.6)  # Upper bound with tolerance
        
        # Test applying the hedge
        hedge_position = Position(
            symbol="SPY",
            position_type="STOCK",
            quantity=hedge_adjustment,
            entry_price=450.0,
            entry_date=datetime.now()
        )
        
        sample_portfolio.add_position(hedge_position)
        
        # New delta should be approximately 50% of original
        new_delta = sample_portfolio.calculate_portfolio_delta()
        assert abs(new_delta) <= portfolio_delta * 0.6
    
    def test_delta_hedger_with_options(self, sample_portfolio, sample_option_data):
        """Test delta hedging using options instead of the underlying."""
        # Load option data
        option_manager = OptionDataManager()
        option_data = option_manager.load_option_data(sample_option_data)
        
        # Create delta hedger in constant mode
        hedger = DeltaHedger(
            mode='constant',
            target_delta=0,  # Target delta-neutral
            tolerance=10.0,  # Allow some delta exposure
            use_options=True
        )
        
        # Get portfolio delta
        portfolio_delta = sample_portfolio.calculate_portfolio_delta()
        
        # Find appropriate options for hedging
        hedge_options = hedger.find_hedge_options(
            portfolio_delta=portfolio_delta,
            option_chain=option_data
        )
        
        # Should recommend at least one option
        assert len(hedge_options) > 0
        
        # First option should be the optimal one
        best_option = hedge_options[0]
        
        # Verify that the option is appropriate
        # If portfolio delta is positive, should use puts or short calls
        if portfolio_delta > 0:
            assert best_option['option_type'] == 'PUT' or best_option['quantity'] < 0
        # If portfolio delta is negative, should use calls or short puts
        else:
            assert best_option['option_type'] == 'CALL' or best_option['quantity'] < 0
        
        # The option delta * quantity should offset portfolio delta to some degree
        assert abs(portfolio_delta + (best_option['delta'] * best_option['quantity'])) < abs(portfolio_delta)
    
    def test_hedging_system_workflow(self, sample_portfolio, sample_option_data):
        """Test the complete hedging workflow."""
        # Load option data
        option_manager = OptionDataManager()
        all_options = option_manager.load_option_data(sample_option_data)
        
        # Create delta hedger
        hedger = DeltaHedger(
            mode='constant',
            target_delta=0,  # Target delta-neutral
            tolerance=10.0,
            use_options=True,
            option_selection_criteria={
                'min_volume': 500,
                'max_delta_distance': 0.2,  # Prefer options with delta close to 0.5/-0.5
                'max_bid_ask_spread_pct': 0.1  # Max 10% bid-ask spread
            }
        )
        
        # Initial portfolio state
        initial_delta = sample_portfolio.calculate_portfolio_delta()
        
        # Step 1: Analyze portfolio exposure
        portfolio_analysis = hedger.analyze_portfolio(sample_portfolio)
        assert 'delta' in portfolio_analysis
        assert 'gamma' in portfolio_analysis
        assert portfolio_analysis['delta'] == initial_delta
        
        # Step 2: Determine hedging needs
        hedge_requirements = hedger.determine_hedge_requirements(portfolio_analysis)
        assert 'target_delta_adjustment' in hedge_requirements
        
        # Step 3: Select hedging instruments
        hedge_options = hedger.find_hedge_options(
            portfolio_delta=initial_delta,
            option_chain=all_options
        )
        
        # Pick the best option
        selected_hedge = hedge_options[0]
        assert 'symbol' in selected_hedge
        assert 'quantity' in selected_hedge
        
        # Create a Position from the selected hedge
        hedge_position = Position(
            symbol=selected_hedge['symbol'],
            position_type="OPTION",
            quantity=selected_hedge['quantity'],
            entry_price=selected_hedge['price'],
            entry_date=datetime.now(),
            expiration=datetime.strptime(selected_hedge['expiration'], '%Y-%m-%d'),
            strike=selected_hedge['strike'],
            option_type=selected_hedge['option_type'],
            delta=selected_hedge['delta'],
            gamma=selected_hedge['gamma'],
            vega=selected_hedge['vega'],
            theta=selected_hedge['theta']
        )
        
        # Step 4: Apply the hedge to the portfolio
        sample_portfolio.add_position(hedge_position)
        
        # Step 5: Verify the new portfolio exposure
        new_delta = sample_portfolio.calculate_portfolio_delta()
        
        # New delta should be closer to target than original delta
        assert abs(new_delta) < abs(initial_delta)
    
    def test_dynamic_hedge_adjustment(self, sample_portfolio, sample_option_data):
        """Test dynamic adjustment of hedges as portfolio changes."""
        # Load option data
        option_manager = OptionDataManager()
        all_options = option_manager.load_option_data(sample_option_data)
        
        # Create delta hedger
        hedger = DeltaHedger(
            mode='ratio',
            target_delta_ratio=0.5,  # Hedge 50% of delta
            tolerance=0.1,
            use_options=True
        )
        
        # Initial portfolio state
        initial_delta = sample_portfolio.calculate_portfolio_delta()
        
        # First round of hedging
        hedge_options = hedger.find_hedge_options(
            portfolio_delta=initial_delta,
            option_chain=all_options
        )
        
        selected_hedge = hedge_options[0]
        hedge_position = Position(
            symbol=selected_hedge['symbol'],
            position_type="OPTION",
            quantity=selected_hedge['quantity'],
            entry_price=selected_hedge['price'],
            entry_date=datetime.now(),
            expiration=datetime.strptime(selected_hedge['expiration'], '%Y-%m-%d'),
            strike=selected_hedge['strike'],
            option_type=selected_hedge['option_type'],
            delta=selected_hedge['delta'],
            gamma=selected_hedge['gamma'],
            vega=selected_hedge['vega'],
            theta=selected_hedge['theta']
        )
        
        sample_portfolio.add_position(hedge_position)
        mid_delta = sample_portfolio.calculate_portfolio_delta()
        
        # Now simulate a market movement that changes deltas
        # Increase delta of all positions by 10%
        for position in sample_portfolio.positions:
            if hasattr(position, 'delta') and position.delta is not None:
                position.delta *= 1.1
        
        # Portfolio delta should now be different
        new_delta = sample_portfolio.calculate_portfolio_delta()
        assert new_delta != mid_delta
        
        # Check if adjustment is needed
        needs_adjustment = hedger.check_hedge_needs(sample_portfolio)
        
        # Determine new hedge requirements
        if needs_adjustment:
            # Find new hedge options
            adjustment_options = hedger.find_hedge_options(
                portfolio_delta=new_delta,
                option_chain=all_options,
                existing_hedges=[pos for pos in sample_portfolio.positions 
                                if pos.position_type == "OPTION" and "SPY" in pos.symbol]
            )
            
            # There should be adjustment options
            assert len(adjustment_options) > 0
            
            # Apply the new hedge
            new_hedge = adjustment_options[0]
            adjustment_position = Position(
                symbol=new_hedge['symbol'],
                position_type="OPTION",
                quantity=new_hedge['quantity'],
                entry_price=new_hedge['price'],
                entry_date=datetime.now(),
                expiration=datetime.strptime(new_hedge['expiration'], '%Y-%m-%d'),
                strike=new_hedge['strike'],
                option_type=new_hedge['option_type'],
                delta=new_hedge['delta'],
                gamma=new_hedge['gamma'],
                vega=new_hedge['vega'],
                theta=new_hedge['theta']
            )
            
            sample_portfolio.add_position(adjustment_position)
            
            # Final portfolio delta should be closer to target
            final_delta = sample_portfolio.calculate_portfolio_delta()
            assert abs(final_delta - 0.5 * initial_delta) < abs(new_delta - 0.5 * initial_delta)


if __name__ == "__main__":
    # This allows running the tests directly from this file
    pytest.main(["-v", __file__]) 