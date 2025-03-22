"""
Integration tests for Portfolio and Risk Management interaction.
These tests validate how the portfolio management and risk assessment components work together.
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the required modules
try:
    from core.portfolio import Portfolio, Position
    from core.risk import RiskManager, RiskMetrics
except ImportError:
    pytest.skip("Core modules not available for testing", allow_module_level=True)


@pytest.fixture
def sample_portfolio():
    """Create a realistic portfolio for testing."""
    portfolio = Portfolio("Test Integration Portfolio")
    
    # Create stock positions
    portfolio.add_position(
        Position(
            symbol="AAPL",
            position_type="STOCK",
            quantity=500,
            entry_price=150.0,
            entry_date=datetime.now() - timedelta(days=30)
        )
    )
    
    portfolio.add_position(
        Position(
            symbol="MSFT",
            position_type="STOCK",
            quantity=300,
            entry_price=250.0,
            entry_date=datetime.now() - timedelta(days=20)
        )
    )
    
    portfolio.add_position(
        Position(
            symbol="AMZN",
            position_type="STOCK",
            quantity=50,
            entry_price=3500.0,
            entry_date=datetime.now() - timedelta(days=15)
        )
    )
    
    # Create option positions
    # Long call options
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
    
    # Short put options
    portfolio.add_position(
        Position(
            symbol="MSFT220121P00240000",
            position_type="OPTION",
            quantity=-20,
            entry_price=4.0,
            entry_date=datetime.now() - timedelta(days=7),
            expiration=datetime.now() + timedelta(days=30),
            strike=240.0,
            option_type="PUT",
            delta=-0.35,
            gamma=0.03,
            vega=0.15,
            theta=-0.08
        )
    )
    
    # Add a deep ITM call
    portfolio.add_position(
        Position(
            symbol="AMZN220121C03000000",
            position_type="OPTION",
            quantity=5,
            entry_price=600.0,
            entry_date=datetime.now() - timedelta(days=5),
            expiration=datetime.now() + timedelta(days=30),
            strike=3000.0,
            option_type="CALL",
            delta=0.95,
            gamma=0.01,
            vega=0.05,
            theta=-0.05
        )
    )
    
    return portfolio


@pytest.fixture
def price_history():
    """Create realistic price history data for testing."""
    # Create a price history DataFrame with 252 trading days (1 year)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
    
    # Generate realistic price paths with some correlation
    np.random.seed(42)  # For reproducibility
    
    # Base random walks with drift
    aapl_returns = np.random.normal(0.0005, 0.015, 252)  # Slight positive drift
    msft_returns = np.random.normal(0.0006, 0.016, 252)
    amzn_returns = np.random.normal(0.0007, 0.020, 252)
    
    # Add correlation
    corr_factor = np.random.normal(0, 0.01, 252)
    aapl_returns += corr_factor
    msft_returns += corr_factor * 1.1
    amzn_returns += corr_factor * 1.2
    
    # Convert returns to prices
    aapl_prices = 100 * np.cumprod(1 + aapl_returns)
    msft_prices = 200 * np.cumprod(1 + msft_returns)
    amzn_prices = 3000 * np.cumprod(1 + amzn_returns)
    
    # Scale to end at current prices
    aapl_prices = aapl_prices * (150 / aapl_prices[-1])
    msft_prices = msft_prices * (250 / msft_prices[-1])
    amzn_prices = amzn_prices * (3500 / amzn_prices[-1])
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'AAPL': aapl_prices,
        'MSFT': msft_prices,
        'AMZN': amzn_prices
    })
    
    df.set_index('date', inplace=True)
    return df


@pytest.fixture
def current_prices():
    """Current market prices for all instruments."""
    return {
        "AAPL": 160.0,
        "MSFT": 260.0,
        "AMZN": 3600.0,
        "AAPL220121C00160000": 7.5,
        "MSFT220121P00240000": 3.0,
        "AMZN220121C03000000": 650.0
    }


class TestPortfolioRiskIntegration:
    """Integration tests for portfolio and risk management."""
    
    def test_portfolio_construction_and_valuation(self, sample_portfolio, current_prices):
        """Test creating a portfolio and valuing it."""
        # Verify portfolio construction
        assert len(sample_portfolio.positions) == 6
        
        # Test portfolio valuation
        portfolio_value = sample_portfolio.calculate_total_value(current_prices)
        
        # Calculate expected value manually
        expected_value = (
            500 * 160.0 +        # AAPL stock
            300 * 260.0 +        # MSFT stock
            50 * 3600.0 +        # AMZN stock
            30 * 7.5 +           # AAPL calls
            -20 * 3.0 +          # Short MSFT puts
            5 * 650.0            # AMZN calls
        )
        
        assert abs(portfolio_value - expected_value) < 0.01
        
        # Test portfolio P&L
        portfolio_pnl = sample_portfolio.calculate_total_pnl(current_prices)
        
        # Calculate expected P&L manually
        expected_pnl = (
            500 * (160.0 - 150.0) +               # AAPL stock
            300 * (260.0 - 250.0) +               # MSFT stock
            50 * (3600.0 - 3500.0) +              # AMZN stock
            30 * (7.5 - 5.0) +                    # AAPL calls
            -20 * (3.0 - 4.0) +                   # Short MSFT puts
            5 * (650.0 - 600.0)                   # AMZN calls
        )
        
        assert abs(portfolio_pnl - expected_pnl) < 0.01
    
    def test_portfolio_risk_calculations(self, sample_portfolio, price_history, current_prices):
        """Test portfolio risk calculations."""
        # Create risk manager
        risk_manager = RiskManager(sample_portfolio, price_history)
        
        # Test portfolio VaR calculation
        portfolio_var = risk_manager.calculate_portfolio_var(
            current_prices=current_prices,
            confidence=0.95
        )
        
        # VaR should be negative and reasonable
        assert portfolio_var < 0
        
        # Calculate portfolio value for comparison
        portfolio_value = sample_portfolio.calculate_total_value(current_prices)
        
        # VaR as a percentage of portfolio value should be reasonable
        var_pct = abs(portfolio_var) / portfolio_value
        assert 0.01 <= var_pct <= 0.2
        
        # Test expected shortfall calculation
        portfolio_es = risk_manager.calculate_portfolio_expected_shortfall(
            current_prices=current_prices,
            confidence=0.95
        )
        
        # ES should be more extreme than VaR
        assert portfolio_es <= portfolio_var
        
        # ES as a percentage of portfolio value should be reasonable
        es_pct = abs(portfolio_es) / portfolio_value
        assert 0.02 <= es_pct <= 0.3
    
    def test_portfolio_greeks_aggregation(self, sample_portfolio):
        """Test calculation and aggregation of Greeks across the portfolio."""
        # Create price history
        price_history = pd.DataFrame({
            'AAPL': [150.0, 151.0, 149.0, 152.0, 153.0],
            'MSFT': [250.0, 252.0, 248.0, 251.0, 253.0],
            'AMZN': [3500.0, 3520.0, 3480.0, 3550.0, 3600.0]
        })
        
        risk_manager = RiskManager(sample_portfolio, price_history)
        
        # Calculate portfolio Greeks
        greeks = risk_manager.calculate_portfolio_greeks()
        
        # Expected delta calculation
        expected_delta = (
            500 +                 # AAPL stock
            300 +                 # MSFT stock
            50 +                  # AMZN stock
            30 * 0.45 +           # AAPL calls
            -20 * (-0.35) +       # Short MSFT puts
            5 * 0.95              # AMZN calls
        )
        
        assert abs(greeks['delta'] - expected_delta) < 0.01
        
        # Expected gamma calculation (only options contribute)
        expected_gamma = (
            30 * 0.05 +          # AAPL calls
            -20 * 0.03 +         # Short MSFT puts
            5 * 0.01             # AMZN calls
        )
        
        assert abs(greeks['gamma'] - expected_gamma) < 0.01
        
        # Expected vega calculation (only options contribute)
        expected_vega = (
            30 * 0.2 +           # AAPL calls
            -20 * 0.15 +         # Short MSFT puts
            5 * 0.05             # AMZN calls
        )
        
        assert abs(greeks['vega'] - expected_vega) < 0.01
        
        # Test gamma exposure with a 1% market move
        market_impact = greeks['gamma'] * (0.01 ** 2) * portfolio_value / 2
        assert market_impact != 0
    
    def test_stress_testing(self, sample_portfolio, price_history, current_prices):
        """Test stress testing scenarios."""
        risk_manager = RiskManager(sample_portfolio, price_history)
        
        # Define stress scenarios
        scenarios = {
            'market_crash': {'AAPL': -0.2, 'MSFT': -0.25, 'AMZN': -0.3},
            'tech_boom': {'AAPL': 0.15, 'MSFT': 0.2, 'AMZN': 0.25},
            'apple_outperformance': {'AAPL': 0.1, 'MSFT': -0.05, 'AMZN': -0.05},
            'amazon_crash': {'AAPL': 0.05, 'MSFT': 0.05, 'AMZN': -0.4}
        }
        
        # Run stress tests
        stress_results = risk_manager.stress_test(current_prices, scenarios)
        
        # Should have results for all scenarios
        assert set(stress_results.keys()) == set(scenarios.keys())
        
        # Portfolio value for comparisons
        portfolio_value = sample_portfolio.calculate_total_value(current_prices)
        
        # Market crash should have significant negative impact
        assert stress_results['market_crash'] < 0
        assert abs(stress_results['market_crash']) > 0.1 * portfolio_value
        
        # Tech boom should have positive impact
        assert stress_results['tech_boom'] > 0
        
        # Apple outperformance impact depends on portfolio composition
        # Since we have more AAPL exposure than MSFT+AMZN, impact should be positive
        assert stress_results['apple_outperformance'] > 0
    
    def test_risk_adjusted_performance(self, sample_portfolio, price_history, current_prices):
        """Test calculation of risk-adjusted performance metrics."""
        # Initialize risk manager
        risk_manager = RiskManager(sample_portfolio, price_history)
        
        # Calculate portfolio returns
        portfolio_returns = risk_manager.calculate_portfolio_returns(current_prices)
        
        # Calculate risk-adjusted metrics
        risk_adjusted_metrics = risk_manager.calculate_risk_adjusted_metrics(portfolio_returns)
        
        # Verify metrics exist
        assert 'sharpe_ratio' in risk_adjusted_metrics
        assert 'sortino_ratio' in risk_adjusted_metrics
        assert 'max_drawdown' in risk_adjusted_metrics
        
        # Sharpe ratio should be a reasonable value (typically between -3 and +4)
        assert -3.0 <= risk_adjusted_metrics['sharpe_ratio'] <= 4.0
        
        # Sortino ratio should generally be higher than Sharpe ratio
        assert risk_adjusted_metrics['sortino_ratio'] >= risk_adjusted_metrics['sharpe_ratio']
        
        # Max drawdown should be between 0 and 1
        assert 0.0 <= risk_adjusted_metrics['max_drawdown'] <= 1.0
    
    def test_portfolio_risk_attribution(self, sample_portfolio, price_history, current_prices):
        """Test risk attribution by asset and risk factor."""
        risk_manager = RiskManager(sample_portfolio, price_history)
        
        # Calculate risk attribution
        risk_attribution = risk_manager.calculate_risk_attribution(current_prices)
        
        # Verify structure
        assert 'by_asset' in risk_attribution
        assert 'by_factor' in risk_attribution
        
        # All assets in portfolio should have risk attribution
        for symbol in [pos.symbol for pos in sample_portfolio.positions]:
            assert symbol in risk_attribution['by_asset']
        
        # Check factors
        assert 'equity' in risk_attribution['by_factor']
        assert 'interest_rate' in risk_attribution['by_factor']
        assert 'volatility' in risk_attribution['by_factor']
        
        # Total risk attribution should sum to approximately 100%
        assert 95.0 <= sum(risk_attribution['by_asset'].values()) <= 105.0
        assert 95.0 <= sum(risk_attribution['by_factor'].values()) <= 105.0
    
    def test_hedging_recommendations(self, sample_portfolio, price_history, current_prices):
        """Test generation of hedging recommendations."""
        risk_manager = RiskManager(sample_portfolio, price_history)
        
        # Get portfolio delta
        portfolio_delta = sample_portfolio.calculate_portfolio_delta()
        
        # Get hedging recommendations
        hedging_recs = risk_manager.generate_hedging_recommendations(current_prices)
        
        # If portfolio has positive delta, should recommend short positions to hedge
        if portfolio_delta > 0:
            assert hedging_recs['delta_hedge_quantity'] < 0
        else:
            assert hedging_recs['delta_hedge_quantity'] > 0
        
        # Check that recommendations include options for different types of hedges
        assert 'delta_hedge' in hedging_recs
        assert 'vega_hedge' in hedging_recs
        
        # Recommended hedge size should be proportional to exposure
        assert abs(hedging_recs['delta_hedge_quantity']) <= abs(portfolio_delta) * 1.2
    
    def test_margin_requirement_calculation(self, sample_portfolio, current_prices):
        """Test calculation of margin requirements for the portfolio."""
        # Initialize risk manager with minimal price history
        price_history = pd.DataFrame({
            'AAPL': [150.0, 151.0],
            'MSFT': [250.0, 252.0],
            'AMZN': [3500.0, 3520.0]
        })
        
        risk_manager = RiskManager(sample_portfolio, price_history)
        
        # Calculate margin requirements
        margin_req = risk_manager.calculate_margin_requirements(current_prices)
        
        # Margin should be positive and less than portfolio value for a diversified portfolio
        assert margin_req > 0
        portfolio_value = sample_portfolio.calculate_total_value(current_prices)
        assert margin_req < portfolio_value
        
        # Calculate margin for component parts
        stock_margin = margin_req['stock_margin']
        option_margin = margin_req['option_margin']
        
        # Portfolio margin should include benefits from hedging
        assert margin_req['total_margin'] <= stock_margin + option_margin
        
        # Margin should be reasonably proportional to portfolio value
        margin_ratio = margin_req['total_margin'] / portfolio_value
        assert 0.15 <= margin_ratio <= 0.6  # Typical range for a mixed portfolio


if __name__ == "__main__":
    # This allows running the tests directly from this file
    pytest.main(["-v", __file__]) 