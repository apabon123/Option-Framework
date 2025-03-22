"""
Unit tests for the risk assessment module.
These tests validate risk metrics calculations and risk assessment tools.
"""

import pytest
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the required modules
try:
    from core.risk import RiskManager, RiskMetrics
    from core.portfolio import Portfolio, Position
except ImportError:
    pytest.skip("Core modules not available for testing", allow_module_level=True)


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio with multiple positions for testing."""
    portfolio = Portfolio("Test Portfolio")
    
    # Add some stock positions
    portfolio.add_position(
        Position(
            symbol="AAPL",
            position_type="STOCK",
            quantity=100,
            entry_price=150.0,
            entry_date=datetime.now()
        )
    )
    
    portfolio.add_position(
        Position(
            symbol="MSFT",
            position_type="STOCK",
            quantity=50,
            entry_price=250.0,
            entry_date=datetime.now()
        )
    )
    
    # Add some option positions
    portfolio.add_position(
        Position(
            symbol="AAPL220121C00160000",
            position_type="OPTION",
            quantity=10,
            entry_price=5.0,
            entry_date=datetime.now(),
            expiration=datetime.now() + timedelta(days=30),
            strike=160.0,
            option_type="CALL",
            delta=0.5,
            gamma=0.05,
            vega=0.2,
            theta=-0.1
        )
    )
    
    portfolio.add_position(
        Position(
            symbol="MSFT220121P00240000",
            position_type="OPTION",
            quantity=-5,
            entry_price=4.0,
            entry_date=datetime.now(),
            expiration=datetime.now() + timedelta(days=30),
            strike=240.0,
            option_type="PUT",
            delta=-0.4,
            gamma=0.03,
            vega=0.15,
            theta=-0.08
        )
    )
    
    return portfolio


@pytest.fixture
def price_history():
    """Create sample price history data for testing."""
    # Create a price history DataFrame with 252 trading days (1 year)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
    
    # AAPL price history (starting at 120, ending at 150)
    aapl_prices = np.linspace(120, 150, 252) + np.random.normal(0, 2, 252)
    
    # MSFT price history (starting at 200, ending at 250)
    msft_prices = np.linspace(200, 250, 252) + np.random.normal(0, 3, 252)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'AAPL': aapl_prices,
        'MSFT': msft_prices
    })
    
    df.set_index('date', inplace=True)
    return df


class TestRiskMetrics:
    """Tests for the RiskMetrics class."""
    
    def test_calculate_returns(self, price_history):
        """Test calculation of returns from price history."""
        risk_metrics = RiskMetrics()
        
        # Calculate daily returns
        returns = risk_metrics.calculate_returns(price_history)
        
        # Should have one less row than price_history
        assert len(returns) == len(price_history) - 1
        
        # Returns should be calculated correctly
        for symbol in ['AAPL', 'MSFT']:
            for i in range(1, len(price_history)):
                expected_return = price_history[symbol].iloc[i] / price_history[symbol].iloc[i-1] - 1
                assert abs(returns[symbol].iloc[i-1] - expected_return) < 0.0001
    
    def test_calculate_volatility(self, price_history):
        """Test volatility calculation."""
        risk_metrics = RiskMetrics()
        
        # Calculate returns first
        returns = risk_metrics.calculate_returns(price_history)
        
        # Test different time windows
        for window in [21, 63, 252]:  # 1 month, 3 months, 1 year
            vol = risk_metrics.calculate_volatility(returns, window=window)
            
            # Check that volatility is calculated for all symbols
            assert 'AAPL' in vol
            assert 'MSFT' in vol
            
            # Volatility should be positive
            assert vol['AAPL'] > 0
            assert vol['MSFT'] > 0
            
            # Volatility should be in a reasonable range for stocks (5%-50% annualized)
            assert 0.05 <= vol['AAPL'] * np.sqrt(252) <= 0.5
            assert 0.05 <= vol['MSFT'] * np.sqrt(252) <= 0.5
    
    def test_calculate_correlation(self, price_history):
        """Test correlation calculation."""
        risk_metrics = RiskMetrics()
        
        # Calculate returns first
        returns = risk_metrics.calculate_returns(price_history)
        
        # Calculate correlation matrix
        corr = risk_metrics.calculate_correlation(returns)
        
        # Should be a DataFrame with both symbols as index and columns
        assert isinstance(corr, pd.DataFrame)
        assert list(corr.index) == ['AAPL', 'MSFT']
        assert list(corr.columns) == ['AAPL', 'MSFT']
        
        # Diagonal should be 1.0 (perfect correlation with self)
        assert corr.loc['AAPL', 'AAPL'] == 1.0
        assert corr.loc['MSFT', 'MSFT'] == 1.0
        
        # Correlation should be symmetric
        assert corr.loc['AAPL', 'MSFT'] == corr.loc['MSFT', 'AAPL']
        
        # Correlation should be between -1 and 1
        assert -1.0 <= corr.loc['AAPL', 'MSFT'] <= 1.0
    
    def test_calculate_beta(self, price_history):
        """Test beta calculation."""
        risk_metrics = RiskMetrics()
        
        # Calculate returns first
        returns = risk_metrics.calculate_returns(price_history)
        
        # Calculate beta relative to AAPL as benchmark
        beta = risk_metrics.calculate_beta(returns, benchmark='AAPL')
        
        # AAPL's beta to itself should be 1.0
        assert abs(beta['AAPL'] - 1.0) < 0.0001
        
        # MSFT's beta should be a reasonable value
        assert -2.0 <= beta['MSFT'] <= 5.0
    
    def test_calculate_var(self, price_history):
        """Test Value at Risk (VaR) calculation."""
        risk_metrics = RiskMetrics()
        
        # Calculate returns first
        returns = risk_metrics.calculate_returns(price_history)
        
        # Test VaR calculation with different confidence levels
        for confidence in [0.95, 0.99]:
            var = risk_metrics.calculate_var(returns, confidence=confidence)
            
            # Check that VaR is calculated for all symbols
            assert 'AAPL' in var
            assert 'MSFT' in var
            
            # VaR should be negative (it's a loss)
            assert var['AAPL'] < 0
            assert var['MSFT'] < 0
            
            # Higher confidence level should lead to more negative VaR
            if confidence > 0.95:
                var_95 = risk_metrics.calculate_var(returns, confidence=0.95)
                assert var['AAPL'] < var_95['AAPL']
                assert var['MSFT'] < var_95['MSFT']
    
    def test_calculate_expected_shortfall(self, price_history):
        """Test Expected Shortfall (ES) calculation."""
        risk_metrics = RiskMetrics()
        
        # Calculate returns first
        returns = risk_metrics.calculate_returns(price_history)
        
        # Calculate Expected Shortfall
        es = risk_metrics.calculate_expected_shortfall(returns, confidence=0.95)
        
        # Check that ES is calculated for all symbols
        assert 'AAPL' in es
        assert 'MSFT' in es
        
        # ES should be negative and more extreme than VaR
        var = risk_metrics.calculate_var(returns, confidence=0.95)
        assert es['AAPL'] < 0
        assert es['MSFT'] < 0
        assert es['AAPL'] <= var['AAPL']
        assert es['MSFT'] <= var['MSFT']


class TestRiskManager:
    """Tests for the RiskManager class."""
    
    def test_initialization(self, sample_portfolio, price_history):
        """Test risk manager initialization."""
        risk_manager = RiskManager(sample_portfolio, price_history)
        
        assert risk_manager.portfolio == sample_portfolio
        assert risk_manager.price_history.equals(price_history)
        assert hasattr(risk_manager, 'risk_metrics')
    
    def test_portfolio_var(self, sample_portfolio, price_history):
        """Test portfolio Value at Risk calculation."""
        risk_manager = RiskManager(sample_portfolio, price_history)
        
        # Current market values for positions
        current_prices = {
            "AAPL": 160.0,
            "MSFT": 260.0,
            "AAPL220121C00160000": 7.0,
            "MSFT220121P00240000": 3.0
        }
        
        # Calculate portfolio VaR
        portfolio_var = risk_manager.calculate_portfolio_var(
            current_prices=current_prices,
            confidence=0.95
        )
        
        # VaR should be negative and in a reasonable range
        assert portfolio_var < 0
        
        # Calculate portfolio value for comparison
        portfolio_value = sample_portfolio.calculate_total_value(current_prices)
        
        # VaR should be a percentage of portfolio value (typically 1-10% for a 95% confidence)
        var_pct = abs(portfolio_var) / portfolio_value
        assert 0.01 <= var_pct <= 0.2
    
    def test_portfolio_expected_shortfall(self, sample_portfolio, price_history):
        """Test portfolio Expected Shortfall calculation."""
        risk_manager = RiskManager(sample_portfolio, price_history)
        
        # Current market values for positions
        current_prices = {
            "AAPL": 160.0,
            "MSFT": 260.0,
            "AAPL220121C00160000": 7.0,
            "MSFT220121P00240000": 3.0
        }
        
        # Calculate portfolio Expected Shortfall
        portfolio_es = risk_manager.calculate_portfolio_expected_shortfall(
            current_prices=current_prices,
            confidence=0.95
        )
        
        # ES should be negative and more extreme than VaR
        portfolio_var = risk_manager.calculate_portfolio_var(
            current_prices=current_prices,
            confidence=0.95
        )
        assert portfolio_es < 0
        assert portfolio_es <= portfolio_var
    
    def test_stress_test(self, sample_portfolio):
        """Test portfolio stress testing."""
        # Create a simple price history
        price_history = pd.DataFrame({
            'AAPL': [150.0, 151.0, 149.0, 152.0, 153.0],
            'MSFT': [250.0, 252.0, 248.0, 251.0, 253.0]
        })
        
        risk_manager = RiskManager(sample_portfolio, price_history)
        
        # Current market values for positions
        current_prices = {
            "AAPL": 160.0,
            "MSFT": 260.0,
            "AAPL220121C00160000": 7.0,
            "MSFT220121P00240000": 3.0
        }
        
        # Define stress scenarios
        scenarios = {
            'market_crash': {'AAPL': -0.2, 'MSFT': -0.25},
            'tech_boom': {'AAPL': 0.15, 'MSFT': 0.2},
            'aapl_crash': {'AAPL': -0.3, 'MSFT': -0.05}
        }
        
        # Run stress tests
        stress_results = risk_manager.stress_test(current_prices, scenarios)
        
        # Should have results for all scenarios
        assert set(stress_results.keys()) == set(scenarios.keys())
        
        # Check that each scenario has a portfolio impact value
        for scenario, impact in stress_results.items():
            assert isinstance(impact, float)
            
            # Market crash should have negative impact
            if scenario == 'market_crash':
                assert impact < 0
            
            # Tech boom should have positive impact
            if scenario == 'tech_boom':
                assert impact > 0
    
    def test_calculate_portfolio_greeks(self, sample_portfolio):
        """Test calculation of portfolio Greeks."""
        price_history = pd.DataFrame({
            'AAPL': [150.0, 151.0, 149.0, 152.0, 153.0],
            'MSFT': [250.0, 252.0, 248.0, 251.0, 253.0]
        })
        
        risk_manager = RiskManager(sample_portfolio, price_history)
        
        # Calculate portfolio Greeks
        greeks = risk_manager.calculate_portfolio_greeks()
        
        # Should have values for all major Greeks
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks
        assert 'theta' in greeks
        
        # Delta should match our expected calculation based on position deltas
        expected_delta = (
            100 +           # AAPL stock
            50 +            # MSFT stock
            10 * 0.5 +      # AAPL calls
            -5 * (-0.4)     # Short MSFT puts
        )
        assert abs(greeks['delta'] - expected_delta) < 0.01
        
        # Gamma should be the sum of option gammas
        expected_gamma = 10 * 0.05 + (-5) * 0.03
        assert abs(greeks['gamma'] - expected_gamma) < 0.01
        
        # Vega should be the sum of option vegas
        expected_vega = 10 * 0.2 + (-5) * 0.15
        assert abs(greeks['vega'] - expected_vega) < 0.01
        
        # Theta should be the sum of option thetas
        expected_theta = 10 * (-0.1) + (-5) * (-0.08)
        assert abs(greeks['theta'] - expected_theta) < 0.01
    
    def test_option_risk_metrics(self, sample_portfolio):
        """Test option-specific risk metrics."""
        price_history = pd.DataFrame({
            'AAPL': [150.0, 151.0, 149.0, 152.0, 153.0],
            'MSFT': [250.0, 252.0, 248.0, 251.0, 253.0]
        })
        
        risk_manager = RiskManager(sample_portfolio, price_history)
        
        # Calculate option risk metrics
        option_risks = risk_manager.calculate_option_risk_metrics()
        
        # Should have results for all option positions
        assert "AAPL220121C00160000" in option_risks
        assert "MSFT220121P00240000" in option_risks
        
        # Each option should have time_to_expiry, implied_volatility, delta
        for option_data in option_risks.values():
            assert 'time_to_expiry' in option_data
            assert 'delta' in option_data
            
            # Time to expiry should be positive for non-expired options
            assert option_data['time_to_expiry'] > 0
    
    def test_risk_limits(self, sample_portfolio, price_history):
        """Test risk limit checks."""
        risk_manager = RiskManager(sample_portfolio, price_history)
        
        # Set up risk limits
        risk_limits = {
            'max_position_size': 100000,
            'max_position_concentration': 40,  # percent
            'max_portfolio_delta': 200,
            'max_portfolio_var_pct': 10  # percent
        }
        
        # Current market values for positions
        current_prices = {
            "AAPL": 160.0,
            "MSFT": 260.0,
            "AAPL220121C00160000": 7.0,
            "MSFT220121P00240000": 3.0
        }
        
        # Check risk limits
        limit_checks = risk_manager.check_risk_limits(current_prices, risk_limits)
        
        # Should have results for all limits
        assert set(limit_checks.keys()) == set(risk_limits.keys())
        
        # Each check should have a boolean result
        for limit, check in limit_checks.items():
            assert isinstance(check, bool)


if __name__ == "__main__":
    # This allows running the tests directly from this file
    pytest.main(["-v", __file__]) 