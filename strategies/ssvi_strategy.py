"""
SSVI-Based Relative Value Trading Strategy

This strategy identifies and executes option trades based on SSVI model parameterization
and z-score analysis of implied volatility surfaces.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta

# Import core modules
from core.portfolio import Portfolio, Position
from data_managers.option_data_manager import OptionDataManager
from analysis.relative_value.ssvi_model import SSVIModel

class SSVIStrategy:
    """
    A trading strategy that identifies relative value opportunities based on SSVI model.
    
    This strategy:
    1. Fits the SSVI model to option market data
    2. Identifies options that deviate significantly from the model
    3. Creates trades based on rich/cheap signals
    4. Manages positions and implements risk management rules
    """
    
    def __init__(self, 
                config: Dict[str, Any],
                option_data_manager: Optional[OptionDataManager] = None,
                ssvi_model: Optional[SSVIModel] = None,
                portfolio: Optional[Portfolio] = None,
                logger: Optional[logging.Logger] = None):
        """
        Initialize the SSVI strategy.
        
        Args:
            config: Strategy configuration
            option_data_manager: Option data manager instance
            ssvi_model: SSVI model instance
            portfolio: Portfolio instance
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Store configuration
        self.config = config
        self.trade_params = config.get('trade_params', {})
        self.risk_params = config.get('risk_params', {})
        self.model_params = config.get('model_params', {})
        
        # Initialize or store components
        self.option_data_manager = option_data_manager or OptionDataManager()
        self.ssvi_model = ssvi_model or SSVIModel(logger=self.logger)
        self.portfolio = portfolio or Portfolio("SSVI Strategy Portfolio")
        
        # Strategy state
        self.last_update_time = None
        self.signal_history = []
        self.trade_log = []
        self.status = "initialized"
        
        # Parse threshold parameters
        self.zscore_threshold = self.trade_params.get('zscore_threshold', 1.5)
        self.min_dte = self.trade_params.get('min_dte', 7)
        self.max_dte = self.trade_params.get('max_dte', 90)
        self.max_legs = self.trade_params.get('max_legs', 3)
        self.min_liquidity = self.trade_params.get('min_liquidity', 100)  # Min open interest
        
        # Risk management parameters
        self.max_position_size = self.risk_params.get('max_position_size', 10)
        self.max_delta_exposure = self.risk_params.get('max_delta_exposure', 100)
        self.max_vega_exposure = self.risk_params.get('max_vega_exposure', 1000)
        self.delta_hedge_freq = self.risk_params.get('delta_hedge_freq', 'daily')
        
        self.logger.info(f"SSVI Strategy initialized with zscore threshold: {self.zscore_threshold}")
    
    def update(self, option_chain: pd.DataFrame, underlying_price: float) -> bool:
        """
        Update the strategy with new market data.
        
        Args:
            option_chain: Current option chain data
            underlying_price: Current underlying price
            
        Returns:
            bool: True if update was successful
        """
        self.logger.info(f"Updating SSVI strategy with {len(option_chain)} options")
        
        try:
            # Store current timestamp
            current_time = datetime.now()
            
            # Filter options based on liquidity and DTE
            filtered_chain = self._filter_options(option_chain)
            
            if filtered_chain.empty:
                self.logger.warning("No valid options after filtering")
                return False
                
            # Fit the SSVI model
            self.ssvi_model.fit(
                filtered_chain, 
                underlying_price, 
                method=self.model_params.get('fit_method', 'global_then_local'),
                max_iterations=self.model_params.get('max_iterations', 1000)
            )
            
            # Identify relative value opportunities
            rv_signals = self.ssvi_model.identify_rv_opportunities(
                filtered_chain,
                underlying_price,
                zscore_threshold=self.zscore_threshold
            )
            
            # Store signals in history
            if not rv_signals.empty:
                signal_record = {
                    'timestamp': current_time,
                    'underlying_price': underlying_price,
                    'signals': rv_signals,
                    'param_zscores': self.ssvi_model.calculate_param_zscores()
                }
                self.signal_history.append(signal_record)
                
                # Keep history manageable
                if len(self.signal_history) > 100:
                    self.signal_history = self.signal_history[-100:]
                    
                self.logger.info(f"Generated {len(rv_signals)} RV signals ({rv_signals['RVSignal'].value_counts().to_dict()})")
            else:
                self.logger.info("No RV signals generated")
                
            # Update tracking variables
            self.last_update_time = current_time
            self.status = "updated"
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating SSVI strategy: {e}")
            self.status = "error"
            return False
    
    def _filter_options(self, option_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Filter options based on liquidity and strategy constraints.
        
        Args:
            option_chain: Full option chain
            
        Returns:
            DataFrame: Filtered option chain
        """
        if option_chain.empty:
            return pd.DataFrame()
            
        # Check that we have the necessary columns
        required_cols = ['Strike', 'DTE', 'IV', 'Type', 'OpenInterest']
        if not all(col in option_chain.columns for col in required_cols):
            self.logger.warning(f"Missing columns in option chain: {[col for col in required_cols if col not in option_chain.columns]}")
            # If OpenInterest is missing, try to use Volume instead
            if 'OpenInterest' not in option_chain.columns and 'Volume' in option_chain.columns:
                option_chain['OpenInterest'] = option_chain['Volume']
                self.logger.info("Using Volume as a proxy for OpenInterest")
            else:
                # Create default OpenInterest if both are missing
                option_chain['OpenInterest'] = 100
                self.logger.warning("Created default OpenInterest values")
            
        # Apply filters
        filtered = option_chain.copy()
        
        # Filter by DTE
        filtered = filtered[(filtered['DTE'] >= self.min_dte) & (filtered['DTE'] <= self.max_dte)]
        
        # Filter by liquidity (open interest or volume)
        filtered = filtered[filtered['OpenInterest'] >= self.min_liquidity]
        
        # Filter out options with missing IVs
        filtered = filtered.dropna(subset=['IV'])
        
        # Filter by additional criteria
        if 'Delta' in filtered.columns:
            # Filter extremely deep ITM/OTM options
            filtered = filtered[(filtered['Delta'].abs() >= 0.05) & (filtered['Delta'].abs() <= 0.95)]
        
        self.logger.info(f"Filtered from {len(option_chain)} to {len(filtered)} options")
        return filtered
    
    def generate_trades(self) -> List[Dict[str, Any]]:
        """
        Generate trade recommendations based on RV signals.
        
        Returns:
            list: List of trade recommendations
        """
        if not self.signal_history:
            self.logger.warning("No signals available to generate trades")
            return []
            
        # Get the most recent signals
        latest_signals = self.signal_history[-1]
        signals_df = latest_signals['signals']
        underlying_price = latest_signals['underlying_price']
        
        if signals_df.empty:
            return []
            
        # Sort by absolute Z-score (strongest signals first)
        signals_df = signals_df.sort_values(by='ZScore', key=abs, ascending=False)
        
        # Generate both single-leg and multi-leg trades
        single_leg_trades = self._generate_single_leg_trades(signals_df)
        multi_leg_trades = self._generate_multi_leg_trades(signals_df, underlying_price)
        
        # Combine and rank all trades
        all_trades = single_leg_trades + multi_leg_trades
        
        # Sort by expected return / risk
        if all_trades:
            all_trades.sort(key=lambda x: x['expected_return_per_risk'], reverse=True)
        
        self.logger.info(f"Generated {len(all_trades)} potential trades")
        return all_trades
    
    def _generate_single_leg_trades(self, signals_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate single-leg trades from RV signals.
        
        Args:
            signals_df: DataFrame with RV signals
            
        Returns:
            list: List of single-leg trade recommendations
        """
        trades = []
        
        for _, row in signals_df.iterrows():
            if row['RVSignal'] not in ['RICH', 'CHEAP']:
                continue
                
            # Determine trade direction
            if row['RVSignal'] == 'RICH':
                direction = 'SELL'
            else:  # CHEAP
                direction = 'BUY'
                
            # Calculate expected PnL and risk
            # Assume mean reversion of Z-score toward 0
            expected_iv_change = row['IV'] - row['ModelIV']
            iv_change_per_zscore = expected_iv_change / row['ZScore']
            
            # Expected return based on vega * expected IV change
            vega = row.get('Vega', 0.2)  # Default vega if not provided
            expected_return = abs(vega * expected_iv_change * 100)  # Per contract
            
            # Risk based on gamma * expected underlying move
            gamma = row.get('Gamma', 0.01)  # Default gamma if not provided
            theta = row.get('Theta', -0.01)  # Default theta if not provided
            expected_risk = abs(gamma * 5**2 / 2)  # Assume 5-point move
            time_decay_risk = abs(theta * 7)  # 7 days holding period
            
            total_risk = expected_risk + time_decay_risk + 0.2  # Base risk
            
            # Create trade object
            trade = {
                'type': 'single_leg',
                'signal': row['RVSignal'],
                'direction': direction,
                'symbol': row['Symbol'] if 'Symbol' in row else f"{row['Underlying']}{row['DTE']}d{row['Type'][0]}{row['Strike']}",
                'underlying': row['Underlying'] if 'Underlying' in row else 'UNKNOWN',
                'strike': row['Strike'],
                'dte': row['DTE'],
                'option_type': row['Type'],
                'zscore': row['ZScore'],
                'current_iv': row['IV'],
                'model_iv': row['ModelIV'],
                'iv_diff': row['IVDiff'],
                'expected_return': expected_return,
                'expected_risk': total_risk,
                'expected_return_per_risk': expected_return / total_risk if total_risk > 0 else 0,
                'suggested_quantity': min(1, self.max_position_size),
                'delta': row.get('Delta', None),
                'gamma': row.get('Gamma', None),
                'theta': row.get('Theta', None),
                'vega': row.get('Vega', None)
            }
            
            trades.append(trade)
            
        return trades
    
    def _generate_multi_leg_trades(self, signals_df: pd.DataFrame, underlying_price: float) -> List[Dict[str, Any]]:
        """
        Generate multi-leg trades (spreads) from RV signals.
        
        Args:
            signals_df: DataFrame with RV signals
            underlying_price: Current underlying price
            
        Returns:
            list: List of multi-leg trade recommendations
        """
        trades = []
        
        # Group by expiration
        for dte, group in signals_df.groupby('DTE'):
            # Need at least 2 options to create a spread
            if len(group) < 2:
                continue
                
            # Look for opposing signals (RICH/CHEAP) at same expiry
            rich_options = group[group['RVSignal'] == 'RICH']
            cheap_options = group[group['RVSignal'] == 'CHEAP']
            
            # Skip if we don't have both rich and cheap options
            if rich_options.empty or cheap_options.empty:
                continue
                
            # Generate potential 2-leg spreads
            for _, rich_opt in rich_options.iterrows():
                for _, cheap_opt in cheap_options.iterrows():
                    # Skip if same strike and type
                    if rich_opt['Strike'] == cheap_opt['Strike'] and rich_opt['Type'] == cheap_opt['Type']:
                        continue
                        
                    # Calculate expected return and risk
                    rich_vega = rich_opt.get('Vega', 0.2)
                    cheap_vega = cheap_opt.get('Vega', 0.2)
                    rich_delta = rich_opt.get('Delta', 0.5 if rich_opt['Type'] == 'CALL' else -0.5)
                    cheap_delta = cheap_opt.get('Delta', 0.5 if cheap_opt['Type'] == 'CALL' else -0.5)
                    
                    # Calculate ratio to make spread delta-neutral
                    if rich_delta != 0 and cheap_delta != 0:
                        ratio = abs(rich_delta / cheap_delta)
                    else:
                        ratio = 1.0
                        
                    # Expected return based on IV mean reversion
                    expected_return = (
                        abs(rich_vega * rich_opt['IVDiff'] * 100) +  # Sell rich option
                        abs(cheap_vega * cheap_opt['IVDiff'] * 100)   # Buy cheap option
                    )
                    
                    # Risk factors
                    rich_gamma = rich_opt.get('Gamma', 0.01)
                    cheap_gamma = cheap_opt.get('Gamma', 0.01)
                    net_gamma = rich_gamma - cheap_gamma * ratio
                    
                    # Risk based on net exposure
                    expected_risk = abs(net_gamma * 5**2 / 2)  # 5-point move
                    
                    # Add time decay risk
                    rich_theta = rich_opt.get('Theta', -0.02)
                    cheap_theta = cheap_opt.get('Theta', -0.02)
                    net_theta = rich_theta - cheap_theta * ratio
                    time_decay_risk = abs(net_theta * 7)  # 7 days
                    
                    total_risk = max(0.5, expected_risk + time_decay_risk)  # Minimum risk floor
                    
                    # Create trade object
                    trade = {
                        'type': 'vertical_spread',
                        'legs': [
                            {
                                'direction': 'SELL',
                                'symbol': rich_opt.get('Symbol', f"RICH{dte}"),
                                'strike': rich_opt['Strike'],
                                'option_type': rich_opt['Type'],
                                'zscore': rich_opt['ZScore'],
                                'quantity': 1
                            },
                            {
                                'direction': 'BUY',
                                'symbol': cheap_opt.get('Symbol', f"CHEAP{dte}"),
                                'strike': cheap_opt['Strike'],
                                'option_type': cheap_opt['Type'],
                                'zscore': cheap_opt['ZScore'],
                                'quantity': min(10, max(1, round(ratio)))
                            }
                        ],
                        'dte': dte,
                        'underlying': rich_opt.get('Underlying', cheap_opt.get('Underlying', 'UNKNOWN')),
                        'expected_return': expected_return,
                        'expected_risk': total_risk,
                        'expected_return_per_risk': expected_return / total_risk if total_risk > 0 else 0,
                        'delta_neutral_ratio': ratio,
                        'net_delta': rich_delta - cheap_delta * ratio,
                        'net_gamma': net_gamma,
                        'net_vega': rich_vega - cheap_vega * ratio,
                        'net_theta': net_theta
                    }
                    
                    trades.append(trade)
            
            # Generate potential butterfly trades if we have at least 3 options
            if len(group) >= 3:
                calls = group[group['Type'] == 'CALL'].sort_values('Strike')
                puts = group[group['Type'] == 'PUT'].sort_values('Strike')
                
                # Generate call butterflies
                if len(calls) >= 3:
                    for i in range(len(calls) - 2):
                        lower = calls.iloc[i]
                        middle = calls.iloc[i + 1]
                        upper = calls.iloc[i + 2]
                        
                        # Skip if strikes are not evenly spaced
                        if not 0.8 <= (middle['Strike'] - lower['Strike']) / (upper['Strike'] - middle['Strike']) <= 1.2:
                            continue
                            
                        # Calculate if butterfly is rich or cheap based on middle option
                        if middle['RVSignal'] == 'RICH':
                            # Sell middle option (sell butterfly)
                            butterfly_type = 'sell_butterfly'
                            direction_middle = 'SELL'
                            direction_wings = 'BUY'
                            legs_quantity = 2  # 2x middle option
                            zscore = middle['ZScore']
                        elif middle['RVSignal'] == 'CHEAP':
                            # Buy middle option (buy butterfly)
                            butterfly_type = 'buy_butterfly'
                            direction_middle = 'BUY'
                            direction_wings = 'SELL'
                            legs_quantity = 2  # 2x middle option
                            zscore = middle['ZScore']
                        else:
                            continue
                            
                        # Create butterfly trade
                        butterfly_trade = {
                            'type': 'butterfly',
                            'strategy': butterfly_type,
                            'legs': [
                                {
                                    'direction': direction_wings,
                                    'symbol': lower.get('Symbol', f"CALL{lower['Strike']}"),
                                    'strike': lower['Strike'],
                                    'option_type': 'CALL',
                                    'quantity': 1
                                },
                                {
                                    'direction': direction_middle,
                                    'symbol': middle.get('Symbol', f"CALL{middle['Strike']}"),
                                    'strike': middle['Strike'],
                                    'option_type': 'CALL',
                                    'quantity': legs_quantity
                                },
                                {
                                    'direction': direction_wings,
                                    'symbol': upper.get('Symbol', f"CALL{upper['Strike']}"),
                                    'strike': upper['Strike'],
                                    'option_type': 'CALL',
                                    'quantity': 1
                                }
                            ],
                            'dte': dte,
                            'underlying': middle.get('Underlying', 'UNKNOWN'),
                            'zscore': zscore,
                            'expected_return': abs(middle['IVDiff'] * 100),  # Simplified
                            'expected_risk': 1.0,  # Defined risk for butterfly
                            'expected_return_per_risk': abs(middle['IVDiff'] * 100)
                        }
                        
                        trades.append(butterfly_trade)
                
                # Generate put butterflies - similar to call butterflies
                # (Code structure would be similar to call butterflies)
                
        return trades
    
    def execute_trades(self, trades: List[Dict[str, Any]], max_trades: int = 3) -> List[Dict[str, Any]]:
        """
        Execute the recommended trades (in real system, would connect to broker).
        
        Args:
            trades: List of trade recommendations
            max_trades: Maximum number of trades to execute
            
        Returns:
            list: List of executed trades
        """
        executed_trades = []
        
        # Execute only top trades up to max_trades
        for i, trade in enumerate(trades):
            if i >= max_trades:
                break
                
            self.logger.info(f"Executing trade: {trade['type']}")
            
            # In a real system, this would place orders with a broker
            # For this simulation, we'll just record the trades
            
            # Add execution timestamp
            trade['execution_time'] = datetime.now()
            
            # Add to executed trades list
            executed_trades.append(trade)
            
            # Add to trade log
            self.trade_log.append(trade)
            
        return executed_trades
    
    def manage_portfolio(self, current_market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage existing positions based on market data and model updates.
        
        Args:
            current_market_data: Current market data including option chain and prices
            
        Returns:
            dict: Portfolio management actions taken
        """
        actions = {
            'delta_hedges': [],
            'position_exits': [],
            'adjustments': []
        }
        
        # Implement delta hedging if needed
        if self.delta_hedge_freq == 'daily':
            hedge_actions = self._perform_delta_hedging(current_market_data)
            actions['delta_hedges'] = hedge_actions
            
        # Check for exit signals
        exit_actions = self._check_exit_signals(current_market_data)
        actions['position_exits'] = exit_actions
        
        # Adjust positions if needed
        adjustment_actions = self._adjust_positions(current_market_data)
        actions['adjustments'] = adjustment_actions
        
        return actions
    
    def _perform_delta_hedging(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform delta hedging of the portfolio.
        
        Args:
            market_data: Current market data
            
        Returns:
            list: Hedging actions taken
        """
        # Get current portfolio delta
        portfolio_delta = self.portfolio.calculate_portfolio_delta()
        
        # Check if delta exceeds threshold
        if abs(portfolio_delta) < self.max_delta_exposure:
            return []
            
        # Calculate hedge quantity
        underlying_price = market_data.get('underlying_price', 100)
        hedge_quantity = -int(portfolio_delta / 100)  # Convert to shares
        
        # Record hedge action
        hedge_action = {
            'action': 'delta_hedge',
            'instrument': market_data.get('underlying_symbol', 'UNKNOWN'),
            'quantity': hedge_quantity,
            'price': underlying_price,
            'timestamp': datetime.now()
        }
        
        return [hedge_action]
    
    def _check_exit_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for exit signals based on z-score reversions.
        
        Args:
            market_data: Current market data
            
        Returns:
            list: Exit actions
        """
        exit_actions = []
        
        # Get current option chain
        option_chain = market_data.get('option_chain', pd.DataFrame())
        
        if option_chain.empty:
            return []
            
        # Check each trade in the log
        for trade in self.trade_log:
            # Skip already closed trades
            if trade.get('status') == 'closed':
                continue
                
            # Check if zscore has normalized
            if trade['type'] == 'single_leg':
                symbol = trade['symbol']
                
                # Find this option in the current chain
                option_data = option_chain[option_chain['Symbol'] == symbol]
                
                if option_data.empty:
                    # Option not found, might be using a different identifier
                    # Try to match by strike, type, and expiry
                    option_data = option_chain[
                        (option_chain['Strike'] == trade['strike']) & 
                        (option_chain['Type'] == trade['option_type']) & 
                        (option_chain['DTE'] == trade['dte'])
                    ]
                
                if not option_data.empty:
                    # Get current zscore
                    row = option_data.iloc[0]
                    current_iv = row['IV']
                    
                    # Calculate new zscore
                    if 'underlying' in market_data:
                        k = np.log(row['Strike'] / market_data['underlying_price'])
                        t = row['DTE'] / 365.0
                        current_zscore = self.ssvi_model.calculate_zscore(k, t, current_iv)
                    else:
                        current_zscore = None
                        
                    # Check if zscore crossed zero (sign changed)
                    if (current_zscore is not None and 
                        ((trade['zscore'] > 0 and current_zscore < 0) or 
                         (trade['zscore'] < 0 and current_zscore > 0))):
                        
                        exit_action = {
                            'action': 'exit',
                            'trade_id': id(trade),
                            'symbol': symbol,
                            'reason': 'zscore_reversal',
                            'entry_zscore': trade['zscore'],
                            'exit_zscore': current_zscore,
                            'timestamp': datetime.now()
                        }
                        
                        exit_actions.append(exit_action)
                        
                        # Mark trade as closed
                        trade['status'] = 'closed'
                        trade['exit_time'] = datetime.now()
                        trade['exit_zscore'] = current_zscore
                        
        return exit_actions
    
    def _adjust_positions(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Adjust existing positions based on market conditions.
        
        Args:
            market_data: Current market data
            
        Returns:
            list: Adjustment actions
        """
        # Implementation would depend on specific adjustment rules
        # This could include rolling positions, adjusting spreads, etc.
        return []
        
    def backtest(self, 
                historical_data: List[Dict[str, Any]], 
                initial_capital: float = 100000.0) -> Dict[str, Any]:
        """
        Backtest the strategy on historical data.
        
        Args:
            historical_data: List of historical market data points
            initial_capital: Initial capital for backtesting
            
        Returns:
            dict: Backtest results
        """
        # Reset for backtesting
        self.portfolio = Portfolio("SSVI Backtest Portfolio", initial_capital=initial_capital)
        self.signal_history = []
        self.trade_log = []
        
        equity_curve = [initial_capital]
        dates = []
        positions_history = []
        returns = []
        
        # Simulate strategy on each historical data point
        for i, data_point in enumerate(historical_data):
            date = data_point.get('date', f"Day {i}")
            dates.append(date)
            
            # Update strategy
            option_chain = data_point.get('option_chain', pd.DataFrame())
            underlying_price = data_point.get('underlying_price', 0)
            
            if not option_chain.empty and underlying_price > 0:
                # Update the strategy
                self.update(option_chain, underlying_price)
                
                # Generate and execute trades
                trades = self.generate_trades()
                executed_trades = self.execute_trades(trades)
                
                # Manage portfolio
                self.manage_portfolio(data_point)
            
            # Calculate portfolio value for this date
            current_prices = data_point.get('current_prices', {})
            portfolio_value = self.portfolio.calculate_total_value(current_prices)
            equity_curve.append(portfolio_value)
            
            # Calculate return
            if i > 0:
                daily_return = (portfolio_value - equity_curve[-2]) / equity_curve[-2]
                returns.append(daily_return)
            
            # Record positions
            positions_snapshot = self.portfolio.get_positions_snapshot()
            positions_history.append(positions_snapshot)
        
        # Calculate performance metrics
        if len(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            total_return = (equity_curve[-1] - initial_capital) / initial_capital
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            total_return = 0
        
        # Prepare results
        results = {
            'equity_curve': equity_curve,
            'dates': dates,
            'positions_history': positions_history,
            'returns': returns,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'trade_log': self.trade_log,
            'signal_history': self.signal_history
        }
        
        return results
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            float: Maximum drawdown as a percentage
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
            
        # Calculate drawdown series
        max_so_far = equity_curve[0]
        drawdowns = []
        
        for value in equity_curve:
            max_so_far = max(max_so_far, value)
            drawdown = (max_so_far - value) / max_so_far if max_so_far > 0 else 0
            drawdowns.append(drawdown)
            
        return max(drawdowns) if drawdowns else 0.0 