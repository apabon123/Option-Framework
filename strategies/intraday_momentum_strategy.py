"""
Intraday Momentum Strategy
Based on the paper: "Beat the Market - An Effective Intraday Momentum Strategy"

This strategy implements a trading approach that identifies breakouts from
the day's noise range based on opening price behavior and volatility.
"""
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, time
import pandas as pd
import numpy as np

from core.trading_engine import Strategy


class IntradayMomentumStrategy(Strategy):
    """
    Implementation of the Intraday Momentum Strategy for the Option-Framework.
    
    This strategy identifies breakouts from the day's noise range,
    going long when price breaks above the upper bound and short when
    price breaks below the lower bound.
    """
    
    def __init__(self, name: str, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration dictionary
            logger: Logger instance
        """
        super().__init__(name, config, logger)
        
        # Strategy-specific parameters
        strategy_config = config.get('strategy', {})
        self.lookback_days = strategy_config.get('lookback_days', 20)
        self.volatility_multiplier = strategy_config.get('volatility_multiplier', 1.0)
        self.entry_times = strategy_config.get('entry_times', [0, 30])  # Default to trading on hour and half hour
        self.invert_signals = strategy_config.get('invert_signals', False)
        self.min_holding_period = pd.Timedelta(minutes=strategy_config.get('min_holding_period_minutes', 1))
        
        # Validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.lookback_days <= 0:
            self.logger.error("Lookback days must be positive")
            raise ValueError("Lookback days must be positive")

        if self.volatility_multiplier <= 0:
            self.logger.error("Volatility multiplier must be positive")
            raise ValueError("Volatility multiplier must be positive")

        if not self.entry_times:
            self.logger.error("No entry times specified")
            raise ValueError("No entry times specified")
            
        self.logger.info(f"Strategy parameters validated: lookback_days={self.lookback_days}, "
                      f"volatility_multiplier={self.volatility_multiplier}, "
                      f"entry_times={self.entry_times}, invert_signals={self.invert_signals}")
    
    def generate_signals(self, current_date: datetime, daily_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate trading signals for the current date.
        
        Args:
            current_date: Current trading date
            daily_data: Data for the current trading day
            
        Returns:
            list: List of trading signal dictionaries
        """
        self.logger.info(f"Generating signals for {current_date}")
        
        try:
            # Calculate noise area boundaries
            data_with_noise = self.calculate_noise_area(daily_data)
            
            # Define valid trading times - if we have time component in our data
            signals = []
            
            if 'Close' not in data_with_noise.columns:
                self.logger.warning("No 'Close' column found in data")
                return signals
                
            # Check if we have time index or just daily data
            has_intraday = isinstance(data_with_noise.index, pd.DatetimeIndex) and len(data_with_noise) > 1
            
            if has_intraday:
                # For intraday data
                valid_trading_times = []
                for idx, row in data_with_noise.iterrows():
                    if hasattr(idx, 'minute') and hasattr(idx, 'time'):
                        # Check if this is a valid entry time
                        is_valid_minute = idx.minute in self.entry_times
                        # Market open/close checks would go here if we had market hours data
                        valid_trading_times.append(is_valid_minute)
                    else:
                        valid_trading_times.append(False)
                        
                data_with_noise['valid_time'] = valid_trading_times
                
                # Generate signals 
                for idx, row in data_with_noise[data_with_noise['valid_time']].iterrows():
                    if row['Close'] > row.get('upper_bound', float('inf')):
                        direction = -1 if self.invert_signals else 1
                        signals.append({
                            'timestamp': idx,
                            'signal': direction,
                            'type': 'LONG',
                            'price': row['Close'],
                            'strength': abs(row['Close'] - row.get('upper_bound', 0)) / row.get('noise_range', 1),
                            'stop_loss': row.get('stop', row['Close'] * 0.99)
                        })
                    elif row['Close'] < row.get('lower_bound', float('-inf')):
                        direction = 1 if self.invert_signals else -1
                        signals.append({
                            'timestamp': idx,
                            'signal': direction,
                            'type': 'SHORT',
                            'price': row['Close'],
                            'strength': abs(row['Close'] - row.get('lower_bound', 0)) / row.get('noise_range', 1),
                            'stop_loss': row.get('stop', row['Close'] * 1.01)
                        })
            else:
                # For daily data - simplified approach
                if len(data_with_noise) == 0:
                    self.logger.warning("No data available for signal generation")
                    return signals
                    
                latest = data_with_noise.iloc[-1]
                
                # Check for breakout signals based on latest data
                if 'upper_bound' in latest and latest['Close'] > latest['upper_bound']:
                    direction = -1 if self.invert_signals else 1
                    signals.append({
                        'timestamp': current_date,
                        'signal': direction,
                        'type': 'LONG',
                        'price': latest['Close'],
                        'strength': abs(latest['Close'] - latest.get('upper_bound', 0)) / latest.get('noise_range', 1),
                        'stop_loss': latest.get('stop', latest['Close'] * 0.99)
                    })
                elif 'lower_bound' in latest and latest['Close'] < latest['lower_bound']:
                    direction = 1 if self.invert_signals else -1
                    signals.append({
                        'timestamp': current_date,
                        'signal': direction,
                        'type': 'SHORT',
                        'price': latest['Close'],
                        'strength': abs(latest['Close'] - latest.get('lower_bound', 0)) / latest.get('noise_range', 1),
                        'stop_loss': latest.get('stop', latest['Close'] * 1.01)
                    })
            
            # Log signal statistics
            if signals:
                self.logger.info(f"Generated {len(signals)} signals")
                for signal in signals:
                    self.logger.debug(f"Signal: {signal}")
                    
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}", exc_info=True)
            return []
    
    def calculate_noise_area(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the noise area boundaries for price action.
        
        This defines the upper and lower bounds of the expected price range
        based on historical volatility.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with added noise area columns
        """
        df = df.copy()
        
        # Calculate daily volatility if we have enough historical data
        if len(df) > self.lookback_days:
            # Calculate daily range
            if 'High' in df.columns and 'Low' in df.columns:
                df['daily_range'] = df['High'] - df['Low']
                # Calculate average range over lookback period
                df['avg_range'] = df['daily_range'].rolling(window=self.lookback_days, min_periods=5).mean()
            else:
                # Fallback to close volatility if OHLC not available
                df['avg_range'] = df['Close'].rolling(window=self.lookback_days, min_periods=5).std() * 2
        else:
            # Not enough data, use simple approximation
            if 'High' in df.columns and 'Low' in df.columns:
                df['avg_range'] = (df['High'] - df['Low']).mean()
            else:
                df['avg_range'] = df['Close'].std() * 2
        
        # Calculate noise range
        df['noise_range'] = df['avg_range'] * self.volatility_multiplier
        
        # Define upper and lower bounds
        if 'Open' in df.columns:
            reference_price = df['Open']
        else:
            reference_price = df['Close'].shift(1)
            
        df['upper_bound'] = reference_price + (df['noise_range'] / 2)
        df['lower_bound'] = reference_price - (df['noise_range'] / 2)
        
        # Calculate trailing stops
        df = self._calculate_stops(df)
        
        return df
    
    def _calculate_stops(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trailing stops for positions.
        
        For long positions, the stop is placed at the lower bound.
        For short positions, the stop is placed at the upper bound.
        
        Args:
            df: DataFrame with price and noise area data
            
        Returns:
            DataFrame with added stop column
        """
        df = df.copy()
        
        # Initialize stop column
        df['stop'] = np.nan
        
        # If we don't have required columns, return unchanged
        if not all(col in df.columns for col in ['signal', 'upper_bound', 'lower_bound']):
            return df
        
        # Apply stops based on position direction
        for i in range(1, len(df)):
            prev_signal = df.iloc[i-1]['signal']
            
            if prev_signal > 0:  # Long position
                df.iloc[i, df.columns.get_loc('stop')] = df.iloc[i]['lower_bound']
            elif prev_signal < 0:  # Short position
                df.iloc[i, df.columns.get_loc('stop')] = df.iloc[i]['upper_bound']
        
        return df
    
    def check_exit_conditions(self, position: Dict[str, Any], market_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a position should be exited based on market conditions.
        
        Args:
            position: Position to check
            market_data: Current market data for the position's symbol
            
        Returns:
            tuple: (should_exit, reason) - Boolean indicating whether to exit, and the reason
        """
        # Default implementation
        current_price = market_data.get('Close', 0)
        stop_price = position.get('stop_loss')
        
        # Exit if stop loss is hit
        if stop_price is not None:
            if position.get('signal', 0) > 0 and current_price <= stop_price:  # Long position
                return True, "STOP_LOSS"
            elif position.get('signal', 0) < 0 and current_price >= stop_price:  # Short position
                return True, "STOP_LOSS"
        
        # Check for market close (if time data is available)
        timestamp = market_data.get('timestamp')
        if timestamp and hasattr(timestamp, 'time'):
            market_close = time(16, 0)  # 4:00 PM by default
            if timestamp.time() >= market_close:
                return True, "MARKET_CLOSE"
        
        return False, "No exit condition met"
    
    def update_metrics(self, portfolio_metrics: Dict[str, Any]) -> None:
        """
        Update strategy-specific metrics based on portfolio performance.
        
        Args:
            portfolio_metrics: Portfolio performance metrics
        """
        # Store performance metrics
        self.performance_history.append(portfolio_metrics) 