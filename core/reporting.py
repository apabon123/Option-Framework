"""
Reporting Module

This module provides tools for generating reports, visualizations, and 
analyzing trading performance.
"""

import os
import logging
import io
import base64
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class ReportingSystem:
    """
    Reporting system for generating trading system reports.
    
    This class handles the generation of various reports including:
    - HTML reports with interactive visualizations
    - CSV output files
    - Business logic verification files with detailed trading metrics
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        portfolio: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        trading_engine: Optional[Any] = None
    ):
        """
        Initialize the reporting system.
        
        Args:
            config: Configuration dictionary
            portfolio: Optional portfolio instance
            logger: Optional logger instance
            trading_engine: Optional reference to the TradingEngine instance
        """
        self.config = config
        self.portfolio = portfolio
        self.logger = logger or logging.getLogger('reporting')
        self.trading_engine = trading_engine  # Store reference to trading engine
        
        # Create output directory if it doesn't exist
        self.output_dir = config.get('paths', {}).get('output_dir', 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create verification files directory if needed
        verification_dir = config.get('paths', {}).get('verification_output_dir')
        if verification_dir:
            os.makedirs(verification_dir, exist_ok=True)
    
    def _generate_html_report_from_data(
        self, 
        equity_history: Dict[datetime, float], 
        performance_metrics: Dict[str, float],
        trades: List[Dict[str, Any]],
        report_name: str = "strategy_report"
    ) -> str:
        """
        Generate a full HTML report with performance metrics and charts from raw data.
        
        Args:
            equity_history: Dictionary of equity values by date
            performance_metrics: Dictionary of performance metrics
            trades: List of trade dictionaries
            report_name: Base name for the report file
            
        Returns:
            str: File path to the generated HTML report
        """
        self.logger.info("===========================================")
        self.logger.info(f"Generating HTML report '{report_name}'...")
        
        # Create HTML content
        html_content = self._generate_html_content(equity_history, performance_metrics, trades)
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{report_name}_{timestamp}.html"
        file_path = os.path.join(self.output_dir, filename)
        
        with open(file_path, 'w') as f:
            f.write(html_content)
            
        self.logger.info(f"Report saved to {file_path}")
        self.logger.info("===========================================")
        
        return file_path
    
    def generate_html_report(
        self,
        portfolio_or_equity_history,
        strategy_name_or_metrics=None,
        strategy_config_or_trades=None,
        report_name="strategy_report"
    ) -> str:
        """
        Generate an HTML report either from a portfolio object or from raw data.
        This method detects the input type and calls the appropriate implementation.
        
        Args:
            portfolio_or_equity_history: Either a Portfolio object or a dict of equity history
            strategy_name_or_metrics: Either strategy name (str) or performance metrics (dict)
            strategy_config_or_trades: Either strategy config (dict) or trades list
            report_name: Base name for the report file
            
        Returns:
            str: Path to the generated HTML report
        """
        # Check if first argument is a portfolio object
        if hasattr(portfolio_or_equity_history, 'get_portfolio_value') or hasattr(portfolio_or_equity_history, 'positions'):
            # It's a portfolio object
            portfolio = portfolio_or_equity_history
            strategy_name = strategy_name_or_metrics
            strategy_config = strategy_config_or_trades
            
            self.logger.info(f"Generating HTML report for {strategy_name or 'strategy'}...")
            
            # Get equity history from portfolio
            if hasattr(portfolio, 'get_equity_history_as_list'):
                # Use standardized method that returns a list of tuples
                equity_list = portfolio.get_equity_history_as_list()
                equity_history = {date: float(value) for date, value in equity_list}
            elif hasattr(portfolio, 'equity_history'):
                # Convert any non-standard types to simple floats
                equity_history = {date: float(value) for date, value in portfolio.equity_history.items()}
            else:
                # Empty history if none available
                equity_history = {}
            
            # Get performance metrics
            if hasattr(portfolio, 'get_performance_metrics'):
                performance_metrics = portfolio.get_performance_metrics()
            else:
                performance_metrics = {}
            
            # Get trade history
            if hasattr(portfolio, 'get_trade_history'):
                trades = portfolio.get_trade_history()
            else:
                trades = []
            
            # Generate report with the collected data
            return self._generate_html_report_from_data(equity_history, performance_metrics, trades, report_name)
        else:
            # It's raw data
            equity_history = portfolio_or_equity_history
            performance_metrics = strategy_name_or_metrics
            trades = strategy_config_or_trades
            
            return self._generate_html_report_from_data(equity_history, performance_metrics, trades, report_name)
    
    def generate_performance_chart(self, equity_history: Dict[datetime, float]) -> str:
        """
        Generate an equity curve chart and return as base64 encoded image.
        
        Args:
            equity_history: Dictionary of equity values by date
            
        Returns:
            str: Base64 encoded PNG image
        """
        # Sort equity history by date
        dates = sorted(equity_history.keys())
        
        # Ensure all values are simple floats
        equity_values = [float(equity_history[date]) for date in dates]
        
        # Create the plot
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        # Plot equity curve
        ax.plot(dates, equity_values, 'b-', linewidth=2)
        ax.set_title('Equity Curve', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True)
        
        # Format x-axis dates
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        fig.autofmt_xdate()
        
        # Convert to base64 image
        buf = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(buf)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        
        return f"data:image/png;base64,{data}"
    
    def generate_drawdown_chart(self, equity_history: Dict[datetime, float]) -> str:
        """
        Generate a drawdown chart and return as base64 encoded image.
        
        Args:
            equity_history: Dictionary of equity values by date
            
        Returns:
            str: Base64 encoded PNG image
        """
        # Sort equity history by date
        dates = sorted(equity_history.keys())
        
        # Ensure all values are simple floats
        equity_values = [float(equity_history[date]) for date in dates]
        
        # Calculate drawdown series
        equity_series = pd.Series(equity_values, index=dates)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100  # Convert to percentage
        
        # Create the plot
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        # Plot drawdown
        ax.fill_between(dates, drawdown, 0, alpha=0.3, color='r')
        ax.plot(dates, drawdown, 'r-', linewidth=1)
        ax.set_title('Drawdown (%)', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        
        # Invert y-axis since drawdowns are negative
        ax.invert_yaxis()
        
        # Format x-axis dates
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        fig.autofmt_xdate()
        
        # Convert to base64 image
        buf = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(buf)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        
        return f"data:image/png;base64,{data}"
    
    def generate_trade_distribution_chart(self, trades: List[Dict[str, Any]]) -> str:
        """
        Generate a trade distribution chart and return as base64 encoded image.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            str: Base64 encoded PNG image
        """
        # Extract PnL values
        pnl_values = [trade.get('pnl', 0) for trade in trades if 'pnl' in trade]
        
        if not pnl_values:
            return ""
            
        # Create the plot
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        # Plot histogram
        ax.hist(pnl_values, bins=20, alpha=0.7, color='green')
        ax.axvline(0, color='red', linestyle='--')
        ax.set_title('Trade P&L Distribution', fontsize=14)
        ax.set_xlabel('P&L ($)')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        
        # Convert to base64 image
        buf = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(buf)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        
        return f"data:image/png;base64,{data}"
    
    def _generate_html_content(
        self, 
        equity_history: Dict[datetime, float],
        performance_metrics: Dict[str, float],
        trades: List[Dict[str, Any]]
    ) -> str:
        """
        Generate the full HTML report content.
        
        Args:
            equity_history: Dictionary of equity values by date
            performance_metrics: Dictionary of performance metrics
            trades: List of trade dictionaries
            
        Returns:
            str: HTML content
        """
        # Generate charts
        equity_chart = self.generate_performance_chart(equity_history)
        drawdown_chart = self.generate_drawdown_chart(equity_history)
        trade_chart = self.generate_trade_distribution_chart(trades)
        
        # Create HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
                .metric-card {{ background-color: #f5f5f5; border-radius: 5px; padding: 15px; margin: 10px; min-width: 200px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #3a78b5; }}
                .metric-name {{ color: #666; }}
                .chart {{ margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Trading Strategy Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Performance Metrics</h2>
                <div class="metrics">
        """
        
        # Add performance metrics
        for name, value in performance_metrics.items():
            # Format based on metric type
            formatted_value = value
            if name in ['return', 'cagr', 'volatility']:
                formatted_value = f"{value:.2%}"
            elif name in ['sharpe_ratio', 'sortino_ratio']:
                formatted_value = f"{value:.2f}"
            elif name in ['max_drawdown']:
                formatted_value = f"{value:.2%}"
            elif name in ['profit_factor', 'win_rate']:
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = f"{value:,.2f}"
                
            html += f"""
                    <div class="metric-card">
                        <div class="metric-name">{name.replace('_', ' ').title()}</div>
                        <div class="metric-value">{formatted_value}</div>
                    </div>
            """
            
        html += """
                </div>
                
                <h2>Equity Curve</h2>
                <div class="chart">
                    <img src="{}" width="100%" />
                </div>
        """.format(equity_chart)
        
        html += """
                <h2>Drawdown</h2>
                <div class="chart">
                    <img src="{}" width="100%" />
                </div>
        """.format(drawdown_chart)
        
        if trade_chart:
            html += """
                    <h2>Trade Distribution</h2>
                    <div class="chart">
                        <img src="{}" width="100%" />
                    </div>
            """.format(trade_chart)
        
        # Add trade table if there are trades
        if trades:
            html += """
                <h2>Trade History</h2>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Symbol</th>
                        <th>Action</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>P&L</th>
                    </tr>
            """
            
            # Add trade rows
            for trade in trades[-50:]:  # Show last 50 trades
                date = trade.get('date', '')
                if isinstance(date, datetime):
                    date = date.strftime('%Y-%m-%d')
                    
                symbol = trade.get('symbol', '')
                action = trade.get('action', '')
                quantity = trade.get('quantity', 0)
                price = trade.get('price', 0)
                pnl = trade.get('pnl', 0)
                
                pnl_class = "positive" if pnl >= 0 else "negative"
                pnl_sign = "+" if pnl > 0 else ""
                
                html += f"""
                    <tr>
                        <td>{date}</td>
                        <td>{symbol}</td>
                        <td>{action}</td>
                        <td>{quantity}</td>
                        <td>${price:.2f}</td>
                        <td class="{pnl_class}">{pnl_sign}${pnl:.2f}</td>
                    </tr>
                """
                
            html += """
                </table>
            """
            
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def save_trade_log(self, trades: List[Dict[str, Any]], filename: str = "trade_log.csv") -> str:
        """
        Save trade history to a CSV file.
        
        Args:
            trades: List of trade dictionaries
            filename: Filename for the CSV
            
        Returns:
            str: File path to the CSV
        """
        if not trades:
            self.logger.warning("No trades to save")
            return ""
            
        # Convert to DataFrame
        trade_df = pd.DataFrame(trades)
        
        # Save to file
        file_path = os.path.join(self.output_dir, filename)
        trade_df.to_csv(file_path, index=False)
        
        self.logger.info(f"Trade log saved to {file_path}")
        
        return file_path
    
    def save_performance_metrics(
        self, 
        metrics: Dict[str, Any], 
        filename: str = "performance_metrics.csv"
    ) -> str:
        """
        Save performance metrics to a CSV file.
        
        Args:
            metrics: Dictionary of performance metrics
            filename: Filename for the CSV
            
        Returns:
            str: File path to the CSV
        """
        if not metrics:
            self.logger.warning("No metrics to save")
            return ""
            
        # Format for CSV - convert to rows
        rows = []
        for key, value in metrics.items():
            rows.append({"Metric": key, "Value": value})
            
        # Convert to DataFrame
        metrics_df = pd.DataFrame(rows)
        
        # Save to file
        file_path = os.path.join(self.output_dir, filename)
        metrics_df.to_csv(file_path, index=False)
        
        self.logger.info(f"Performance metrics saved to {file_path}")
        
        return file_path
    
    def generate_business_logic_verification_file(self, portfolio, date, pre_trade=False):
        """
        Generate a detailed output file for business logic verification in the required format.
        
        Args:
            portfolio: Portfolio object containing all position data
            date: Current date for the report
            pre_trade: Whether this is a pre-trade or post-trade report
        
        Returns:
            str: Path to the generated file
        """
        report_type = "PRE-TRADE" if pre_trade else "POST-TRADE"
        date_str = date.strftime("%Y-%m-%d")
        
        # Initialize the output content
        output = io.StringIO()
        
        # Header Section
        output.write("=" * 60 + "\n")
        output.write(f"{report_type} Summary [{date_str}]:\n")
        
        # Daily P&L Breakdown
        daily_return = portfolio.get_daily_return()
        daily_return_pct = portfolio.get_daily_return_percent()
        option_pnl = portfolio.get_option_pnl()
        hedge_pnl = portfolio.get_hedge_pnl()
        
        output.write(f"Daily P&L: ${daily_return:.2f} ({daily_return_pct:.2%})\n")
        output.write(f"  Option PnL: ${option_pnl['total']:.2f}\n")
        output.write(f"  Hedge PnL: ${hedge_pnl['total']:.2f}\n")
        
        # Trade and Position Information
        open_trades = len(portfolio.get_open_positions())
        position_exposure = portfolio.get_total_position_exposure()
        nlv = portfolio.get_net_liquidation_value()
        cash_balance = portfolio.get_cash_balance()
        total_liability = portfolio.get_total_liability()
        self_hedge = portfolio.get_hedge_value()
        
        output.write(f"Open Trades: {open_trades}\n")
        output.write(f"Total Position Exposure: {position_exposure:.1%} of NLV\n")
        output.write(f"Net Liq: ${nlv:.2f}\n")
        output.write(f"  Cash Balance: ${cash_balance:.2f}\n")
        output.write(f"  Total Liability: ${total_liability:.2f}\n")
        output.write(f"  Self Hedge (Hedge PnL): ${self_hedge:.2f}\n")
        
        # Margin Information
        total_margin = portfolio.get_total_margin_requirement()
        available_margin = portfolio.get_available_margin()
        margin_based_leverage = portfolio.get_margin_based_leverage()
        
        output.write(f"Total Margin Requirement: ${total_margin:.2f}\n")
        output.write(f"Available Margin: ${available_margin:.2f}\n")
        output.write(f"Margin-Based Leverage: {margin_based_leverage:.2f}\n")
        
        # Portfolio Greek Risk Metrics
        option_delta = portfolio.get_option_delta()
        hedge_delta = portfolio.get_hedge_delta()
        total_delta = portfolio.get_total_delta()
        gamma = portfolio.get_gamma()
        theta = portfolio.get_theta()
        vega = portfolio.get_vega()
        
        output.write("\nPortfolio Greek Risk:\n")
        output.write(f"  Option Delta: {option_delta:.3f} (${option_delta * 100 * 100:.2f})\n")
        output.write(f"  Hedge Delta: {hedge_delta:.3f} (${hedge_delta * 100 * 100:.2f})\n")
        output.write(f"  Total Delta: {total_delta:.3f} (${total_delta * 100 * 100:.2f})\n")
        output.write(f"  Gamma: {gamma:.6f} (${gamma * 100 * 100:.2f} per 1% move)\n")
        output.write(f"  Theta: ${theta:.2f} per day\n")
        output.write(f"  Vega: ${vega:.2f} per 1% IV\n")
        
        # Rolling Metrics
        metrics = portfolio.get_rolling_metrics()
        
        output.write("\nRolling Metrics:\n")
        output.write(f"  Expanding Window (all obs, min 5 required): Sharpe: {metrics.get('expanding_sharpe', 0):.2f}, " +
                    f"Volatility: {metrics.get('expanding_volatility', 0):.2%}\n")
        output.write(f"  Short Window (21 days, rolling): Sharpe: {metrics.get('short_sharpe', 0):.2f}, " +
                    f"Volatility: {metrics.get('short_volatility', 0):.2%}\n")
        output.write(f"  Medium Window (63 days, rolling): Sharpe: {metrics.get('medium_sharpe', 0):.2f}, " +
                    f"Volatility: {metrics.get('medium_volatility', 0):.2%}\n")
        output.write(f"  Long Window (252 days, rolling): Sharpe: {metrics.get('long_sharpe', 0):.2f}, " +
                    f"Volatility: {metrics.get('long_volatility', 0):.2%}\n")
        
        output.write("-" * 50 + "\n")
        
        # Open Trades Table
        open_positions = portfolio.get_open_positions()
        if open_positions:
            output.write("\nOpen Trades Table:\n")
            output.write("-" * 120 + "\n")
            output.write(f"{'Symbol':<15} {'Contracts':>10} {'Entry':>7} {'Current':>8} {'Value':>10} {'NLV%':>5} {'Underlying':>10} " +
                         f"{'Delta':>10} {'Gamma':>10} {'Theta':>10} {'Vega':>10} {'Margin':>10} {'DTE':>5}\n")
            output.write("-" * 120 + "\n")
            
            total_value = 0
            total_margin = 0
            
            for pos in open_positions:
                symbol = pos.get_symbol() if hasattr(pos, 'get_symbol') else pos.symbol
                contracts = pos.get_quantity() if hasattr(pos, 'get_quantity') else pos.contracts
                entry_price = pos.get_entry_price() if hasattr(pos, 'get_entry_price') else pos.avg_entry_price
                current_price = pos.get_current_price() if hasattr(pos, 'get_current_price') else pos.current_price
                
                # Calculate position value
                if hasattr(pos, 'get_market_value'):
                    value = pos.get_market_value()
                else:
                    # Check if this is an option position
                    is_option = hasattr(pos, 'option_symbol') or hasattr(pos, 'strike') or hasattr(pos, 'expiration')
                    value = current_price * contracts * (100 if is_option else 1)
                
                nlv_pct = value / nlv * 100 if nlv else 0
                
                # Get underlying price
                if hasattr(pos, 'get_underlying_price'):
                    underlying_price = pos.get_underlying_price()
                else:
                    underlying_price = pos.underlying_price if hasattr(pos, 'underlying_price') else 0
                
                # Get delta
                if hasattr(pos, 'get_delta'):
                    delta = pos.get_delta()
                else:
                    delta = pos.current_delta if hasattr(pos, 'current_delta') else 0
                
                # Get gamma
                if hasattr(pos, 'get_gamma'):
                    gamma = pos.get_gamma()
                else:
                    gamma = pos.current_gamma if hasattr(pos, 'current_gamma') else 0
                
                # Get theta
                if hasattr(pos, 'get_theta'):
                    theta = pos.get_theta()
                else:
                    theta = pos.current_theta if hasattr(pos, 'current_theta') else 0
                
                # Get vega
                if hasattr(pos, 'get_vega'):
                    vega = pos.get_vega()
                else:
                    vega = pos.current_vega if hasattr(pos, 'current_vega') else 0
                
                # Get margin requirement
                if hasattr(pos, 'get_margin_requirement'):
                    margin = pos.get_margin_requirement()
                else:
                    margin = 0
                total_margin += margin
                
                # Get days to expiry
                if hasattr(pos, 'get_days_to_expiry'):
                    dte = pos.get_days_to_expiry()
                else:
                    dte = pos.days_to_expiry if hasattr(pos, 'days_to_expiry') else 0
                
                # Write position data
                output.write(f"{symbol:<15} {contracts:>10} ${entry_price:>5.2f} ${current_price:>6.2f} " +
                           f"${value:>8.2f} {nlv_pct:>4.1f}% ${underlying_price:>8.2f} " +
                           f"{delta:>9.3f} {gamma:>9.6f} ${theta:>8.2f} ${vega:>8.2f} ${margin:>8.0f} {dte:>5}\n")
                
                total_value += value
            
            # Write table footer with totals
            output.write("-" * 120 + "\n")
            output.write(f"{'TOTAL':<15} {'':<10} {'':<7} {'':<8} ${total_value:>8.2f} {(total_value / nlv * 100) if nlv else 0:>4.1f}%{'':<31}${total_margin:>8.0f}\n")
            output.write("-" * 120 + "\n")
        
        output.write("=" * 60 + "\n")
        
        # Logging and Manager Output section - Fixed to properly get the log file path
        # Try to access the log file through various possible methods
        log_file_path = None
        
        # Method 1: Try to get it from a trading_engine reference which might have the LoggingManager
        if hasattr(self, 'trading_engine') and hasattr(self.trading_engine, 'logging_manager'):
            log_file_path = self.trading_engine.logging_manager.get_log_file_path()
        
        # Method 2: Check if logger is actually a LoggingManager
        elif hasattr(self, 'logger') and hasattr(self.logger, 'get_log_file_path'):
            log_file_path = self.logger.get_log_file_path()
        
        # Method 3: Check if we can access the log file directly from the logger
        elif hasattr(self, 'logger') and hasattr(self.logger, 'handlers'):
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler) and hasattr(handler, 'baseFilename'):
                    log_file_path = handler.baseFilename
                    break
        
        if log_file_path and os.path.exists(log_file_path):
            try:
                with open(log_file_path, 'r') as log_file:
                    log_content = log_file.read()
                    # Filter log entries for current date
                    log_date_str = date.strftime("%Y-%m-%d")
                    log_lines = [line for line in log_content.split('\n') if log_date_str in line]
                    
                    # Include key log entries based on tags
                    trade_manager_logs = [line for line in log_lines if "[TradeManager]" in line]
                    risk_scaling_logs = [line for line in log_lines if "[Risk Scaling]" in line]
                    portfolio_logs = [line for line in log_lines if "[Portfolio Rebalancer]" in line]
                    
                    if trade_manager_logs:
                        output.write("\n[TradeManager] Logs:\n")
                        for line in trade_manager_logs:
                            output.write(f"{line}\n")
                    
                    if risk_scaling_logs:
                        output.write("\n[Risk Scaling] Logs:\n")
                        for line in risk_scaling_logs:
                            output.write(f"{line}\n")
                    
                    if portfolio_logs:
                        output.write("\n[Portfolio Rebalancer] Logs:\n")
                        for line in portfolio_logs:
                            output.write(f"{line}\n")
            except Exception as e:
                self.logger.error(f"Error reading log file: {e}")
        
        # Write the output file
        output_dir = self.get_output_directory()
        file_prefix = "pre_trade" if pre_trade else "post_trade"
        output_file_path = os.path.join(output_dir, f"{file_prefix}_summary_{date_str}.txt")
        
        with open(output_file_path, 'w') as f:
            f.write(output.getvalue())
        
        return output_file_path
    
    def run_reports(self, portfolio, engine, current_date=None):
        """
        Generate all reports including HTML report and business logic verification files.
        
        Args:
            portfolio: Portfolio object
            engine: Trading engine instance
            current_date: Current date for the report (defaults to today)
            
        Returns:
            dict: Paths to generated reports
        """
        result = {}
        date = current_date or datetime.now()
        
        try:
            # Generate business logic verification files
            pre_trade_file = self.generate_business_logic_verification_file(
                portfolio, date, pre_trade=True)
            post_trade_file = self.generate_business_logic_verification_file(
                portfolio, date, pre_trade=False)
            
            result['pre_trade_file'] = pre_trade_file
            result['post_trade_file'] = post_trade_file
        except Exception as e:
            self.logger.error(f"Error generating verification files: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        return result
    
    def get_output_directory(self) -> str:
        """
        Get the output directory for reports and verification files.
        
        Returns:
            str: Path to the output directory
        """
        # First try to get the verification_output_dir if it exists
        verification_dir = self.config.get('paths', {}).get('verification_output_dir')
        if verification_dir:
            os.makedirs(verification_dir, exist_ok=True)
            return verification_dir
            
        # Otherwise use the general output_dir
        output_dir = self.config.get('paths', {}).get('output_dir', 'output')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir