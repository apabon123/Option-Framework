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
    Generates reports and visualizations for trading performance.
    
    This class handles creating HTML reports, generating charts, and
    calculating performance metrics for trading strategies.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        portfolio: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the ReportingSystem.
        
        Args:
            config: Configuration dictionary
            portfolio: Portfolio instance for performance tracking
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger('trading_engine')
        self.config = config
        self.portfolio = portfolio
        
        # Extract configuration settings
        self.output_dir = config.get('output_dir', 'reports')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_html_report(
        self, 
        equity_history: Dict[datetime, float], 
        performance_metrics: Dict[str, float],
        trades: List[Dict[str, Any]],
        report_name: str = "strategy_report"
    ) -> str:
        """
        Generate a full HTML report with performance metrics and charts.
        
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
        equity_values = [equity_history[date] for date in dates]
        
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
        equity_values = [equity_history[date] for date in dates]
        
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