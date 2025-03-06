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
        output_dir: str, 
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the ReportingSystem.
        
        Args:
            output_dir: Directory to save reports
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger('trading')
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_html_report(
        self, 
        equity_history: Dict[datetime, float], 
        position_history: Optional[List[Dict[str, Any]]] = None,
        risk_metrics: Optional[Dict[str, Any]] = None,
        greeks_history: Optional[Dict[datetime, Dict[str, float]]] = None,
        config: Optional[Dict[str, Any]] = None,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate an HTML performance report with interactive charts.
        
        Args:
            equity_history: Dictionary of {date: equity_value}
            position_history: List of position records
            risk_metrics: Dictionary of risk metrics
            greeks_history: Dictionary of {date: greeks_data}
            config: Configuration dictionary used for the strategy
            output_file: Output file path (optional)
            
        Returns:
            str: Path to the saved report
        """
        try:
            # Generate output filename if not provided
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = os.path.join(self.output_dir, f"performance_report_{timestamp}.html")
            
            # Convert equity history to DataFrame
            equity_df = pd.DataFrame.from_dict(equity_history, orient='index', columns=['Equity'])
            equity_df.index.name = 'Date'
            equity_df.index = pd.to_datetime(equity_df.index)
            
            # Generate performance metrics
            performance_metrics = self.calculate_performance_metrics(equity_df)
            
            # Generate charts
            equity_chart = self.generate_equity_chart(equity_df)
            drawdown_chart = self.generate_drawdown_chart(equity_df)
            
            # Generate delta chart if greeks_history is provided
            delta_chart = None
            if greeks_history:
                delta_chart = self.generate_greek_chart(greeks_history, equity_df, 'delta')
            
            # Generate returns heatmap
            returns_heatmap = self.generate_returns_heatmap(equity_df)
            
            # Build HTML report
            html_content = self._build_html_report(
                performance_metrics,
                equity_chart,
                drawdown_chart,
                delta_chart,
                returns_heatmap,
                config
            )
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write(html_content)
                
            self.logger.info(f"HTML report saved to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def generate_equity_chart(self, equity_df: pd.DataFrame) -> str:
        """
        Generate a base64-encoded image of the equity curve.
        
        Args:
            equity_df: DataFrame with equity history
            
        Returns:
            str: Base64-encoded PNG image
        """
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot equity curve
        ax.plot(equity_df.index, equity_df['Equity'], linewidth=2, color='#0066cc')
        
        # Add horizontal line at starting equity
        ax.axhline(y=equity_df['Equity'].iloc[0], color='gray', linestyle='--', alpha=0.7)
        
        # Format x-axis to show dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Add grid, title and labels
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('Equity Curve', fontsize=14)
        ax.set_ylabel('Net Liquidation Value ($)')
        ax.set_xlabel('Date')
        
        # Format y-axis with dollar sign
        ax.yaxis.set_major_formatter('${x:,.0f}')
        
        # Calculate and show CAGR
        if len(equity_df) > 1:
            start_value = equity_df['Equity'].iloc[0]
            end_value = equity_df['Equity'].iloc[-1]
            days = (equity_df.index[-1] - equity_df.index[0]).days
            years = days / 365 if days > 0 else 1
            cagr = (end_value / start_value) ** (1 / years) - 1
            
            # Add CAGR annotation
            ax.annotate(f'CAGR: {cagr:.2%}', 
                        xy=(0.02, 0.95), 
                        xycoords='axes fraction',
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Ensure tight layout
        plt.tight_layout()
        
        # Convert plot to PNG image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Encode PNG image to base64 string
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Close plot
        plt.close(fig)
        
        return img_str
    
    def generate_drawdown_chart(self, equity_df: pd.DataFrame) -> str:
        """
        Generate a base64-encoded image of the drawdown chart.
        
        Args:
            equity_df: DataFrame with equity history
            
        Returns:
            str: Base64-encoded PNG image
        """
        # Calculate drawdowns
        if len(equity_df) <= 1:
            # Not enough data for drawdown calculation
            return None
            
        # Calculate returns
        equity_df['Return'] = equity_df['Equity'].pct_change()
        
        # Calculate cumulative returns and drawdowns
        equity_df['Cum_Return'] = (1 + equity_df['Return']).cumprod()
        equity_df['Peak'] = equity_df['Cum_Return'].cummax()
        equity_df['Drawdown'] = (equity_df['Cum_Return'] / equity_df['Peak'] - 1) * 100
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot drawdown
        ax.fill_between(equity_df.index, equity_df['Drawdown'], 0, color='red', alpha=0.3)
        ax.plot(equity_df.index, equity_df['Drawdown'], color='red', linewidth=1)
        
        # Format x-axis to show dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Add grid, title and labels
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('Drawdown Chart', fontsize=14)
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Date')
        
        # Invert y-axis for better visualization (drawdowns are negative)
        ax.invert_yaxis()
        
        # Add max drawdown annotation
        max_dd = equity_df['Drawdown'].min()
        max_dd_date = equity_df.loc[equity_df['Drawdown'] == max_dd].index[0]
        
        ax.annotate(f'Max Drawdown: {max_dd:.2f}%', 
                    xy=(max_dd_date, max_dd), 
                    xytext=(max_dd_date, max_dd * 0.8),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Ensure tight layout
        plt.tight_layout()
        
        # Convert plot to PNG image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Encode PNG image to base64 string
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Close plot
        plt.close(fig)
        
        return img_str
    
    def generate_greek_chart(
        self, 
        greeks_history: Dict[datetime, Dict[str, float]], 
        equity_df: pd.DataFrame, 
        greek: str
    ) -> str:
        """
        Generate a chart for a specific Greek metric over time.
        
        Args:
            greeks_history: Dictionary of {date: greeks_data}
            equity_df: DataFrame with equity history for date alignment
            greek: Which greek to plot ('delta', 'gamma', 'theta', 'vega')
            
        Returns:
            str: Base64-encoded PNG image
        """
        try:
            # Convert greeks history to DataFrame
            dates = []
            values = []
            dollar_values = []
            
            for date, metrics in greeks_history.items():
                dates.append(date)
                # Try different key formats that might exist in the data
                greek_keys = [greek, greek.lower(), greek.upper()]
                dollar_keys = [f'dollar_{greek}', f'dollar_{greek.lower()}']
                
                # Find the greek value
                for key in greek_keys:
                    if key in metrics:
                        values.append(metrics[key])
                        break
                else:
                    values.append(0)
                    
                # Find the dollar value
                for key in dollar_keys:
                    if key in metrics:
                        dollar_values.append(metrics[key])
                        break
                else:
                    dollar_values.append(0)
            
            # Create DataFrame
            greek_df = pd.DataFrame({
                'date': dates,
                f'{greek}': values,
                f'dollar_{greek}': dollar_values
            })
            greek_df['date'] = pd.to_datetime(greek_df['date'])
            greek_df.set_index('date', inplace=True)
            greek_df.sort_index(inplace=True)
            
            # Align with equity_df dates
            aligned_df = pd.merge(greek_df, equity_df, left_index=True, right_index=True, how='outer')
            aligned_df.fillna(method='ffill', inplace=True)
            
            # Calculate percentage of portfolio value
            if 'Equity' in aligned_df.columns and f'dollar_{greek}' in aligned_df.columns:
                aligned_df[f'{greek}_pct'] = aligned_df[f'dollar_{greek}'] / aligned_df['Equity'] * 100
            
            # Create figure with two y-axes
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Plot absolute greek
            color = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel(f'{greek.capitalize()} Value', color=color)
            ax1.plot(aligned_df.index, aligned_df[greek], color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Create second y-axis
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel(f'{greek.capitalize()} % of Portfolio', color=color)
            
            if f'{greek}_pct' in aligned_df.columns:
                ax2.plot(aligned_df.index, aligned_df[f'{greek}_pct'], color=color, linestyle='--')
                ax2.tick_params(axis='y', labelcolor=color)
            
            # Format x-axis to show dates nicely
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
            
            # Add grid, title and labels
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_title(f'{greek.capitalize()} Over Time', fontsize=14)
            
            # Ensure tight layout
            fig.tight_layout()
            
            # Convert plot to PNG image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Encode PNG image to base64 string
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            # Close plot
            plt.close(fig)
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error generating {greek} chart: {e}")
            return None
    
    def generate_returns_heatmap(self, equity_df: pd.DataFrame) -> str:
        """
        Generate an HTML returns heatmap by month and year.
        
        Args:
            equity_df: DataFrame with equity history
            
        Returns:
            str: HTML table for the returns heatmap
        """
        try:
            # Check if we have sufficient data
            if equity_df is None or len(equity_df) < 5 or 'Equity' not in equity_df.columns:
                return "<p>Insufficient data for returns heatmap</p>"
            
            # Calculate daily returns
            equity_df['Return'] = equity_df['Equity'].pct_change()
            
            # Resample to monthly returns
            monthly_returns = equity_df['Return'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            
            # Check if we have enough monthly returns
            if len(monthly_returns) < 2:
                return "<p>Insufficient monthly data for returns heatmap</p>"
            
            # Extract year and month directly from the datetime index
            returns_data = []
            for date, value in monthly_returns.items():
                returns_data.append({
                    'Year': date.year,
                    'Month': date.month,
                    'Return': value
                })
            
            # Convert to DataFrame
            monthly_df = pd.DataFrame(returns_data)
            
            # Check if we have enough unique years/months
            if len(monthly_df['Year'].unique()) < 1 or len(monthly_df['Month'].unique()) < 2:
                return "<p>Insufficient data variation for returns heatmap</p>"
            
            # Create pivot table
            returns_pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
            
            # Define month names for columns
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Map column indices to month names
            col_mapping = {i: month_names[i-1] for i in range(1, 13) if i in returns_pivot.columns}
            returns_pivot = returns_pivot.rename(columns=col_mapping)
            
            # Generate HTML for the heatmap
            html = """
            <div class="returns-heatmap">
                <h3>Monthly Returns Heatmap</h3>
                <table class="heatmap-table">
                    <thead>
                        <tr>
                            <th>Year</th>
            """
            
            # Add month headers
            for month in returns_pivot.columns:
                html += f"<th>{month}</th>"
            
            # Add annual returns column
            html += "<th>Annual</th></tr></thead><tbody>"
            
            # Calculate max absolute return for color scaling
            max_abs_return = max(
                abs(returns_pivot.fillna(0).values.min()),
                abs(returns_pivot.fillna(0).values.max())
            )
            
            if pd.isna(max_abs_return) or max_abs_return == 0:
                max_abs_return = 0.05  # Default to 5% if no valid returns
            
            # Add data rows
            for year in returns_pivot.index:
                html += f"<tr><td>{year}</td>"
                
                # Calculate annual return
                annual_return = (1 + returns_pivot.loc[year].fillna(0)).prod() - 1
                
                # Add monthly returns with color coding
                for month in returns_pivot.columns:
                    if month in returns_pivot.columns and not pd.isna(returns_pivot.loc[year, month]):
                        value = returns_pivot.loc[year, month]
                        # Scale color intensity based on return magnitude
                        intensity = min(abs(value) / max_abs_return * 0.8, 0.8)
                        if value > 0:
                            bgcolor = f"rgba(0, 128, 0, {intensity})"  # Green for positive
                            color = "white" if intensity > 0.5 else "black"
                        else:
                            bgcolor = f"rgba(220, 0, 0, {intensity})"  # Red for negative
                            color = "white" if intensity > 0.5 else "black"
                        
                        html += f'<td style="background-color: {bgcolor}; color: {color}">{value:.2%}</td>'
                    else:
                        html += '<td></td>'
                
                # Add annual return
                if not pd.isna(annual_return):
                    # Color annual return
                    intensity = min(abs(annual_return) / max_abs_return * 0.8, 0.8)
                    if annual_return > 0:
                        bgcolor = f"rgba(0, 128, 0, {intensity})"  # Green for positive
                        color = "white" if intensity > 0.5 else "black"
                    else:
                        bgcolor = f"rgba(220, 0, 0, {intensity})"  # Red for negative
                        color = "white" if intensity > 0.5 else "black"
                    
                    html += f'<td style="background-color: {bgcolor}; color: {color}"><strong>{annual_return:.2%}</strong></td>'
                else:
                    html += '<td></td>'
                
                html += "</tr>"
            
            # Calculate and add average monthly returns
            html += "<tr><td><strong>Avg</strong></td>"
            
            # Add monthly averages
            for month in returns_pivot.columns:
                monthly_avg = returns_pivot[month].mean()
                if not pd.isna(monthly_avg):
                    # Color monthly average
                    intensity = min(abs(monthly_avg) / max_abs_return * 0.8, 0.8)
                    if monthly_avg > 0:
                        bgcolor = f"rgba(0, 128, 0, {intensity})"  # Green for positive
                        color = "white" if intensity > 0.5 else "black"
                    else:
                        bgcolor = f"rgba(220, 0, 0, {intensity})"  # Red for negative
                        color = "white" if intensity > 0.5 else "black"
                    
                    html += f'<td style="background-color: {bgcolor}; color: {color}"><strong>{monthly_avg:.2%}</strong></td>'
                else:
                    html += '<td></td>'
            
            # Calculate and add average annual return
            avg_annual = returns_pivot.fillna(0).apply(lambda x: (1 + x).prod() - 1, axis=1).mean()
            
            if not pd.isna(avg_annual):
                intensity = min(abs(avg_annual) / max_abs_return * 0.8, 0.8)
                if avg_annual > 0:
                    bgcolor = f"rgba(0, 128, 0, {intensity})"  # Green for positive
                    color = "white" if intensity > 0.5 else "black"
                else:
                    bgcolor = f"rgba(220, 0, 0, {intensity})"  # Red for negative
                    color = "white" if intensity > 0.5 else "black"
                
                html += f'<td style="background-color: {bgcolor}; color: {color}"><strong>{avg_annual:.2%}</strong></td>'
            else:
                html += '<td></td>'
            
            html += "</tr></tbody></table></div>"
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error generating returns heatmap: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return f"<p>Error generating returns heatmap: {str(e)}</p>"
    
    def _config_to_html(self, config: Optional[Dict[str, Any]]) -> str:
        """
        Convert configuration dictionary to HTML table.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            str: HTML table
        """
        if not config:
            return "<p>No configuration data available</p>"
            
        html = """
        <div class="config-section">
            <h3>Strategy Configuration</h3>
            <div class="accordion">
        """
        
        # Define categories to organize config settings
        categories = {
            'portfolio': 'Portfolio Settings',
            'risk': 'Risk Management',
            'strategy': 'Strategy Parameters',
            'data': 'Data Settings',
            'paths': 'File Paths'
        }
        
        # Create accordion sections for each category
        for i, (category_key, category_name) in enumerate(categories.items()):
            if category_key in config:
                html += f"""
                <div class="accordion-item">
                    <h2 class="accordion-header" id="heading{i}">
                        <button class="accordion-button collapsed" type="button" 
                                data-bs-toggle="collapse" data-bs-target="#collapse{i}">
                            {category_name}
                        </button>
                    </h2>
                    <div id="collapse{i}" class="accordion-collapse collapse" 
                         aria-labelledby="heading{i}">
                        <div class="accordion-body">
                            <table class="table table-sm table-striped">
                                <thead>
                                    <tr>
                                        <th>Parameter</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                """
                
                for param, value in config[category_key].items():
                    # Format the value for display
                    if isinstance(value, float):
                        if 0 < value < 1:  # Format small values as percentages
                            formatted_value = f"{value:.2%}"
                        else:
                            formatted_value = f"{value:,.3f}"
                    elif isinstance(value, int):
                        formatted_value = f"{value:,}"
                    elif isinstance(value, (datetime, pd.Timestamp)):
                        formatted_value = value.strftime('%Y-%m-%d')
                    elif isinstance(value, bool):
                        formatted_value = "Yes" if value else "No"
                    elif isinstance(value, (list, tuple)):
                        formatted_value = ", ".join(str(v) for v in value)
                    elif isinstance(value, dict):
                        formatted_value = "<pre>" + str(value) + "</pre>"
                    else:
                        formatted_value = str(value)
                        
                    html += f"""
                                    <tr>
                                        <td>{param}</td>
                                        <td>{formatted_value}</td>
                                    </tr>
                    """
                
                html += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def calculate_performance_metrics(self, equity_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics from equity history.
        
        Args:
            equity_df: DataFrame with equity history
            
        Returns:
            dict: Dictionary of performance metrics
        """
        metrics = {}
        
        try:
            # Basic metrics
            metrics['start_value'] = equity_df['Equity'].iloc[0]
            metrics['end_value'] = equity_df['Equity'].iloc[-1]
            metrics['total_return'] = (metrics['end_value'] / metrics['start_value']) - 1
            
            # Need at least 2 data points for remaining calculations
            if len(equity_df) < 2:
                return metrics
                
            # Calculate daily returns
            daily_returns = equity_df['Equity'].pct_change().dropna()
            
            # Annualized metrics
            days = (equity_df.index[-1] - equity_df.index[0]).days
            years = max(days / 365, 0.01)  # Avoid division by zero
            
            metrics['cagr'] = (metrics['end_value'] / metrics['start_value']) ** (1 / years) - 1
            metrics['volatility'] = daily_returns.std() * np.sqrt(252)  # Annualized
            
            if metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = (daily_returns.mean() * 252) / metrics['volatility']
            else:
                metrics['sharpe_ratio'] = 0
                
            # Drawdown analysis
            cum_returns = (1 + daily_returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max - 1)
            metrics['max_drawdown'] = drawdown.min()
            
            # Calculate winning days percentage
            metrics['winning_days'] = len(daily_returns[daily_returns > 0]) / len(daily_returns)
            
            # Calculate Sortino ratio (downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std() * np.sqrt(252)
                if downside_deviation > 0:
                    metrics['sortino_ratio'] = (daily_returns.mean() * 252) / downside_deviation
                else:
                    metrics['sortino_ratio'] = 0
            else:
                metrics['sortino_ratio'] = 0
                
            # Calculate Calmar ratio (return / max drawdown)
            if metrics['max_drawdown'] != 0:
                metrics['calmar_ratio'] = metrics['cagr'] / abs(metrics['max_drawdown'])
            else:
                metrics['calmar_ratio'] = 0
                
            # Average daily return
            metrics['avg_daily_return'] = daily_returns.mean()
            
            # Max gain and loss days
            metrics['max_daily_gain'] = daily_returns.max()
            metrics['max_daily_loss'] = daily_returns.min()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return metrics
    
    def _build_html_report(
        self, 
        metrics: Dict[str, float], 
        equity_chart: str, 
        drawdown_chart: Optional[str] = None, 
        delta_chart: Optional[str] = None,
        returns_heatmap: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build the complete HTML report.
        
        Args:
            metrics: Performance metrics
            equity_chart: Base64-encoded equity chart
            drawdown_chart: Base64-encoded drawdown chart
            delta_chart: Base64-encoded delta chart
            returns_heatmap: HTML string for returns heatmap
            config: Configuration dictionary
            
        Returns:
            str: Complete HTML report
        """
        # Format metrics for display
        formatted_metrics = []
        
        # Helper function to format a value
        def format_value(key, value):
            if key in ['total_return', 'cagr', 'volatility', 'max_drawdown', 'winning_days',
                      'avg_daily_return', 'max_daily_gain', 'max_daily_loss']:
                return f"{value:.2%}"
            elif key in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
                return f"{value:.2f}"
            elif key in ['start_value', 'end_value']:
                return f"${value:,.2f}"
            else:
                return str(value)
        
        # Add key metrics to the formatted list
        key_metrics = [
            ('total_return', 'Total Return'),
            ('cagr', 'CAGR'),
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('volatility', 'Volatility (Annualized)'),
            ('max_drawdown', 'Maximum Drawdown'),
            ('sortino_ratio', 'Sortino Ratio'),
            ('calmar_ratio', 'Calmar Ratio'),
            ('winning_days', 'Winning Days %')
        ]
        
        for key, label in key_metrics:
            if key in metrics:
                formatted_metrics.append({
                    'key': key,
                    'label': label,
                    'value': metrics[key],
                    'formatted': format_value(key, metrics[key])
                })
        
        # Build HTML content
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Trading Strategy Performance Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1, h2, h3 { color: #2c3e50; margin-top: 1.5rem; }
                .card { border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }
                .card-header { background-color: #f8f9fa; font-weight: bold; }
                .metrics-row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
                .metric-card { flex: 1; min-width: 200px; text-align: center; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-title { font-weight: bold; margin-bottom: 5px; color: #6c757d; }
                .metric-value { font-size: 24px; color: #0066cc; }
                .chart-container { margin: 20px 0; text-align: center; }
                .chart-container img { max-width: 100%; height: auto; }
                .positive { color: #28a745; }
                .negative { color: #dc3545; }
                .heatmap-table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
                .heatmap-table th, .heatmap-table td { padding: 6px 8px; text-align: center; border: 1px solid #dee2e6; }
                .heatmap-table th { background-color: #f8f9fa; font-weight: bold; }
                .heatmap-table td { width: 7%; }
                .heatmap-table tr td:first-child { text-align: center; font-weight: bold; background-color: #f8f9fa; }
                .accordion-button { padding: 0.5rem 1rem; }
                .accordion-body { padding: 1rem; }
                .table { margin-bottom: 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="mb-4">Trading Strategy Performance Report</h1>
                <div class="card">
                    <div class="card-header">Performance Summary</div>
                    <div class="card-body">
                        <div class="row">
        """
        
        # Add metrics in cards
        for i, metric in enumerate(formatted_metrics[:6]):  # First 6 metrics in 2 rows of 3
            if i % 3 == 0 and i > 0:
                html += '</div><div class="row mt-3">'
                
            value_class = ""
            if metric['key'] in ['total_return', 'cagr', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'winning_days']:
                value_class = "positive" if metric['value'] >= 0 else "negative"
            elif metric['key'] in ['volatility', 'max_drawdown']:
                value_class = "negative"  # These are generally better when lower
                
            html += f"""
                            <div class="col-md-4">
                                <div class="metric-card">
                                    <div class="metric-title">{metric['label']}</div>
                                    <div class="metric-value {value_class}">{metric['formatted']}</div>
                                </div>
                            </div>
            """
        
        html += """
                        </div>
                    </div>
                </div>
        """
        
        # Add equity chart
        if equity_chart:
            html += """
                <div class="card">
                    <div class="card-header">Equity Curve</div>
                    <div class="card-body chart-container">
                        <img src="data:image/png;base64,""" + equity_chart + """" alt="Equity Curve" class="img-fluid">
                    </div>
                </div>
            """
        
        # Add drawdown chart
        if drawdown_chart:
            html += """
                <div class="card">
                    <div class="card-header">Drawdown Analysis</div>
                    <div class="card-body chart-container">
                        <img src="data:image/png;base64,""" + drawdown_chart + """" alt="Drawdown Chart" class="img-fluid">
                    </div>
                </div>
            """
        
        # Add delta chart
        if delta_chart:
            html += """
                <div class="card">
                    <div class="card-header">Portfolio Delta</div>
                    <div class="card-body chart-container">
                        <img src="data:image/png;base64,""" + delta_chart + """" alt="Delta Chart" class="img-fluid">
                    </div>
                </div>
            """
        
        # Add returns heatmap
        if returns_heatmap:
            html += """
                <div class="card">
                    <div class="card-header">Returns Analysis</div>
                    <div class="card-body">
                        """ + returns_heatmap + """
                    </div>
                </div>
            """
        
        # Add configuration section
        if config:
            html += """
                <div class="card">
                    <div class="card-header">Strategy Configuration</div>
                    <div class="card-body">
                        """ + self._config_to_html(config) + """
                    </div>
                </div>
            """
        
        # Add detailed metrics table
        html += """
                <div class="card">
                    <div class="card-header">Detailed Performance Metrics</div>
                    <div class="card-body">
                        <table class="table table-striped table-hover">
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Value</th>
                                </tr>
                            </thead>
                            <tbody>
        """
        
        # Add all metrics to the table
        for key, value in metrics.items():
            label = key.replace('_', ' ').title()
            formatted = format_value(key, value)
            
            html += f"""
                                <tr>
                                    <td>{label}</td>
                                    <td>{formatted}</td>
                                </tr>
            """
        
        html += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                // Initialize tooltips
                var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
                var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
                    return new bootstrap.Tooltip(tooltipTriggerEl)
                });
                
                // Make all external links open in a new tab
                document.addEventListener('DOMContentLoaded', function() {
                    var links = document.querySelectorAll('a[href^="http"]');
                    links.forEach(function(link) {
                        link.setAttribute('target', '_blank');
                        link.setAttribute('rel', 'noopener noreferrer');
                    });
                });
            </script>
        </body>
        </html>
        """
        
        return html
    
    def save_trade_log(self, trade_log: pd.DataFrame, output_file: Optional[str] = None) -> str:
        """
        Save the trade log to a CSV file.
        
        Args:
            trade_log: DataFrame of trade log
            output_file: Output file path (optional)
            
        Returns:
            str: Path to the saved file
        """
        try:
            # Generate output filename if not provided
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = os.path.join(self.output_dir, f"trade_log_{timestamp}.csv")
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to CSV
            trade_log.to_csv(output_file, index=False)
            self.logger.info(f"Trade log saved to {output_file}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error saving trade log: {e}")
            return None