# Creating Trading Strategies

This guide explains how to create new trading strategies using the Options Trading Framework, with emphasis on configuring the risk management components.

## Strategy Structure

A complete strategy in the Options Trading Framework consists of:

1. **Strategy Logic**: The core trading rules or models
2. **Position Sizing**: How much to trade
3. **Risk Scaling**: When to increase or decrease exposure
4. **Hedging Configuration**: How to hedge positions
5. **Execution Parameters**: How to enter and exit positions

## Creating a Basic Strategy

### 1. Create a Strategy Configuration File

Start by creating a new YAML file in the `config/strategy/` directory:

```yaml
# config/strategy/my_put_selling_strategy.yaml
strategy:
  name: "Put Selling Strategy"
  description: "Mechanical put selling strategy with risk management"
  
  # Strategy Parameters
  parameters:
    underlying: "SPY"
    dte_min: 30
    dte_max: 45
    delta_target: 0.20
    entry_days: ["Monday", "Wednesday", "Friday"]
    max_open_positions: 10
    profit_target_pct: 0.50
    stop_loss_pct: 1.50
    
  # Position Sizing Configuration
  position_sizing:
    enabled: true
    max_position_size_pct: 0.05
    min_position_size: 1
    max_leverage: 1.5
    conservative_pct: 0.85
    
  # Risk Scaling Configuration
  risk_scaling:
    enabled: true
    method: "sharpe"
    rolling_window: 21
    min_investment: 0.25
    sharpe:
      min_sharpe: 0.5
      target_sharpe: 1.5
    
  # Hedging Configuration
  hedging:
    enabled: true
    hedge_method: "delta"
    delta:
      hedge_ratio: 0.5
      use_stock: true
```

### 2. Create a Strategy Class (Optional)

For more complex strategies, you can create a Python class:

```python
# strategies/put_selling_strategy.py
from core.strategy import Strategy
from core.position_sizer import PositionSizer
from core.risk_scaler import RiskScaler
from utils.logger import get_logger

class PutSellingStrategy(Strategy):
    def __init__(self, config, trading_engine):
        super().__init__(config, trading_engine)
        self.logger = get_logger(__name__)
        
        # Extract strategy parameters
        self.underlying = self.params.get("underlying", "SPY")
        self.dte_min = self.params.get("dte_min", 30)
        self.dte_max = self.params.get("dte_max", 45)
        self.delta_target = self.params.get("delta_target", 0.20)
        self.entry_days = self.params.get("entry_days", ["Monday", "Wednesday", "Friday"])
        self.max_open_positions = self.params.get("max_open_positions", 10)
        self.profit_target_pct = self.params.get("profit_target_pct", 0.50)
        self.stop_loss_pct = self.params.get("stop_loss_pct", 1.50)
    
    def process_market_data(self, data):
        """Process new market data and generate signals"""
        # Implementation of strategy logic
        
    def find_entry_opportunities(self):
        """Find new entry opportunities"""
        # Implementation of entry logic
        
    def manage_exits(self):
        """Manage existing positions for exits"""
        # Implementation of exit logic
```

## Configuring Risk Management Components

### Position Sizing Configuration

Position sizing determines how many contracts or shares to trade:

```yaml
position_sizing:
  enabled: true
  max_position_size_pct: 0.05    # Maximum 5% of account per position
  min_position_size: 1           # Minimum 1 contract
  max_leverage: 1.5              # Maximum 1.5x leverage
  conservative_pct: 0.85         # 85% safety buffer on margin
  
  # Advanced margin settings
  margin:
    use_broker_requirements: true
    hedge_benefit_pct: 0.8
    maintenance_margin_pct: 0.75
```

### Risk Scaling Configuration

Risk scaling adjusts overall exposure based on performance:

```yaml
risk_scaling:
  enabled: true
  method: "sharpe"              # Use Sharpe ratio for scaling
  rolling_window: 21            # 21-day rolling window
  min_investment: 0.25          # Minimum 25% investment
  
  # Sharpe ratio settings
  sharpe:
    min_sharpe: 0.5             # Minimum Sharpe for scaling
    target_sharpe: 1.5          # Target Sharpe for full investment
    risk_free_rate: 0.02        # 2% risk-free rate
```

Alternative risk scaling methods:

```yaml
# Volatility-based scaling
risk_scaling:
  enabled: true
  method: "volatility"
  rolling_window: 21
  volatility:
    target_volatility: 0.15
    min_variance_ratio: 0.5
    max_variance_ratio: 2.0
```

```yaml
# Combined scaling (multiple methods)
risk_scaling:
  enabled: true
  method: "combined"
  rolling_window: 21
  combined:
    weights: [0.33, 0.33, 0.34]  # Weights for [volatility, sharpe, adaptive]
```

### Hedging Configuration

Configure how your strategy manages risk through hedging:

```yaml
hedging:
  enabled: true
  hedge_method: "delta"           # Delta-based hedging
  rebalance_threshold: 0.1        # Rebalance when delta changes by 10%
  
  # Delta hedging settings
  delta:
    hedge_ratio: 0.5              # Hedge 50% of delta exposure
    use_stock: true               # Use stock for hedging
    use_options: false            # Don't use options for hedging
```

Alternative hedging methods:

```yaml
# Beta-based hedging
hedging:
  enabled: true
  hedge_method: "beta"
  beta:
    hedge_instrument: "SPY"       # Use SPY for hedging
    beta_lookback: 60             # 60-day lookback for beta calculation
    hedge_ratio: 0.8              # Hedge 80% of beta exposure
```

## Strategy Integration

### How Components Work Together

1. **Trading Engine** coordinates all components
2. **Position Sizer** calculates appropriate position sizes
3. **Risk Scaler** adjusts overall exposure
4. **Margin Manager** calculates margin requirements
5. **Hedging Manager** manages hedge positions

### Example Integration

```python
def execute_entry_signal(self, signal):
    # Get the risk scaling factor
    risk_factor = self.risk_scaler.get_scaling_factor(
        self.portfolio.get_returns_data()
    )
    
    # Calculate appropriate position size
    position_size = self.position_sizer.calculate_position_size(
        symbol=signal['symbol'],
        price=signal['price'],
        risk_factor=risk_factor,
        option_data=signal.get('option_data')
    )
    
    # Execute the trade
    if position_size > 0:
        self.execute_trade(
            symbol=signal['symbol'],
            direction=signal['direction'],
            quantity=position_size,
            order_type=signal.get('order_type', 'LIMIT'),
            price=signal['price']
        )
        
        # Add hedge if necessary
        if self.hedging_manager.is_enabled():
            hedge_instructions = self.hedging_manager.get_hedge_instructions(
                symbol=signal['symbol'],
                quantity=position_size,
                option_data=signal.get('option_data')
            )
            
            if hedge_instructions:
                for instruction in hedge_instructions:
                    self.execute_trade(**instruction)
```

## Advanced Strategy Configuration

### Environment-Specific Settings

You can specify different settings for backtesting, paper trading, and live environments:

```yaml
# Environment-specific overrides
environments:
  backtest:
    position_sizing:
      max_leverage: 2.0
      
  paper:
    position_sizing:
      max_leverage: 1.5
      
  live:
    position_sizing:
      max_leverage: 1.0
      conservative_pct: 0.9
```

### Custom Risk Parameters by Symbol

Apply different risk parameters to specific symbols:

```yaml
position_sizing:
  custom_rules:
    - symbol: "SPY"
      max_position_size_pct: 0.10  # Allow larger SPY positions
      
    - symbol: "TSLA"
      max_position_size_pct: 0.03  # Limit TSLA positions
      conservative_pct: 0.95       # More conservative with TSLA
```

### Conditional Risk Parameters

Apply risk parameters based on conditions:

```yaml
position_sizing:
  custom_rules:
    - condition: "vix > 25"
      max_position_size_pct: 0.03  # Smaller positions when VIX is high
      
    - condition: "days_to_earnings < 10"
      max_position_size_pct: 0.02  # Smaller positions near earnings
```

## Example Strategies

### Conservative Put Selling

```yaml
# config/strategy/conservative_put_selling.yaml
strategy:
  name: "Conservative Put Selling"
  
  parameters:
    underlying: "SPY"
    dte_min: 30
    dte_max: 45
    delta_target: 0.15  # Lower delta for less risk
    
  position_sizing:
    max_position_size_pct: 0.03
    max_leverage: 1.0
    
  risk_scaling:
    method: "sharpe"
    min_investment: 0.1
    
  hedging:
    enabled: true
    hedge_method: "delta"
    delta:
      hedge_ratio: 1.0  # Full delta hedge
```

### Aggressive Put Selling

```yaml
# config/strategy/aggressive_put_selling.yaml
strategy:
  name: "Aggressive Put Selling"
  
  parameters:
    underlying: ["SPY", "QQQ", "IWM"]
    dte_min: 15
    dte_max: 30
    delta_target: 0.30  # Higher delta for more premium
    
  position_sizing:
    max_position_size_pct: 0.1
    max_leverage: 2.0
    
  risk_scaling:
    method: "volatility"
    min_investment: 0.5
    
  hedging:
    enabled: false  # No hedging
```

## Testing Your Strategy

### 1. Backtest Your Strategy

Run a backtest to evaluate performance:

```bash
python run_backtest.py --strategy=my_put_selling_strategy --start=2020-01-01 --end=2021-12-31
```

### 2. Paper Trade Your Strategy

Test your strategy with paper trading:

```bash
python run_paper_trading.py --strategy=my_put_selling_strategy
```

### 3. Deploy for Live Trading

When ready, deploy for live trading:

```bash
python run_live_trading.py --strategy=my_put_selling_strategy
```

## Tips for Strategy Development

1. **Start Conservative**: Begin with conservative settings and gradually increase risk
2. **Test Thoroughly**: Backtest across different market conditions
3. **Monitor Closely**: When live trading, monitor performance and risk metrics
4. **Iterate Gradually**: Make small changes and observe the impact
5. **Document Performance**: Keep detailed records of strategy performance

## Common Pitfalls and Solutions

### Excessive Leverage

**Problem**: Strategy uses too much leverage during volatile periods.

**Solution**: 
- Add volatility-based risk scaling
- Set stricter max_leverage limits
- Implement additional margin safety buffers

### Insufficient Hedging

**Problem**: Strategy suffers large drawdowns during market corrections.

**Solution**:
- Enable delta hedging with higher hedge_ratio
- Implement portfolio-level hedging
- Add VIX-based risk scaling

### Over-optimization

**Problem**: Strategy performs well in backtest but poorly in live trading.

**Solution**:
- Test on out-of-sample data
- Use cross-validation techniques
- Keep strategy parameters simple
- Focus on robustness over optimization 