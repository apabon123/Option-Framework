# Configuration Reference

This document provides a comprehensive reference for all configuration options in the Options Trading Framework, with a focus on risk management components.

## Config File Structure

Configuration files are organized in YAML format with the following structure:

```yaml
strategy:
  name: "My Strategy"
  description: "Description of the strategy"
  
  # Core strategy parameters
  parameters:
    # Strategy-specific parameters
    
  # Position sizing configuration
  position_sizing:
    # Position sizing parameters
    
  # Risk scaling configuration  
  risk_scaling:
    # Risk scaling parameters
    
  # Hedging configuration
  hedging:
    # Hedging parameters
    
  # Margin configuration
  margin:
    # Margin parameters
    
  # Execution configuration
  execution:
    # Execution parameters
```

## Position Sizing Configuration

### Basic Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable/disable position sizing |
| `max_position_size_pct` | float | `0.1` | Maximum position size as % of account |
| `min_position_size` | int | `1` | Minimum position size (contracts or shares) |
| `max_leverage` | float | `2.0` | Maximum account leverage |
| `max_margin_usage_pct` | float | `0.8` | Maximum % of account to allocate to margin |
| `conservative_pct` | float | `0.85` | Safety buffer on margin calculations |

### Margin Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `margin.use_broker_requirements` | boolean | `true` | Use broker margin requirements when available |
| `margin.maintenance_margin_pct` | float | `0.75` | % of initial margin for maintenance margin |
| `margin.hedge_benefit_pct` | float | `0.8` | % of hedge value applied to margin reduction |
| `margin.uncorrelated_margin_pct` | float | `0.25` | % reduction for uncorrelated positions |
| `margin.include_open_orders` | boolean | `true` | Include pending orders in margin calculation |

### Volatility-Based Sizing

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `volatility_sizing.enabled` | boolean | `false` | Enable/disable volatility-based position sizing |
| `volatility_sizing.target_risk_pct` | float | `0.01` | Target daily risk per position (as % of account) |
| `volatility_sizing.max_risk_multiplier` | float | `3.0` | Maximum volatility adjustment multiplier |
| `volatility_sizing.volatility_lookback` | int | `20` | Days to look back for volatility calculation |

### Custom Rules

Custom position sizing rules can be specified for specific symbols or conditions:

```yaml
position_sizing:
  custom_rules:
    - symbol: "SPY"
      max_position_size: 100    
      min_position_size: 5      
    - condition: "market_cap > 10e9"
      max_position_size_pct: 0.3
```

## Risk Scaling Configuration

### Basic Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable/disable risk scaling |
| `method` | string | `"sharpe"` | Risk scaling method: "sharpe", "volatility", "adaptive", or "combined" |
| `rolling_window` | int | `21` | Rolling window for performance calculations (trading days) |
| `target_z` | float | `0` | Z-score for full exposure |
| `min_z` | float | `-2.0` | Z-score for minimum exposure |
| `min_investment` | float | `0.25` | Minimum investment level (as a fraction of full size) |
| `max_scaling` | float | `1.5` | Maximum risk scaling factor |
| `min_scaling` | float | `0.1` | Minimum risk scaling factor |

### Method-Specific Options

#### Sharpe Ratio Scaling

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `sharpe.min_sharpe` | float | `0.5` | Minimum Sharpe ratio |
| `sharpe.target_sharpe` | float | `1.5` | Target Sharpe ratio |
| `sharpe.risk_free_rate` | float | `0.02` | Annual risk-free rate |

#### Volatility Targeting

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `volatility.target_volatility` | float | `0.15` | Target annualized volatility |
| `volatility.min_variance_ratio` | float | `0.5` | Minimum scaling based on volatility |
| `volatility.max_variance_ratio` | float | `2.0` | Maximum scaling based on volatility |

#### Adaptive Scaling

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `adaptive.max_heat` | float | `1.0` | Maximum heat level |
| `adaptive.cooldown_rate` | float | `0.05` | Rate of heat reduction after gains |
| `adaptive.heatup_rate` | float | `0.02` | Rate of heat increase after losses |
| `adaptive.recovery_factor` | float | `2.0` | Amplification factor for recovery |

#### Combined Scaling

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `combined.weights` | list[float] | `[0.33, 0.33, 0.34]` | Weights for [volatility, sharpe, adaptive] methods |

## Hedging Configuration

### Basic Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable/disable automatic hedging |
| `hedge_method` | string | `"delta"` | Hedging method: "delta", "beta", or "custom" |
| `rebalance_threshold` | float | `0.1` | Threshold for hedge rebalancing (as fraction of position) |
| `max_hedge_cost` | float | `0.05` | Maximum cost for hedging as % of position value |

### Delta Hedging Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `delta.hedge_ratio` | float | `1.0` | Ratio of delta to hedge (1.0 = full hedge) |
| `delta.use_stock` | boolean | `true` | Use stock for delta hedging |
| `delta.use_options` | boolean | `false` | Use options for delta hedging |
| `delta.min_option_days` | int | `14` | Minimum days to expiration for hedge options |

### Beta Hedging Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `beta.hedge_instrument` | string | `"SPY"` | Instrument to use for beta hedging |
| `beta.beta_lookback` | int | `60` | Lookback period for beta calculation (trading days) |
| `beta.hedge_ratio` | float | `0.8` | Ratio of beta to hedge (0.8 = 80% hedge) |

## Margin Configuration

### Basic Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `calculate_margin` | boolean | `true` | Calculate margin requirements |
| `margin_safety_multiplier` | float | `1.1` | Safety multiplier for margin calculations |
| `update_frequency` | string | `"daily"` | Frequency of margin updates: "trade", "daily", "weekly" |

### Options Margin Parameters

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `options.naked_call_margin_pct` | float | `0.2` | Margin % for naked call options |
| `options.naked_put_margin_pct` | float | `0.15` | Margin % for naked put options |
| `options.spread_margin_pct` | float | `1.0` | Margin % of max loss for spreads |
| `options.use_span_margin` | boolean | `true` | Use SPAN-like margin calculations |

### Stock Margin Parameters

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `stock.long_margin_pct` | float | `0.5` | Margin % for long stock positions |
| `stock.short_margin_pct` | float | `1.5` | Margin % for short stock positions |

### Futures Margin Parameters

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `futures.use_exchange_margins` | boolean | `true` | Use exchange-provided margin requirements |
| `futures.default_margin_per_contract` | float | `5000` | Default margin per contract if exchange data unavailable |
| `futures.spread_discount_pct` | float | `0.8` | Discount % for futures spreads |

## Example Configurations

### Conservative Configuration

```yaml
position_sizing:
  enabled: true
  max_position_size_pct: 0.05
  max_leverage: 1.0
  conservative_pct: 0.9
  
risk_scaling:
  enabled: true
  method: "sharpe"
  min_investment: 0.1
  max_scaling: 1.0
  sharpe:
    min_sharpe: 1.0
    target_sharpe: 2.0
    
hedging:
  enabled: true
  hedge_method: "delta"
  delta:
    hedge_ratio: 1.0
```

### Aggressive Configuration

```yaml
position_sizing:
  enabled: true
  max_position_size_pct: 0.25
  max_leverage: 3.0
  conservative_pct: 0.7
  
risk_scaling:
  enabled: true
  method: "volatility"
  min_investment: 0.5
  max_scaling: 2.0
  volatility:
    target_volatility: 0.25
    
hedging:
  enabled: false
```

### Balanced Configuration

```yaml
position_sizing:
  enabled: true
  max_position_size_pct: 0.1
  max_leverage: 1.5
  conservative_pct: 0.8
  
risk_scaling:
  enabled: true
  method: "combined"
  min_investment: 0.25
  combined:
    weights: [0.4, 0.4, 0.2]
    
hedging:
  enabled: true
  hedge_method: "delta"
  delta:
    hedge_ratio: 0.5
```

## Environment-Specific Configurations

You can specify different configurations for backtesting, paper trading, and live trading:

```yaml
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

## Configuration Files

### Main Configuration Files

- `config/config.yaml`: Global configuration
- `config/strategy/*.yaml`: Strategy-specific configurations

### Template Files

- `config/position_sizing_template.yaml`: Position sizing template
- `config/risk_scaling_template.yaml`: Risk scaling template
- `config/hedging_template.yaml`: Hedging template

## Configuration Validation

The framework validates all configuration options at startup and provides detailed error messages for invalid configurations. Common validation checks include:

- Value ranges (e.g., percentages between 0 and 1)
- Allowed values for enumerated options
- Required fields
- Type checking
- Consistency between related parameters 