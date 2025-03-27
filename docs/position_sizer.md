# Position Sizer

The `PositionSizer` component determines appropriate position sizes for trading strategies based on account value, margin requirements, and risk parameters.

## Overview

Position sizing is a critical aspect of trading system design that balances risk exposure with potential returns. The `PositionSizer` calculates position sizes based on:

1. Available margin
2. Account value (Net Liquidation Value)
3. Risk parameters (max leverage, max position size)
4. Instrument-specific characteristics (price, margin requirements)
5. Risk scaling factors (from the RiskScaler)

## Position Sizing Methods

### Margin-Based Sizing

The primary method calculates position sizes based on margin requirements and available capital.

#### How It Works:
1. Calculates margin per contract for the target instrument
2. Determines available margin based on account value and risk parameters
3. Applies risk scaling factors to adjust position size
4. Applies minimum and maximum position size constraints

#### Margin Considerations:
- For options, the system calculates SPAN-like margin requirements
- For stock positions, standard Reg T margin is applied
- For futures, exchange-specific initial margin requirements are used
- Hedged positions receive margin offset benefits based on delta relationships

## Configuration

Position sizing is configured through the `position_sizing` section in your strategy configuration:

```yaml
position_sizing:
  enabled: true
  max_position_size_pct: 0.2    # Maximum % of account for any single position
  min_position_size: 1          # Minimum position size (contracts/shares)
  max_leverage: 2.0             # Maximum account leverage
  max_margin_usage_pct: 0.8     # Maximum % of account to allocate to margin 
  conservative_pct: 0.85        # Safety buffer on margin calculations
  margin:
    use_broker_requirements: true     # Use brokerage margin requirements when available
    maintenance_margin_pct: 0.75      # % of initial margin for maintenance
    hedge_benefit_pct: 0.8            # % of hedge value applied to margin reduction
    uncorrelated_margin_pct: 0.25     # % reduction for uncorrelated positions
    include_open_orders: true         # Include pending orders in margin calculation
```

### Advanced Configuration Options

#### Volatility-Based Sizing

For strategies sensitive to volatility, additional volatility-based sizing parameters can be specified:

```yaml
position_sizing:
  # ... basic configuration ...
  volatility_sizing:
    enabled: true
    target_risk_pct: 0.01       # Target 1% daily risk per position
    max_risk_multiplier: 3.0    # Maximum volatility adjustment
    volatility_lookback: 20     # Days to look back for volatility calculation
```

#### Custom Sizing Rules

Custom position sizing rules can be applied to specific symbols or conditions:

```yaml
position_sizing:
  # ... basic configuration ...
  custom_rules:
    - symbol: "SPY"
      max_position_size: 100    # Maximum position size for SPY
      min_position_size: 5      # Minimum position size for SPY
    - condition: "market_cap > 10e9"
      max_position_size_pct: 0.3  # Larger allocation for large-cap stocks
```

## Enabling/Disabling Position Sizing

Position sizing can be entirely disabled by setting `enabled: false` in the configuration:

```yaml
position_sizing:
  enabled: false
```

When disabled, the system will use the requested position size directly, only applying basic validation checks.

## Integration with Risk Scaler

The `PositionSizer` integrates with the `RiskScaler` by applying risk scaling factors to position sizes:

1. The `TradingEngine` obtains a risk scaling factor from the `RiskScaler`
2. The factor is passed to the `PositionSizer` during position size calculation
3. The final position size is scaled proportionally to the risk factor

## Hedge Detection and Margin Offsets

The `PositionSizer` can detect hedged positions and calculate appropriate margin offsets:

### For Options and Stock:
- Options positions are analyzed for their delta exposure
- Stock positions that offset option delta receive hedge margin benefits
- The system calculates the effective hedge value and reduces margin requirements

### For Futures:
- Offsetting futures positions receive standard exchange margin offsets
- Calendar spreads and inter-commodity spreads are recognized for margin benefits

## Logging and Monitoring

The `PositionSizer` provides detailed logging of:
- Margin calculations per position
- Position size decisions with reasoning
- Applied constraints (minimum/maximum sizes)
- Risk scaling adjustments
- Hedge benefits and margin offsets

## Example Scenarios

### Conservative Position Sizing

```yaml
position_sizing:
  enabled: true
  max_position_size_pct: 0.05    # Very small maximum position size
  max_leverage: 1.0              # No leverage
  max_margin_usage_pct: 0.5      # Only use half of available margin
  conservative_pct: 0.90         # Highly conservative margin buffer
```

### Aggressive Position Sizing

```yaml
position_sizing:
  enabled: true
  max_position_size_pct: 0.5     # Larger maximum position size
  max_leverage: 4.0              # Higher leverage
  max_margin_usage_pct: 0.95     # Use most of available margin
  conservative_pct: 0.70         # Less conservative margin buffer
```

### Options-Specific Configuration

```yaml
position_sizing:
  enabled: true
  max_position_size_pct: 0.1
  margin:
    use_broker_requirements: true
    hedge_benefit_pct: 1.0        # Full hedge benefit for options
    maintenance_margin_pct: 0.6   # Lower maintenance margin for options
  volatility_sizing:
    enabled: true                 # Enable vol-based sizing for options
    target_risk_pct: 0.005        # Lower risk per position
```

## Technical Implementation

### Margin Calculation Logic

The `PositionSizer` uses a multi-step process for margin calculations:

1. Calculate initial margin for each position
2. Identify potential hedge relationships
3. Apply hedge benefits to reduce margin requirements
4. Sum adjusted margin across all positions
5. Apply portfolio-level risk constraints

### Position Size Calculation

For a given trade, position sizing follows this process:

1. Calculate margin per contract
2. Determine maximum position size based on account limits
3. Apply risk scaling factor from RiskScaler
4. Apply minimum and maximum position size constraints
5. Return final position size with detailed calculation notes

### Performance Considerations

- Margin calculations can be computationally intensive for large portfolios
- The system caches margin calculations where possible
- For high-frequency strategies, consider enabling the `use_cached_margins` option 