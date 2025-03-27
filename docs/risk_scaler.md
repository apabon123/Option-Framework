# Risk Scaler

The `RiskScaler` component dynamically adjusts position sizes based on market conditions and strategy performance.

## Overview

Risk scaling is a technique that adjusts position sizes to manage risk exposure in changing market conditions. The `RiskScaler` analyzes performance metrics to determine an appropriate scaling factor that is applied to position sizes.

## Risk Scaling Methods

The `RiskScaler` supports four distinct methods for calculating risk scaling factors:

### 1. Sharpe Ratio Scaling

Adjusts position sizes based on the recent Sharpe ratio of the strategy relative to its historical performance.

#### How It Works:
1. Calculates the rolling Sharpe ratio over the configured window
2. Computes a z-score by comparing the current Sharpe to historical values
3. Applies linear scaling between min_investment and full exposure based on the z-score

#### Configuration:
```yaml
risk_scaling:
  method: "sharpe"
  rolling_window: 21
  target_z: 0        # Z-score for full exposure
  min_z: -2.0        # Z-score for minimum exposure
  min_investment: 0.25
  sharpe:
    min_sharpe: 0.5      # Minimum Sharpe ratio
    target_sharpe: 1.5   # Target Sharpe ratio
    risk_free_rate: 0.02 # Annual risk-free rate
```

#### Best For:
- Strategies that prioritize risk-adjusted returns
- Markets with varying volatility regimes
- Long-term position management

### 2. Volatility Targeting

Scales position sizes to maintain a target level of portfolio volatility, reducing exposure in high-volatility environments and increasing it during low-volatility periods.

#### How It Works:
1. Calculates realized volatility over the configured window
2. Computes the ratio of target volatility to realized volatility
3. Applies constraints to prevent extreme position sizes

#### Configuration:
```yaml
risk_scaling:
  method: "volatility"
  rolling_window: 21
  volatility:
    target_volatility: 0.15   # Target annualized volatility (15%)
    min_variance_ratio: 0.5   # Minimum scaling
    max_variance_ratio: 2.0   # Maximum scaling
```

#### Best For:
- Strategies targeting consistent volatility
- Markets with sharp regime changes
- Options strategies sensitive to volatility

### 3. Adaptive Scaling

Adjusts position sizes based on recent performance and drawdowns, reducing exposure after losses and gradually increasing it during recovery periods.

#### How It Works:
1. Maintains a "heat level" that increases with losses and decreases with gains
2. Calculates a drawdown-based recovery ratio
3. Combines these factors to determine the scaling factor

#### Configuration:
```yaml
risk_scaling:
  method: "adaptive"
  adaptive:
    max_heat: 1.0          # Maximum heat level
    cooldown_rate: 0.05    # Rate of heat reduction after gains
    heatup_rate: 0.02      # Rate of heat increase after losses
    recovery_factor: 2.0   # Amplification factor for recovery
```

#### Best For:
- Trend-following strategies
- Highly volatile markets
- Risk-sensitive approaches

### 4. Combined Scaling

Uses a weighted average of multiple scaling methods to create a balanced approach to risk management.

#### How It Works:
1. Calculates scaling factors using each individual method
2. Applies weights to each method's result
3. Combines them into a single scaling factor

#### Configuration:
```yaml
risk_scaling:
  method: "combined"
  combined:
    weights: [0.33, 0.33, 0.34]  # Weights for [volatility, sharpe, adaptive]
```

#### Best For:
- Complex strategies with multiple objectives
- Environments where no single risk metric is sufficient
- Robust risk management across different market conditions

## Implementation Details

### Enabling/Disabling Risk Scaling

Risk scaling can be entirely disabled by setting `enabled: false` in the configuration:

```yaml
risk_scaling:
  enabled: false  # Disables all risk scaling
```

When disabled, the `RiskScaler` will always return a scaling factor of 1.0, effectively using the normal position sizes calculated by the PositionSizer.

### Handling Missing Data

The `RiskScaler` handles cases where there is insufficient historical data:
- If no returns data is available, it returns a neutral scaling factor of 1.0
- For methods requiring a minimum amount of data, it uses expanding windows when applicable
- NaN values are properly handled in statistical calculations

### Logging and Monitoring

The `RiskScaler` provides detailed logging of:
- Initialized parameters
- Calculation details for each scaling method
- Historical risk scaling factors
- Interpretation of scaling factors

## Integration with Position Sizer

The `RiskScaler` is designed to work seamlessly with the `PositionSizer`:

1. Trading Engine obtains returns data from the Portfolio
2. Trading Engine requests a risk scaling factor from the RiskScaler
3. The scaling factor is passed to the PositionSizer
4. PositionSizer adjusts position sizes based on the scaling factor

## Performance Considerations

- Calculation of metrics like Sharpe ratio and volatility can be computationally intensive
- For high-frequency strategies, consider increasing the `rolling_window` parameter
- The `risk_scaling_history` property tracks all scaling factors for later analysis

## Example Scenarios

### Conservative Scaling

```yaml
risk_scaling:
  enabled: true
  method: "sharpe"
  min_investment: 0.10  # Much lower minimum during poor performance
  sharpe:
    min_sharpe: 1.0     # Higher bar for full investment
    target_sharpe: 2.0  # Very high target
```

### Aggressive Scaling

```yaml
risk_scaling:
  enabled: true
  method: "volatility"
  volatility:
    target_volatility: 0.25  # Higher target volatility
    min_variance_ratio: 0.3  # Lower minimum
    max_variance_ratio: 3.0  # Higher maximum
```

### Balanced Approach

```yaml
risk_scaling:
  enabled: true
  method: "combined"
  min_investment: 0.25
  combined:
    weights: [0.4, 0.4, 0.2]  # Higher weights on volatility and Sharpe
``` 