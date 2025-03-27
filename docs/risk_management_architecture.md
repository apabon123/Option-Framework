# Risk Management Architecture

## Overview

The risk management system is designed to control position sizing and risk scaling in the trading engine. The architecture has been restructured to separate concerns between position sizing (margin-based calculations) and risk scaling (performance-based adjustments).

## Components

### 1. Position Sizer (`core/position_sizer.py`)

The PositionSizer is responsible for calculating appropriate position sizes based on margin requirements and portfolio constraints.

#### Key Features:
- Calculates position sizes based on available margin
- Applies portfolio-level constraints like max leverage and max position size
- Integrates with the margin calculator to determine margin requirements
- Can utilize risk scaling factors from the RiskScaler
- Works with the hedging system to account for hedge benefits in margin calculations

#### Configuration:
```yaml
position_sizing:
  enabled: true                  # Enable/disable dynamic position sizing
  max_position_size_pct: 0.25    # Max position size as % of portfolio
  min_position_size: 1           # Minimum position size in contracts
  hedge_for_margin: true         # Consider hedging when calculating margin
  apply_margin_constraints: true # Apply margin-based position constraints
  use_risk_scaling: true         # Apply risk scaling factor to position sizing
```

### 2. Risk Scaler (`core/risk_scaler.py`)

The RiskScaler is responsible for adjusting position sizes based on market conditions and strategy performance.

#### Risk Scaling Methods:
- **Sharpe Ratio**: Scales position sizes based on recent Sharpe ratio performance
- **Volatility**: Targets a specific volatility level by scaling positions inversely to market volatility
- **Adaptive**: Adjusts risk based on recent drawdowns and recovery metrics
- **Combined**: Uses a weighted combination of multiple risk scaling methods

#### Configuration:
```yaml
risk_scaling:
  enabled: true                # Enable/disable risk scaling
  method: "sharpe"             # Method: "sharpe", "volatility", "adaptive", "combined"
  rolling_window: 21           # Days in calculation window
  target_z: 0                  # Z-score for full position sizing
  min_z: -2.0                  # Z-score for minimum position sizing
  min_investment: 0.25         # Minimum scaling factor (% of normal size)
  
  # Method-specific parameters
  sharpe: {...}                # Sharpe ratio method parameters
  volatility: {...}            # Volatility targeting parameters
  adaptive: {...}              # Adaptive method parameters
  combined: {...}              # Combined method parameters
```

## Flow Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Trading Engine ├────►│  Position Sizer ├────►│   Risk Scaler   │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
        │                        │                       ▲
        │                        │                       │
        │                        ▼                       │
        │               ┌─────────────────┐              │
        │               │                 │              │
        │               │ Margin Manager  │              │
        │               │                 │              │
        │               └─────────────────┘              │
        │                                                │
        │                                                │
        │               ┌─────────────────┐              │
        └──────────────►│                 ├──────────────┘
                        │ Performance     │
                        │ Metrics         │
                        └─────────────────┘
```

## Interaction with Trading Engine

1. Trading Engine receives a signal to trade
2. Engine calls Position Sizer to determine appropriate position size
3. Position Sizer requests risk scaling factor from Risk Scaler (if enabled)
4. Position Sizer calculates margin requirements (directly or via Margin Manager)
5. Position Sizer returns the final position size to Trading Engine
6. Trading Engine executes the trade with the calculated position size

## Benefits of the New Architecture

1. **Separation of Concerns**: Clear division between margin-based position sizing and risk-based scaling
2. **Flexibility**: Can enable/disable risk scaling independently from position sizing
3. **Multiple Risk Models**: Choose from different risk scaling methodologies
4. **Configurability**: Extensive configuration options for each component
5. **Maintainability**: Easier to modify or extend individual components

## Configuration Examples

### Basic Configuration (No Risk Scaling)

```yaml
position_sizing:
  enabled: true
  max_position_size_pct: 0.25
  min_position_size: 1
  use_risk_scaling: false

risk_scaling:
  enabled: false
```

### Volatility-Based Scaling

```yaml
position_sizing:
  enabled: true
  max_position_size_pct: 0.25
  use_risk_scaling: true

risk_scaling:
  enabled: true
  method: "volatility"
  volatility:
    target_volatility: 0.15  # Target 15% annualized volatility
    min_variance_ratio: 0.5
    max_variance_ratio: 2.0
```

### Sharpe-Based with Tighter Constraints

```yaml
position_sizing:
  enabled: true
  max_position_size_pct: 0.15  # More conservative position limit
  use_risk_scaling: true

risk_scaling:
  enabled: true
  method: "sharpe"
  min_investment: 0.15  # Lower minimum in bad conditions
  sharpe:
    min_sharpe: 0.75    # Higher minimum Sharpe requirement
    target_sharpe: 2.0  # Higher target Sharpe
```

## Migration from Previous Architecture

The previous architecture used a single `RiskManager` class that combined both position sizing and risk scaling functionality. The new architecture separates these concerns into distinct classes:

- `RiskManager` → `PositionSizer` and `RiskScaler`

For backward compatibility, the Trading Engine maintains a reference to the Position Sizer as `risk_manager`. This allows existing code to continue functioning without modification while transitioning to the new architecture. 