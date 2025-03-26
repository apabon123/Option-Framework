# Option Framework Trading Flow

This document outlines the complete trading lifecycle implemented in the Option Framework.

## Complete Trading Lifecycle

The framework implements a comprehensive trading flow with distinct phases for both EOD (End of Day) and intraday strategies:

```
┌─────────────────────┐
│  PRETRADE ANALYSIS  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│PORTFOLIO REBALANCING│
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ STRATEGY EVALUATION │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  HEDGING ANALYSIS   │◄───┐
└──────────┬──────────┘    │ Optional
           ▼               │ Component
┌─────────────────────┐    │
│    MARGIN CALC      │    │
└──────────┬──────────┘    │
           ▼               │
┌─────────────────────┐    │
│   POSITION SIZING   │    │
└──────────┬──────────┘    │
           ▼               │
┌─────────────────────┐    │
│   TRADE EXECUTION   │────┘
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│POSITION MANAGEMENT  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ POSTTRADE ANALYSIS  │
└─────────────────────┘
```

### 1. PRETRADE Analysis

**Purpose**: Evaluate the current market and portfolio state before making trading decisions.

**Implementation**:
- For EOD strategies: Snapshot prices and risk metrics at the close of business
- For intraday strategies: Snapshot at defined intervals
- Evaluate existing positions against exit/stop criteria
- Close positions that meet exit conditions

**Code Implementation**:
- `_process_trading_day` method starts with `_update_positions` to update market data
- `_log_pre_trade_summary` logs current portfolio status, positions, and P&L
- Exit conditions are checked with `_check_exit_conditions` method

### 2. Portfolio Rebalancing

**Purpose**: Ensure the portfolio is within margin and risk limits before adding new positions.

**Implementation**:
- Check if portfolio is within margin limits
- If exceeding limits, systematically close positions to return to target levels
- For intraday strategies: May use simplified rules or skip if not applicable

**Code Implementation**:
- Margin management is performed in `_execute_trading_activities` with the "MARGIN MANAGEMENT" section
- `margin_manager.analyze_margin_status(current_date)` identifies if rebalancing is needed
- For excessive margin utilization, positions are systematically reduced

### 3. Strategy Evaluation

**Purpose**: Identify potential new trading opportunities based on market conditions.

**Implementation**:
- Strategy identifies potential new trades based on signals
- Evaluates option chains for opportunities matching strategy criteria
- Ranks potential trades based on risk/reward metrics

**Code Implementation**:
- Strategy signal generation in `_execute_trading_activities` ("STRATEGY SIGNAL GENERATION" section)
- `strategy.generate_signals(current_date, daily_data)` identifies new trade candidates
- Signals include relevant information like symbol, direction, and strategy-specific attributes

### 4. Hedging Analysis (OPTIONAL)

**Purpose**: Determine hedging requirements for new trades to manage risk exposure.

**Implementation**:
- If strategy has hedging enabled:
  - Hedging Manager determines hedging requirements for the new trade
  - Calculates corresponding hedge instrument quantities
- If hedging is disabled, skip this step
- For certain intraday strategies: May use simplified hedging approaches

**Code Implementation**:
- Hedging is integrated within `_execute_signals` method
- `hedging_manager.create_theoretical_hedge_position(temp_position)` calculates hedging requirements
- The "HEDGING ADJUSTMENTS" section in `_execute_trading_activities` handles portfolio-level hedging

### 5. Initial Margin Calculation

**Purpose**: Calculate the margin required for the trade package to ensure sufficient capital.

**Implementation**:
- Calculate margin for the specific trade package:
  - For unhedged strategies: Calculate margin for trade alone
  - For hedged strategies: Calculate combined margin for trade + its hedge
- For futures/equities: Use appropriate margin rules for those instruments

**Code Implementation**:
- `_position_sizing` method integrates with margin calculation
- `SPANMarginCalculator` calculates the margin for both options and hedges
- The system accounts for hedging benefits in margin calculations

### 6. Position Sizing

**Purpose**: Determine the appropriate position size based on risk parameters and margin requirements.

**Implementation**:
- Determine appropriate size based on margin parameters and portfolio metrics
- Ensure trade package size is within configured margin limits
- Scale positions based on risk factors like volatility

**Code Implementation**:
- `_position_sizing` method determines appropriate position size
- Risk manager is consulted via `risk_manager.calculate_position_size()`
- Position size is adjusted to remain within margin constraints

### 7. Trade Execution

**Purpose**: Execute the trades with proper order management.

**Implementation**:
- Execute the primary trade
- If hedging is enabled, execute the corresponding hedge trade
- Apply slippage and transaction cost models

**Code Implementation**:
- Trade execution flow in `_execute_signals` method
- Primary trade execution with `portfolio.add_position()`
- If hedging is enabled, hedge position is also added to the portfolio

### 8. Position Management

**Purpose**: Track and manage the newly added positions within the portfolio.

**Implementation**:
- Add the new position to the portfolio
- If hedged, add the hedge position to the portfolio
- Link related positions for management purposes

**Code Implementation**:
- Positions are added to the portfolio with all required tracking information
- Hedges are linked to their primary positions
- Position inventory maintains a centralized record of all positions

### 9. POSTTRADE Analysis

**Purpose**: Evaluate the portfolio after trading to ensure all metrics are within limits.

**Implementation**:
- Recalculate complete portfolio margin including new positions
- Verify all margin metrics are within limits
- For EOD strategies: Complete end-of-day reconciliation
- For intraday strategies: May skip or simplify if positions are closed same day

**Code Implementation**:
- `_log_post_trade_summary` performs the post-trade analysis
- Margin is recalculated with the new positions
- EOD reconciliation with `portfolio.record_daily_metrics(current_date)`

## Terminology Notes

- **Trade**: The primary financial instrument transaction (option, future, equity)
- **Hedge**: The instrument used to offset exposure from the primary trade
- **Trade package**: The combination of primary trade + hedge when applicable
- **Portfolio**: The full collection of all positions and hedges

## Framework Adaptability

The framework is adaptable to different instrument types with appropriate margin calculation methods for each:

- **Options**: Specialized SPAN-style margin calculations with proper Greek-based risk assessment
- **Equities**: Regulation T or portfolio margin calculations
- **Futures**: SPAN margin with scaling based on contract specifications

Each instrument type has dedicated calculation methods implemented in the margin system. 