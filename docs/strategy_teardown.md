# Strategy Teardown: Contrarian Options Alpha

## 1. Strategy Thesis

Markets overreact to negative news. When a fundamentally sound company experiences a temporary price decline driven by sector rotation, sentiment contagion, or short-term fear, short-dated put options become mispriced. This strategy systematically identifies these overreactions and captures the mean-reversion premium.

**Core hypothesis:** Temporary dips in large-cap equities create predictable option mispricing that reverts within 1-5 trading days.

**Edge sources:**
- Behavioral: retail panic selling creates transient supply/demand imbalances
- Structural: market makers widen spreads during volatility, creating entry opportunities
- Informational: AI sentiment analysis distinguishes temporary dips from fundamental deterioration faster than consensus

---

## 2. Signal Pipeline Mathematics

### 2.1 Relative Strength Index (RSI)

The RSI uses Wilder's Exponential Moving Average:

```
RS = EMA(gains, period) / EMA(losses, period)
RSI = 100 - 100 / (1 + RS)
```

Where the EMA uses Wilder's smoothing factor `alpha = 1/period`:
```
EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}
```

Seeded with a simple average over the first `period` observations.

**Entry signal:** RSI < 30 (oversold territory)
**Exit signal:** RSI > 50 (mean reversion complete)

Properties:
- Bounded [0, 100]
- Memory: O(1) — only stores previous average gain/loss
- Computation: O(1) per update
- Default period: 14

### 2.2 Bollinger Bands

Bollinger Bands define a volatility envelope around a simple moving average:

```
Middle = SMA(price, period)
Upper = Middle + multiplier * StdDev(price, period)
Lower = Middle - multiplier * StdDev(price, period)
```

The signal value measures how far the current price has fallen below the lower band:

```
bb_signal = max(0, (Lower - price) / (Upper - Lower))
```

A value > 0 means the price is below the lower band — a statistically unusual event suggesting mean reversion.

Properties:
- Memory: O(period) — maintains a rolling window
- Computation: O(1) amortized per update
- Default: period=20, multiplier=2.0

### 2.3 Volume Spike Detection

Volume spikes indicate unusual market activity, often accompanying overreactions:

```
avg_volume = SMA(volume, lookback)
ratio = current_volume / avg_volume
is_spike = ratio >= spike_threshold AND ready()
```

Properties:
- Memory: O(lookback)
- Default: lookback=20, spike_threshold=2.0

### 2.4 Greeks Filter

Options-specific filter ensuring favorable contract characteristics:

| Criterion | Range | Rationale |
|-----------|-------|-----------|
| |Delta| | 0.20 - 0.40 | Sufficient directional exposure without excessive premium |
| IV Percentile | < 50% | Avoid buying overpriced volatility |
| Bid-Ask Spread | < 20% of mid | Ensures reasonable execution cost |

Score computation:
```
delta_score = 1.0 - |delta - 0.30| / 0.10
iv_score = 1.0 - iv_percentile / 100.0
spread_score = 1.0 - spread_pct / max_spread
greeks_score = (delta_score + iv_score + spread_score) / 3.0
```

### 2.5 Composite Signal Scoring

The four indicators combine into a weighted composite:

```
composite = w_rsi * rsi_score + w_bb * bb_score + w_vol * vol_score + w_greeks * greeks_score
```

Default weights:
| Weight | Value | Component |
|--------|-------|-----------|
| w_rsi | 0.30 | RSI oversold depth |
| w_bb | 0.25 | Bollinger Band deviation |
| w_vol | 0.25 | Volume spike ratio |
| w_greeks | 0.20 | Greeks filter score |

**Minimum composite threshold:** 0.65

---

## 3. Entry and Exit Rules

### Entry Conditions (ALL must be true)

1. RSI < oversold threshold (default: 30)
2. Price < lower Bollinger Band
3. Volume spike detected (ratio >= 2.0x average)
4. Greeks filter passes (delta, IV, spread criteria)
5. Composite score >= min_composite (default: 0.65)
6. Sentiment = TEMPORARY_DIP with confidence >= 0.7
7. Risk manager approves (daily loss, max positions, circuit breaker)

### Exit Conditions (priority-ordered)

1. **Stop-loss:** P&L <= -50% of entry price
2. **DTE expiry:** Days to expiration <= 0
3. **Mean reversion:** RSI > exit_rsi threshold (default: 50)

### Contract Selection

| Parameter | Range |
|-----------|-------|
| DTE | 1-5 days |
| Option Type | Puts |
| Delta | 0.20-0.40 |
| IV Percentile | < 50% |
| Bid-Ask Spread | < 20% of mid |
| Price | $0.05 - $0.30 |

---

## 4. Position Sizing: Kelly Criterion

### Full Kelly

```
f* = (p * b - q) / b
```

Where: p = win probability, q = 1 - p, b = avg_win / avg_loss

### Half-Kelly

```
f = kelly_fraction * f*    (kelly_fraction = 0.5)
```

Half-Kelly reduces variance by ~75% while sacrificing only ~25% of theoretical growth rate.

### Position Size Calculation

```python
kelly_bet = kelly_fraction * (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
bet_pct = min(kelly_bet, max_bet_pct)    # max_bet = 20%
dollar_size = bet_pct * bankroll
contracts = max(1, floor(dollar_size / (option_price * 100)))
```

---

## 5. Risk Management

### Pre-Trade Risk Gates

| Gate | Check | Default Limit |
|------|-------|---------------|
| 1 | Daily P&L | >= -$50 |
| 2 | Open positions | <= 5 |
| 3 | Single position size | <= $20 |
| 4 | Circuit breaker | Not tripped |

### Circuit Breaker

Rolling-window mechanism:
```
window_size = 20 trades
min_win_rate = 0.40
is_tripped = (trades >= window_size) AND (wins / window_size < min_win_rate)
```

### Live vs Paper Limits

| Parameter | Paper | Live |
|-----------|-------|------|
| Daily loss limit | -$50 | -$30 |
| Max positions | 5 | 3 |
| Max single position | $20 | $15 |
| Min win rate | 0.40 | 0.45 |
| Position scale | 1.0x | 0.5x |

---

## 6. Walk-Forward Optimization

**Anchored walk-forward** with expanding training windows:

```
Train: 252 days (1 year), anchored at start
Test: 63 days (1 quarter)
Step: 63 days forward
```

### Parameter Grid

| Parameter | Search Range | Step |
|-----------|-------------|------|
| RSI period | 10-20 | 2 |
| RSI oversold | 25-35 | 5 |
| BB period | 15-25 | 5 |
| BB multiplier | 1.5-2.5 | 0.5 |
| Volume lookback | 15-25 | 5 |
| Spike threshold | 1.5-2.5 | 0.5 |

Final metrics are averaged across all OOS folds.

---

## 7. Monte Carlo Simulation

### Bootstrap Methodology

1. Extract realized trade returns
2. For N=10,000 simulations: sample trades with replacement, compute equity curve
3. Report quantiles: 5th, 25th, 50th, 75th, 95th percentile

---

## 8. Parameter Sensitivity

### High-Impact Parameters
1. **RSI oversold threshold** — Moving 30→25 reduces signal frequency ~40%, improves win rate ~5%
2. **Min composite score** — Higher thresholds = fewer but higher-quality signals
3. **Kelly fraction** — Directly scales position sizes and volatility

### Stability Zones
- RSI period: 12-16
- RSI oversold: 28-35
- Min composite: 0.60-0.70
- Kelly fraction: 0.3-0.5

---

## 9. Limitations

- **Liquidity risk:** Short-dated options can have wide spreads in fast markets
- **Slippage:** Entry signals trigger during high-volatility periods
- **Sentiment latency:** Claude API adds 0.5-2s per classification
- **Small samples:** Strict filters produce few trades; walk-forward windows may be thin
- **Curve-fitting:** Multiple tunable parameters risk overfitting
- **Mean reversion failure:** Core thesis breaks during regime changes
- **Capital constraints:** $200-500 capital limits diversification
- **Regulatory:** PDT rules apply under $25K; brokers may close near-expiry positions
