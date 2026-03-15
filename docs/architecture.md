# Architecture: Contrarian Options Alpha Engine

## 1. System Overview

Two-layer architecture: high-performance C++20 core for signal computation and risk management, connected to Python orchestration via pybind11.

```
Python Layer: Screener → Sentiment → Dashboard → Alerts
                    ↓
         Trading Orchestrator (Paper/Live)
                    ↓ pybind11
C++20 Core: Strategy | Execution | Risk | Market Data
```

**Why this split?**
- C++ handles the hot path: indicator updates, signal scoring, risk checks (sub-microsecond)
- Python handles the slow path: API calls, broker communication, UI (100ms+ tolerance)
- pybind11 bridges with zero-copy for primitive types

---

## 2. C++ Module Design

### 2.1 Common Module (coe::common)

- **Types:** Price (double), Quantity (int32_t), Timestamp (chrono::nanoseconds), Symbol (string), Side, OptionType enums
- **Error Handling:** Result<T> = variant<T, ErrorCode> — no exceptions in C++ core
- **Config:** YAML-based with dotted key access, non-copyable/move-only
- **Logging:** spdlog with stdout + rotating file sinks

### 2.2 Strategy Module (coe::strategy)

Concept-constrained indicators:
```cpp
template <typename T>
concept Indicator = requires(T t, double v) {
    { t.update(v) }; { t.value() } -> same_as<double>;
    { t.ready() } -> same_as<bool>; { t.reset() };
};
```

Four indicators: RSI (O(1) memory), BollingerBands (O(period)), VolumeSpike (O(lookback)), GreeksFilter.

**SignalScorer** owns all indicators, emits Signal when composite >= threshold.

### 2.3 Execution Module (coe::execution)

- **Order state machine:** New → PendingSend → Sent → PartialFill → Filled | Cancelled | Rejected
- **OrderManager:** submit, cancel, on_fill with state validation
- **PositionTracker:** fill-driven updates, mark-to-market P&L
- **KellyPositionSizer:** half-Kelly with configurable fraction and max bet

### 2.4 Risk Module (coe::risk)

Three-layer stack:

| Layer | Component | Check |
|-------|-----------|-------|
| 1 | DailyPnLTracker | Cumulative daily P&L vs limit |
| 2 | RiskManager | Position count + single-position size |
| 3 | CircuitBreaker | Rolling win rate vs threshold |

RiskManager.check_new_order() runs 4 sequential gates.

### 2.5 Market Data Module (coe::market_data)

- **SPSCRingBuffer<T, N>:** Lock-free, cache-line aligned, 13-29M msg/sec
- **Message types:** Quote, Trade, OptionsQuote as variant
- **WebSocket client:** Boost.Beast + OpenSSL, pushes to ring buffer

---

## 3. pybind11 Bridge

### Type Mapping

| C++ | Python |
|-----|--------|
| Result<T> | Returns T or raises CoeError |
| Result<monostate> | Returns None or raises CoeError |
| optional<T> | T or None |
| Timestamp | int (nanoseconds) |
| vector<T> | list[T] |
| Config.get<T> | get_int, get_double, get_string, get_bool |

### Module Structure

```
_coe_engine (.so/.dylib)
├── init_common()    → Side, OptionType, ErrorCode, Config, CoeError
├── init_strategy()  → RSI, BollingerBands, VolumeSpike, GreeksFilter, Signal, SignalScorer
├── init_execution() → Order, OrderManager, Position, PositionTracker, KellyPositionSizer
└── init_risk()      → RiskLimits, DailyPnLTracker, CircuitBreaker, RiskManager
```

CoeError wraps ErrorCode and is shared across translation units via module attribute lookup.

---

## 4. Event Flow

```
Market Data (Polygon/IBKR)
    → Screener (Python): pre-filter by volume, market cap, RSI
    → Sentiment Filter (Python): Claude AI classification
    → Technical Analyzer: C++ SignalScorer via pybind11
    → Options Chain Analyzer (Python): select best contract
    → Risk Manager (C++): 4 sequential gates
    → Kelly Position Sizer (C++): half-Kelly sizing
    → Order Bridge (Python): C++ Order → ib_insync → IBKR TWS
    → Fill Callback: updates OrderManager + PositionTracker
    → Dashboard (Flask + SocketIO) + Alerts (Telegram/Discord)
```

---

## 5. Risk Layering

### Pre-Trade (synchronous)

```
Gate 1: daily_pnl > daily_loss_limit?     → DailyLossExceeded
Gate 2: open_positions < max_positions?     → PositionLimitExceeded
Gate 3: order_value < max_single_position?  → InsufficientMargin
Gate 4: circuit_breaker.is_tripped()?       → CircuitBreakerTripped
```

### Post-Trade (asynchronous)
- Fill callback updates PnLTracker and CircuitBreaker
- Position reconciler every 5 minutes
- Dashboard reflects real-time P&L

### Live Trading Additions
- Startup confirmation prompt
- Capital ceiling ($200-500)
- Position scale factor (0.5x)
- Tighter risk limits

---

## 6. Python Orchestration

### Signal Pipeline

| Module | Responsibility | External Dep |
|--------|---------------|-------------|
| signals/screener.py | Scan for oversold large-caps | Polygon.io |
| signals/sentiment.py | Classify dip type via AI | Claude API |
| signals/technicals.py | C++ signal scoring | pybind11 |
| signals/options_chain.py | Fetch and filter options | Polygon/IBKR |

### Broker Integration

| Module | Responsibility | External Dep |
|--------|---------------|-------------|
| broker/connection.py | IBKR connectivity + heartbeat | ib_insync |
| broker/order_bridge.py | C++ Order ↔ ib_insync | ib_insync |
| broker/reconciliation.py | Position sync | ib_insync |
| broker/paper_trader.py | Main trading loop | All above |
| broker/live_trader.py | Live with safety constraints | paper_trader |

### Graceful Degradation

Import guards: HAS_ENGINE (C++ bindings), HAS_IB_INSYNC (broker), HAS_SIGNALS (signal modules). Each failure path provides clear error messages or falls back to pure-Python alternatives.

---

## 7. Dashboard Architecture

```
Flask App (app.py)
    ├── REST API: /api/{positions,trades,equity,signals,metrics}
    ├── WebSocket: Flask-SocketIO real-time push
    └── Templates: 2x2 grid (P&L, positions, equity, signals)

Trading Engine → update_positions() / add_trade() / update_equity()
                 → Updates module-level state + emits WebSocket event
```

Runs in separate thread; Flask-SocketIO handles thread safety.
