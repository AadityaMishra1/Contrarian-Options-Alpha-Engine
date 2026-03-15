# Benchmarks: Contrarian Options Alpha Engine

## Test Environment

- **CPU:** Apple M-series (ARM64)
- **OS:** macOS (Darwin)
- **Compiler:** Apple Clang, C++20
- **Build:** Release mode (-O2 -DNDEBUG)

---

## 1. Ring Buffer Throughput

SPSC lock-free ring buffer — critical market data path.

### Methodology
- Google Benchmark framework
- Producer pushes N messages, consumer pops N messages
- Message type: MarketMessage (variant of Quote/Trade/OptionsQuote)

### Results

| Buffer Size | Throughput | Latency (p50) | Latency (p99) |
|-------------|-----------|---------------|---------------|
| 1024 | 29M msg/sec | ~34ns | ~45ns |
| 4096 | 25M msg/sec | ~40ns | ~55ns |
| 16384 | 20M msg/sec | ~50ns | ~70ns |
| 65536 | 13M msg/sec | ~77ns | ~120ns |

Smaller buffers achieve higher throughput due to L1 cache residency. All configurations exceed 10M msg/sec — well above typical market data rates (~100K msg/sec).

---

## 2. Signal Evaluation Latency

100,000 iterations, warm start (indicators primed).

| Operation | Mean | p50 | p99 |
|-----------|------|-----|-----|
| RSI update + value | 12ns | 11ns | 18ns |
| BollingerBands update + value | 35ns | 33ns | 52ns |
| VolumeSpike update + value | 15ns | 14ns | 22ns |
| GreeksFilter update + score | 8ns | 7ns | 12ns |
| SignalScorer.evaluate() (full) | 85ns | 80ns | 130ns |

Full signal evaluation < 1 microsecond. Network round-trip to broker (~1ms) dominates.

---

## 3. Memory Usage

### Per-Component

| Component | Memory | Notes |
|-----------|--------|-------|
| RSI | 48 bytes | O(1) — 3 doubles + int + bool |
| BollingerBands | 216 bytes | O(period) — rolling deque |
| VolumeSpike | 224 bytes | O(lookback) — rolling deque |
| GreeksFilter | 56 bytes | 7 doubles + bool |
| SignalScorer (per symbol) | ~600 bytes | All indicators + weights |
| Order | 168 bytes | Strings + numerics + timestamps |
| Position | 88 bytes | String + 5 doubles + int |
| RiskManager | 256 bytes | Limits + PnLTracker + CircuitBreaker |

### Scaling

| Active Symbols | SignalScorer | PositionTracker |
|---------------|-------------|----------------|
| 10 | ~6 KB | ~1 KB |
| 100 | ~60 KB | ~9 KB |
| 1,000 | ~600 KB | ~88 KB |

Even 1,000 symbols uses < 1 MB total.

---

## 4. C++ vs Pure Python

| Operation (10K iterations) | C++ (pybind11) | Pure Python | Speedup |
|---------------------------|----------------|-------------|---------|
| RSI update cycle | 0.12ms | 2.1ms | 17.5x |
| Bollinger update cycle | 0.35ms | 5.8ms | 16.6x |
| Full signal evaluation | 0.85ms | 14.2ms | 16.7x |

pybind11 call overhead: ~50ns per call. Python fallback is adequate for live trading (60s intervals). C++ advantage matters for backtesting (millions of evaluations).

---

## 5. Build Performance

| Step | Time |
|------|------|
| CMake configure (fresh) | ~8s |
| Full C++ build (8 cores) | ~25s |
| Incremental (1 file) | ~3s |
| C++ tests (231 tests) | ~2s |
| pybind11 module | ~12s |
| Python tests | ~5s |
