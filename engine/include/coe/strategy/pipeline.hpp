#pragma once

#include <coe/strategy/bollinger.hpp>
#include <coe/strategy/greeks_filter.hpp>
#include <coe/strategy/rsi.hpp>
#include <coe/strategy/signal.hpp>
#include <coe/strategy/volume_spike.hpp>
#include <coe/common/config.hpp>
#include <coe/common/types.hpp>

#include <optional>

namespace coe::strategy {

/// Non-template facade that owns all four indicator/filter objects and
/// orchestrates the complete signal-scoring pipeline for a single symbol.
///
/// Lifecycle:
///   1. Construct with a Config (reads all parameters from YAML).
///   2. Feed market data via update_price(), update_volume(), update_greeks().
///   3. Call evaluate() after each relevant update to obtain a Signal when
///      the composite score clears the configured threshold.
///
/// Configuration keys consumed (all under "strategy.*"):
///   rsi.period, bollinger.period, bollinger.multiplier,
///   volume.lookback, volume.spike_threshold,
///   greeks.delta_min, greeks.delta_max, greeks.iv_percentile_max,
///   greeks.spread_pct_max,
///   scoring.weight_rsi, scoring.weight_bollinger, scoring.weight_volume,
///   scoring.weight_greeks, scoring.min_composite
class SignalScorer {
public:
    /// Construct and configure from a YAML Config object.
    explicit SignalScorer(const coe::common::Config& config);

    // Non-copyable (owns stateful indicator objects), movable.
    SignalScorer(const SignalScorer&)            = delete;
    SignalScorer& operator=(const SignalScorer&) = delete;
    SignalScorer(SignalScorer&&)                 = default;
    SignalScorer& operator=(SignalScorer&&)      = default;
    ~SignalScorer()                              = default;

    // ── Data ingestion ──────────────────────────────────────────────────────

    /// Feed a new price tick.  Updates RSI and BollingerBands.
    /// @param sym  Symbol (used only for consistency checking in future work).
    /// @param price  Mid-price or last-trade price.
    /// @param ts   Event timestamp.
    void update_price(const coe::common::Symbol& sym,
                      double                      price,
                      coe::common::Timestamp      ts) noexcept;

    /// Feed a new volume observation.  Updates VolumeSpike.
    void update_volume(const coe::common::Symbol& sym,
                       double                      volume) noexcept;

    /// Feed updated option Greeks and quote.  Updates GreeksFilter.
    void update_greeks(const coe::common::Symbol& sym,
                       double                      delta,
                       double                      iv_pct,
                       double                      bid,
                       double                      ask) noexcept;

    // ── Signal evaluation ───────────────────────────────────────────────────

    /// Compute the composite score and return a Signal if all indicators are
    /// ready and the composite score >= min_composite.
    ///
    /// @returns std::nullopt when the pipeline is still warming up or the
    ///          score is below threshold.
    [[nodiscard]] std::optional<Signal>
    evaluate(const coe::common::Symbol& sym) const noexcept;

    // ── Accessors for inspection / testing ─────────────────────────────────

    [[nodiscard]] const RSI&            rsi()     const noexcept { return rsi_; }
    [[nodiscard]] const BollingerBands& bb()      const noexcept { return bb_; }
    [[nodiscard]] const VolumeSpike&    volume()  const noexcept { return volume_; }
    [[nodiscard]] const GreeksFilter&   greeks()  const noexcept { return greeks_; }

    /// Reset all indicators back to their initial state.
    void reset() noexcept;

private:
    // ── Indicators ──────────────────────────────────────────────────────────
    RSI            rsi_;
    BollingerBands bb_;
    VolumeSpike    volume_;
    GreeksFilter   greeks_;

    // ── Scoring weights (read from config) ──────────────────────────────────
    double weight_rsi_       {0.30};
    double weight_bollinger_ {0.25};
    double weight_volume_    {0.25};
    double weight_greeks_    {0.20};
    double min_composite_    {0.65};

    // ── Thresholds for per-component score translation ───────────────────────
    double rsi_oversold_  {30.0};
    double rsi_overbought_{70.0};

    // ── Last known state ─────────────────────────────────────────────────────
    mutable coe::common::Timestamp last_ts_{};

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Map RSI value to a [0,1] contrarian score.
    /// Deep oversold (RSI << 30) => score near 1.
    /// Deep overbought (RSI >> 70) => score near 1 (put side).
    /// Neutral (RSI ~ 50) => score 0.
    [[nodiscard]] double rsi_to_score(double rsi_val) const noexcept;

    /// Map Bollinger value() (std-devs below lower band) to [0,1].
    [[nodiscard]] static double bb_to_score(double bb_val) noexcept;

    /// Map volume ratio to [0,1].
    [[nodiscard]] double volume_to_score(double ratio) const noexcept;
};

} // namespace coe::strategy
