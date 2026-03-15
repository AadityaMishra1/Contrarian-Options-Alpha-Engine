#include <coe/strategy/pipeline.hpp>

#include <algorithm>
#include <cmath>

namespace coe::strategy {

// ── Constructor ──────────────────────────────────────────────────────────────

SignalScorer::SignalScorer(const coe::common::Config& config)
    : rsi_{config.get<int>("strategy.rsi.period", 14)}
    , bb_{config.get<int>("strategy.bollinger.period", 20),
          config.get<double>("strategy.bollinger.multiplier", 2.0)}
    , volume_{config.get<int>("strategy.volume.lookback", 20),
              config.get<double>("strategy.volume.spike_threshold", 2.0)}
    , greeks_{config.get<double>("strategy.greeks.delta_min",         0.20),
              config.get<double>("strategy.greeks.delta_max",         0.40),
              config.get<double>("strategy.greeks.iv_percentile_max", 50.0),
              config.get<double>("strategy.greeks.spread_pct_max",    20.0)}
    , weight_rsi_      {config.get<double>("strategy.scoring.weight_rsi",       0.30)}
    , weight_bollinger_{config.get<double>("strategy.scoring.weight_bollinger", 0.25)}
    , weight_volume_   {config.get<double>("strategy.scoring.weight_volume",    0.25)}
    , weight_greeks_   {config.get<double>("strategy.scoring.weight_greeks",    0.20)}
    , min_composite_   {config.get<double>("strategy.scoring.min_composite",    0.65)}
    , rsi_oversold_    {config.get<double>("strategy.rsi.oversold",             30.0)}
    , rsi_overbought_  {config.get<double>("strategy.rsi.overbought",           70.0)}
{}

// ── Data ingestion ───────────────────────────────────────────────────────────

void SignalScorer::update_price(const coe::common::Symbol& /*sym*/,
                                double                      price,
                                coe::common::Timestamp      ts) noexcept {
    last_ts_ = ts;
    rsi_.update(price);
    bb_.update(price);
}

void SignalScorer::update_volume(const coe::common::Symbol& /*sym*/,
                                 double                      volume) noexcept {
    volume_.update(volume);
}

void SignalScorer::update_greeks(const coe::common::Symbol& /*sym*/,
                                 double delta,
                                 double iv_pct,
                                 double bid,
                                 double ask) noexcept {
    greeks_.update(delta, iv_pct, bid, ask);
}

// ── Score helpers ────────────────────────────────────────────────────────────

double SignalScorer::rsi_to_score(double rsi_val) const noexcept {
    // Oversold regime: RSI in [0, oversold] maps linearly to [1, 0].
    if (rsi_val <= rsi_oversold_) {
        return 1.0 - (rsi_val / rsi_oversold_);
    }
    // Overbought regime: RSI in [overbought, 100] maps linearly to [0, 1].
    if (rsi_val >= rsi_overbought_) {
        return (rsi_val - rsi_overbought_) / (100.0 - rsi_overbought_);
    }
    // Neutral zone: score 0.
    return 0.0;
}

double SignalScorer::bb_to_score(double bb_val) noexcept {
    // bb_val is std-devs below lower band.  Clamp to [0, 3] and normalise.
    if (bb_val <= 0.0) return 0.0;
    constexpr double max_devs = 3.0;
    return std::min(bb_val / max_devs, 1.0);
}

double SignalScorer::volume_to_score(double ratio) const noexcept {
    // Ratio == 1 (average) => 0, ratio == spike_threshold => 1, clamp above.
    const double threshold = volume_.spike_threshold();
    if (ratio <= 1.0) return 0.0;
    return std::min((ratio - 1.0) / (threshold - 1.0), 1.0);
}

// ── Evaluation ───────────────────────────────────────────────────────────────

std::optional<Signal>
SignalScorer::evaluate(const coe::common::Symbol& sym) const noexcept {
    // All four components must be ready before we emit any signal.
    if (!rsi_.ready() || !bb_.ready() || !volume_.ready()) {
        return std::nullopt;
    }

    const double rsi_val    = rsi_.value();
    const double bb_val     = bb_.value();
    const double vol_ratio  = volume_.value();

    const double rsi_s     = rsi_to_score(rsi_val);
    const double bb_s      = bb_to_score(bb_val);
    const double volume_s  = volume_to_score(vol_ratio);
    const double greeks_s  = greeks_.score(); // 0 if filter fails or no data

    const double composite = weight_rsi_       * rsi_s
                           + weight_bollinger_ * bb_s
                           + weight_volume_    * volume_s
                           + weight_greeks_    * greeks_s;

    if (composite < min_composite_) {
        return std::nullopt;
    }

    // Determine side: oversold RSI or price below lower band => Buy.
    // (Overbought => Sell, useful for put-buying in a contrarian bear setup.)
    const coe::common::Side side =
        (rsi_val <= rsi_oversold_ || bb_val > 0.0)
            ? coe::common::Side::Buy
            : coe::common::Side::Sell;

    return Signal{
        .symbol          = sym,
        .side            = side,
        .composite_score = composite,
        .rsi_score       = rsi_s,
        .bb_score        = bb_s,
        .volume_score    = volume_s,
        .greeks_score    = greeks_s,
        .ts              = last_ts_,
    };
}

// ── Reset ────────────────────────────────────────────────────────────────────

void SignalScorer::reset() noexcept {
    rsi_.reset();
    bb_.reset();
    volume_.reset();
    last_ts_ = coe::common::Timestamp{};
}

} // namespace coe::strategy
