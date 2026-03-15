#pragma once

#include <coe/common/types.hpp>

namespace coe::strategy {

/// A fully-scored trading signal emitted by SignalScorer::evaluate().
///
/// All score fields are normalised to [0, 1].
/// composite_score is the weighted sum of the four per-indicator scores and
/// must reach the configured min_composite threshold before a Signal is emitted.
struct Signal {
    /// Underlying or option symbol this signal relates to.
    coe::common::Symbol symbol;

    /// Suggested trade direction derived from contrarian logic (oversold => Buy).
    coe::common::Side side;

    /// Weighted composite of the four component scores.
    double composite_score {0.0};

    /// Contribution from the RSI indicator (0 = neutral, 1 = deep oversold/overbought).
    double rsi_score {0.0};

    /// Contribution from Bollinger Bands (0 = inside bands, 1 = far below lower band).
    double bb_score {0.0};

    /// Contribution from volume analysis (0 = normal volume, 1 = extreme spike).
    double volume_score {0.0};

    /// Contribution from the Greeks/liquidity filter (0 = failing, 1 = ideal).
    double greeks_score {0.0};

    /// Wall-clock timestamp at which evaluate() was called, in nanoseconds.
    coe::common::Timestamp ts;
};

} // namespace coe::strategy
