#pragma once

#include <cstdint>

namespace coe::risk {

/// Aggregate of all configurable risk thresholds for the contrarian engine.
///
/// All monetary values are expressed in USD thousands (e.g. daily_loss_limit
/// of -50.0 means -$50k).  Position counts are raw contract counts.
struct RiskLimits {
    /// Maximum cumulative daily loss before new orders are blocked.
    /// Must be <= 0.0.  Orders are blocked when daily_pnl <= this value.
    double daily_loss_limit{-50.0};

    /// Maximum number of concurrently open positions.
    /// Orders are blocked when current_positions >= this value.
    int32_t max_positions{5};

    /// Maximum notional value of any single new order, in USD thousands.
    /// Orders are blocked when order_value > this value.
    double max_single_position{20.0};

    /// Rolling window size (number of trades) used by the circuit breaker.
    int32_t circuit_breaker_window{20};

    /// Minimum required win rate over the circuit-breaker window.
    /// The circuit breaker trips when the observed win rate falls below this.
    double min_win_rate{0.40};
};

} // namespace coe::risk
