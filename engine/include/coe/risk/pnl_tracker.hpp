#pragma once

#include <cstdint>

namespace coe::risk {

/// Tracks cumulative realised PnL within a single trading day.
///
/// This class is intentionally not thread-safe; callers must serialise access
/// if the tracker is shared across threads.
class DailyPnLTracker {
public:
    DailyPnLTracker() = default;

    /// Accumulates a closed-trade PnL contribution into the daily total.
    ///
    /// @param pnl  Realised PnL for the trade (negative for a loss).
    void record_trade(double pnl) noexcept;

    /// Returns the cumulative PnL for the current trading day.
    [[nodiscard]] double daily_pnl() const noexcept;

    /// Returns the number of trades recorded since the last reset().
    [[nodiscard]] int32_t trade_count() const noexcept;

    /// Returns true when the daily PnL has reached or breached @p limit.
    ///
    /// "Breached" means daily_pnl() <= limit (i.e. a loss at least as large
    /// as the configured threshold).
    ///
    /// @param limit  The loss threshold (expected to be <= 0.0).
    [[nodiscard]] bool is_limit_breached(double limit) const noexcept;

    /// Resets all state for a new trading day.
    void reset() noexcept;

private:
    double  daily_total_{0.0};
    int32_t trade_count_{0};
};

} // namespace coe::risk
