#pragma once

#include "coe/risk/circuit_breaker.hpp"
#include "coe/risk/limits.hpp"
#include "coe/risk/pnl_tracker.hpp"
#include "coe/common/error.hpp"
#include "coe/execution/order.hpp"

#include <variant>

namespace coe::risk {

/// Facade that composes all risk controls into a single decision gate.
///
/// Order acceptance is contingent on four sequential checks:
///   1. Daily PnL has not breached the configured loss limit.
///   2. The number of open positions is below the configured maximum.
///   3. The notional value of the new order does not exceed the single-position cap.
///   4. The rolling-window circuit breaker has not tripped.
///
/// Checks are evaluated in that order; the first failure short-circuits and
/// returns the corresponding ErrorCode.
///
/// This class is intentionally not thread-safe; callers must serialise access
/// if the manager is shared across threads.
class RiskManager {
public:
    /// Constructs the manager with the given risk thresholds.
    explicit RiskManager(const RiskLimits& limits);

    // Non-copyable, movable.
    RiskManager(const RiskManager&)            = delete;
    RiskManager& operator=(const RiskManager&) = delete;
    RiskManager(RiskManager&&)                 = default;
    RiskManager& operator=(RiskManager&&)      = default;

    ~RiskManager() = default;

    // ── Order gate ──────────────────────────────────────────────────────────

    /// Evaluates all risk checks before a new order is submitted.
    ///
    /// @param order              The proposed order (used for contextual
    ///                           information; not mutated).
    /// @param current_positions  Number of currently open positions.
    /// @param order_value        Notional value of the proposed order in USD
    ///                           thousands (must be >= 0).
    ///
    /// @return std::monostate{}  on success.
    /// @return ErrorCode::DailyLossExceeded      if daily PnL <= loss limit.
    /// @return ErrorCode::PositionLimitExceeded  if at or above max positions.
    /// @return ErrorCode::InsufficientMargin     if order_value > single-position cap.
    /// @return ErrorCode::CircuitBreakerTripped  if the circuit breaker has tripped.
    [[nodiscard]] coe::common::Result<std::monostate> check_new_order(
        const coe::execution::Order& order,
        int32_t current_positions,
        double  order_value) const;

    // ── Trade feedback ──────────────────────────────────────────────────────

    /// Called when a position is fully closed.
    ///
    /// Updates both the daily PnL tracker and the circuit-breaker window.
    ///
    /// @param pnl     Realised PnL of the closed trade (negative for a loss).
    /// @param is_win  true if the trade resulted in a net profit.
    void on_trade_closed(double pnl, bool is_win) noexcept;

    // ── Day management ──────────────────────────────────────────────────────

    /// Resets the daily PnL accumulator for a new trading day.
    ///
    /// The circuit-breaker window is intentionally preserved across day
    /// boundaries so that recent performance history is not discarded.
    void reset_daily() noexcept;

    // ── Observability ───────────────────────────────────────────────────────

    /// Returns the cumulative realised PnL for the current trading day.
    [[nodiscard]] double daily_pnl() const noexcept;

    /// Returns the current rolling win rate as seen by the circuit breaker.
    [[nodiscard]] double win_rate() const noexcept;

    /// Returns true when the circuit breaker has tripped.
    [[nodiscard]] bool is_circuit_breaker_tripped() const noexcept;

    /// Returns a const reference to the active risk limits.
    [[nodiscard]] const RiskLimits& limits() const noexcept;

private:
    RiskLimits      limits_;
    DailyPnLTracker pnl_tracker_;
    CircuitBreaker  circuit_breaker_;
};

} // namespace coe::risk
