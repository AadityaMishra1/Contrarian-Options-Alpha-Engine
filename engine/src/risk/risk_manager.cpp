#include "coe/risk/risk_manager.hpp"

namespace coe::risk {

// ── Construction ──────────────────────────────────────────────────────────────

RiskManager::RiskManager(const RiskLimits& limits)
    : limits_{limits}
    , pnl_tracker_{}
    , circuit_breaker_{limits.circuit_breaker_window, limits.min_win_rate}
{}

// ── Order gate ────────────────────────────────────────────────────────────────

coe::common::Result<std::monostate> RiskManager::check_new_order(
    const coe::execution::Order& /*order*/,
    int32_t current_positions,
    double  order_value) const
{
    using coe::common::ErrorCode;

    // Check 1: daily loss limit.
    // The tracker's is_limit_breached() returns true when daily_pnl <= limit,
    // meaning the running loss has met or exceeded the configured threshold.
    if (pnl_tracker_.is_limit_breached(limits_.daily_loss_limit)) {
        return ErrorCode::DailyLossExceeded;
    }

    // Check 2: open-position count.
    // Reject if adding this order would meet or exceed the maximum.
    if (current_positions >= limits_.max_positions) {
        return ErrorCode::PositionLimitExceeded;
    }

    // Check 3: single-position notional cap.
    // Reject if the order's notional value exceeds the configured ceiling.
    if (order_value > limits_.max_single_position) {
        return ErrorCode::InsufficientMargin;
    }

    // Check 4: circuit breaker.
    // Reject if the rolling win rate has fallen below min_win_rate.
    if (circuit_breaker_.is_tripped()) {
        return ErrorCode::CircuitBreakerTripped;
    }

    return std::monostate{};
}

// ── Trade feedback ────────────────────────────────────────────────────────────

void RiskManager::on_trade_closed(double pnl, bool is_win) noexcept {
    pnl_tracker_.record_trade(pnl);
    circuit_breaker_.record_trade(is_win);
}

// ── Day management ────────────────────────────────────────────────────────────

void RiskManager::reset_daily() noexcept {
    pnl_tracker_.reset();
    // The circuit-breaker window deliberately carries over across day
    // boundaries — losing streaks that span midnight should remain visible.
}

// ── Observability ─────────────────────────────────────────────────────────────

double RiskManager::daily_pnl() const noexcept {
    return pnl_tracker_.daily_pnl();
}

double RiskManager::win_rate() const noexcept {
    return circuit_breaker_.current_win_rate();
}

bool RiskManager::is_circuit_breaker_tripped() const noexcept {
    return circuit_breaker_.is_tripped();
}

const RiskLimits& RiskManager::limits() const noexcept {
    return limits_;
}

} // namespace coe::risk
