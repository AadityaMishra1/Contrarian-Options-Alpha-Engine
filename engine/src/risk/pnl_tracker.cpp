#include "coe/risk/pnl_tracker.hpp"

namespace coe::risk {

void DailyPnLTracker::record_trade(double pnl) noexcept {
    daily_total_ += pnl;
    ++trade_count_;
}

double DailyPnLTracker::daily_pnl() const noexcept {
    return daily_total_;
}

int32_t DailyPnLTracker::trade_count() const noexcept {
    return trade_count_;
}

bool DailyPnLTracker::is_limit_breached(double limit) const noexcept {
    // A breach occurs when the running loss equals or exceeds the threshold
    // magnitude, i.e. the PnL has gone at-or-below the (negative) limit.
    return daily_total_ <= limit;
}

void DailyPnLTracker::reset() noexcept {
    daily_total_ = 0.0;
    trade_count_ = 0;
}

} // namespace coe::risk
