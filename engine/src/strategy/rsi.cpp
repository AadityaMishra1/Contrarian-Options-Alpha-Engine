#include <coe/strategy/rsi.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace coe::strategy {

RSI::RSI(int period) noexcept : period_{period} {}

void RSI::update(double price) noexcept {
    ++count_;

    if (count_ == 1) {
        // Seed the previous price; no gain/loss to record yet.
        prev_price_ = price;
        return;
    }

    const double change = price - prev_price_;
    prev_price_         = price;

    const double gain = (change > 0.0) ? change : 0.0;
    const double loss = (change < 0.0) ? -change : 0.0;

    if (count_ == 2) {
        // Bootstrap: treat first difference as the initial smoothed value.
        avg_gain_ = gain;
        avg_loss_ = loss;
        return;
    }

    // Wilder's smoothing: EMA with alpha = 1/period.
    const double inv_period = 1.0 / static_cast<double>(period_);
    avg_gain_ = avg_gain_ * (1.0 - inv_period) + gain * inv_period;
    avg_loss_ = avg_loss_ * (1.0 - inv_period) + loss * inv_period;
}

double RSI::value() const noexcept {
    if (!ready()) {
        return 50.0; // neutral sentinel before warm-up
    }
    if (avg_loss_ == 0.0) {
        return 100.0;
    }
    const double rs = avg_gain_ / avg_loss_;
    return 100.0 - (100.0 / (1.0 + rs));
}

bool RSI::ready() const noexcept {
    // We need (period + 1) prices: one to seed prev_price_ and period more to
    // build the smoothed averages up through the first real RSI computation.
    return count_ > period_;
}

void RSI::reset() noexcept {
    count_      = 0;
    prev_price_ = 0.0;
    avg_gain_   = 0.0;
    avg_loss_   = 0.0;
}

} // namespace coe::strategy
