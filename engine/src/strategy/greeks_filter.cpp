#include <coe/strategy/greeks_filter.hpp>

#include <algorithm>
#include <cmath>

namespace coe::strategy {

GreeksFilter::GreeksFilter(double delta_min,
                           double delta_max,
                           double iv_pct_max,
                           double spread_pct_max) noexcept
    : delta_min_{delta_min}
    , delta_max_{delta_max}
    , iv_pct_max_{iv_pct_max}
    , spread_pct_max_{spread_pct_max}
{}

void GreeksFilter::update(double delta,
                          double iv_percentile,
                          double bid,
                          double ask) noexcept {
    abs_delta_ = std::abs(delta);
    iv_pct_    = iv_percentile;

    // Spread as a percentage of the mid-price.
    const double mid = (bid + ask) * 0.5;
    if (mid > 0.0) {
        spread_pct_ = ((ask - bid) / mid) * 100.0;
    } else {
        spread_pct_ = spread_pct_max_; // treat invalid quote as worst-case
    }

    has_data_ = true;
}

bool GreeksFilter::passes() const noexcept {
    if (!has_data_) return false;

    const bool delta_ok  = (abs_delta_ >= delta_min_) && (abs_delta_ <= delta_max_);
    const bool iv_ok     = (iv_pct_ < iv_pct_max_);
    const bool spread_ok = (spread_pct_ < spread_pct_max_);

    return delta_ok && iv_ok && spread_ok;
}

double GreeksFilter::delta_partial_score() const noexcept {
    if (abs_delta_ < delta_min_ || abs_delta_ > delta_max_) return 0.0;

    // Score is highest at the centre of [delta_min, delta_max] and falls
    // linearly to 0 at the edges.
    const double centre = (delta_min_ + delta_max_) * 0.5;
    const double half   = (delta_max_ - delta_min_) * 0.5;
    if (half == 0.0) return 1.0;
    return 1.0 - (std::abs(abs_delta_ - centre) / half);
}

double GreeksFilter::iv_partial_score() const noexcept {
    if (iv_pct_ >= iv_pct_max_) return 0.0;
    // Lower IV percentile is better: score = 1 - iv_pct / iv_pct_max.
    return 1.0 - (iv_pct_ / iv_pct_max_);
}

double GreeksFilter::spread_partial_score() const noexcept {
    if (spread_pct_ >= spread_pct_max_) return 0.0;
    // Tighter spread is better.
    return 1.0 - (spread_pct_ / spread_pct_max_);
}

double GreeksFilter::score() const noexcept {
    if (!passes()) return 0.0;

    constexpr double weight = 1.0 / 3.0;
    return weight * delta_partial_score()
         + weight * iv_partial_score()
         + weight * spread_partial_score();
}

} // namespace coe::strategy
