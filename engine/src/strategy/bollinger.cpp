#include <coe/strategy/bollinger.hpp>

#include <cmath>

namespace coe::strategy {

BollingerBands::BollingerBands(int period, double multiplier) noexcept
    : period_{period}, multiplier_{multiplier}
{}

void BollingerBands::update(double price) noexcept {
    last_ = price;

    window_.push_back(price);
    sum_    += price;
    sum_sq_ += price * price;

    if (static_cast<int>(window_.size()) > period_) {
        const double evicted = window_.front();
        window_.pop_front();
        sum_    -= evicted;
        sum_sq_ -= evicted * evicted;
    }
}

double BollingerBands::stddev() const noexcept {
    if (window_.empty()) return 0.0;

    const double n    = static_cast<double>(window_.size());
    const double mean = sum_ / n;
    // Variance using the computational formula: E[x^2] - (E[x])^2
    const double variance = (sum_sq_ / n) - (mean * mean);
    // Guard against floating-point negative epsilon.
    return (variance > 0.0) ? std::sqrt(variance) : 0.0;
}

double BollingerBands::middle() const noexcept {
    if (!ready()) return 0.0;
    return sum_ / static_cast<double>(period_);
}

double BollingerBands::upper() const noexcept {
    if (!ready()) return 0.0;
    return middle() + multiplier_ * stddev();
}

double BollingerBands::lower() const noexcept {
    if (!ready()) return 0.0;
    return middle() - multiplier_ * stddev();
}

double BollingerBands::value() const noexcept {
    if (!ready()) return 0.0;

    const double sd = stddev();
    if (sd == 0.0) return 0.0;

    // Positive result: price is below the lower band by this many std-devs.
    const double lb = lower();
    return (lb - last_) / sd;
}

bool BollingerBands::ready() const noexcept {
    return static_cast<int>(window_.size()) == period_;
}

void BollingerBands::reset() noexcept {
    window_.clear();
    sum_    = 0.0;
    sum_sq_ = 0.0;
    last_   = 0.0;
}

} // namespace coe::strategy
