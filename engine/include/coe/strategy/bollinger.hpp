#pragma once

#include <coe/strategy/concepts.hpp>

#include <deque>

namespace coe::strategy {

/// Bollinger Bands with a running-sum/sum-of-squares window.
///
/// Memory: O(period) — maintains a sliding deque of exactly `period` prices.
/// Complexity: O(1) per update (running sum avoids O(n) recalculation).
///
/// value() semantics (contrarian signal):
///   Returns how many standard deviations *below* the lower band the current
///   price sits.  A positive result means the price has broken below the lower
///   band and the magnitude indicates the depth of that breach.
///   Zero or negative means the price is inside or above the bands.
///
/// Not ready => value() / upper() / lower() / middle() all return 0.0.
class BollingerBands {
public:
    /// @param period      Look-back window length (default 20).
    /// @param multiplier  Band width in standard deviations (default 2.0).
    explicit BollingerBands(int period = 20, double multiplier = 2.0) noexcept;

    BollingerBands(const BollingerBands&)            = default;
    BollingerBands& operator=(const BollingerBands&) = default;
    BollingerBands(BollingerBands&&)                 = default;
    BollingerBands& operator=(BollingerBands&&)      = default;
    ~BollingerBands()                                = default;

    /// Ingest the next price observation.
    void update(double price) noexcept;

    /// Std-deviation distance below the lower band (positive = contrarian signal).
    [[nodiscard]] double value() const noexcept;

    /// Upper band: middle + multiplier * stddev.
    [[nodiscard]] double upper() const noexcept;

    /// Lower band: middle - multiplier * stddev.
    [[nodiscard]] double lower() const noexcept;

    /// Middle band: simple moving average over the window.
    [[nodiscard]] double middle() const noexcept;

    /// True after `period` prices have been consumed.
    [[nodiscard]] bool ready() const noexcept;

    /// Reset to construction state.
    void reset() noexcept;

private:
    /// Compute the population standard deviation from running sums.
    [[nodiscard]] double stddev() const noexcept;

    int               period_;
    double            multiplier_;
    std::deque<double> window_;
    double            sum_    {0.0};
    double            sum_sq_ {0.0};
    double            last_   {0.0}; // most recently ingested price
};

static_assert(Indicator<BollingerBands>, "BollingerBands must satisfy the Indicator concept");

} // namespace coe::strategy
