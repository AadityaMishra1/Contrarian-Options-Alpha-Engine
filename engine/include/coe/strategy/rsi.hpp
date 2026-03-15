#pragma once

#include <coe/strategy/concepts.hpp>

namespace coe::strategy {

/// Relative Strength Index using Wilder's exponential smoothing.
///
/// Memory: O(1) — only four scalars are retained regardless of history length.
/// Complexity: O(1) per update.
///
/// Formula:
///   RS  = avg_gain / avg_loss
///   RSI = 100 - (100 / (1 + RS))
///
/// Edge cases:
///   avg_loss == 0 => RSI = 100 (pure upward momentum)
///   Not ready     => value() returns 50.0 (neutral sentinel)
class RSI {
public:
    /// @param period  Wilder smoothing length (default 14).
    explicit RSI(int period = 14) noexcept;

    // Rule-of-five: all defaulted — no owning resources.
    RSI(const RSI&)            = default;
    RSI& operator=(const RSI&) = default;
    RSI(RSI&&)                 = default;
    RSI& operator=(RSI&&)      = default;
    ~RSI()                     = default;

    /// Ingest a new closing price.  First call seeds prev_price_ only.
    void update(double price) noexcept;

    /// Current RSI value in [0, 100].  Returns 50.0 until ready().
    [[nodiscard]] double value() const noexcept;

    /// True once (period + 1) prices have been consumed.
    [[nodiscard]] bool ready() const noexcept;

    /// Reset to construction state.
    void reset() noexcept;

    /// The smoothing period supplied at construction.
    [[nodiscard]] int period() const noexcept { return period_; }

private:
    int    period_;
    int    count_      {0};
    double prev_price_ {0.0};
    double avg_gain_   {0.0};
    double avg_loss_   {0.0};
};

static_assert(Indicator<RSI>, "RSI must satisfy the Indicator concept");

} // namespace coe::strategy
