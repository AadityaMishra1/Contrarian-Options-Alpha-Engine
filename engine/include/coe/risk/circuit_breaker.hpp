#pragma once

#include <deque>

namespace coe::risk {

/// Rolling-window circuit breaker that trips when the recent win rate of
/// closed trades falls below a configurable threshold.
///
/// The window is populated with the most recent @p window_size trade outcomes.
/// The breaker only evaluates once the window is fully populated, so it will
/// never trip on fewer than @p window_size trades.
///
/// This class is intentionally not thread-safe; callers must serialise access
/// if the breaker is shared across threads.
class CircuitBreaker {
public:
    /// Constructs the circuit breaker.
    ///
    /// @param window_size   Number of recent trades to consider (>= 1).
    /// @param min_win_rate  Minimum acceptable fraction of wins in [0, 1].
    explicit CircuitBreaker(int window_size = 20,
                            double min_win_rate = 0.40) noexcept;

    /// Records a new trade outcome and advances the rolling window.
    ///
    /// When the deque is already at capacity the oldest entry is evicted before
    /// the new outcome is appended, so the window always represents the most
    /// recent @p window_size trades.
    ///
    /// @param is_win  true if the trade was profitable.
    void record_trade(bool is_win) noexcept;

    /// Returns true when the circuit breaker has tripped.
    ///
    /// The breaker is tripped when all of the following hold:
    ///   - The rolling window is fully populated (trades_.size() >= window_size_)
    ///   - The current win rate is strictly less than min_win_rate_
    [[nodiscard]] bool is_tripped() const noexcept;

    /// Returns the win rate over the current rolling window.
    ///
    /// Returns 1.0 when the window is empty (conservative: not tripped).
    [[nodiscard]] double current_win_rate() const noexcept;

    /// Returns the number of trades currently in the rolling window.
    [[nodiscard]] int window_fill() const noexcept;

    /// Resets all state; the window is cleared and win count zeroed.
    void reset() noexcept;

private:
    std::deque<bool> trades_;
    int              win_count_{0};
    int              window_size_;
    double           min_win_rate_;
};

} // namespace coe::risk
