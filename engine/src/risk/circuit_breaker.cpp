#include "coe/risk/circuit_breaker.hpp"

#include <cassert>

namespace coe::risk {

CircuitBreaker::CircuitBreaker(int window_size, double min_win_rate) noexcept
    : window_size_{window_size}
    , min_win_rate_{min_win_rate}
{
    assert(window_size_ >= 1 && "CircuitBreaker window_size must be >= 1");
    assert(min_win_rate_ >= 0.0 && min_win_rate_ <= 1.0 &&
           "CircuitBreaker min_win_rate must be in [0, 1]");
}

void CircuitBreaker::record_trade(bool is_win) noexcept {
    // Evict the oldest outcome if the window is already full.
    if (static_cast<int>(trades_.size()) >= window_size_) {
        if (trades_.front()) {
            --win_count_;
        }
        trades_.pop_front();
    }

    trades_.push_back(is_win);
    if (is_win) {
        ++win_count_;
    }
}

bool CircuitBreaker::is_tripped() const noexcept {
    // Do not trip until the window has been fully populated at least once.
    if (static_cast<int>(trades_.size()) < window_size_) {
        return false;
    }
    return current_win_rate() < min_win_rate_;
}

double CircuitBreaker::current_win_rate() const noexcept {
    if (trades_.empty()) {
        // Conservative: treat an empty window as 100% win rate so that the
        // breaker never trips before any trades are observed.
        return 1.0;
    }
    return static_cast<double>(win_count_) /
           static_cast<double>(trades_.size());
}

int CircuitBreaker::window_fill() const noexcept {
    return static_cast<int>(trades_.size());
}

void CircuitBreaker::reset() noexcept {
    trades_.clear();
    win_count_ = 0;
}

} // namespace coe::risk
