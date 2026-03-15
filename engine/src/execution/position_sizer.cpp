#include "coe/execution/position_sizer.hpp"

#include <algorithm>
#include <cmath>

namespace coe::execution {

// ── Constructor ───────────────────────────────────────────────────────────────

KellyPositionSizer::KellyPositionSizer(double kelly_fraction,
                                       double max_bet) noexcept
    : kelly_fraction_(kelly_fraction),
      max_bet_(max_bet) {}

// ── calculate_size ────────────────────────────────────────────────────────────

double KellyPositionSizer::calculate_size(double win_rate,
                                          double avg_win,
                                          double avg_loss,
                                          double bankroll) const noexcept {
    // Guard against degenerate inputs that would produce undefined or
    // meaningless results.
    if (win_rate  <= 0.0 || win_rate  >= 1.0) { return 0.0; }
    if (avg_win   <= 0.0)                      { return 0.0; }
    if (avg_loss  <= 0.0)                      { return 0.0; }
    if (bankroll  <= 0.0)                      { return 0.0; }
    if (kelly_fraction_ <= 0.0)                { return 0.0; }

    const double q = 1.0 - win_rate;

    // Payoff ratio: how many dollars are won per dollar risked.
    const double b = avg_win / avg_loss;

    // Full Kelly fraction of bankroll to wager.
    // Derivation: maximise E[ln(W)] => f* = (b*p - q) / b
    const double f_star = (b * win_rate - q) / b;

    // Negative or zero Kelly implies no edge — do not bet.
    if (f_star <= 0.0) {
        return 0.0;
    }

    // Scale by the fractional Kelly factor (default 0.5 = half Kelly).
    const double f_scaled = kelly_fraction_ * f_star;

    // Dollar amount before the max-bet cap.
    const double dollar_bet = f_scaled * bankroll;

    // Cap to max_bet_ percent of bankroll.
    const double cap = (max_bet_ / 100.0) * bankroll;

    return std::min(dollar_bet, cap);
}

// ── contracts ─────────────────────────────────────────────────────────────────

Quantity KellyPositionSizer::contracts(double size,
                                       Price  option_price) const noexcept {
    // A zero or negative size means no position should be opened.
    if (size <= 0.0 || option_price <= 0.0) {
        return 0;
    }

    // Each standard US equity option contract covers 100 shares.
    constexpr double kContractMultiplier = 100.0;

    const double raw       = size / (option_price * kContractMultiplier);
    const auto   floored   = static_cast<Quantity>(std::floor(raw));

    // Always allocate at least one contract when the trader has a valid size.
    return std::max(floored, static_cast<Quantity>(1));
}

} // namespace coe::execution
