#pragma once

#include "coe/common/types.hpp"

namespace coe::execution {

using coe::common::Price;
using coe::common::Quantity;

// ── KellyPositionSizer ───────────────────────────────────────────────────────

/// Calculates position sizes using the fractional Kelly criterion.
///
/// The full Kelly formula sizes a bet to maximise the expected logarithmic
/// growth of bankroll.  In practice a fraction (typically 0.5 = "half Kelly")
/// is applied to reduce variance while retaining most of the growth benefit.
///
/// Formula:
///   b  = avg_win / avg_loss   (payoff ratio)
///   f* = (b * p - q) / b      (full Kelly fraction of bankroll)
///   f  = kelly_fraction * f*  (scaled Kelly)
///   dollar_bet = min(f * bankroll, max_bet_pct% of bankroll)
///
/// If f* <= 0 the edge is negative or zero and the method returns 0 (no bet).
class KellyPositionSizer {
public:
    /// @param kelly_fraction  Scaling factor applied to the raw Kelly fraction.
    ///                        0.5 (half-Kelly) is the conventional default.
    ///                        Must be in (0, 1].
    /// @param max_bet         Maximum percentage of bankroll to risk on a single
    ///                        position (e.g. 20.0 = 20 %).  Must be > 0.
    explicit KellyPositionSizer(double kelly_fraction = 0.5,
                                double max_bet        = 20.0) noexcept;

    // ── Size calculation ───────────────────────────────────────────────────

    /// Compute the dollar amount to commit to a trade.
    ///
    /// @param win_rate   Historical win rate in [0, 1] (e.g. 0.55 = 55 %).
    /// @param avg_win    Average profit per winning trade in USD (> 0).
    /// @param avg_loss   Average loss per losing trade in USD (> 0).
    /// @param bankroll   Current total account equity in USD (> 0).
    /// @return Dollar amount to invest.  Returns 0.0 when the edge is
    ///         non-positive (f* <= 0) or when any input is invalid.
    [[nodiscard]] double calculate_size(double win_rate,
                                        double avg_win,
                                        double avg_loss,
                                        double bankroll) const noexcept;

    /// Convert a dollar size to a whole number of option contracts.
    ///
    /// contracts = floor(size / (option_price * 100))
    /// Result is clamped to a minimum of 1 contract when size > 0 (i.e. when
    /// the caller has a positive edge and a non-zero dollar allocation).
    ///
    /// @param size         Dollar amount returned by calculate_size().
    /// @param option_price Mid-market price of the option per share in USD.
    /// @return Whole number of contracts (>= 1 if size > 0, else 0).
    [[nodiscard]] Quantity contracts(double size,
                                     Price  option_price) const noexcept;

    // ── Accessors ──────────────────────────────────────────────────────────

    [[nodiscard]] double kelly_fraction() const noexcept { return kelly_fraction_; }
    [[nodiscard]] double max_bet()        const noexcept { return max_bet_; }

private:
    double kelly_fraction_;
    double max_bet_;          ///< Maximum bet as a percentage of bankroll.
};

} // namespace coe::execution
