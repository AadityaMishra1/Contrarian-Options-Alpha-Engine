#pragma once

#include "coe/common/types.hpp"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace coe::execution {

using coe::common::Price;
using coe::common::Quantity;
using coe::common::Side;
using coe::common::Symbol;

// ── Position aggregate ───────────────────────────────────────────────────────

/// Tracks the net exposure for a single option symbol.
///
/// Convention:
///   - A Buy fill increases quantity; a Sell fill decreases it.
///   - When quantity reaches zero the position is considered flat but is
///     retained in the map so that realized_pnl history is preserved.
///   - unrealized_pnl = (current_price - avg_entry) * quantity * 100
///     (each equity option contract covers 100 shares).
struct Position {
    /// Instrument identifier (OSI symbol or underlying).
    Symbol symbol;

    /// Side of the net position (Buy = long, Sell = short).
    Side side{Side::Buy};

    /// Net number of contracts held.  Positive = long, 0 = flat.
    Quantity quantity{0};

    /// Volume-weighted average entry price paid per contract (in USD).
    Price avg_entry{0.0};

    /// Most recently applied mark price (last known mid or settlement).
    Price current_price{0.0};

    /// Mark-to-market PnL in USD: (current_price - avg_entry) * qty * 100.
    /// Sign is positive when the position is in profit.
    double unrealized_pnl{0.0};

    /// Cumulative closed PnL in USD for this symbol (survives partial closes).
    double realized_pnl{0.0};
};

// ── PositionTracker ──────────────────────────────────────────────────────────

/// Maintains a per-symbol book of option positions and calculates running PnL.
///
/// Thread-safety: NOT thread-safe.  Caller is responsible for synchronisation.
class PositionTracker {
public:
    PositionTracker() = default;

    // Non-copyable, movable.
    PositionTracker(const PositionTracker&)            = delete;
    PositionTracker& operator=(const PositionTracker&) = delete;
    PositionTracker(PositionTracker&&)                 = default;
    PositionTracker& operator=(PositionTracker&&)      = default;

    ~PositionTracker() = default;

    // ── Fill ingestion ─────────────────────────────────────────────────────

    /// Apply a fill event to the position book.
    ///
    /// - Buy  fills increase the net long position.
    /// - Sell fills first close the existing long (realising PnL), then open
    ///   a short for any residual quantity.
    ///
    /// If no position exists for sym a new one is created.
    void on_fill(const Symbol& sym, Side side, Quantity qty, Price price);

    // ── Mark-to-market ─────────────────────────────────────────────────────

    /// Update the mark price for sym and recalculate unrealized_pnl.
    /// No-op if no position for sym exists.
    void update_mark(const Symbol& sym, Price mark);

    // ── Queries ────────────────────────────────────────────────────────────

    /// Retrieve a copy of the position for sym, or std::nullopt if absent.
    [[nodiscard]] std::optional<Position> get_position(const Symbol& sym) const;

    /// Return a snapshot of all tracked positions (including flat ones).
    [[nodiscard]] std::vector<Position> get_all_positions() const;

    /// Number of symbols whose quantity is strictly non-zero (open positions).
    [[nodiscard]] int32_t open_position_count() const;

    /// Sum of unrealized_pnl across all positions, in USD.
    [[nodiscard]] double total_unrealized_pnl() const;

    /// Sum of realized_pnl across all positions, in USD.
    [[nodiscard]] double total_realized_pnl() const;

private:
    /// Recalculate pos.unrealized_pnl from current fields.
    static void recalculate_unrealized(Position& pos) noexcept;

    std::unordered_map<Symbol, Position> positions_;
};

} // namespace coe::execution
