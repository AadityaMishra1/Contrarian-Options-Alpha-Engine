#include "coe/execution/position.hpp"

#include <algorithm>
#include <numeric>

namespace coe::execution {

// ── Internal helpers ─────────────────────────────────────────────────────────

namespace {

/// Multiplier for options contracts (standard US equity option = 100 shares).
constexpr double kContractMultiplier = 100.0;

} // anonymous namespace

// ── recalculate_unrealized ───────────────────────────────────────────────────

void PositionTracker::recalculate_unrealized(Position& pos) noexcept {
    if (pos.quantity == 0) {
        pos.unrealized_pnl = 0.0;
        return;
    }

    const double direction = (pos.side == Side::Buy) ? 1.0 : -1.0;
    pos.unrealized_pnl     = direction *
                             (pos.current_price - pos.avg_entry) *
                             static_cast<double>(pos.quantity) *
                             kContractMultiplier;
}

// ── on_fill ──────────────────────────────────────────────────────────────────

void PositionTracker::on_fill(const Symbol& sym, Side side,
                              Quantity qty, Price price) {
    auto [it, inserted] = positions_.try_emplace(sym);
    Position& pos       = it->second;

    if (inserted) {
        pos.symbol = sym;
    }

    if (pos.quantity == 0) {
        // ── Opening a new position ─────────────────────────────────────────
        pos.side      = side;
        pos.quantity  = qty;
        pos.avg_entry = price;
    } else if (pos.side == side) {
        // ── Adding to an existing position (same direction) ────────────────
        // Recalculate VWAP entry price.
        const double total_old = pos.avg_entry * static_cast<double>(pos.quantity);
        const double new_leg   = price        * static_cast<double>(qty);
        pos.quantity  += qty;
        pos.avg_entry  = (total_old + new_leg) / static_cast<double>(pos.quantity);
    } else {
        // ── Closing (or reversing) an existing position ────────────────────
        const Quantity close_qty = std::min(qty, pos.quantity);
        const Quantity open_qty  = qty - close_qty;

        // Realise PnL for the closed portion.
        const double direction    = (pos.side == Side::Buy) ? 1.0 : -1.0;
        const double closed_value = direction *
                                    (price - pos.avg_entry) *
                                    static_cast<double>(close_qty) *
                                    kContractMultiplier;
        pos.realized_pnl += closed_value;
        pos.quantity      -= close_qty;

        if (pos.quantity == 0 && open_qty > 0) {
            // Full close followed by a reversal: open the opposing leg.
            pos.side      = side;
            pos.quantity  = open_qty;
            pos.avg_entry = price;
        }
        // If quantity dropped to zero and open_qty == 0 the position is flat;
        // avg_entry is left as-is for historical reference.
    }

    recalculate_unrealized(pos);
}

// ── update_mark ──────────────────────────────────────────────────────────────

void PositionTracker::update_mark(const Symbol& sym, Price mark) {
    const auto it = positions_.find(sym);
    if (it == positions_.end()) {
        return;
    }

    Position& pos   = it->second;
    pos.current_price = mark;
    recalculate_unrealized(pos);
}

// ── get_position ─────────────────────────────────────────────────────────────

std::optional<Position> PositionTracker::get_position(const Symbol& sym) const {
    const auto it = positions_.find(sym);
    if (it == positions_.end()) {
        return std::nullopt;
    }
    return it->second;
}

// ── get_all_positions ─────────────────────────────────────────────────────────

std::vector<Position> PositionTracker::get_all_positions() const {
    std::vector<Position> result;
    result.reserve(positions_.size());
    for (const auto& [sym, pos] : positions_) {
        result.push_back(pos);
    }
    return result;
}

// ── open_position_count ──────────────────────────────────────────────────────

int32_t PositionTracker::open_position_count() const {
    int32_t count = 0;
    for (const auto& [sym, pos] : positions_) {
        if (pos.quantity != 0) {
            ++count;
        }
    }
    return count;
}

// ── total_unrealized_pnl ─────────────────────────────────────────────────────

double PositionTracker::total_unrealized_pnl() const {
    double total = 0.0;
    for (const auto& [sym, pos] : positions_) {
        total += pos.unrealized_pnl;
    }
    return total;
}

// ── total_realized_pnl ──────────────────────────────────────────────────────

double PositionTracker::total_realized_pnl() const {
    double total = 0.0;
    for (const auto& [sym, pos] : positions_) {
        total += pos.realized_pnl;
    }
    return total;
}

} // namespace coe::execution
