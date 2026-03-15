#include "coe/execution/order_manager.hpp"

#include <chrono>

namespace coe::execution {

// ── Internal helpers ─────────────────────────────────────────────────────────

namespace {

/// Returns the current wall-clock time as a Timestamp (nanoseconds since epoch).
[[nodiscard]] Timestamp now() noexcept {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
}

/// Returns true when the given state is terminal (no further transitions
/// are possible).
[[nodiscard]] constexpr bool is_terminal(OrderState s) noexcept {
    return s == OrderState::Filled   ||
           s == OrderState::Cancelled ||
           s == OrderState::Rejected;
}

} // anonymous namespace

// ── State machine ────────────────────────────────────────────────────────────

bool OrderManager::is_valid_transition(OrderState current,
                                       OrderState next) noexcept {
    // Terminal states accept no further transitions.
    if (is_terminal(current)) {
        return false;
    }

    switch (current) {
        case OrderState::New:
            return next == OrderState::PendingSend ||
                   next == OrderState::Cancelled;

        case OrderState::PendingSend:
            return next == OrderState::Sent      ||
                   next == OrderState::Rejected   ||
                   next == OrderState::Cancelled;

        case OrderState::Sent:
            return next == OrderState::PartialFill ||
                   next == OrderState::Filled      ||
                   next == OrderState::Cancelled   ||
                   next == OrderState::Rejected;

        case OrderState::PartialFill:
            return next == OrderState::PartialFill ||
                   next == OrderState::Filled      ||
                   next == OrderState::Cancelled;

        // Terminal states: handled above, listed here for compiler exhaustiveness.
        case OrderState::Filled:
        case OrderState::Cancelled:
        case OrderState::Rejected:
            return false;
    }
    return false;
}

// ── submit ───────────────────────────────────────────────────────────────────

coe::common::Result<uint64_t> OrderManager::submit(Order order) {
    // ── Input validation ────────────────────────────────────────────────────
    if (order.symbol.empty()) {
        return coe::common::ErrorCode::InvalidParameter;
    }
    if (order.quantity <= 0) {
        return coe::common::ErrorCode::InvalidParameter;
    }
    if (order.order_type == OrderType::Limit && order.limit_price <= 0.0) {
        return coe::common::ErrorCode::InvalidParameter;
    }

    // ── Assign identity and initial state ───────────────────────────────────
    order.id          = next_order_id();
    order.state       = OrderState::New;
    order.filled_qty  = 0;
    order.avg_fill_price = 0.0;

    const Timestamp ts = now();
    order.created = ts;
    order.updated = ts;

    const uint64_t id = order.id;
    orders_.emplace(id, std::move(order));

    return id;
}

// ── cancel ───────────────────────────────────────────────────────────────────

VoidResult OrderManager::cancel(uint64_t order_id) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return coe::common::ErrorCode::OrderRejected;
    }

    Order& o = it->second;

    if (!is_valid_transition(o.state, OrderState::Cancelled)) {
        return coe::common::ErrorCode::InvalidOrderState;
    }

    o.state   = OrderState::Cancelled;
    o.updated = now();

    return std::monostate{};
}

// ── on_fill ──────────────────────────────────────────────────────────────────

VoidResult OrderManager::on_fill(uint64_t order_id, Quantity qty, Price price) {
    if (qty <= 0 || price <= 0.0) {
        return coe::common::ErrorCode::InvalidParameter;
    }

    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return coe::common::ErrorCode::OrderRejected;
    }

    Order& o = it->second;

    // Fills are only valid from Sent or PartialFill.
    if (o.state != OrderState::Sent && o.state != OrderState::PartialFill) {
        return coe::common::ErrorCode::InvalidOrderState;
    }

    // Guard against overfill.
    const Quantity remaining = o.quantity - o.filled_qty;
    if (qty > remaining) {
        return coe::common::ErrorCode::InvalidParameter;
    }

    // Update running volume-weighted average fill price.
    // vwap_new = (vwap_old * old_qty + price * new_qty) / (old_qty + new_qty)
    const double total_old_value = o.avg_fill_price * static_cast<double>(o.filled_qty);
    const double new_fill_value  = price * static_cast<double>(qty);
    o.filled_qty    += qty;
    o.avg_fill_price = (total_old_value + new_fill_value) /
                       static_cast<double>(o.filled_qty);

    // Determine the resulting state.
    const OrderState next_state = (o.filled_qty == o.quantity)
                                      ? OrderState::Filled
                                      : OrderState::PartialFill;

    o.state   = next_state;
    o.updated = now();

    return std::monostate{};
}

// ── get_order ────────────────────────────────────────────────────────────────

std::optional<Order> OrderManager::get_order(uint64_t id) const {
    const auto it = orders_.find(id);
    if (it == orders_.end()) {
        return std::nullopt;
    }
    return it->second;
}

// ── get_open_orders ──────────────────────────────────────────────────────────

std::vector<Order> OrderManager::get_open_orders() const {
    std::vector<Order> result;
    result.reserve(orders_.size());

    for (const auto& [id, order] : orders_) {
        if (!is_terminal(order.state)) {
            result.push_back(order);
        }
    }

    return result;
}

} // namespace coe::execution
