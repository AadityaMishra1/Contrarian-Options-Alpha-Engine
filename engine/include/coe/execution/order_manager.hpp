#pragma once

#include "coe/common/error.hpp"
#include "coe/common/types.hpp"
#include "coe/execution/order.hpp"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

namespace coe::execution {

using coe::common::ErrorCode;
using coe::common::Price;
using coe::common::Quantity;

/// Result alias for operations that carry no success payload.
/// std::variant<void, ErrorCode> is ill-formed, so std::monostate is used
/// as the success type instead.
using VoidResult = std::variant<std::monostate, ErrorCode>;

// ── OrderManager ────────────────────────────────────────────────────────────

/// Owns and manages the lifecycle of all orders in the execution engine.
///
/// The manager enforces a strict state machine: only transitions listed in the
/// OrderState documentation are permitted.  Any attempt to move an order to an
/// incompatible state returns ErrorCode::InvalidOrderState.
///
/// Thread-safety: NOT thread-safe.  The caller (e.g. a single-threaded event
/// loop) is responsible for external synchronisation if multiple threads call
/// into the same instance.
class OrderManager {
public:
    OrderManager() = default;

    // Non-copyable, movable.
    OrderManager(const OrderManager&)            = delete;
    OrderManager& operator=(const OrderManager&) = delete;
    OrderManager(OrderManager&&)                 = default;
    OrderManager& operator=(OrderManager&&)      = default;

    ~OrderManager() = default;

    // ── Submission ─────────────────────────────────────────────────────────

    /// Validate and store a new order.
    ///
    /// Assigns a fresh id via next_order_id(), forces state to New, stamps
    /// created/updated with the current system clock, then inserts the order
    /// into the internal map.
    ///
    /// Validation rules:
    ///   - quantity > 0
    ///   - Limit orders must have limit_price > 0
    ///   - symbol must be non-empty
    ///
    /// @return The assigned order id on success, or ErrorCode::InvalidParameter
    ///         / ErrorCode::OrderRejected on failure.
    [[nodiscard]] coe::common::Result<uint64_t> submit(Order order);

    // ── Cancellation ───────────────────────────────────────────────────────

    /// Request cancellation of an existing order.
    ///
    /// Cancellation is only valid from: New, PendingSend, Sent.
    /// Cancelling an order that is already Filled, Cancelled, or Rejected
    /// returns ErrorCode::InvalidOrderState.
    ///
    /// @return monostate on success, ErrorCode on failure.
    [[nodiscard]] VoidResult cancel(uint64_t order_id);

    // ── Fill notification ──────────────────────────────────────────────────

    /// Apply a (possibly partial) fill to an existing order.
    ///
    /// Updates filled_qty and recalculates avg_fill_price using a running
    /// volume-weighted average.  Transitions the order to PartialFill when
    /// filled_qty < quantity, or Filled when filled_qty == quantity.
    ///
    /// Preconditions:
    ///   - qty > 0
    ///   - price > 0
    ///   - order must be in state Sent or PartialFill
    ///
    /// @return monostate on success, ErrorCode on failure.
    [[nodiscard]] VoidResult on_fill(uint64_t order_id, Quantity qty, Price price);

    // ── Queries ────────────────────────────────────────────────────────────

    /// Look up an order by id.  Returns std::nullopt if not found.
    [[nodiscard]] std::optional<Order> get_order(uint64_t id) const;

    /// Return a snapshot of all orders that are not in a terminal state
    /// (i.e. not Filled, Cancelled, or Rejected).
    [[nodiscard]] std::vector<Order> get_open_orders() const;

    /// Total number of orders ever submitted to this manager (including closed).
    [[nodiscard]] std::size_t order_count() const noexcept { return orders_.size(); }

private:
    /// Returns true when transitioning to `next` from `current` is permitted
    /// by the state machine.
    [[nodiscard]] static bool is_valid_transition(OrderState current,
                                                  OrderState next) noexcept;

    std::unordered_map<uint64_t, Order> orders_;
};

} // namespace coe::execution
