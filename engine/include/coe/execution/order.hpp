#pragma once

#include "coe/common/types.hpp"

#include <atomic>
#include <cstdint>

namespace coe::execution {

using coe::common::OptionType;
using coe::common::Price;
using coe::common::Quantity;
using coe::common::Side;
using coe::common::Symbol;
using coe::common::Timestamp;

// ── Order classification ────────────────────────────────────────────────────

enum class OrderType : uint8_t {
    Market,
    Limit,
};

// ── Order lifecycle states ──────────────────────────────────────────────────

/// Valid transitions:
///   New        -> PendingSend, Cancelled
///   PendingSend -> Sent, Rejected, Cancelled
///   Sent       -> PartialFill, Filled, Cancelled, Rejected
///   PartialFill -> PartialFill, Filled, Cancelled
///   Filled     -> (terminal)
///   Cancelled  -> (terminal)
///   Rejected   -> (terminal)
enum class OrderState : uint8_t {
    New,
    PendingSend,
    Sent,
    PartialFill,
    Filled,
    Cancelled,
    Rejected,
};

// ── Order aggregate ─────────────────────────────────────────────────────────

/// Represents a single options order from submission through settlement.
struct Order {
    /// Monotonically increasing identifier assigned at submission time.
    uint64_t id{0};

    /// OSI-format or underlying symbol identifying the instrument.
    Symbol symbol;

    /// Direction of the order.
    Side side{Side::Buy};

    /// Whether the order is on a Call or Put.
    OptionType option_type{OptionType::Call};

    /// Strike price in USD.
    Price strike{0.0};

    /// Execution type — market or limit.
    OrderType order_type{OrderType::Market};

    /// Limit price in USD (ignored for Market orders).
    Price limit_price{0.0};

    /// Total number of contracts requested.
    Quantity quantity{0};

    /// Cumulative number of contracts filled so far.
    Quantity filled_qty{0};

    /// Volume-weighted average fill price in USD.
    Price avg_fill_price{0.0};

    /// Current lifecycle state.
    OrderState state{OrderState::New};

    /// Nanosecond timestamp when the order was first submitted.
    Timestamp created{};

    /// Nanosecond timestamp of the most recent state change.
    Timestamp updated{};
};

// ── Monotonic order-ID generator ────────────────────────────────────────────

/// Returns a process-unique, monotonically increasing order identifier.
/// Thread-safe; uses a relaxed atomic fetch_add because strict ordering across
/// independent orders is not required — only uniqueness.
[[nodiscard]] inline uint64_t next_order_id() noexcept {
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

} // namespace coe::execution
