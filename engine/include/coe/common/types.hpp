#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>

namespace coe::common {

// ── Primitive aliases ──────────────────────────────────────────────────────

using Price     = double;
using Quantity  = int32_t;
using Timestamp = std::chrono::nanoseconds;
using Symbol    = std::string;

// ── Enumerations ───────────────────────────────────────────────────────────

enum class Side : uint8_t {
    Buy,
    Sell,
};

enum class OptionType : uint8_t {
    Call,
    Put,
};

// ── toString helpers ───────────────────────────────────────────────────────

[[nodiscard]] constexpr std::string_view toString(Side side) noexcept {
    switch (side) {
        case Side::Buy:  return "Buy";
        case Side::Sell: return "Sell";
    }
    return "Unknown";
}

[[nodiscard]] constexpr std::string_view toString(OptionType type) noexcept {
    switch (type) {
        case OptionType::Call: return "Call";
        case OptionType::Put:  return "Put";
    }
    return "Unknown";
}

// ── Option contract ────────────────────────────────────────────────────────

/// Represents a single exchange-listed option contract.
struct OptionContract {
    /// Underlying equity or index symbol (e.g. "SPY").
    Symbol underlying;

    /// Whether this is a Call or Put.
    OptionType type;

    /// Strike price in USD.
    Price strike;

    /// Expiry expressed as nanoseconds since the Unix epoch.
    Timestamp expiry;

    /// Days-to-expiration at the time of contract construction.
    int32_t dte;
};

} // namespace coe::common
