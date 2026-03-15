#pragma once

#include <cstdint>
#include <string_view>
#include <variant>

namespace coe::common {

// ── Error codes ────────────────────────────────────────────────────────────

/// All recoverable failure modes that can be returned through Result<T>.
enum class ErrorCode : uint16_t {
    Ok = 0,

    // Configuration
    ConfigNotFound,
    ConfigParseError,
    InvalidParameter,

    // Ring buffer
    RingBufferFull,
    RingBufferEmpty,

    // Connectivity
    WebSocketError,
    ConnectionFailed,

    // Order management
    OrderRejected,

    // Risk controls
    PositionLimitExceeded,
    DailyLossExceeded,
    CircuitBreakerTripped,
    InsufficientMargin,
    InvalidOrderState,

    Unknown,
};

/// Human-readable label for an ErrorCode.
[[nodiscard]] constexpr std::string_view toString(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::Ok:                   return "Ok";
        case ErrorCode::ConfigNotFound:       return "ConfigNotFound";
        case ErrorCode::ConfigParseError:     return "ConfigParseError";
        case ErrorCode::InvalidParameter:     return "InvalidParameter";
        case ErrorCode::RingBufferFull:       return "RingBufferFull";
        case ErrorCode::RingBufferEmpty:      return "RingBufferEmpty";
        case ErrorCode::WebSocketError:       return "WebSocketError";
        case ErrorCode::ConnectionFailed:     return "ConnectionFailed";
        case ErrorCode::OrderRejected:        return "OrderRejected";
        case ErrorCode::PositionLimitExceeded:return "PositionLimitExceeded";
        case ErrorCode::DailyLossExceeded:    return "DailyLossExceeded";
        case ErrorCode::CircuitBreakerTripped:return "CircuitBreakerTripped";
        case ErrorCode::InsufficientMargin:   return "InsufficientMargin";
        case ErrorCode::InvalidOrderState:    return "InvalidOrderState";
        case ErrorCode::Unknown:              return "Unknown";
    }
    return "Unknown";
}

// ── Result<T> ─────────────────────────────────────────────────────────────

/// A discriminated union that holds either a value of type T or an ErrorCode.
/// The first alternative is always the success value; the second is the error.
template <typename T>
using Result = std::variant<T, ErrorCode>;

// ── Concept constraints ────────────────────────────────────────────────────
namespace detail {

template <typename>
struct is_result_impl : std::false_type {};

template <typename T>
struct is_result_impl<std::variant<T, ErrorCode>> : std::true_type {
    using value_type = T;
};

} // namespace detail

template <typename R>
concept ResultType = detail::is_result_impl<R>::value;

// ── Free-function helpers ──────────────────────────────────────────────────

/// Returns true when the Result holds the success value.
template <ResultType R>
[[nodiscard]] constexpr bool is_ok(const R& result) noexcept {
    return std::holds_alternative<typename detail::is_result_impl<R>::value_type>(result);
}

/// Extracts the success value.  Caller must ensure is_ok() == true.
template <ResultType R>
[[nodiscard]] constexpr decltype(auto) get_value(R& result) {
    return std::get<typename detail::is_result_impl<R>::value_type>(result);
}

/// Const overload of get_value.
template <ResultType R>
[[nodiscard]] constexpr decltype(auto) get_value(const R& result) {
    return std::get<typename detail::is_result_impl<R>::value_type>(result);
}

/// Extracts the ErrorCode.  Caller must ensure is_ok() == false.
template <ResultType R>
[[nodiscard]] constexpr ErrorCode get_error(const R& result) noexcept {
    return std::get<ErrorCode>(result);
}

} // namespace coe::common
