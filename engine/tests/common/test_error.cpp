#include <gtest/gtest.h>
#include <coe/common/error.hpp>

#include <string_view>
#include <variant>

using namespace coe::common;

// ── ErrorCode enum values ─────────────────────────────────────────────────────

TEST(ErrorCode, OkIsZero) {
    EXPECT_EQ(static_cast<uint16_t>(ErrorCode::Ok), 0u);
}

TEST(ErrorCode, DistinctCodes) {
    // Verify a sample of codes are unique.
    EXPECT_NE(ErrorCode::Ok,                   ErrorCode::ConfigNotFound);
    EXPECT_NE(ErrorCode::ConfigNotFound,        ErrorCode::ConfigParseError);
    EXPECT_NE(ErrorCode::RingBufferFull,        ErrorCode::RingBufferEmpty);
    EXPECT_NE(ErrorCode::DailyLossExceeded,     ErrorCode::CircuitBreakerTripped);
    EXPECT_NE(ErrorCode::InvalidOrderState,     ErrorCode::OrderRejected);
    EXPECT_NE(ErrorCode::PositionLimitExceeded, ErrorCode::InsufficientMargin);
}

// ── toString for ErrorCode ────────────────────────────────────────────────────

TEST(ErrorCode, ToStringOk) {
    EXPECT_EQ(toString(ErrorCode::Ok), "Ok");
}

TEST(ErrorCode, ToStringConfigNotFound) {
    EXPECT_EQ(toString(ErrorCode::ConfigNotFound), "ConfigNotFound");
}

TEST(ErrorCode, ToStringConfigParseError) {
    EXPECT_EQ(toString(ErrorCode::ConfigParseError), "ConfigParseError");
}

TEST(ErrorCode, ToStringInvalidParameter) {
    EXPECT_EQ(toString(ErrorCode::InvalidParameter), "InvalidParameter");
}

TEST(ErrorCode, ToStringRingBufferFull) {
    EXPECT_EQ(toString(ErrorCode::RingBufferFull), "RingBufferFull");
}

TEST(ErrorCode, ToStringRingBufferEmpty) {
    EXPECT_EQ(toString(ErrorCode::RingBufferEmpty), "RingBufferEmpty");
}

TEST(ErrorCode, ToStringOrderRejected) {
    EXPECT_EQ(toString(ErrorCode::OrderRejected), "OrderRejected");
}

TEST(ErrorCode, ToStringPositionLimitExceeded) {
    EXPECT_EQ(toString(ErrorCode::PositionLimitExceeded), "PositionLimitExceeded");
}

TEST(ErrorCode, ToStringDailyLossExceeded) {
    EXPECT_EQ(toString(ErrorCode::DailyLossExceeded), "DailyLossExceeded");
}

TEST(ErrorCode, ToStringCircuitBreakerTripped) {
    EXPECT_EQ(toString(ErrorCode::CircuitBreakerTripped), "CircuitBreakerTripped");
}

TEST(ErrorCode, ToStringInsufficientMargin) {
    EXPECT_EQ(toString(ErrorCode::InsufficientMargin), "InsufficientMargin");
}

TEST(ErrorCode, ToStringInvalidOrderState) {
    EXPECT_EQ(toString(ErrorCode::InvalidOrderState), "InvalidOrderState");
}

TEST(ErrorCode, ToStringUnknown) {
    EXPECT_EQ(toString(ErrorCode::Unknown), "Unknown");
}

// ── Result<T> success path ────────────────────────────────────────────────────

TEST(Result, IsOkWhenHoldingValue) {
    Result<int> r{42};
    EXPECT_TRUE(is_ok(r));
}

TEST(Result, GetValueReturnsCorrectValue) {
    Result<int> r{42};
    EXPECT_EQ(get_value(r), 42);
}

TEST(Result, IsOkFalseWhenHoldingError) {
    Result<int> r{ErrorCode::ConfigNotFound};
    EXPECT_FALSE(is_ok(r));
}

TEST(Result, GetErrorReturnsErrorCode) {
    Result<int> r{ErrorCode::RingBufferFull};
    EXPECT_EQ(get_error(r), ErrorCode::RingBufferFull);
}

TEST(Result, DoubleSuccess) {
    Result<double> r{3.14};
    EXPECT_TRUE(is_ok(r));
    EXPECT_DOUBLE_EQ(get_value(r), 3.14);
}

TEST(Result, StringSuccess) {
    Result<std::string> r{std::string{"hello"}};
    EXPECT_TRUE(is_ok(r));
    EXPECT_EQ(get_value(r), "hello");
}

TEST(Result, ErrorCodePreservedForAllCodes) {
    const std::initializer_list<ErrorCode> codes = {
        ErrorCode::ConfigNotFound,
        ErrorCode::ConfigParseError,
        ErrorCode::InvalidParameter,
        ErrorCode::RingBufferFull,
        ErrorCode::RingBufferEmpty,
        ErrorCode::WebSocketError,
        ErrorCode::ConnectionFailed,
        ErrorCode::OrderRejected,
        ErrorCode::PositionLimitExceeded,
        ErrorCode::DailyLossExceeded,
        ErrorCode::CircuitBreakerTripped,
        ErrorCode::InsufficientMargin,
        ErrorCode::InvalidOrderState,
        ErrorCode::Unknown,
    };

    for (const auto code : codes) {
        Result<int> r{code};
        EXPECT_FALSE(is_ok(r));
        EXPECT_EQ(get_error(r), code);
    }
}

TEST(Result, MutateValueInPlace) {
    Result<int> r{10};
    EXPECT_TRUE(is_ok(r));
    get_value(r) = 99;
    EXPECT_EQ(get_value(r), 99);
}

// ── ResultType concept ────────────────────────────────────────────────────────

TEST(ResultType, IntResultSatisfiesConcept) {
    static_assert(ResultType<Result<int>>);
}

TEST(ResultType, DoubleResultSatisfiesConcept) {
    static_assert(ResultType<Result<double>>);
}

TEST(ResultType, StringResultSatisfiesConcept) {
    static_assert(ResultType<Result<std::string>>);
}
