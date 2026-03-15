#include <gtest/gtest.h>
#include <coe/common/types.hpp>

#include <chrono>
#include <string_view>

using namespace coe::common;

// ── Side enum ────────────────────────────────────────────────────────────────

TEST(SideEnum, DistinctValues) {
    EXPECT_NE(static_cast<uint8_t>(Side::Buy), static_cast<uint8_t>(Side::Sell));
}

TEST(SideEnum, ToStringBuy) {
    EXPECT_EQ(toString(Side::Buy), "Buy");
}

TEST(SideEnum, ToStringSell) {
    EXPECT_EQ(toString(Side::Sell), "Sell");
}

TEST(SideEnum, ToStringIsConstexpr) {
    // Verify constexpr evaluation at compile time.
    static_assert(toString(Side::Buy)  == std::string_view{"Buy"});
    static_assert(toString(Side::Sell) == std::string_view{"Sell"});
}

// ── OptionType enum ───────────────────────────────────────────────────────────

TEST(OptionTypeEnum, DistinctValues) {
    EXPECT_NE(static_cast<uint8_t>(OptionType::Call), static_cast<uint8_t>(OptionType::Put));
}

TEST(OptionTypeEnum, ToStringCall) {
    EXPECT_EQ(toString(OptionType::Call), "Call");
}

TEST(OptionTypeEnum, ToStringPut) {
    EXPECT_EQ(toString(OptionType::Put), "Put");
}

TEST(OptionTypeEnum, ToStringIsConstexpr) {
    static_assert(toString(OptionType::Call) == std::string_view{"Call"});
    static_assert(toString(OptionType::Put)  == std::string_view{"Put"});
}

// ── OptionContract construction ───────────────────────────────────────────────

TEST(OptionContract, AggregateInitialisation) {
    const Timestamp expiry = std::chrono::nanoseconds{1700000000LL * 1000000000LL};
    const OptionContract contract{
        .underlying = "SPY",
        .type       = OptionType::Call,
        .strike     = 450.0,
        .expiry     = expiry,
        .dte        = 30,
    };

    EXPECT_EQ(contract.underlying, "SPY");
    EXPECT_EQ(contract.type,       OptionType::Call);
    EXPECT_DOUBLE_EQ(contract.strike, 450.0);
    EXPECT_EQ(contract.expiry, expiry);
    EXPECT_EQ(contract.dte, 30);
}

TEST(OptionContract, PutContract) {
    const OptionContract put{
        .underlying = "AAPL",
        .type       = OptionType::Put,
        .strike     = 175.0,
        .expiry     = std::chrono::nanoseconds{0},
        .dte        = 7,
    };

    EXPECT_EQ(put.type,  OptionType::Put);
    EXPECT_EQ(toString(put.type), "Put");
    EXPECT_DOUBLE_EQ(put.strike, 175.0);
}

TEST(OptionContract, ZeroDteMeansExpired) {
    const OptionContract contract{
        .underlying = "QQQ",
        .type       = OptionType::Call,
        .strike     = 400.0,
        .expiry     = std::chrono::nanoseconds{0},
        .dte        = 0,
    };
    EXPECT_EQ(contract.dte, 0);
}

TEST(OptionContract, LargeStrikePrice) {
    const OptionContract contract{
        .underlying = "AMZN",
        .type       = OptionType::Call,
        .strike     = 3500.0,
        .expiry     = std::chrono::nanoseconds{0},
        .dte        = 60,
    };
    EXPECT_DOUBLE_EQ(contract.strike, 3500.0);
}

// ── Type alias sanity ─────────────────────────────────────────────────────────

TEST(TypeAliases, PriceIsDouble) {
    Price p = 3.14;
    EXPECT_DOUBLE_EQ(p, 3.14);
}

TEST(TypeAliases, QuantityIsInt32) {
    Quantity q = -5;
    EXPECT_EQ(q, -5);
}

TEST(TypeAliases, TimestampIsNanoseconds) {
    Timestamp ts = std::chrono::nanoseconds{123456789LL};
    EXPECT_EQ(ts.count(), 123456789LL);
}
