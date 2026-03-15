#include <gtest/gtest.h>
#include <coe/market_data/message_types.hpp>

#include <chrono>
#include <string>
#include <variant>

using namespace coe::md;
using namespace std::chrono_literals;

// ── Quote construction and field access ──────────────────────────────────────

TEST(Quote, FieldsRoundTrip) {
    const Quote q{
        .symbol   = "SPY",
        .bid      = 449.95,
        .ask      = 450.05,
        .bid_size = 500,
        .ask_size = 300,
        .ts       = std::chrono::nanoseconds{1'000'000'000LL},
    };

    EXPECT_EQ(q.symbol, "SPY");
    EXPECT_DOUBLE_EQ(q.bid, 449.95);
    EXPECT_DOUBLE_EQ(q.ask, 450.05);
    EXPECT_EQ(q.bid_size, 500);
    EXPECT_EQ(q.ask_size, 300);
    EXPECT_EQ(q.ts.count(), 1'000'000'000LL);
}

// ── Trade construction and field access ──────────────────────────────────────

TEST(Trade, FieldsRoundTrip) {
    const Trade t{
        .symbol = "AAPL",
        .price  = 182.50,
        .volume = 1000,
        .ts     = std::chrono::nanoseconds{9'999'999LL},
    };

    EXPECT_EQ(t.symbol, "AAPL");
    EXPECT_DOUBLE_EQ(t.price, 182.50);
    EXPECT_EQ(t.volume, 1000);
    EXPECT_EQ(t.ts.count(), 9'999'999LL);
}

// ── OptionsQuote construction and field access ────────────────────────────────

TEST(OptionsQuote, FieldsRoundTrip) {
    const OptionsQuote oq{
        .symbol     = "SPY240119C00450000",
        .underlying = "SPY",
        .type       = OptionType::Call,
        .strike     = 450.0,
        .dte        = 30,
        .bid        = 3.10,
        .ask        = 3.20,
        .delta      = 0.35,
        .gamma      = 0.04,
        .theta      = -0.05,
        .vega       = 0.12,
        .iv         = 0.22,
        .ts         = std::chrono::nanoseconds{0},
    };

    EXPECT_EQ(oq.symbol,     "SPY240119C00450000");
    EXPECT_EQ(oq.underlying, "SPY");
    EXPECT_EQ(oq.type,       OptionType::Call);
    EXPECT_DOUBLE_EQ(oq.strike, 450.0);
    EXPECT_EQ(oq.dte, 30);
    EXPECT_DOUBLE_EQ(oq.bid,   3.10);
    EXPECT_DOUBLE_EQ(oq.ask,   3.20);
    EXPECT_DOUBLE_EQ(oq.delta, 0.35);
    EXPECT_DOUBLE_EQ(oq.gamma, 0.04);
    EXPECT_DOUBLE_EQ(oq.theta, -0.05);
    EXPECT_DOUBLE_EQ(oq.vega,  0.12);
    EXPECT_DOUBLE_EQ(oq.iv,    0.22);
}

// ── MarketMessage variant: holds each type ────────────────────────────────────

TEST(MarketMessage, HoldsQuote) {
    MarketMessage msg = Quote{.symbol = "QQQ", .bid = 390.0, .ask = 390.10,
                               .bid_size = 100, .ask_size = 100,
                               .ts = std::chrono::nanoseconds{0}};
    EXPECT_TRUE(std::holds_alternative<Quote>(msg));
    EXPECT_FALSE(std::holds_alternative<Trade>(msg));
    EXPECT_FALSE(std::holds_alternative<OptionsQuote>(msg));
}

TEST(MarketMessage, HoldsTrade) {
    MarketMessage msg = Trade{.symbol = "TSLA", .price = 220.0,
                               .volume = 500, .ts = std::chrono::nanoseconds{0}};
    EXPECT_TRUE(std::holds_alternative<Trade>(msg));
    EXPECT_FALSE(std::holds_alternative<Quote>(msg));
    EXPECT_FALSE(std::holds_alternative<OptionsQuote>(msg));
}

TEST(MarketMessage, HoldsOptionsQuote) {
    MarketMessage msg = OptionsQuote{
        .symbol = "TSLA231215P00200000", .underlying = "TSLA",
        .type = OptionType::Put, .strike = 200.0, .dte = 15,
        .bid = 1.50, .ask = 1.60,
        .delta = -0.30, .gamma = 0.03, .theta = -0.04, .vega = 0.09,
        .iv = 0.45, .ts = std::chrono::nanoseconds{0}
    };
    EXPECT_TRUE(std::holds_alternative<OptionsQuote>(msg));
}

// ── std::visit dispatches to the correct type ─────────────────────────────────

TEST(MarketMessage, VisitExtractsQuoteSymbol) {
    MarketMessage msg = Quote{.symbol = "SPY", .bid = 1.0, .ask = 1.1,
                               .bid_size = 1, .ask_size = 1,
                               .ts = std::chrono::nanoseconds{0}};
    std::string visited_symbol;
    std::visit([&](const auto& m) { visited_symbol = m.symbol; }, msg);
    EXPECT_EQ(visited_symbol, "SPY");
}

TEST(MarketMessage, VisitExtractsTradePrice) {
    MarketMessage msg = Trade{.symbol = "AAPL", .price = 182.0,
                               .volume = 10, .ts = std::chrono::nanoseconds{0}};
    double visited_price = 0.0;
    std::visit([&](const auto& m) {
        if constexpr (std::is_same_v<std::decay_t<decltype(m)>, Trade>) {
            visited_price = m.price;
        }
    }, msg);
    EXPECT_DOUBLE_EQ(visited_price, 182.0);
}

TEST(MarketMessage, VisitExtractsOptionsQuoteDelta) {
    MarketMessage msg = OptionsQuote{
        .symbol = "X", .underlying = "X",
        .type = OptionType::Call, .strike = 100.0, .dte = 5,
        .bid = 1.0, .ask = 1.1,
        .delta = 0.28, .gamma = 0.01, .theta = -0.02, .vega = 0.05,
        .iv = 0.30, .ts = std::chrono::nanoseconds{0}
    };

    double visited_delta = 0.0;
    std::visit([&](const auto& m) {
        if constexpr (std::is_same_v<std::decay_t<decltype(m)>, OptionsQuote>) {
            visited_delta = m.delta;
        }
    }, msg);
    EXPECT_DOUBLE_EQ(visited_delta, 0.28);
}

// ── Variant reassignment ──────────────────────────────────────────────────────

TEST(MarketMessage, CanBeReassigned) {
    MarketMessage msg = Quote{.symbol = "A", .bid = 1.0, .ask = 1.1,
                               .bid_size = 1, .ask_size = 1,
                               .ts = std::chrono::nanoseconds{0}};
    EXPECT_TRUE(std::holds_alternative<Quote>(msg));

    msg = Trade{.symbol = "B", .price = 10.0, .volume = 1,
                .ts = std::chrono::nanoseconds{0}};
    EXPECT_TRUE(std::holds_alternative<Trade>(msg));
}
