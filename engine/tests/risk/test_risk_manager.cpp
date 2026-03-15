#include <gtest/gtest.h>
#include <coe/risk/risk_manager.hpp>
#include <coe/execution/order.hpp>

using namespace coe::risk;
using namespace coe::execution;
using coe::common::ErrorCode;
using coe::common::is_ok;
using coe::common::get_error;

// ── Fixture ───────────────────────────────────────────────────────────────────

class RiskManagerTest : public ::testing::Test {
protected:
    RiskLimits limits{
        .daily_loss_limit      = -50.0,
        .max_positions         = 5,
        .max_single_position   = 20.0,
        .circuit_breaker_window = 20,
        .min_win_rate          = 0.40,
    };
    RiskManager rm{limits};

    Order make_order(const std::string& sym = "AAPL") const {
        Order o;
        o.symbol   = sym;
        o.side     = Side::Buy;
        o.quantity = 1;
        return o;
    }
};

// ── check_new_order: all clear ────────────────────────────────────────────────

TEST_F(RiskManagerTest, AllLimitsOkPassesThrough) {
    auto result = rm.check_new_order(make_order(), 0, 10.0);
    EXPECT_TRUE(is_ok(result));
}

TEST_F(RiskManagerTest, PassesWithPositionsBelowMax) {
    // 4 open positions with max=5 → still allowed.
    auto result = rm.check_new_order(make_order(), 4, 10.0);
    EXPECT_TRUE(is_ok(result));
}

// ── Daily loss limit ──────────────────────────────────────────────────────────

TEST_F(RiskManagerTest, DailyLossExceededBlocksOrder) {
    rm.on_trade_closed(-60.0, false);
    auto result = rm.check_new_order(make_order(), 0, 10.0);
    ASSERT_FALSE(is_ok(result));
    EXPECT_EQ(get_error(result), ErrorCode::DailyLossExceeded);
}

TEST_F(RiskManagerTest, DailyLossExactlyAtLimitIsBlocked) {
    // Limit = -50. daily_pnl = -50 <= -50 → blocked.
    rm.on_trade_closed(-50.0, false);
    auto result = rm.check_new_order(make_order(), 0, 10.0);
    ASSERT_FALSE(is_ok(result));
    EXPECT_EQ(get_error(result), ErrorCode::DailyLossExceeded);
}

TEST_F(RiskManagerTest, DailyLossJustAboveLimitAllowed) {
    rm.on_trade_closed(-49.0, false);
    auto result = rm.check_new_order(make_order(), 0, 10.0);
    EXPECT_TRUE(is_ok(result));
}

// ── Position limit ────────────────────────────────────────────────────────────

TEST_F(RiskManagerTest, PositionLimitExceededAtMax) {
    // current_positions == max_positions (5) → blocked.
    auto result = rm.check_new_order(make_order(), 5, 10.0);
    ASSERT_FALSE(is_ok(result));
    EXPECT_EQ(get_error(result), ErrorCode::PositionLimitExceeded);
}

TEST_F(RiskManagerTest, PositionLimitExceededAboveMax) {
    auto result = rm.check_new_order(make_order(), 10, 10.0);
    ASSERT_FALSE(is_ok(result));
    EXPECT_EQ(get_error(result), ErrorCode::PositionLimitExceeded);
}

// ── Single-position notional cap ─────────────────────────────────────────────

TEST_F(RiskManagerTest, InsufficientMarginWhenOrderValueExceedsCap) {
    // max_single_position = 20.0; order_value = 25.0 > 20.0 → blocked.
    auto result = rm.check_new_order(make_order(), 0, 25.0);
    ASSERT_FALSE(is_ok(result));
    EXPECT_EQ(get_error(result), ErrorCode::InsufficientMargin);
}

TEST_F(RiskManagerTest, InsufficientMarginExactlyAtCapPasses) {
    // order_value == max_single_position: check is >, so 20.0 is allowed.
    auto result = rm.check_new_order(make_order(), 0, 20.0);
    EXPECT_TRUE(is_ok(result));
}

// ── Circuit breaker ───────────────────────────────────────────────────────────

TEST_F(RiskManagerTest, CircuitBreakerTrippedBlocksOrder) {
    // 20 consecutive losses → win rate = 0 < 0.40 → tripped.
    for (int i = 0; i < 20; ++i) rm.on_trade_closed(1.0, false);
    auto result = rm.check_new_order(make_order(), 0, 10.0);
    ASSERT_FALSE(is_ok(result));
    EXPECT_EQ(get_error(result), ErrorCode::CircuitBreakerTripped);
}

TEST_F(RiskManagerTest, CircuitBreakerNotTrippedWithSufficientWins) {
    // 8 wins + 12 losses out of 20 = 0.40 win rate — exactly at threshold.
    for (int i = 0; i < 8;  ++i) rm.on_trade_closed(1.0, true);
    for (int i = 0; i < 12; ++i) rm.on_trade_closed(-1.0, false);
    auto result = rm.check_new_order(make_order(), 0, 10.0);
    EXPECT_TRUE(is_ok(result));
}

// ── Check order priority: DailyLoss checked before PositionLimit ─────────────

TEST_F(RiskManagerTest, DailyLossCheckedBeforePositionLimit) {
    // Both limits breached — DailyLoss is checked first per the documented order.
    rm.on_trade_closed(-60.0, false);
    auto result = rm.check_new_order(make_order(), 10, 30.0);
    ASSERT_FALSE(is_ok(result));
    EXPECT_EQ(get_error(result), ErrorCode::DailyLossExceeded);
}

// ── on_trade_closed updates both PnL tracker and circuit breaker ─────────────

TEST_F(RiskManagerTest, OnTradeClosedUpdatesDailyPnl) {
    rm.on_trade_closed(15.0, true);
    EXPECT_DOUBLE_EQ(rm.daily_pnl(), 15.0);
}

TEST_F(RiskManagerTest, OnTradeClosedUpdatesWinRate) {
    // Feed 10 wins → window not full → conservative win rate = 1.0.
    for (int i = 0; i < 10; ++i) rm.on_trade_closed(1.0, true);
    EXPECT_DOUBLE_EQ(rm.win_rate(), 1.0);
}

TEST_F(RiskManagerTest, OnTradeClosedLossUpdatesDailyPnl) {
    rm.on_trade_closed(-20.0, false);
    EXPECT_DOUBLE_EQ(rm.daily_pnl(), -20.0);
}

// ── reset_daily resets PnL but preserves circuit breaker ─────────────────────

TEST_F(RiskManagerTest, ResetDailyClearsPnl) {
    rm.on_trade_closed(-60.0, false);
    rm.reset_daily();
    EXPECT_DOUBLE_EQ(rm.daily_pnl(), 0.0);
}

TEST_F(RiskManagerTest, ResetDailyAllowsNewOrdersAfterLossLimit) {
    rm.on_trade_closed(-60.0, false);
    rm.reset_daily();
    auto result = rm.check_new_order(make_order(), 0, 10.0);
    EXPECT_TRUE(is_ok(result));
}

TEST_F(RiskManagerTest, ResetDailyPreservesCircuitBreakerState) {
    // Trip the circuit breaker.
    for (int i = 0; i < 20; ++i) rm.on_trade_closed(0.0, false);
    ASSERT_TRUE(rm.is_circuit_breaker_tripped());

    // Reset daily — breaker must survive.
    rm.reset_daily();
    EXPECT_TRUE(rm.is_circuit_breaker_tripped());
}

// ── Observability ─────────────────────────────────────────────────────────────

TEST_F(RiskManagerTest, LimitsAccessorMatchesConstruction) {
    EXPECT_DOUBLE_EQ(rm.limits().daily_loss_limit, -50.0);
    EXPECT_EQ(rm.limits().max_positions, 5);
    EXPECT_DOUBLE_EQ(rm.limits().max_single_position, 20.0);
}

TEST_F(RiskManagerTest, IsCircuitBreakerTrippedFalseInitially) {
    EXPECT_FALSE(rm.is_circuit_breaker_tripped());
}
