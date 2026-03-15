#include <gtest/gtest.h>
#include <coe/risk/pnl_tracker.hpp>

using coe::risk::DailyPnLTracker;

// ── Initial state ─────────────────────────────────────────────────────────────

TEST(DailyPnLTracker, InitiallyZero) {
    DailyPnLTracker tracker;
    EXPECT_DOUBLE_EQ(tracker.daily_pnl(), 0.0);
}

TEST(DailyPnLTracker, InitialTradeCountIsZero) {
    DailyPnLTracker tracker;
    EXPECT_EQ(tracker.trade_count(), 0);
}

TEST(DailyPnLTracker, NoLimitBreachWhenZero) {
    DailyPnLTracker tracker;
    EXPECT_FALSE(tracker.is_limit_breached(-50.0));
}

// ── record_trade accumulation ────────────────────────────────────────────────

TEST(DailyPnLTracker, RecordTradesAccumulate) {
    DailyPnLTracker tracker;
    tracker.record_trade(10.0);
    tracker.record_trade(-5.0);
    tracker.record_trade(3.0);
    EXPECT_DOUBLE_EQ(tracker.daily_pnl(), 8.0);
}

TEST(DailyPnLTracker, RecordWinIncreasesTotal) {
    DailyPnLTracker tracker;
    tracker.record_trade(25.0);
    EXPECT_DOUBLE_EQ(tracker.daily_pnl(), 25.0);
}

TEST(DailyPnLTracker, RecordLossDecreasesTotal) {
    DailyPnLTracker tracker;
    tracker.record_trade(-15.0);
    EXPECT_DOUBLE_EQ(tracker.daily_pnl(), -15.0);
}

TEST(DailyPnLTracker, TradeCountIncrementsOnEachRecord) {
    DailyPnLTracker tracker;
    tracker.record_trade(5.0);
    EXPECT_EQ(tracker.trade_count(), 1);
    tracker.record_trade(-2.0);
    EXPECT_EQ(tracker.trade_count(), 2);
    tracker.record_trade(1.0);
    EXPECT_EQ(tracker.trade_count(), 3);
}

// ── Limit breach detection ────────────────────────────────────────────────────

TEST(DailyPnLTracker, LimitBreach) {
    DailyPnLTracker tracker;
    tracker.record_trade(-30.0);
    EXPECT_FALSE(tracker.is_limit_breached(-50.0));

    tracker.record_trade(-25.0);
    EXPECT_TRUE(tracker.is_limit_breached(-50.0)); // -55 <= -50
}

TEST(DailyPnLTracker, ExactlyAtLimitIsBreach) {
    DailyPnLTracker tracker;
    // daily_pnl == limit: is_limit_breached uses <=, so exact match counts.
    tracker.record_trade(-50.0);
    EXPECT_TRUE(tracker.is_limit_breached(-50.0));
}

TEST(DailyPnLTracker, JustAboveLimitIsNotBreached) {
    DailyPnLTracker tracker;
    tracker.record_trade(-49.99);
    EXPECT_FALSE(tracker.is_limit_breached(-50.0));
}

TEST(DailyPnLTracker, MultipleLossesBreachLimit) {
    DailyPnLTracker tracker;
    for (int i = 0; i < 6; ++i) {
        tracker.record_trade(-10.0); // -60 after 6 trades
    }
    EXPECT_TRUE(tracker.is_limit_breached(-50.0));
}

TEST(DailyPnLTracker, WinsDoNotBreach) {
    DailyPnLTracker tracker;
    tracker.record_trade(100.0);
    EXPECT_FALSE(tracker.is_limit_breached(-50.0));
}

// ── Reset ─────────────────────────────────────────────────────────────────────

TEST(DailyPnLTracker, Reset) {
    DailyPnLTracker tracker;
    tracker.record_trade(-100.0);
    tracker.reset();
    EXPECT_DOUBLE_EQ(tracker.daily_pnl(), 0.0);
    EXPECT_FALSE(tracker.is_limit_breached(-50.0));
}

TEST(DailyPnLTracker, ResetClearsTradeCount) {
    DailyPnLTracker tracker;
    tracker.record_trade(10.0);
    tracker.record_trade(20.0);
    EXPECT_EQ(tracker.trade_count(), 2);
    tracker.reset();
    EXPECT_EQ(tracker.trade_count(), 0);
}

TEST(DailyPnLTracker, CanAccumulateAgainAfterReset) {
    DailyPnLTracker tracker;
    tracker.record_trade(-100.0);
    tracker.reset();
    tracker.record_trade(5.0);
    EXPECT_DOUBLE_EQ(tracker.daily_pnl(), 5.0);
    EXPECT_EQ(tracker.trade_count(), 1);
}
