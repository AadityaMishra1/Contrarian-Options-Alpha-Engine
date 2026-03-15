#include <gtest/gtest.h>
#include <coe/execution/position.hpp>

#include <cmath>
#include <optional>

using namespace coe::execution;

// ── No position before any fill ───────────────────────────────────────────────

TEST(PositionTracker, NoPositionBeforeAnyFill) {
    PositionTracker tracker;
    EXPECT_FALSE(tracker.get_position("SPY").has_value());
    EXPECT_EQ(tracker.open_position_count(), 0);
}

// ── on_fill creates a position ────────────────────────────────────────────────

TEST(PositionTracker, OpenNewPosition) {
    PositionTracker tracker;
    tracker.on_fill("AAPL", Side::Buy, 5, 1.50);

    auto pos = tracker.get_position("AAPL");
    ASSERT_TRUE(pos.has_value());
    EXPECT_EQ(pos->symbol,   "AAPL");
    EXPECT_EQ(pos->quantity, 5);
    EXPECT_DOUBLE_EQ(pos->avg_entry, 1.50);
    EXPECT_EQ(pos->side, Side::Buy);
}

TEST(PositionTracker, OnFillIncreasesOpenPositionCount) {
    PositionTracker tracker;
    tracker.on_fill("SPY", Side::Buy, 1, 450.0);
    EXPECT_EQ(tracker.open_position_count(), 1);
}

// ── Multiple fills average entry price (VWAP) ────────────────────────────────

TEST(PositionTracker, AddToPosition) {
    PositionTracker tracker;
    tracker.on_fill("AAPL", Side::Buy, 5, 1.00);
    tracker.on_fill("AAPL", Side::Buy, 5, 2.00);

    auto pos = tracker.get_position("AAPL");
    ASSERT_TRUE(pos.has_value());
    EXPECT_EQ(pos->quantity, 10);
    EXPECT_DOUBLE_EQ(pos->avg_entry, 1.50);
}

TEST(PositionTracker, MultipleFillsVwapAvgEntry) {
    PositionTracker tracker;
    // 2 @ 100, 2 @ 200 → avg = (200+400)/4 = 150.
    tracker.on_fill("SPY", Side::Buy, 2, 100.0);
    tracker.on_fill("SPY", Side::Buy, 2, 200.0);

    auto pos = tracker.get_position("SPY");
    ASSERT_TRUE(pos.has_value());
    EXPECT_EQ(pos->quantity, 4);
    EXPECT_NEAR(pos->avg_entry, 150.0, 1e-9);
}

TEST(PositionTracker, ThreeFillsVwapAvgEntry) {
    PositionTracker tracker;
    // 1@10, 2@20, 3@30 → total=140, qty=6 → avg≈23.333
    tracker.on_fill("AAPL", Side::Buy, 1, 10.0);
    tracker.on_fill("AAPL", Side::Buy, 2, 20.0);
    tracker.on_fill("AAPL", Side::Buy, 3, 30.0);

    auto pos = tracker.get_position("AAPL");
    ASSERT_TRUE(pos.has_value());
    EXPECT_EQ(pos->quantity, 6);
    EXPECT_NEAR(pos->avg_entry, 140.0 / 6.0, 1e-9);
}

// ── update_mark updates unrealized P&L ───────────────────────────────────────

TEST(PositionTracker, UpdateMark) {
    PositionTracker tracker;
    tracker.on_fill("AAPL", Side::Buy, 1, 1.00);
    tracker.update_mark("AAPL", 2.00);

    auto pos = tracker.get_position("AAPL");
    ASSERT_TRUE(pos.has_value());
    EXPECT_DOUBLE_EQ(pos->current_price, 2.00);
    // unrealized = (2.0 - 1.0) * 1 * 100 = 100
    EXPECT_DOUBLE_EQ(pos->unrealized_pnl, 100.0);
}

TEST(PositionTracker, UpdateMarkUpdatesUnrealizedPnl) {
    PositionTracker tracker;
    // Buy 2 contracts @ 3.00.  Mark at 4.00 → unrealized = (4-3)*2*100 = $200.
    tracker.on_fill("SPY", Side::Buy, 2, 3.0);
    tracker.update_mark("SPY", 4.0);

    auto pos = tracker.get_position("SPY");
    ASSERT_TRUE(pos.has_value());
    EXPECT_NEAR(pos->unrealized_pnl, 200.0, 1e-9);
}

TEST(PositionTracker, UpdateMarkNegativePnlWhenPriceFalls) {
    PositionTracker tracker;
    // Buy 1 contract @ 5.00.  Mark at 4.00 → unrealized = (4-5)*1*100 = -$100.
    tracker.on_fill("SPY", Side::Buy, 1, 5.0);
    tracker.update_mark("SPY", 4.0);

    auto pos = tracker.get_position("SPY");
    ASSERT_TRUE(pos.has_value());
    EXPECT_NEAR(pos->unrealized_pnl, -100.0, 1e-9);
}

TEST(PositionTracker, UpdateMarkNoOpForUnknownSymbol) {
    PositionTracker tracker;
    EXPECT_NO_THROW(tracker.update_mark("UNKNOWN", 100.0));
    EXPECT_FALSE(tracker.get_position("UNKNOWN").has_value());
}

TEST(PositionTracker, TotalUnrealizedPnlAcrossPositions) {
    PositionTracker tracker;
    tracker.on_fill("SPY",  Side::Buy, 1, 10.0);
    tracker.on_fill("AAPL", Side::Buy, 1, 20.0);
    tracker.update_mark("SPY",  12.0);   // unrealized = +200
    tracker.update_mark("AAPL", 18.0);   // unrealized = -200
    EXPECT_NEAR(tracker.total_unrealized_pnl(), 0.0, 1e-9);
}

// ── Closing a position moves to realized P&L ─────────────────────────────────

TEST(PositionTracker, ClosePosition) {
    PositionTracker tracker;
    tracker.on_fill("AAPL", Side::Buy,  5, 1.00);
    tracker.on_fill("AAPL", Side::Sell, 5, 2.00);

    auto pos = tracker.get_position("AAPL");
    ASSERT_TRUE(pos.has_value());
    EXPECT_EQ(pos->quantity, 0);
    // realized = (2.0 - 1.0) * 5 * 100 = 500
    EXPECT_DOUBLE_EQ(pos->realized_pnl, 500.0);
}

TEST(PositionTracker, ClosingPositionRealisesLoss) {
    PositionTracker tracker;
    // Buy 1 @ 5.0, sell 1 @ 3.0 → realized = (3-5)*1*100 = -$200.
    tracker.on_fill("SPY", Side::Buy,  1, 5.0);
    tracker.on_fill("SPY", Side::Sell, 1, 3.0);

    auto pos = tracker.get_position("SPY");
    ASSERT_TRUE(pos.has_value());
    EXPECT_NEAR(pos->realized_pnl, -200.0, 1e-9);
}

TEST(PositionTracker, PartialCloseAccumulatesRealizedPnl) {
    PositionTracker tracker;
    // Buy 4 @ 10.0, sell 2 @ 12.0 → realized = (12-10)*2*100 = $400.  2 still open.
    tracker.on_fill("SPY", Side::Buy,  4, 10.0);
    tracker.on_fill("SPY", Side::Sell, 2, 12.0);

    auto pos = tracker.get_position("SPY");
    ASSERT_TRUE(pos.has_value());
    EXPECT_EQ(pos->quantity, 2);
    EXPECT_NEAR(pos->realized_pnl, 400.0, 1e-9);
}

// ── open_position_count ───────────────────────────────────────────────────────

TEST(PositionTracker, OpenPositionCount) {
    PositionTracker tracker;
    tracker.on_fill("AAPL", Side::Buy, 1, 1.00);
    tracker.on_fill("MSFT", Side::Buy, 1, 2.00);
    EXPECT_EQ(tracker.open_position_count(), 2);

    tracker.on_fill("AAPL", Side::Sell, 1, 1.50);
    EXPECT_EQ(tracker.open_position_count(), 1);
}

// ── total_realized_pnl ────────────────────────────────────────────────────────

TEST(PositionTracker, TotalRealizedPnlAcrossMultipleSymbols) {
    PositionTracker tracker;
    tracker.on_fill("SPY",  Side::Buy,  1, 10.0);
    tracker.on_fill("AAPL", Side::Buy,  1, 20.0);
    tracker.on_fill("SPY",  Side::Sell, 1, 12.0);   // realized +200
    tracker.on_fill("AAPL", Side::Sell, 1, 18.0);   // realized -200
    EXPECT_NEAR(tracker.total_realized_pnl(), 0.0, 1e-9);
}

// ── get_all_positions includes flat (closed) positions ───────────────────────

TEST(PositionTracker, GetAllPositionsIncludesFlatPositions) {
    PositionTracker tracker;
    tracker.on_fill("SPY", Side::Buy,  1, 10.0);
    tracker.on_fill("SPY", Side::Sell, 1, 11.0); // flat now
    auto all = tracker.get_all_positions();
    EXPECT_EQ(all.size(), 1u);
    EXPECT_EQ(all[0].symbol, "SPY");
    EXPECT_EQ(all[0].quantity, 0);
}
