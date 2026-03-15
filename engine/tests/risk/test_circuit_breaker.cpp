#include <gtest/gtest.h>
#include <coe/risk/circuit_breaker.hpp>

using coe::risk::CircuitBreaker;

// ── Initial state ─────────────────────────────────────────────────────────────

TEST(CircuitBreaker, NotTrippedOnConstruction) {
    CircuitBreaker cb{20, 0.40};
    EXPECT_FALSE(cb.is_tripped());
}

TEST(CircuitBreaker, WindowFillIsZeroOnConstruction) {
    CircuitBreaker cb{20, 0.40};
    EXPECT_EQ(cb.window_fill(), 0);
}

TEST(CircuitBreaker, WinRateIsOneWhenWindowEmpty) {
    // Empty window → conservative: win rate = 1.0 (not tripped).
    CircuitBreaker cb{20, 0.40};
    EXPECT_DOUBLE_EQ(cb.current_win_rate(), 1.0);
}

// ── Not tripped with all wins ─────────────────────────────────────────────────

TEST(CircuitBreaker, NotTrippedWithAllWins) {
    CircuitBreaker cb{20, 0.40};
    for (int i = 0; i < 20; ++i) cb.record_trade(true);
    EXPECT_FALSE(cb.is_tripped());
    EXPECT_DOUBLE_EQ(cb.current_win_rate(), 1.0);
}

// ── Tripped with all losses ───────────────────────────────────────────────────

TEST(CircuitBreaker, TrippedWithAllLosses) {
    CircuitBreaker cb{20, 0.40};
    for (int i = 0; i < 20; ++i) cb.record_trade(false);
    EXPECT_TRUE(cb.is_tripped());
    EXPECT_DOUBLE_EQ(cb.current_win_rate(), 0.0);
}

// ── Exactly at threshold → NOT tripped ───────────────────────────────────────

TEST(CircuitBreaker, ExactlyAtThreshold) {
    CircuitBreaker cb{20, 0.40};
    // 8 wins out of 20 = 0.40 — not strictly less than 0.40, so NOT tripped.
    for (int i = 0; i < 8;  ++i) cb.record_trade(true);
    for (int i = 0; i < 12; ++i) cb.record_trade(false);
    EXPECT_FALSE(cb.is_tripped());
    EXPECT_NEAR(cb.current_win_rate(), 0.40, 1e-9);
}

// ── Just below threshold → tripped ───────────────────────────────────────────

TEST(CircuitBreaker, JustBelowThreshold) {
    CircuitBreaker cb{20, 0.40};
    // 7 wins out of 20 = 0.35 < 0.40 → tripped.
    for (int i = 0; i < 7;  ++i) cb.record_trade(true);
    for (int i = 0; i < 13; ++i) cb.record_trade(false);
    EXPECT_TRUE(cb.is_tripped());
    EXPECT_NEAR(cb.current_win_rate(), 7.0 / 20.0, 1e-9);
}

// ── Does not trip before window is full ──────────────────────────────────────

TEST(CircuitBreaker, NotTrippedBeforeWindowFull) {
    CircuitBreaker cb{20, 0.40};
    // Feed 19 losses — window not full yet, so never trips.
    for (int i = 0; i < 19; ++i) cb.record_trade(false);
    EXPECT_FALSE(cb.is_tripped());
    EXPECT_EQ(cb.window_fill(), 19);
}

TEST(CircuitBreaker, TripsOnExactlyWindowSizeTrades) {
    CircuitBreaker cb{20, 0.40};
    // 20th trade makes window full and triggers evaluation.
    for (int i = 0; i < 19; ++i) cb.record_trade(false);
    EXPECT_FALSE(cb.is_tripped()); // still 19
    cb.record_trade(false);        // 20th loss
    EXPECT_TRUE(cb.is_tripped());
}

// ── Rolling window evicts old entries ────────────────────────────────────────

TEST(CircuitBreaker, OldTradesEvictedFromWindow) {
    CircuitBreaker cb{5, 0.40};
    // Fill with 5 losses → tripped.
    for (int i = 0; i < 5; ++i) cb.record_trade(false);
    ASSERT_TRUE(cb.is_tripped());

    // Feed 5 wins — they evict all losses → win rate = 1.0 → no longer tripped.
    for (int i = 0; i < 5; ++i) cb.record_trade(true);
    EXPECT_FALSE(cb.is_tripped());
    EXPECT_DOUBLE_EQ(cb.current_win_rate(), 1.0);
}

// ── window_fill respects capacity ────────────────────────────────────────────

TEST(CircuitBreaker, WindowFillCapsAtWindowSize) {
    CircuitBreaker cb{5, 0.40};
    for (int i = 0; i < 10; ++i) cb.record_trade(true);
    EXPECT_EQ(cb.window_fill(), 5);
}

// ── Reset ─────────────────────────────────────────────────────────────────────

TEST(CircuitBreaker, Reset) {
    CircuitBreaker cb{20, 0.40};
    for (int i = 0; i < 20; ++i) cb.record_trade(false);
    ASSERT_TRUE(cb.is_tripped());

    cb.reset();

    EXPECT_FALSE(cb.is_tripped());
    EXPECT_EQ(cb.window_fill(), 0);
    EXPECT_DOUBLE_EQ(cb.current_win_rate(), 1.0);
}

TEST(CircuitBreaker, CanAccumulateAfterReset) {
    CircuitBreaker cb{3, 0.40};
    for (int i = 0; i < 3; ++i) cb.record_trade(false);
    ASSERT_TRUE(cb.is_tripped());

    cb.reset();

    // Now fill with all wins — should not trip.
    for (int i = 0; i < 3; ++i) cb.record_trade(true);
    EXPECT_FALSE(cb.is_tripped());
    EXPECT_DOUBLE_EQ(cb.current_win_rate(), 1.0);
}
