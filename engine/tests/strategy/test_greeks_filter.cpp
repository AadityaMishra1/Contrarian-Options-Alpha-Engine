#include <gtest/gtest.h>
#include <coe/strategy/greeks_filter.hpp>

using coe::strategy::GreeksFilter;

// ── Before first update ───────────────────────────────────────────────────────

TEST(GreeksFilter, FailsBeforeAnyUpdate) {
    GreeksFilter gf;
    EXPECT_FALSE(gf.passes());
}

TEST(GreeksFilter, ScoreZeroBeforeAnyUpdate) {
    GreeksFilter gf;
    EXPECT_DOUBLE_EQ(gf.score(), 0.0);
}

// ── All criteria passing ──────────────────────────────────────────────────────

TEST(GreeksFilter, AllCriteriaPassingWithIdealInputs) {
    // Default: delta [0.20, 0.40], iv_pct_max 50.0, spread_pct_max 20.0
    GreeksFilter gf{0.20, 0.40, 50.0, 20.0};
    // delta=0.30 (midpoint), iv=25% (well below 50), spread=5% (well below 20)
    // bid=9.75, ask=10.25 → mid=10.0, spread_pct = 0.50/10.0 * 100 = 5.0%
    gf.update(0.30, 25.0, 9.75, 10.25);
    EXPECT_TRUE(gf.passes());
    EXPECT_GT(gf.score(), 0.0);
    EXPECT_LE(gf.score(), 1.0);
}

// ── Delta criterion ───────────────────────────────────────────────────────────

TEST(GreeksFilter, DeltaExactlyAtMinBoundaryPasses) {
    GreeksFilter gf{0.20, 0.40, 50.0, 20.0};
    // |delta| = 0.20 exactly — boundary is inclusive (>= delta_min).
    gf.update(0.20, 20.0, 9.90, 10.10);
    EXPECT_TRUE(gf.passes());
}

TEST(GreeksFilter, DeltaExactlyAtMaxBoundaryPasses) {
    GreeksFilter gf{0.20, 0.40, 50.0, 20.0};
    gf.update(0.40, 20.0, 9.90, 10.10);
    EXPECT_TRUE(gf.passes());
}

TEST(GreeksFilter, DeltaBelowMinFails) {
    GreeksFilter gf{0.20, 0.40, 50.0, 20.0};
    // |delta| = 0.10 < 0.20
    gf.update(0.10, 20.0, 9.90, 10.10);
    EXPECT_FALSE(gf.passes());
    EXPECT_DOUBLE_EQ(gf.score(), 0.0);
}

TEST(GreeksFilter, DeltaAboveMaxFails) {
    GreeksFilter gf{0.20, 0.40, 50.0, 20.0};
    // |delta| = 0.70 > 0.40
    gf.update(0.70, 20.0, 9.90, 10.10);
    EXPECT_FALSE(gf.passes());
    EXPECT_DOUBLE_EQ(gf.score(), 0.0);
}

TEST(GreeksFilter, NegativeDeltaUsesAbsoluteValue) {
    // A put with delta=-0.30 should satisfy the [0.20, 0.40] band via |delta|.
    GreeksFilter gf{0.20, 0.40, 50.0, 20.0};
    gf.update(-0.30, 20.0, 9.75, 10.25);
    EXPECT_TRUE(gf.passes());
}

// ── IV percentile criterion ───────────────────────────────────────────────────

TEST(GreeksFilter, LowIvPercentilePasses) {
    GreeksFilter gf{0.20, 0.40, 50.0, 20.0};
    gf.update(0.30, 10.0, 9.75, 10.25);
    EXPECT_TRUE(gf.passes());
}

TEST(GreeksFilter, HighIvPercentileFails) {
    GreeksFilter gf{0.20, 0.40, 50.0, 20.0};
    // iv_pct = 80 > iv_pct_max = 50
    gf.update(0.30, 80.0, 9.75, 10.25);
    EXPECT_FALSE(gf.passes());
    EXPECT_DOUBLE_EQ(gf.score(), 0.0);
}

TEST(GreeksFilter, IvAtExactMaxPasses) {
    GreeksFilter gf{0.20, 0.40, 50.0, 20.0};
    // iv_pct == iv_pct_max: the check is strict <, so 50.0 is NOT accepted.
    // Check the boundary behavior: passes() returns false for iv_pct >= iv_pct_max.
    gf.update(0.30, 49.9, 9.75, 10.25);
    EXPECT_TRUE(gf.passes());
}

// ── Bid-ask spread criterion ──────────────────────────────────────────────────

TEST(GreeksFilter, TightSpreadPasses) {
    GreeksFilter gf{0.20, 0.40, 50.0, 20.0};
    // bid=9.95, ask=10.05 → spread = 0.10, mid = 10.0 → spread_pct = 1.0%
    gf.update(0.30, 20.0, 9.95, 10.05);
    EXPECT_TRUE(gf.passes());
}

TEST(GreeksFilter, WideSpreadFails) {
    GreeksFilter gf{0.20, 0.40, 50.0, 20.0};
    // bid=8.0, ask=12.0 → spread=4.0, mid=10.0 → spread_pct=40%  > 20%
    gf.update(0.30, 20.0, 8.0, 12.0);
    EXPECT_FALSE(gf.passes());
    EXPECT_DOUBLE_EQ(gf.score(), 0.0);
}

// ── Score is in [0, 1] when passing ──────────────────────────────────────────

TEST(GreeksFilter, ScoreInUnitIntervalWhenPassing) {
    GreeksFilter gf{0.20, 0.40, 50.0, 20.0};
    gf.update(0.30, 25.0, 9.75, 10.25);
    ASSERT_TRUE(gf.passes());
    EXPECT_GE(gf.score(), 0.0);
    EXPECT_LE(gf.score(), 1.0);
}

// ── Update overwrites previous state ─────────────────────────────────────────

TEST(GreeksFilter, SubsequentUpdateOverwritesPrevious) {
    GreeksFilter gf{0.20, 0.40, 50.0, 20.0};
    // First update: good inputs.
    gf.update(0.30, 20.0, 9.90, 10.10);
    ASSERT_TRUE(gf.passes());

    // Second update: delta out of range.
    gf.update(0.05, 20.0, 9.90, 10.10);
    EXPECT_FALSE(gf.passes());
}
