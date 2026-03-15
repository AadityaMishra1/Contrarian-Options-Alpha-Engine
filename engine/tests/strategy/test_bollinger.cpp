#include <gtest/gtest.h>
#include <coe/strategy/bollinger.hpp>

#include <cmath>

using coe::strategy::BollingerBands;

// ── Warm-up requirement ───────────────────────────────────────────────────────

TEST(BollingerBands, NotReadyAfterZeroUpdates) {
    BollingerBands bb{20, 2.0};
    EXPECT_FALSE(bb.ready());
}

TEST(BollingerBands, NotReadyUntilPeriodReached) {
    BollingerBands bb{20, 2.0};
    for (int i = 0; i < 19; ++i) {
        bb.update(100.0);
        EXPECT_FALSE(bb.ready()) << "Should not be ready after " << (i + 1) << " updates";
    }
}

TEST(BollingerBands, ReadyAfterExactlyPeriodUpdates) {
    BollingerBands bb{20, 2.0};
    for (int i = 0; i < 20; ++i) {
        bb.update(100.0);
    }
    EXPECT_TRUE(bb.ready());
}

// ── Before ready: all accessors return 0.0 ───────────────────────────────────

TEST(BollingerBands, AllAccessorsZeroBeforeReady) {
    BollingerBands bb{20, 2.0};
    bb.update(100.0);
    EXPECT_DOUBLE_EQ(bb.value(),  0.0);
    EXPECT_DOUBLE_EQ(bb.upper(),  0.0);
    EXPECT_DOUBLE_EQ(bb.lower(),  0.0);
    EXPECT_DOUBLE_EQ(bb.middle(), 0.0);
}

// ── Constant price: stddev = 0, bands collapse to mean ───────────────────────

TEST(BollingerBands, ConstantPriceBandsCollapseToMean) {
    BollingerBands bb{20, 2.0};
    for (int i = 0; i < 20; ++i) {
        bb.update(50.0);
    }
    ASSERT_TRUE(bb.ready());
    EXPECT_NEAR(bb.middle(), 50.0, 1e-9);
    EXPECT_NEAR(bb.upper(),  50.0, 1e-9);
    EXPECT_NEAR(bb.lower(),  50.0, 1e-9);
}

// ── Ascending prices: upper > middle > lower ─────────────────────────────────

TEST(BollingerBands, AscendingPricesUpperMiddleLowerRelation) {
    BollingerBands bb{20, 2.0};
    for (int i = 0; i < 20; ++i) {
        bb.update(100.0 + static_cast<double>(i));
    }
    ASSERT_TRUE(bb.ready());
    EXPECT_GT(bb.upper(),  bb.middle());
    EXPECT_GT(bb.middle(), bb.lower());
}

// ── Middle band equals SMA ────────────────────────────────────────────────────

TEST(BollingerBands, MiddleBandEqualsSma) {
    BollingerBands bb{5, 2.0};
    // Feed prices 10, 20, 30, 40, 50 → SMA = 30.
    for (int v : {10, 20, 30, 40, 50}) {
        bb.update(static_cast<double>(v));
    }
    ASSERT_TRUE(bb.ready());
    EXPECT_NEAR(bb.middle(), 30.0, 1e-9);
}

// ── Upper and lower bands symmetric around middle ────────────────────────────

TEST(BollingerBands, BandsSymmetricAroundMiddle) {
    BollingerBands bb{5, 2.0};
    for (int i = 1; i <= 5; ++i) {
        bb.update(static_cast<double>(i) * 10.0);
    }
    ASSERT_TRUE(bb.ready());
    const double mid    = bb.middle();
    const double spread = bb.upper() - mid;
    EXPECT_NEAR(mid - bb.lower(), spread, 1e-9);
}

// ── value() is positive when price is below lower band ───────────────────────

TEST(BollingerBands, ValuePositiveWhenBelowLowerBand) {
    BollingerBands bb{20, 2.0};
    // Warm up with prices around 100.
    for (int i = 0; i < 20; ++i) {
        bb.update(100.0 + static_cast<double>(i % 3) - 1.0);
    }
    ASSERT_TRUE(bb.ready());
    // Feed a price far below the lower band.
    bb.update(50.0);
    EXPECT_GT(bb.value(), 0.0);
}

// ── value() is non-positive when price is inside or above bands ───────────────

TEST(BollingerBands, ValueNonPositiveWhenInsideBands) {
    BollingerBands bb{20, 2.0};
    // Feed constant price — stddev=0 so value is (mid - mid) / eps = 0.
    for (int i = 0; i < 20; ++i) {
        bb.update(100.0);
    }
    ASSERT_TRUE(bb.ready());
    // Price is on the mean: not below the lower band.
    bb.update(100.0);
    EXPECT_LE(bb.value(), 0.0);
}

// ── Sliding window drops oldest price ────────────────────────────────────────

TEST(BollingerBands, SlidingWindowUpdatesMiddle) {
    BollingerBands bb{3, 2.0};
    bb.update(10.0);
    bb.update(20.0);
    bb.update(30.0); // window: [10, 20, 30] → mid = 20
    ASSERT_TRUE(bb.ready());
    EXPECT_NEAR(bb.middle(), 20.0, 1e-9);

    bb.update(40.0); // window: [20, 30, 40] → mid = 30
    EXPECT_NEAR(bb.middle(), 30.0, 1e-9);
}

// ── Reset ─────────────────────────────────────────────────────────────────────

TEST(BollingerBands, ResetClearsState) {
    BollingerBands bb{20, 2.0};
    for (int i = 0; i < 20; ++i) { bb.update(100.0); }
    ASSERT_TRUE(bb.ready());

    bb.reset();

    EXPECT_FALSE(bb.ready());
    EXPECT_DOUBLE_EQ(bb.value(),  0.0);
    EXPECT_DOUBLE_EQ(bb.upper(),  0.0);
    EXPECT_DOUBLE_EQ(bb.lower(),  0.0);
    EXPECT_DOUBLE_EQ(bb.middle(), 0.0);
}
