#include <gtest/gtest.h>
#include <coe/strategy/rsi.hpp>

#include <cmath>

using coe::strategy::RSI;

// ── Warm-up requirement ───────────────────────────────────────────────────────

TEST(RSI, NotReadyAfterZeroUpdates) {
    RSI rsi{14};
    EXPECT_FALSE(rsi.ready());
}

TEST(RSI, NotReadyAfterPeriodUpdates) {
    // Ready requires period+1 prices (first call seeds prev_price only).
    RSI rsi{14};
    for (int i = 0; i < 14; ++i) {
        rsi.update(100.0 + static_cast<double>(i));
        EXPECT_FALSE(rsi.ready()) << "Should not be ready after " << (i + 1) << " updates";
    }
}

TEST(RSI, ReadyAfterPeriodPlusOneUpdates) {
    RSI rsi{14};
    for (int i = 0; i <= 14; ++i) {
        rsi.update(100.0 + static_cast<double>(i));
    }
    EXPECT_TRUE(rsi.ready());
}

TEST(RSI, SentinelValueBeforeReady) {
    RSI rsi{14};
    EXPECT_DOUBLE_EQ(rsi.value(), 50.0);
    rsi.update(100.0);
    EXPECT_DOUBLE_EQ(rsi.value(), 50.0);
}

// ── Period accessor ───────────────────────────────────────────────────────────

TEST(RSI, PeriodAccessor) {
    RSI rsi{21};
    EXPECT_EQ(rsi.period(), 21);
}

// ── Pure up-moves → RSI near 100 ─────────────────────────────────────────────

TEST(RSI, AllUpMovesProducesHighRsi) {
    RSI rsi{14};
    // Feed 30 strictly increasing prices — no down moves, so avg_loss stays 0.
    for (int i = 0; i <= 30; ++i) {
        rsi.update(100.0 + static_cast<double>(i));
    }
    ASSERT_TRUE(rsi.ready());
    // avg_loss == 0 => RSI defined as 100 by the implementation.
    EXPECT_DOUBLE_EQ(rsi.value(), 100.0);
}

// ── Pure down-moves → RSI near 0 ─────────────────────────────────────────────

TEST(RSI, AllDownMovesProducesLowRsi) {
    RSI rsi{14};
    // Feed 30 strictly decreasing prices — no up moves, so avg_gain stays 0.
    for (int i = 0; i <= 30; ++i) {
        rsi.update(100.0 - static_cast<double>(i));
    }
    ASSERT_TRUE(rsi.ready());
    // avg_gain == 0 => RS = 0 => RSI = 100 - 100/(1+0) = 0.
    EXPECT_NEAR(rsi.value(), 0.0, 1e-9);
}

// ── Known price series: alternating gains/losses → RSI ~50 ───────────────────

TEST(RSI, AlternatingPricesProducesMidRangeRsi) {
    RSI rsi{14};
    // Alternate +1 and -1 moves: equal up and down pressure.
    double price = 100.0;
    for (int i = 0; i <= 30; ++i) {
        rsi.update(price);
        price += (i % 2 == 0) ? 1.0 : -1.0;
    }
    ASSERT_TRUE(rsi.ready());
    // Should be close to 50 (symmetric pressure).
    EXPECT_GT(rsi.value(), 30.0);
    EXPECT_LT(rsi.value(), 70.0);
}

// ── RSI value is bounded to [0, 100] ─────────────────────────────────────────

TEST(RSI, ValueAlwaysInBounds) {
    RSI rsi{14};
    // Feed a zigzag of varying magnitudes.
    for (int i = 0; i <= 50; ++i) {
        rsi.update(100.0 + (i % 7 == 0 ? 10.0 : -3.0));
        if (rsi.ready()) {
            EXPECT_GE(rsi.value(), 0.0);
            EXPECT_LE(rsi.value(), 100.0);
        }
    }
}

// ── Reset ─────────────────────────────────────────────────────────────────────

TEST(RSI, ResetClearsState) {
    RSI rsi{14};
    for (int i = 0; i <= 14; ++i) {
        rsi.update(100.0 + static_cast<double>(i));
    }
    ASSERT_TRUE(rsi.ready());

    rsi.reset();

    EXPECT_FALSE(rsi.ready());
    EXPECT_DOUBLE_EQ(rsi.value(), 50.0);
}

TEST(RSI, ReadyAgainAfterResetAndReload) {
    RSI rsi{14};
    for (int i = 0; i <= 14; ++i) { rsi.update(100.0 + static_cast<double>(i)); }
    rsi.reset();
    for (int i = 0; i <= 14; ++i) { rsi.update(200.0 + static_cast<double>(i)); }
    EXPECT_TRUE(rsi.ready());
    EXPECT_DOUBLE_EQ(rsi.value(), 100.0); // all up moves again
}

// ── Custom period ─────────────────────────────────────────────────────────────

TEST(RSI, CustomPeriod9ReadinessThreshold) {
    RSI rsi{9};
    for (int i = 0; i < 9; ++i) {
        rsi.update(100.0 + static_cast<double>(i));
        EXPECT_FALSE(rsi.ready());
    }
    rsi.update(109.0);
    EXPECT_TRUE(rsi.ready());
}
