#include <gtest/gtest.h>
#include <coe/execution/position_sizer.hpp>

#include <cmath>

using namespace coe::execution;

// ── Accessors ─────────────────────────────────────────────────────────────────

TEST(KellyPositionSizer, KellyFractionAccessor) {
    KellyPositionSizer sizer{0.5, 20.0};
    EXPECT_DOUBLE_EQ(sizer.kelly_fraction(), 0.5);
}

TEST(KellyPositionSizer, MaxBetAccessor) {
    KellyPositionSizer sizer{0.5, 20.0};
    EXPECT_DOUBLE_EQ(sizer.max_bet(), 20.0);
}

// ── Positive edge ─────────────────────────────────────────────────────────────

TEST(KellyPositionSizer, PositiveEdge) {
    KellyPositionSizer sizer{0.5, 20.0}; // half-Kelly, 20% max
    // win_rate=0.6, avg_win=2.0, avg_loss=1.0, bankroll=1000
    // b = 2.0/1.0 = 2.0
    // f* = (2.0*0.6 - 0.4) / 2.0 = (1.2 - 0.4) / 2.0 = 0.4
    // f_scaled = 0.5 * 0.4 = 0.2
    // dollar_bet = 0.2 * 1000 = 200
    // cap = 20/100 * 1000 = 200 → result = min(200, 200) = 200
    double size = sizer.calculate_size(0.6, 2.0, 1.0, 1000.0);
    EXPECT_GT(size, 0.0);
    EXPECT_LE(size, 200.0);
}

TEST(KellyPositionSizer, PositiveEdgeReturnsBelowMaxBet) {
    KellyPositionSizer sizer{1.0, 10.0}; // full Kelly, 10% max
    double bankroll = 50000.0;
    double size = sizer.calculate_size(0.55, 1.0, 1.0, bankroll);
    EXPECT_GT(size, 0.0);
    // Must never exceed max_bet% of bankroll.
    EXPECT_LE(size, 0.10 * bankroll + 1e-9);
}

TEST(KellyPositionSizer, HalfKellyHalvesSizeVsFullKelly) {
    KellyPositionSizer half{0.5, 100.0};
    KellyPositionSizer full{1.0, 100.0};
    double bankroll = 10000.0;
    double half_size = half.calculate_size(0.55, 1.0, 1.0, bankroll);
    double full_size = full.calculate_size(0.55, 1.0, 1.0, bankroll);
    // half-Kelly should produce roughly half the size of full-Kelly
    // (unless the max_bet cap is hit for full-Kelly).
    if (full_size < 100.0 * bankroll) {
        EXPECT_NEAR(half_size, full_size * 0.5, full_size * 0.05);
    }
}

// ── Negative edge → returns 0 ─────────────────────────────────────────────────

TEST(KellyPositionSizer, NegativeEdge) {
    KellyPositionSizer sizer{0.5, 20.0};
    // win_rate=0.3, avg_win=1.0, avg_loss=1.0 → f*=(1*0.3-0.7)/1=-0.4 → 0
    double size = sizer.calculate_size(0.3, 1.0, 1.0, 1000.0);
    EXPECT_DOUBLE_EQ(size, 0.0);
}

TEST(KellyPositionSizer, ZeroEdgeReturnsZero) {
    KellyPositionSizer sizer{0.5, 20.0};
    // Breakeven: win_rate=0.5, avg_win=avg_loss=1.0 → f*=(1*0.5-0.5)/1=0 → 0
    double size = sizer.calculate_size(0.5, 1.0, 1.0, 1000.0);
    EXPECT_DOUBLE_EQ(size, 0.0);
}

// ── Max-bet cap ───────────────────────────────────────────────────────────────

TEST(KellyPositionSizer, MaxBetCapsResult) {
    // Very high edge — Kelly fraction would suggest large bet.
    // max_bet=5% ensures the result is capped.
    KellyPositionSizer sizer{1.0, 5.0};
    double bankroll = 100000.0;
    double size = sizer.calculate_size(0.9, 10.0, 1.0, bankroll);
    EXPECT_LE(size, 0.05 * bankroll + 1e-9);
}

// ── contracts() calculation ───────────────────────────────────────────────────

TEST(KellyPositionSizer, ContractsCalculation) {
    KellyPositionSizer sizer{0.5, 20.0};
    // size=500, option_price=0.50 → 500/(0.5*100) = 10 contracts
    EXPECT_EQ(sizer.contracts(500.0, 0.50), 10);
}

TEST(KellyPositionSizer, ContractsLargeSize) {
    KellyPositionSizer sizer{0.5, 20.0};
    // size=1000, option_price=2.0 → 1000/(2.0*100) = 5 contracts
    EXPECT_EQ(sizer.contracts(1000.0, 2.0), 5);
}

TEST(KellyPositionSizer, MinOneContract) {
    KellyPositionSizer sizer{0.5, 20.0};
    // size=10, option_price=5.0 → 10/(5*100)=0.02 → floor=0 → clamp to 1
    EXPECT_EQ(sizer.contracts(10.0, 5.0), 1);
}

TEST(KellyPositionSizer, ZeroSizeReturnsZeroContracts) {
    KellyPositionSizer sizer{0.5, 20.0};
    EXPECT_EQ(sizer.contracts(0.0, 1.0), 0);
}

TEST(KellyPositionSizer, ContractsFloorsDivision) {
    KellyPositionSizer sizer{0.5, 20.0};
    // size=250, option_price=1.0 → 250/(1.0*100) = 2.5 → floor = 2
    EXPECT_EQ(sizer.contracts(250.0, 1.0), 2);
}

// ── Invalid inputs ────────────────────────────────────────────────────────────

TEST(KellyPositionSizer, ZeroBankrollReturnsZero) {
    KellyPositionSizer sizer{0.5, 20.0};
    double size = sizer.calculate_size(0.6, 2.0, 1.0, 0.0);
    EXPECT_DOUBLE_EQ(size, 0.0);
}

TEST(KellyPositionSizer, ZeroAvgLossReturnsZero) {
    KellyPositionSizer sizer{0.5, 20.0};
    // avg_loss = 0 is invalid (division by zero risk).
    double size = sizer.calculate_size(0.6, 2.0, 0.0, 1000.0);
    EXPECT_DOUBLE_EQ(size, 0.0);
}
