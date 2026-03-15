#include <gtest/gtest.h>
#include <coe/strategy/volume_spike.hpp>

using coe::strategy::VolumeSpike;

// ── Warm-up requirement ───────────────────────────────────────────────────────

TEST(VolumeSpike, NotReadyAfterZeroUpdates) {
    VolumeSpike vs{20, 2.0};
    EXPECT_FALSE(vs.ready());
}

TEST(VolumeSpike, NotReadyUntilLookbackReached) {
    VolumeSpike vs{20, 2.0};
    for (int i = 0; i < 19; ++i) {
        vs.update(1000.0);
        EXPECT_FALSE(vs.ready()) << "Should not be ready after " << (i + 1) << " updates";
    }
}

TEST(VolumeSpike, ReadyAfterExactlyLookbackUpdates) {
    VolumeSpike vs{20, 2.0};
    for (int i = 0; i < 20; ++i) {
        vs.update(1000.0);
    }
    EXPECT_TRUE(vs.ready());
}

// ── Before ready: sentinel values ────────────────────────────────────────────

TEST(VolumeSpike, ValueZeroBeforeReady) {
    VolumeSpike vs{20, 2.0};
    vs.update(5000.0);
    EXPECT_DOUBLE_EQ(vs.value(), 0.0);
}

TEST(VolumeSpike, IsSpikeReturnsFalseBeforeReady) {
    VolumeSpike vs{20, 2.0};
    vs.update(9999.0);
    EXPECT_FALSE(vs.is_spike());
}

// ── Spike detection ───────────────────────────────────────────────────────────

TEST(VolumeSpike, ConstantVolumeThenTripleSpikeIsDetected) {
    VolumeSpike vs{20, 2.0};
    // Fill window with 1000 units.
    for (int i = 0; i < 20; ++i) {
        vs.update(1000.0);
    }
    ASSERT_TRUE(vs.ready());
    // value() at this point is current_volume / avg = 1000/1000 = 1.0 → no spike.
    EXPECT_DOUBLE_EQ(vs.value(), 1.0);
    EXPECT_FALSE(vs.is_spike());

    // Now feed a 3x spike.
    vs.update(3000.0);
    // After the spike the window slides: one 1000 is evicted, 3000 enters.
    // avg = (19*1000 + 3000) / 20 = 22000/20 = 1100; ratio = 3000/1100 ≈ 2.727.
    EXPECT_GT(vs.value(), 2.0);
    EXPECT_TRUE(vs.is_spike());
}

TEST(VolumeSpike, NormalVolumeNoSpike) {
    VolumeSpike vs{20, 2.0};
    for (int i = 0; i < 20; ++i) {
        vs.update(1000.0);
    }
    ASSERT_TRUE(vs.ready());
    // Feed a volume exactly at the average — ratio = 1.0, well below threshold 2.0.
    vs.update(1000.0);
    EXPECT_DOUBLE_EQ(vs.value(), 1.0);
    EXPECT_FALSE(vs.is_spike());
}

TEST(VolumeSpike, SlightlyBelowThresholdNoSpike) {
    VolumeSpike vs{10, 2.0};
    for (int i = 0; i < 10; ++i) {
        vs.update(100.0);
    }
    ASSERT_TRUE(vs.ready());
    // Feed 1.99x: ratio < 2.0 → no spike.
    vs.update(199.0);
    EXPECT_FALSE(vs.is_spike());
}

TEST(VolumeSpike, ExactlyAtThresholdIsSpike) {
    VolumeSpike vs{10, 2.0};
    for (int i = 0; i < 10; ++i) {
        vs.update(100.0);
    }
    ASSERT_TRUE(vs.ready());
    // After one update of 200: window becomes [100*9, 200], avg = (900+200)/10 = 110.
    // ratio = 200/110 ≈ 1.818 — not quite 2x due to rolling.
    // Instead feed an extreme value so ratio clearly >= threshold.
    vs.update(400.0); // avg = (900+400)/10 = 130; ratio = 400/130 ≈ 3.08 → spike
    EXPECT_TRUE(vs.is_spike());
}

// ── spike_threshold accessor ──────────────────────────────────────────────────

TEST(VolumeSpike, SpikeThresholdAccessor) {
    VolumeSpike vs{20, 3.5};
    EXPECT_DOUBLE_EQ(vs.spike_threshold(), 3.5);
}

// ── value() ratio calculation ─────────────────────────────────────────────────

TEST(VolumeSpike, ValueEqualsCurrentOverAverage) {
    VolumeSpike vs{4, 2.0};
    vs.update(100.0);
    vs.update(100.0);
    vs.update(100.0);
    vs.update(100.0);
    ASSERT_TRUE(vs.ready());
    // current = 100, avg = 100 → ratio = 1.0
    EXPECT_DOUBLE_EQ(vs.value(), 1.0);
}

// ── Reset ─────────────────────────────────────────────────────────────────────

TEST(VolumeSpike, ResetClearsState) {
    VolumeSpike vs{20, 2.0};
    for (int i = 0; i < 20; ++i) { vs.update(1000.0); }
    vs.update(5000.0);
    ASSERT_TRUE(vs.ready());
    ASSERT_TRUE(vs.is_spike());

    vs.reset();

    EXPECT_FALSE(vs.ready());
    EXPECT_FALSE(vs.is_spike());
    EXPECT_DOUBLE_EQ(vs.value(), 0.0);
}

TEST(VolumeSpike, ReadyAgainAfterReset) {
    VolumeSpike vs{5, 2.0};
    for (int i = 0; i < 5; ++i) { vs.update(500.0); }
    vs.reset();
    for (int i = 0; i < 5; ++i) { vs.update(500.0); }
    EXPECT_TRUE(vs.ready());
}
