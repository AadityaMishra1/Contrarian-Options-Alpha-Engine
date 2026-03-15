#include <gtest/gtest.h>
#include <coe/strategy/pipeline.hpp>
#include <coe/common/config.hpp>
#include <coe/common/error.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>

using namespace coe::strategy;
using namespace coe::common;
using coe::common::Timestamp;

// ── RAII temp YAML file ───────────────────────────────────────────────────────

struct TempStrategyYaml {
    std::string path;

    explicit TempStrategyYaml(const std::string& p = "/tmp/coe_pipeline_test.yaml") : path(p) {
        std::ofstream ofs(path);
        ofs << R"(
strategy:
  rsi:
    period: 5
  bollinger:
    period: 5
    multiplier: 2.0
  volume:
    lookback: 5
    spike_threshold: 2.0
  greeks:
    delta_min: 0.10
    delta_max: 0.60
    iv_percentile_max: 90.0
    spread_pct_max: 50.0
  scoring:
    weight_rsi: 0.30
    weight_bollinger: 0.25
    weight_volume: 0.25
    weight_greeks: 0.20
    min_composite: 0.01
)";
    }

    ~TempStrategyYaml() { std::remove(path.c_str()); }
};

// ── Helper: load a Config from the temp file ─────────────────────────────────

static Config load_config(const std::string& path) {
    auto result = Config::load(path);
    if (!is_ok(result)) {
        throw std::runtime_error("Failed to load test config");
    }
    return std::move(get_value(result));
}

// ── Construction ─────────────────────────────────────────────────────────────

TEST(SignalScorer, ConstructsFromConfig) {
    TempStrategyYaml tmp;
    auto cfg = load_config(tmp.path);
    EXPECT_NO_THROW({ SignalScorer scorer{cfg}; });
}

// ── Indicators are exposed via accessors ─────────────────────────────────────

TEST(SignalScorer, IndicatorAccessorsAccessible) {
    TempStrategyYaml tmp;
    auto cfg = load_config(tmp.path);
    SignalScorer scorer{cfg};
    // Access all four indicator/filter references — they must compile.
    [[maybe_unused]] const RSI&            rsi    = scorer.rsi();
    [[maybe_unused]] const BollingerBands& bb     = scorer.bb();
    [[maybe_unused]] const VolumeSpike&    volume = scorer.volume();
    [[maybe_unused]] const GreeksFilter&   greeks = scorer.greeks();
}

// ── No signal before warm-up ─────────────────────────────────────────────────

TEST(SignalScorer, NoSignalBeforeIndicatorsReady) {
    TempStrategyYaml tmp;
    auto cfg = load_config(tmp.path);
    SignalScorer scorer{cfg};

    // Feed only 3 price updates (period=5 requires 5 for BB, 6 for RSI).
    const Timestamp ts{std::chrono::nanoseconds{0}};
    for (int i = 0; i < 3; ++i) {
        scorer.update_price("SPY", 100.0 + static_cast<double>(i), ts);
    }
    auto sig = scorer.evaluate("SPY");
    EXPECT_FALSE(sig.has_value());
}

// ── Signal can be generated once all indicators are warm ─────────────────────

// We use period=5 for all indicators and min_composite=0.01 (very low threshold)
// so almost any valid score should emit a signal.

TEST(SignalScorer, SignalEmittedWhenAllIndicatorsReady) {
    TempStrategyYaml tmp;
    auto cfg = load_config(tmp.path);
    SignalScorer scorer{cfg};

    const Timestamp ts{std::chrono::nanoseconds{1000LL}};

    // Warm up RSI (needs period+1 = 6) and BB (needs period = 5) with strictly
    // decreasing prices to build an oversold RSI signal.
    for (int i = 0; i < 10; ++i) {
        scorer.update_price("SPY", 100.0 - static_cast<double>(i) * 2.0, ts);
    }

    // Warm up volume (needs 5) with constant then spike.
    for (int i = 0; i < 5; ++i) {
        scorer.update_volume("SPY", 1000.0);
    }
    scorer.update_volume("SPY", 5000.0); // spike

    // Greeks in the passing zone.
    scorer.update_greeks("SPY", 0.30, 30.0, 9.75, 10.25);

    auto sig = scorer.evaluate("SPY");
    EXPECT_TRUE(sig.has_value());
}

// ── Signal fields are populated correctly ────────────────────────────────────

TEST(SignalScorer, SignalFieldsPopulatedOnEmission) {
    TempStrategyYaml tmp;
    auto cfg = load_config(tmp.path);
    SignalScorer scorer{cfg};

    const Timestamp ts{std::chrono::nanoseconds{42LL}};
    for (int i = 0; i < 10; ++i) {
        scorer.update_price("SPY", 100.0 - static_cast<double>(i) * 2.0, ts);
    }
    for (int i = 0; i < 5; ++i) { scorer.update_volume("SPY", 1000.0); }
    scorer.update_volume("SPY", 5000.0);
    scorer.update_greeks("SPY", 0.30, 30.0, 9.75, 10.25);

    auto sig = scorer.evaluate("SPY");
    ASSERT_TRUE(sig.has_value());
    EXPECT_EQ(sig->symbol, "SPY");
    EXPECT_GE(sig->composite_score, 0.0);
    EXPECT_LE(sig->composite_score, 1.0);
    EXPECT_GE(sig->rsi_score,     0.0);
    EXPECT_GE(sig->bb_score,      0.0);
    EXPECT_GE(sig->volume_score,  0.0);
    EXPECT_GE(sig->greeks_score,  0.0);
}

// ── No signal when composite is below threshold ───────────────────────────────

TEST(SignalScorer, NoSignalWhenCompositeIsBelowThreshold) {
    // Use a very high min_composite threshold so the signal is blocked.
    const std::string path = "/tmp/coe_pipeline_high_threshold.yaml";
    {
        std::ofstream ofs(path);
        ofs << R"(
strategy:
  rsi:
    period: 5
  bollinger:
    period: 5
    multiplier: 2.0
  volume:
    lookback: 5
    spike_threshold: 2.0
  greeks:
    delta_min: 0.20
    delta_max: 0.40
    iv_percentile_max: 50.0
    spread_pct_max: 20.0
  scoring:
    weight_rsi: 0.25
    weight_bollinger: 0.25
    weight_volume: 0.25
    weight_greeks: 0.25
    min_composite: 0.99
)";
    }

    auto res = Config::load(path);
    ASSERT_TRUE(is_ok(res));
    SignalScorer scorer{get_value(res)};
    std::remove(path.c_str());

    // Feed mixed data that won't produce a near-perfect composite score.
    const Timestamp ts{std::chrono::nanoseconds{0}};
    for (int i = 0; i < 10; ++i) {
        scorer.update_price("SPY", 100.0 + static_cast<double>(i % 3), ts);
    }
    for (int i = 0; i < 5; ++i) { scorer.update_volume("SPY", 1000.0); }
    scorer.update_volume("SPY", 1050.0); // no meaningful spike
    scorer.update_greeks("SPY", 0.30, 25.0, 9.75, 10.25);

    auto sig = scorer.evaluate("SPY");
    EXPECT_FALSE(sig.has_value());
}

// ── Reset brings all indicators back to not-ready ────────────────────────────

TEST(SignalScorer, ResetBringsIndicatorsBackToNotReady) {
    TempStrategyYaml tmp;
    auto cfg = load_config(tmp.path);
    SignalScorer scorer{cfg};

    const Timestamp ts{std::chrono::nanoseconds{0}};
    for (int i = 0; i < 10; ++i) {
        scorer.update_price("SPY", 100.0 + static_cast<double>(i), ts);
    }
    EXPECT_TRUE(scorer.rsi().ready());

    scorer.reset();

    EXPECT_FALSE(scorer.rsi().ready());
    EXPECT_FALSE(scorer.bb().ready());
    EXPECT_FALSE(scorer.volume().ready());
}
