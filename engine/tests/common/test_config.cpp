#include <gtest/gtest.h>
#include <coe/common/config.hpp>
#include <coe/common/error.hpp>

#include <cstdio>
#include <fstream>
#include <string>

using namespace coe::common;

// ── RAII helper: writes a temp file and removes it on scope exit ──────────────

struct TempYaml {
    std::string path;

    explicit TempYaml(const std::string& content, const std::string& name = "/tmp/coe_test_config.yaml") {
        path = name;
        std::ofstream ofs(path);
        ofs << content;
    }

    ~TempYaml() {
        std::remove(path.c_str());
    }
};

// ── Factory: load() success ───────────────────────────────────────────────────

TEST(Config, LoadSucceedsOnValidYaml) {
    TempYaml tmp{R"(
strategy:
  rsi:
    period: 14
)"};
    auto result = Config::load(tmp.path);
    EXPECT_TRUE(is_ok(result));
}

TEST(Config, LoadPreservesPath) {
    TempYaml tmp{"key: value\n"};
    auto result = Config::load(tmp.path);
    ASSERT_TRUE(is_ok(result));
    EXPECT_EQ(get_value(result).path(), tmp.path);
}

// ── Factory: load() failure ───────────────────────────────────────────────────

TEST(Config, LoadReturnsConfigNotFoundForMissingFile) {
    auto result = Config::load("/tmp/this_file_should_not_exist_coe.yaml");
    EXPECT_FALSE(is_ok(result));
    EXPECT_EQ(get_error(result), ErrorCode::ConfigNotFound);
}

TEST(Config, LoadReturnsConfigParseErrorForMalformedYaml) {
    TempYaml tmp{"key: [unclosed bracket\n", "/tmp/coe_bad_yaml.yaml"};
    auto result = Config::load(tmp.path);
    EXPECT_FALSE(is_ok(result));
    EXPECT_EQ(get_error(result), ErrorCode::ConfigParseError);
}

// ── Dotted-key access ─────────────────────────────────────────────────────────

TEST(Config, GetDoubleNestedKey) {
    TempYaml tmp{R"(
strategy:
  rsi:
    period: 21
)"};
    auto result = Config::load(tmp.path);
    ASSERT_TRUE(is_ok(result));
    const auto& cfg = get_value(result);
    EXPECT_DOUBLE_EQ(cfg.get<double>("strategy.rsi.period", 14.0), 21.0);
}

TEST(Config, GetIntNestedKey) {
    TempYaml tmp{R"(
strategy:
  bollinger:
    period: 20
)"};
    auto result = Config::load(tmp.path);
    ASSERT_TRUE(is_ok(result));
    const auto& cfg = get_value(result);
    EXPECT_EQ(cfg.get<int>("strategy.bollinger.period", 10), 20);
}

TEST(Config, GetStringValue) {
    TempYaml tmp{R"(
engine:
  name: contrarian
)"};
    auto result = Config::load(tmp.path);
    ASSERT_TRUE(is_ok(result));
    const auto& cfg = get_value(result);
    EXPECT_EQ(cfg.get<std::string>("engine.name", "default"), "contrarian");
}

TEST(Config, GetBoolValue) {
    TempYaml tmp{R"(
engine:
  live: true
)"};
    auto result = Config::load(tmp.path);
    ASSERT_TRUE(is_ok(result));
    const auto& cfg = get_value(result);
    EXPECT_TRUE(cfg.get<bool>("engine.live", false));
}

// ── Default fallback when key is missing ─────────────────────────────────────

TEST(Config, MissingTopLevelKeyReturnsDefault) {
    TempYaml tmp{"existing: 1\n"};
    auto result = Config::load(tmp.path);
    ASSERT_TRUE(is_ok(result));
    const auto& cfg = get_value(result);
    EXPECT_DOUBLE_EQ(cfg.get<double>("nonexistent", 99.5), 99.5);
}

TEST(Config, MissingNestedKeyReturnsDefault) {
    TempYaml tmp{R"(
strategy:
  rsi:
    period: 14
)"};
    auto result = Config::load(tmp.path);
    ASSERT_TRUE(is_ok(result));
    const auto& cfg = get_value(result);
    // "strategy.rsi.overbought" does not exist.
    EXPECT_DOUBLE_EQ(cfg.get<double>("strategy.rsi.overbought", 70.0), 70.0);
}

TEST(Config, DefaultIntWhenKeyAbsent) {
    TempYaml tmp{"placeholder: 0\n"};
    auto result = Config::load(tmp.path);
    ASSERT_TRUE(is_ok(result));
    const auto& cfg = get_value(result);
    EXPECT_EQ(cfg.get<int>("risk.max_positions", 5), 5);
}

// ── Full strategy config round-trip ──────────────────────────────────────────

TEST(Config, FullStrategyConfig) {
    TempYaml tmp{R"(
strategy:
  rsi:
    period: 14
  bollinger:
    period: 20
    multiplier: 2.0
  volume:
    lookback: 20
    spike_threshold: 2.5
  greeks:
    delta_min: 0.20
    delta_max: 0.40
    iv_percentile_max: 50.0
    spread_pct_max: 20.0
  scoring:
    weight_rsi: 0.30
    weight_bollinger: 0.25
    weight_volume: 0.25
    weight_greeks: 0.20
    min_composite: 0.65
)", "/tmp/coe_full_strategy.yaml"};

    auto result = Config::load(tmp.path);
    ASSERT_TRUE(is_ok(result));
    const auto& cfg = get_value(result);

    EXPECT_EQ(cfg.get<int>("strategy.rsi.period", 0), 14);
    EXPECT_EQ(cfg.get<int>("strategy.bollinger.period", 0), 20);
    EXPECT_DOUBLE_EQ(cfg.get<double>("strategy.bollinger.multiplier", 0.0), 2.0);
    EXPECT_EQ(cfg.get<int>("strategy.volume.lookback", 0), 20);
    EXPECT_DOUBLE_EQ(cfg.get<double>("strategy.volume.spike_threshold", 0.0), 2.5);
    EXPECT_DOUBLE_EQ(cfg.get<double>("strategy.greeks.delta_min", 0.0), 0.20);
    EXPECT_DOUBLE_EQ(cfg.get<double>("strategy.greeks.delta_max", 0.0), 0.40);
    EXPECT_DOUBLE_EQ(cfg.get<double>("strategy.scoring.weight_rsi", 0.0), 0.30);
    EXPECT_DOUBLE_EQ(cfg.get<double>("strategy.scoring.min_composite", 0.0), 0.65);
}

// ── Type mismatch falls back to default ──────────────────────────────────────

TEST(Config, TypeMismatchReturnsDefault) {
    TempYaml tmp{R"(
key: "not_a_number"
)"};
    auto result = Config::load(tmp.path);
    ASSERT_TRUE(is_ok(result));
    const auto& cfg = get_value(result);
    // Requesting an int where the value is a string should return the default.
    EXPECT_EQ(cfg.get<int>("key", 42), 42);
}
