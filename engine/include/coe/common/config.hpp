#pragma once

#include <coe/common/error.hpp>

#include <yaml-cpp/yaml.h>

#include <string>
#include <string_view>

namespace coe::common {

/// Wraps a YAML document and provides typed access via dotted key paths.
///
/// Example:
/// @code
///   auto res = Config::load("config/strategy.yaml");
///   if (is_ok(res)) {
///       auto& cfg = get_value(res);
///       int period = cfg.get<int>("strategy.rsi.period", 14);
///   }
/// @endcode
class Config {
public:
    /// Construct from a YAML file on disk.  Prefer the load() factory which
    /// returns a Result<Config> and translates YAML exceptions into ErrorCodes.
    explicit Config(std::string_view path);

    // Non-copyable, movable.
    Config(const Config&)            = delete;
    Config& operator=(const Config&) = delete;
    Config(Config&&)                 = default;
    Config& operator=(Config&&)      = default;

    ~Config() = default;

    // ── Typed accessor ─────────────────────────────────────────────────────

    /// Navigate a dotted key path (e.g. "strategy.rsi.period") and return the
    /// value converted to T.  Falls back to @p default_val when the key is
    /// absent or the type conversion fails.
    template <typename T>
    [[nodiscard]] T get(std::string_view dotted_key, T default_val) const noexcept {
        try {
            YAML::Node node = navigate(dotted_key);
            if (node && node.IsDefined()) {
                return node.as<T>();
            }
        } catch (...) {
            // Missing key or type mismatch — fall through to default.
        }
        return default_val;
    }

    /// Returns the file path this Config was loaded from.
    [[nodiscard]] std::string_view path() const noexcept { return path_; }

    // ── Factory ────────────────────────────────────────────────────────────

    /// Load a YAML file and return a Result<Config>.
    /// Returns ErrorCode::ConfigNotFound  if the file cannot be opened.
    /// Returns ErrorCode::ConfigParseError if the YAML is malformed.
    [[nodiscard]] static Result<Config> load(std::string_view path);

private:
    /// Walk the YAML tree along a '.'-delimited key sequence.
    [[nodiscard]] YAML::Node navigate(std::string_view dotted_key) const;

    YAML::Node  root_;
    std::string path_;
};

} // namespace coe::common
