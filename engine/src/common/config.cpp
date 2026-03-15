#include <coe/common/config.hpp>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace coe::common {

// ── Constructor ─────────────────────────────────────────────────────────────

Config::Config(std::string_view path)
    : root_(YAML::LoadFile(std::string(path))),
      path_(path) {}

// ── Private helpers ─────────────────────────────────────────────────────────

YAML::Node Config::navigate(std::string_view dotted_key) const {
    // Split the dotted key into individual segments.
    std::vector<std::string> segments;
    {
        std::string key_str(dotted_key);
        std::istringstream ss(key_str);
        std::string token;
        while (std::getline(ss, token, '.')) {
            if (!token.empty()) {
                segments.push_back(std::move(token));
            }
        }
    }

    if (segments.empty()) {
        return YAML::Node{};
    }

    // Walk the YAML tree one level at a time.
    YAML::Node current = YAML::Clone(root_);
    for (const auto& seg : segments) {
        if (!current.IsMap()) {
            return YAML::Node{};
        }
        current = current[seg];
        if (!current.IsDefined()) {
            return YAML::Node{};
        }
    }

    return current;
}

// ── Factory ─────────────────────────────────────────────────────────────────

Result<Config> Config::load(std::string_view path) {
    // Check existence before attempting to parse so we can distinguish the two
    // failure modes with distinct error codes.
    const std::filesystem::path fspath(path);
    if (!std::filesystem::exists(fspath)) {
        return ErrorCode::ConfigNotFound;
    }

    try {
        Config cfg(path);
        return cfg;
    } catch (const YAML::ParserException&) {
        return ErrorCode::ConfigParseError;
    } catch (const YAML::BadFile&) {
        return ErrorCode::ConfigNotFound;
    } catch (const std::exception&) {
        return ErrorCode::ConfigParseError;
    }
}

} // namespace coe::common
