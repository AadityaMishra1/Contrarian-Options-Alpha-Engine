#pragma once

#include <spdlog/spdlog.h>

#include <memory>
#include <string>

namespace coe::common {

/// Initialise the named logger and register it with the spdlog global registry.
///
/// Two sinks are attached:
///   - A coloured stdout console sink.
///   - A size-rotating file sink writing to "logs/<log_name>.log" (5 MB / 3
///     files).
///
/// After calling this function the logger is retrievable anywhere in the
/// process via spdlog::get(log_name).
///
/// @param log_name  Name used both for the registry key and the log file name.
/// @param level     Minimum severity that will be emitted (default: info).
/// @returns         Shared pointer to the newly created logger.
std::shared_ptr<spdlog::logger> init_logging(
    const std::string&      log_name,
    spdlog::level::level_enum level = spdlog::level::info);

} // namespace coe::common
