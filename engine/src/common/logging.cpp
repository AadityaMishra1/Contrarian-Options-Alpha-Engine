#include <coe/common/logging.hpp>

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace coe::common {

std::shared_ptr<spdlog::logger> init_logging(
    const std::string&        log_name,
    spdlog::level::level_enum level)
{
    // If a logger with this name is already registered, return it as-is so that
    // calling init_logging() more than once is idempotent.
    if (auto existing = spdlog::get(log_name)) {
        return existing;
    }

    // Ensure the logs/ directory exists.
    const std::filesystem::path log_dir("logs");
    std::filesystem::create_directories(log_dir);

    // ── Sinks ──────────────────────────────────────────────────────────────

    // Coloured console output (stdout).
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(level);

    // Rotating file: 5 MB per file, 3 rotated files kept.
    constexpr std::size_t max_file_size = 5ULL * 1024ULL * 1024ULL; // 5 MB
    constexpr std::size_t max_files     = 3;

    const std::string log_file = (log_dir / (log_name + ".log")).string();
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_file, max_file_size, max_files);
    file_sink->set_level(level);

    // ── Logger assembly ────────────────────────────────────────────────────

    std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
    auto logger = std::make_shared<spdlog::logger>(log_name, sinks.begin(), sinks.end());

    // Common format: timestamp | logger name | level (coloured) | message
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
    logger->set_level(level);

    // Flush immediately on warnings and above so no messages are lost on crash.
    logger->flush_on(spdlog::level::warn);

    // Register so other translation units can retrieve it via spdlog::get().
    spdlog::register_logger(logger);

    return logger;
}

} // namespace coe::common
