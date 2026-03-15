cmake_minimum_required(VERSION 3.24)
project(contrarian_options_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warnings
add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion)

# ── FetchContent dependencies ───────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.15.0
  GIT_SHALLOW    TRUE
)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.0
  GIT_SHALLOW    TRUE
)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(spdlog googletest nlohmann_json)

# ── System dependencies ─────────────────────────────────────────────────
find_package(Boost 1.80 REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)

# ── Include path ────────────────────────────────────────────────────────
include_directories(${PROJECT_SOURCE_DIR}/include)

# ── Library modules ─────────────────────────────────────────────────────
add_subdirectory(src/common)
add_subdirectory(src/market_data)
add_subdirectory(src/strategy)
add_subdirectory(src/execution)
add_subdirectory(src/risk)

option(BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" OFF)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()

# ── Tests ───────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests/common)
add_subdirectory(tests/market_data)
add_subdirectory(tests/strategy)
add_subdirectory(tests/execution)
add_subdirectory(tests/risk)

# ── Benchmarks ──────────────────────────────────────────────────────────
add_subdirectory(benchmarks)
