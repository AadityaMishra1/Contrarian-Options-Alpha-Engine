add_library(coe_common
  config.cpp
  logging.cpp
)

target_link_libraries(coe_common PUBLIC
  spdlog::spdlog
  yaml-cpp::yaml-cpp
  fmt::fmt
)

target_include_directories(coe_common PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)
