add_executable(test_common
  test_config.cpp
  test_types.cpp
  test_error.cpp
)

target_link_libraries(test_common PRIVATE
  coe_common
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_common)
