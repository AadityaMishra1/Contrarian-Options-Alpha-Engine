add_executable(test_strategy
  test_rsi.cpp
  test_bollinger.cpp
  test_volume_spike.cpp
  test_greeks_filter.cpp
  test_pipeline.cpp
)

target_link_libraries(test_strategy PRIVATE
  coe_strategy
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_strategy)
