add_executable(test_market_data
  test_ring_buffer.cpp
  test_message_types.cpp
)

target_link_libraries(test_market_data PRIVATE
  coe_market_data
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_market_data)
