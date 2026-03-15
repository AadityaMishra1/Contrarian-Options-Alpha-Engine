add_executable(test_execution
  test_order_manager.cpp
  test_position.cpp
  test_position_sizer.cpp
)

target_link_libraries(test_execution PRIVATE
  coe_execution
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_execution)
