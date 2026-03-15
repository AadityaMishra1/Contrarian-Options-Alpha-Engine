add_executable(test_risk
  test_pnl_tracker.cpp
  test_circuit_breaker.cpp
  test_risk_manager.cpp
)

target_link_libraries(test_risk PRIVATE
  coe_risk
  coe_execution
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_risk)
