add_library(coe_risk
  pnl_tracker.cpp
  circuit_breaker.cpp
  risk_manager.cpp
)

target_link_libraries(coe_risk PUBLIC
  coe_common
  coe_execution
)
