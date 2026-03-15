add_library(coe_execution
  order_manager.cpp
  position.cpp
  position_sizer.cpp
)

target_link_libraries(coe_execution PUBLIC
  coe_common
)
