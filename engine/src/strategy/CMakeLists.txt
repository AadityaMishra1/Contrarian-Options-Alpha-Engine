add_library(coe_strategy
  rsi.cpp
  bollinger.cpp
  volume_spike.cpp
  greeks_filter.cpp
  pipeline.cpp
)

target_link_libraries(coe_strategy PUBLIC
  coe_common
)
