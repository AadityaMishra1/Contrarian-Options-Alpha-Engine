add_executable(bench_ring_buffer
  bench_ring_buffer.cpp
)

target_link_libraries(bench_ring_buffer PRIVATE
  coe_market_data
)
