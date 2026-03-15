find_package(OpenSSL REQUIRED)

add_library(coe_market_data
  ws_client.cpp
)

target_link_libraries(coe_market_data PUBLIC
  coe_common
  Boost::headers
  nlohmann_json::nlohmann_json
  OpenSSL::SSL
  OpenSSL::Crypto
)
