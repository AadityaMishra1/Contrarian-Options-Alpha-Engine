find_package(pybind11 REQUIRED)

pybind11_add_module(_coe_engine
    module.cpp
    bind_common.cpp
    bind_strategy.cpp
    bind_execution.cpp
    bind_risk.cpp
)

target_link_libraries(_coe_engine PRIVATE
    coe_common
    coe_strategy
    coe_execution
    coe_risk
)

target_include_directories(_coe_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
