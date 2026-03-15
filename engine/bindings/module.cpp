#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations — each bind_*.cpp defines one of these.
void init_common(py::module_& m);
void init_strategy(py::module_& m);
void init_execution(py::module_& m);
void init_risk(py::module_& m);

PYBIND11_MODULE(_coe_engine, m) {
    m.doc() = "Contrarian Options Alpha Engine — C++20 Python bindings";

    init_common(m);
    init_strategy(m);
    init_execution(m);
    init_risk(m);
}
