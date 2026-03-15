#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <coe/common/error.hpp>
#include <coe/execution/order.hpp>
#include <coe/risk/limits.hpp>
#include <coe/risk/pnl_tracker.hpp>
#include <coe/risk/circuit_breaker.hpp>
#include <coe/risk/risk_manager.hpp>

#include <variant>

namespace py = pybind11;

using coe::common::ErrorCode;
using coe::common::is_ok;
using coe::common::get_value;
using coe::common::get_error;
using coe::common::toString;
using namespace coe::risk;

// ---------------------------------------------------------------------------
// Same pattern as bind_execution: look up CoeError by module attribute.
// ---------------------------------------------------------------------------
static void raise_coe(py::module_& m, ErrorCode code) {
    py::object exc_type = m.attr("_coe_error_type");
    PyErr_SetObject(
        exc_type.ptr(),
        py::make_tuple(static_cast<int>(code),
                       std::string(toString(code))).ptr()
    );
    throw py::error_already_set{};
}

void init_risk(py::module_& m) {
    // ── RiskLimits struct ──────────────────────────────────────────────────
    py::class_<RiskLimits>(m, "RiskLimits")
        .def(py::init([]() { return RiskLimits{}; }))
        .def(py::init([](double  daily_loss_limit,
                          int32_t max_positions,
                          double  max_single_position,
                          int32_t circuit_breaker_window,
                          double  min_win_rate) {
            RiskLimits rl;
            rl.daily_loss_limit        = daily_loss_limit;
            rl.max_positions           = max_positions;
            rl.max_single_position     = max_single_position;
            rl.circuit_breaker_window  = circuit_breaker_window;
            rl.min_win_rate            = min_win_rate;
            return rl;
        }),
        py::arg("daily_loss_limit")       = -50.0,
        py::arg("max_positions")          = 5,
        py::arg("max_single_position")    = 20.0,
        py::arg("circuit_breaker_window") = 20,
        py::arg("min_win_rate")           = 0.40)
        .def_readwrite("daily_loss_limit",       &RiskLimits::daily_loss_limit)
        .def_readwrite("max_positions",          &RiskLimits::max_positions)
        .def_readwrite("max_single_position",    &RiskLimits::max_single_position)
        .def_readwrite("circuit_breaker_window", &RiskLimits::circuit_breaker_window)
        .def_readwrite("min_win_rate",           &RiskLimits::min_win_rate)
        .def("__repr__", [](const RiskLimits& r) {
            return "RiskLimits(daily_loss_limit=" + std::to_string(r.daily_loss_limit) +
                   ", max_positions=" + std::to_string(r.max_positions) +
                   ", max_single_position=" + std::to_string(r.max_single_position) +
                   ", circuit_breaker_window=" + std::to_string(r.circuit_breaker_window) +
                   ", min_win_rate=" + std::to_string(r.min_win_rate) + ")";
        });

    // ── DailyPnLTracker ────────────────────────────────────────────────────
    py::class_<DailyPnLTracker>(m, "DailyPnLTracker")
        .def(py::init<>())
        .def("record_trade",      &DailyPnLTracker::record_trade, py::arg("pnl"))
        .def("daily_pnl",         &DailyPnLTracker::daily_pnl)
        .def("trade_count",       &DailyPnLTracker::trade_count)
        .def("is_limit_breached", &DailyPnLTracker::is_limit_breached, py::arg("limit"))
        .def("reset",             &DailyPnLTracker::reset)
        .def("__repr__", [](const DailyPnLTracker& d) {
            return "DailyPnLTracker(daily_pnl=" + std::to_string(d.daily_pnl()) +
                   ", trade_count=" + std::to_string(d.trade_count()) + ")";
        });

    // ── CircuitBreaker ─────────────────────────────────────────────────────
    py::class_<CircuitBreaker>(m, "CircuitBreaker")
        .def(py::init<int, double>(),
             py::arg("window_size")  = 20,
             py::arg("min_win_rate") = 0.40)
        .def("record_trade",       &CircuitBreaker::record_trade,    py::arg("is_win"))
        .def("is_tripped",         &CircuitBreaker::is_tripped)
        .def("current_win_rate",   &CircuitBreaker::current_win_rate)
        .def("window_fill",        &CircuitBreaker::window_fill)
        .def("reset",              &CircuitBreaker::reset)
        .def("__repr__", [](const CircuitBreaker& cb) {
            return "CircuitBreaker(win_rate=" + std::to_string(cb.current_win_rate()) +
                   ", window_fill=" + std::to_string(cb.window_fill()) +
                   ", is_tripped=" + (cb.is_tripped() ? "True" : "False") + ")";
        });

    // ── RiskManager ────────────────────────────────────────────────────────
    py::class_<RiskManager>(m, "RiskManager")
        .def(py::init<const RiskLimits&>(), py::arg("limits"))

        // check_new_order() -> None on success, raises CoeError on rejection.
        .def("check_new_order",
             [&m](const RiskManager& rm,
                  const coe::execution::Order& order,
                  int32_t current_positions,
                  double  order_value) {
            auto res = rm.check_new_order(order, current_positions, order_value);
            if (!is_ok(res)) {
                raise_coe(m, get_error(res));
            }
        },
        py::arg("order"),
        py::arg("current_positions"),
        py::arg("order_value"))

        .def("on_trade_closed",           &RiskManager::on_trade_closed,
             py::arg("pnl"), py::arg("is_win"))
        .def("reset_daily",               &RiskManager::reset_daily)
        .def("daily_pnl",                 &RiskManager::daily_pnl)
        .def("win_rate",                  &RiskManager::win_rate)
        .def("is_circuit_breaker_tripped",&RiskManager::is_circuit_breaker_tripped)
        .def_property_readonly("limits",  &RiskManager::limits)
        .def("__repr__", [](const RiskManager& rm) {
            return "RiskManager(daily_pnl=" + std::to_string(rm.daily_pnl()) +
                   ", win_rate=" + std::to_string(rm.win_rate()) +
                   ", circuit_tripped=" +
                   (rm.is_circuit_breaker_tripped() ? "True" : "False") + ")";
        });
}
