#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <coe/common/error.hpp>
#include <coe/common/types.hpp>
#include <coe/execution/order.hpp>
#include <coe/execution/order_manager.hpp>
#include <coe/execution/position.hpp>
#include <coe/execution/position_sizer.hpp>

#include <variant>

namespace py = pybind11;

using namespace coe::execution;
using coe::common::ErrorCode;
using coe::common::is_ok;
using coe::common::get_value;
using coe::common::get_error;
using coe::common::toString;

// ---------------------------------------------------------------------------
// Raise the module-level CoeError.  We look up the type by name at runtime
// so that this TU does not need to link against bind_common's statics.
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

// Macro to keep the boilerplate tidy inside lambdas.
#define RAISE_COE(m_ref, code) raise_coe(m_ref, code)

void init_execution(py::module_& m) {
    // ── OrderType enum ─────────────────────────────────────────────────────
    py::enum_<OrderType>(m, "OrderType")
        .value("Market", OrderType::Market)
        .value("Limit",  OrderType::Limit)
        .def("__str__", [](OrderType t) -> std::string {
            switch (t) {
                case OrderType::Market: return "Market";
                case OrderType::Limit:  return "Limit";
            }
            return "Unknown";
        });

    // ── OrderState enum ────────────────────────────────────────────────────
    py::enum_<OrderState>(m, "OrderState")
        .value("New",         OrderState::New)
        .value("PendingSend", OrderState::PendingSend)
        .value("Sent",        OrderState::Sent)
        .value("PartialFill", OrderState::PartialFill)
        .value("Filled",      OrderState::Filled)
        .value("Cancelled",   OrderState::Cancelled)
        .value("Rejected",    OrderState::Rejected)
        .def("__str__", [](OrderState s) -> std::string {
            switch (s) {
                case OrderState::New:         return "New";
                case OrderState::PendingSend: return "PendingSend";
                case OrderState::Sent:        return "Sent";
                case OrderState::PartialFill: return "PartialFill";
                case OrderState::Filled:      return "Filled";
                case OrderState::Cancelled:   return "Cancelled";
                case OrderState::Rejected:    return "Rejected";
            }
            return "Unknown";
        });

    // ── Order struct ───────────────────────────────────────────────────────
    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("id",             &Order::id)
        .def_readwrite("symbol",         &Order::symbol)
        .def_readwrite("side",           &Order::side)
        .def_readwrite("option_type",    &Order::option_type)
        .def_readwrite("strike",         &Order::strike)
        .def_readwrite("order_type",     &Order::order_type)
        .def_readwrite("limit_price",    &Order::limit_price)
        .def_readwrite("quantity",       &Order::quantity)
        .def_readwrite("filled_qty",     &Order::filled_qty)
        .def_readwrite("avg_fill_price", &Order::avg_fill_price)
        .def_readwrite("state",          &Order::state)
        // Timestamps exposed as integer nanoseconds.
        .def_property("created_ns",
            [](const Order& o) { return o.created.count(); },
            [](Order& o, int64_t ns) { o.created = coe::common::Timestamp{ns}; })
        .def_property("updated_ns",
            [](const Order& o) { return o.updated.count(); },
            [](Order& o, int64_t ns) { o.updated = coe::common::Timestamp{ns}; })
        .def("__repr__", [](const Order& o) {
            return "Order(id=" + std::to_string(o.id) +
                   ", symbol='" + o.symbol + "'" +
                   ", side=" + std::string(toString(o.side)) +
                   ", qty=" + std::to_string(o.quantity) +
                   ", filled=" + std::to_string(o.filled_qty) +
                   ", state=" + [&]{
                       switch (o.state) {
                           case OrderState::New:         return "New";
                           case OrderState::PendingSend: return "PendingSend";
                           case OrderState::Sent:        return "Sent";
                           case OrderState::PartialFill: return "PartialFill";
                           case OrderState::Filled:      return "Filled";
                           case OrderState::Cancelled:   return "Cancelled";
                           case OrderState::Rejected:    return "Rejected";
                       }
                       return "Unknown";
                   }() + ")";
        });

    // ── OrderManager ───────────────────────────────────────────────────────
    py::class_<OrderManager>(m, "OrderManager")
        .def(py::init<>())

        // submit() -> int  (raises CoeError on failure)
        .def("submit", [&m](OrderManager& om, Order order) -> uint64_t {
            auto res = om.submit(std::move(order));
            if (is_ok(res)) {
                return get_value(res);
            }
            raise_coe(m, get_error(res));
            __builtin_unreachable();
        }, py::arg("order"))

        // cancel() -> None  (raises CoeError on failure)
        .def("cancel", [&m](OrderManager& om, uint64_t id) {
            auto res = om.cancel(id);
            if (!std::holds_alternative<std::monostate>(res)) {
                raise_coe(m, std::get<ErrorCode>(res));
            }
        }, py::arg("order_id"))

        // on_fill() -> None  (raises CoeError on failure)
        .def("on_fill", [&m](OrderManager& om,
                              uint64_t id,
                              int32_t  qty,
                              double   price) {
            auto res = om.on_fill(id, qty, price);
            if (!std::holds_alternative<std::monostate>(res)) {
                raise_coe(m, std::get<ErrorCode>(res));
            }
        }, py::arg("order_id"), py::arg("qty"), py::arg("price"))

        // get_order() -> Optional[Order]
        .def("get_order", [](const OrderManager& om, uint64_t id) -> py::object {
            auto opt = om.get_order(id);
            if (opt.has_value()) return py::cast(*opt);
            return py::none();
        }, py::arg("order_id"))

        // get_open_orders() -> list[Order]
        .def("get_open_orders", &OrderManager::get_open_orders)

        .def("order_count", &OrderManager::order_count);

    // ── Position struct ────────────────────────────────────────────────────
    py::class_<Position>(m, "Position")
        .def(py::init<>())
        .def_readonly("symbol",         &Position::symbol)
        .def_readonly("side",           &Position::side)
        .def_readonly("quantity",       &Position::quantity)
        .def_readonly("avg_entry",      &Position::avg_entry)
        .def_readonly("current_price",  &Position::current_price)
        .def_readonly("unrealized_pnl", &Position::unrealized_pnl)
        .def_readonly("realized_pnl",   &Position::realized_pnl)
        .def("__repr__", [](const Position& p) {
            return "Position(symbol='" + p.symbol +
                   "', qty=" + std::to_string(p.quantity) +
                   ", avg_entry=" + std::to_string(p.avg_entry) +
                   ", unrealized_pnl=" + std::to_string(p.unrealized_pnl) +
                   ", realized_pnl=" + std::to_string(p.realized_pnl) + ")";
        });

    // ── PositionTracker ────────────────────────────────────────────────────
    py::class_<PositionTracker>(m, "PositionTracker")
        .def(py::init<>())
        .def("on_fill", &PositionTracker::on_fill,
             py::arg("symbol"), py::arg("side"), py::arg("qty"), py::arg("price"))
        .def("update_mark", &PositionTracker::update_mark,
             py::arg("symbol"), py::arg("mark"))
        .def("get_position", [](const PositionTracker& pt,
                                 const coe::common::Symbol& sym) -> py::object {
            auto opt = pt.get_position(sym);
            if (opt.has_value()) return py::cast(*opt);
            return py::none();
        }, py::arg("symbol"))
        .def("get_all_positions",     &PositionTracker::get_all_positions)
        .def("open_position_count",   &PositionTracker::open_position_count)
        .def("total_unrealized_pnl",  &PositionTracker::total_unrealized_pnl)
        .def("total_realized_pnl",    &PositionTracker::total_realized_pnl);

    // ── KellyPositionSizer ─────────────────────────────────────────────────
    py::class_<KellyPositionSizer>(m, "KellyPositionSizer")
        .def(py::init<double, double>(),
             py::arg("kelly_fraction") = 0.5,
             py::arg("max_bet")        = 20.0)
        .def("calculate_size", &KellyPositionSizer::calculate_size,
             py::arg("win_rate"),
             py::arg("avg_win"),
             py::arg("avg_loss"),
             py::arg("bankroll"))
        .def("contracts", &KellyPositionSizer::contracts,
             py::arg("size"), py::arg("option_price"))
        .def_property_readonly("kelly_fraction", &KellyPositionSizer::kelly_fraction)
        .def_property_readonly("max_bet",         &KellyPositionSizer::max_bet);

    // ── next_order_id free function ────────────────────────────────────────
    m.def("next_order_id", &next_order_id,
          "Return a process-unique, monotonically increasing order identifier.");
}
