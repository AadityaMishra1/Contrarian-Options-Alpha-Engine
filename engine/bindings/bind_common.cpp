#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <coe/common/types.hpp>
#include <coe/common/error.hpp>
#include <coe/common/config.hpp>

namespace py = pybind11;

using namespace coe::common;

// ---------------------------------------------------------------------------
// Module-level exception type.  Stored as a py::object so that bind_strategy,
// bind_execution, and bind_risk can raise it via PyErr_SetObject.
// ---------------------------------------------------------------------------
static py::object g_coe_error_type;

/// Raise the Python CoeError exception with the given ErrorCode.
void raise_coe_error(ErrorCode code) {
    PyErr_SetObject(
        g_coe_error_type.ptr(),
        py::make_tuple(static_cast<int>(code), std::string(toString(code))).ptr()
    );
    throw py::error_already_set{};
}

/// Unwrap a Result<T>: return the value on success, raise CoeError on failure.
template <typename T>
T unwrap(Result<T> result) {
    if (is_ok(result)) {
        return get_value(result);
    }
    raise_coe_error(get_error(result));
    // Unreachable — raise_coe_error always throws.
    __builtin_unreachable();
}

/// Unwrap a VoidResult (Result<std::monostate>): returns None on success,
/// raises CoeError on failure.
void unwrap_void(std::variant<std::monostate, ErrorCode> result) {
    if (!std::holds_alternative<std::monostate>(result)) {
        raise_coe_error(std::get<ErrorCode>(result));
    }
}

// Make helpers accessible from the other bind_*.cpp translation units.
void raise_coe_error_pub(ErrorCode code) { raise_coe_error(code); }

// Provide unwrap helpers as free functions so other TUs can include them via
// a minimal shared header.  Because the definitions are in this TU, the
// symbols are emitted once and linked into the single extension module.

template uint64_t         unwrap<uint64_t>(Result<uint64_t>);
template Config            unwrap<Config>(Result<Config>);

// ---------------------------------------------------------------------------

void init_common(py::module_& m) {
    // ── Side enum ──────────────────────────────────────────────────────────
    py::enum_<Side>(m, "Side")
        .value("Buy",  Side::Buy)
        .value("Sell", Side::Sell)
        .def("__str__", [](Side s) { return std::string(toString(s)); })
        .def("__repr__", [](Side s) {
            return std::string("Side.") + std::string(toString(s));
        });

    // ── OptionType enum ────────────────────────────────────────────────────
    py::enum_<OptionType>(m, "OptionType")
        .value("Call", OptionType::Call)
        .value("Put",  OptionType::Put)
        .def("__str__", [](OptionType t) { return std::string(toString(t)); })
        .def("__repr__", [](OptionType t) {
            return std::string("OptionType.") + std::string(toString(t));
        });

    // ── ErrorCode enum ─────────────────────────────────────────────────────
    py::enum_<ErrorCode>(m, "ErrorCode")
        .value("Ok",                    ErrorCode::Ok)
        .value("ConfigNotFound",        ErrorCode::ConfigNotFound)
        .value("ConfigParseError",      ErrorCode::ConfigParseError)
        .value("InvalidParameter",      ErrorCode::InvalidParameter)
        .value("RingBufferFull",        ErrorCode::RingBufferFull)
        .value("RingBufferEmpty",       ErrorCode::RingBufferEmpty)
        .value("WebSocketError",        ErrorCode::WebSocketError)
        .value("ConnectionFailed",      ErrorCode::ConnectionFailed)
        .value("OrderRejected",         ErrorCode::OrderRejected)
        .value("PositionLimitExceeded", ErrorCode::PositionLimitExceeded)
        .value("DailyLossExceeded",     ErrorCode::DailyLossExceeded)
        .value("CircuitBreakerTripped", ErrorCode::CircuitBreakerTripped)
        .value("InsufficientMargin",    ErrorCode::InsufficientMargin)
        .value("InvalidOrderState",     ErrorCode::InvalidOrderState)
        .value("Unknown",               ErrorCode::Unknown)
        .def("__str__", [](ErrorCode c) { return std::string(toString(c)); })
        .def("__repr__", [](ErrorCode c) {
            return std::string("ErrorCode.") + std::string(toString(c));
        });

    // ── CoeError exception ─────────────────────────────────────────────────
    // Derive from RuntimeError so that except RuntimeError also catches it.
    g_coe_error_type = py::exception<std::runtime_error>(m, "CoeError");

    // Store a reference accessible from the other bind_*.cpp files via
    // module attribute lookup at raise time.
    m.attr("_coe_error_type") = g_coe_error_type;

    // ── OptionContract struct ──────────────────────────────────────────────
    py::class_<OptionContract>(m, "OptionContract")
        .def(py::init<>())
        .def(py::init([](const std::string& underlying,
                         OptionType         type,
                         double             strike,
                         int64_t            expiry_ns,
                         int32_t            dte) {
            OptionContract c;
            c.underlying = underlying;
            c.type       = type;
            c.strike     = strike;
            c.expiry     = Timestamp{expiry_ns};
            c.dte        = dte;
            return c;
        }),
        py::arg("underlying"),
        py::arg("type"),
        py::arg("strike"),
        py::arg("expiry_ns"),
        py::arg("dte"))
        .def_readwrite("underlying", &OptionContract::underlying)
        .def_readwrite("type",       &OptionContract::type)
        .def_readwrite("strike",     &OptionContract::strike)
        .def_property("expiry_ns",
            [](const OptionContract& c) {
                return c.expiry.count();
            },
            [](OptionContract& c, int64_t ns) {
                c.expiry = Timestamp{ns};
            })
        .def_readwrite("dte", &OptionContract::dte)
        .def("__repr__", [](const OptionContract& c) {
            return "OptionContract(underlying='" + c.underlying +
                   "', type=" + std::string(toString(c.type)) +
                   ", strike=" + std::to_string(c.strike) +
                   ", dte="    + std::to_string(c.dte) + ")";
        });

    // ── Config class ───────────────────────────────────────────────────────
    py::class_<Config>(m, "Config")
        // Direct constructor — may throw; exceptions propagate naturally.
        .def(py::init<std::string_view>(), py::arg("path"))

        // Factory that goes through Result<Config> and raises CoeError.
        .def_static("load", [](const std::string& path) -> Config {
            auto res = Config::load(path);
            if (is_ok(res)) {
                return std::move(get_value(res));
            }
            raise_coe_error(get_error(res));
            __builtin_unreachable();
        }, py::arg("path"))

        .def("get_int", [](const Config& c, const std::string& key, int def) {
            return c.get<int>(key, def);
        }, py::arg("key"), py::arg("default") = 0)

        .def("get_double", [](const Config& c, const std::string& key, double def) {
            return c.get<double>(key, def);
        }, py::arg("key"), py::arg("default") = 0.0)

        .def("get_string", [](const Config& c,
                               const std::string& key,
                               const std::string& def) {
            return c.get<std::string>(key, def);
        }, py::arg("key"), py::arg("default") = std::string{})

        .def("get_bool", [](const Config& c, const std::string& key, bool def) {
            return c.get<bool>(key, def);
        }, py::arg("key"), py::arg("default") = false)

        .def("path", [](const Config& c) {
            return std::string(c.path());
        });
}
