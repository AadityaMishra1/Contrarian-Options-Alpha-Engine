#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>

#include <coe/common/types.hpp>
#include <coe/common/config.hpp>
#include <coe/strategy/rsi.hpp>
#include <coe/strategy/bollinger.hpp>
#include <coe/strategy/volume_spike.hpp>
#include <coe/strategy/greeks_filter.hpp>
#include <coe/strategy/pipeline.hpp>
#include <coe/strategy/signal.hpp>

#include <optional>

namespace py = pybind11;

using namespace coe::strategy;
using coe::common::Symbol;
using coe::common::Timestamp;

void init_strategy(py::module_& m) {
    // ── RSI ────────────────────────────────────────────────────────────────
    py::class_<RSI>(m, "RSI")
        .def(py::init<int>(), py::arg("period") = 14)
        .def("update", &RSI::update,  py::arg("price"))
        .def("value",  &RSI::value)
        .def("ready",  &RSI::ready)
        .def("reset",  &RSI::reset)
        .def_property_readonly("period", &RSI::period)
        .def("__repr__", [](const RSI& r) {
            return "RSI(period=" + std::to_string(r.period()) +
                   ", value="    + std::to_string(r.value()) +
                   ", ready="    + (r.ready() ? "True" : "False") + ")";
        });

    // ── BollingerBands ─────────────────────────────────────────────────────
    py::class_<BollingerBands>(m, "BollingerBands")
        .def(py::init<int, double>(),
             py::arg("period")     = 20,
             py::arg("multiplier") = 2.0)
        .def("update", &BollingerBands::update, py::arg("price"))
        .def("value",  &BollingerBands::value)
        .def("upper",  &BollingerBands::upper)
        .def("lower",  &BollingerBands::lower)
        .def("middle", &BollingerBands::middle)
        .def("ready",  &BollingerBands::ready)
        .def("reset",  &BollingerBands::reset)
        .def("__repr__", [](const BollingerBands& bb) {
            return "BollingerBands(middle=" + std::to_string(bb.middle()) +
                   ", upper="  + std::to_string(bb.upper()) +
                   ", lower="  + std::to_string(bb.lower()) +
                   ", ready="  + (bb.ready() ? "True" : "False") + ")";
        });

    // ── VolumeSpike ────────────────────────────────────────────────────────
    py::class_<VolumeSpike>(m, "VolumeSpike")
        .def(py::init<int, double>(),
             py::arg("lookback")         = 20,
             py::arg("spike_threshold")  = 2.0)
        .def("update",           &VolumeSpike::update,          py::arg("volume"))
        .def("value",            &VolumeSpike::value)
        .def("ready",            &VolumeSpike::ready)
        .def("reset",            &VolumeSpike::reset)
        .def("is_spike",         &VolumeSpike::is_spike)
        .def_property_readonly("spike_threshold", &VolumeSpike::spike_threshold)
        .def("__repr__", [](const VolumeSpike& v) {
            return "VolumeSpike(value=" + std::to_string(v.value()) +
                   ", threshold=" + std::to_string(v.spike_threshold()) +
                   ", is_spike=" + (v.is_spike() ? "True" : "False") + ")";
        });

    // ── GreeksFilter ───────────────────────────────────────────────────────
    py::class_<GreeksFilter>(m, "GreeksFilter")
        .def(py::init<double, double, double, double>(),
             py::arg("delta_min")      = 0.20,
             py::arg("delta_max")      = 0.40,
             py::arg("iv_pct_max")     = 50.0,
             py::arg("spread_pct_max") = 20.0)
        .def("update", &GreeksFilter::update,
             py::arg("delta"),
             py::arg("iv_percentile"),
             py::arg("bid"),
             py::arg("ask"))
        .def("passes", &GreeksFilter::passes)
        .def("score",  &GreeksFilter::score)
        .def("__repr__", [](const GreeksFilter& g) {
            return "GreeksFilter(passes=" + std::string(g.passes() ? "True" : "False") +
                   ", score=" + std::to_string(g.score()) + ")";
        });

    // ── Signal struct ──────────────────────────────────────────────────────
    py::class_<Signal>(m, "Signal")
        .def(py::init<>())
        .def_readonly("symbol",          &Signal::symbol)
        .def_readonly("side",            &Signal::side)
        .def_readonly("composite_score", &Signal::composite_score)
        .def_readonly("rsi_score",       &Signal::rsi_score)
        .def_readonly("bb_score",        &Signal::bb_score)
        .def_readonly("volume_score",    &Signal::volume_score)
        .def_readonly("greeks_score",    &Signal::greeks_score)
        // Expose timestamp as nanoseconds integer for easy Python handling.
        .def_property_readonly("ts_ns", [](const Signal& s) {
            return s.ts.count();
        })
        .def("__repr__", [](const Signal& s) {
            return "Signal(symbol='" + s.symbol +
                   "', side=" + std::string(coe::common::toString(s.side)) +
                   ", composite_score=" + std::to_string(s.composite_score) +
                   ", rsi_score=" + std::to_string(s.rsi_score) +
                   ", bb_score=" + std::to_string(s.bb_score) +
                   ", volume_score=" + std::to_string(s.volume_score) +
                   ", greeks_score=" + std::to_string(s.greeks_score) + ")";
        });

    // ── SignalScorer ───────────────────────────────────────────────────────
    py::class_<SignalScorer>(m, "SignalScorer")
        .def(py::init<const coe::common::Config&>(), py::arg("config"))

        .def("update_price", [](SignalScorer& sc,
                                 const Symbol& sym,
                                 double        price,
                                 int64_t       ts_ns) {
            sc.update_price(sym, price, Timestamp{ts_ns});
        }, py::arg("symbol"), py::arg("price"), py::arg("ts_ns"))

        .def("update_volume", &SignalScorer::update_volume,
             py::arg("symbol"), py::arg("volume"))

        .def("update_greeks", &SignalScorer::update_greeks,
             py::arg("symbol"),
             py::arg("delta"),
             py::arg("iv_pct"),
             py::arg("bid"),
             py::arg("ask"))

        // evaluate() returns std::optional<Signal> — map to None or Signal.
        .def("evaluate", [](const SignalScorer& sc, const Symbol& sym)
             -> py::object {
            std::optional<Signal> sig = sc.evaluate(sym);
            if (sig.has_value()) {
                return py::cast(*sig);
            }
            return py::none();
        }, py::arg("symbol"))

        .def("reset", &SignalScorer::reset)

        // Read-only access to sub-indicators for inspection.
        .def_property_readonly("rsi",    &SignalScorer::rsi,    py::return_value_policy::reference_internal)
        .def_property_readonly("bb",     &SignalScorer::bb,     py::return_value_policy::reference_internal)
        .def_property_readonly("volume", &SignalScorer::volume, py::return_value_policy::reference_internal)
        .def_property_readonly("greeks", &SignalScorer::greeks, py::return_value_policy::reference_internal);
}
