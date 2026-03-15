#pragma once

#include <concepts>

namespace coe::strategy {

/// Concept satisfied by any type that acts as a streaming market indicator.
///
/// Requirements:
///   - update(double): consume the next data point
///   - value() const -> double: return the current indicator value
///   - ready() const -> bool: true once enough data has been accumulated
///   - reset(): clear internal state back to the initial condition
template <typename T>
concept Indicator = requires(T t, const T ct, double v) {
    { t.update(v) }       -> std::same_as<void>;
    { ct.value() }        -> std::same_as<double>;
    { ct.ready() }        -> std::same_as<bool>;
    { t.reset() }         -> std::same_as<void>;
};

/// Concept satisfied by any type that acts as a gate or scoring filter.
///
/// Requirements:
///   - passes() const -> bool: true when the current state meets all criteria
///   - score() const -> double: normalised [0, 1] quality score
template <typename T>
concept Filter = requires(const T ct) {
    { ct.passes() } -> std::same_as<bool>;
    { ct.score() }  -> std::same_as<double>;
};

} // namespace coe::strategy
