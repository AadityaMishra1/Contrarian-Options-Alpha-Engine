#pragma once

#include <coe/strategy/concepts.hpp>

#include <deque>

namespace coe::strategy {

/// Rolling-window volume spike detector.
///
/// Memory: O(lookback) — a sliding deque of the last `lookback` volume samples.
/// Complexity: O(1) per update via running sum.
///
/// value() returns current_volume / average_volume.
/// A ratio >= spike_threshold signals unusual activity.
///
/// Not ready => value() returns 0.0, is_spike() returns false.
class VolumeSpike {
public:
    /// @param lookback         Rolling window size (default 20).
    /// @param spike_threshold  Ratio at which volume is considered a spike (default 2.0).
    explicit VolumeSpike(int lookback = 20, double spike_threshold = 2.0) noexcept;

    VolumeSpike(const VolumeSpike&)            = default;
    VolumeSpike& operator=(const VolumeSpike&) = default;
    VolumeSpike(VolumeSpike&&)                 = default;
    VolumeSpike& operator=(VolumeSpike&&)      = default;
    ~VolumeSpike()                             = default;

    /// Ingest the next volume observation.
    void update(double volume) noexcept;

    /// Ratio of the most recent volume to the rolling average.  Returns 0.0 before ready().
    [[nodiscard]] double value() const noexcept;

    /// True after `lookback` volume samples have been consumed.
    [[nodiscard]] bool ready() const noexcept;

    /// Reset to construction state.
    void reset() noexcept;

    /// True when value() >= spike_threshold and ready().
    [[nodiscard]] bool is_spike() const noexcept;

    /// The configured spike threshold.
    [[nodiscard]] double spike_threshold() const noexcept { return spike_threshold_; }

private:
    int                lookback_;
    double             spike_threshold_;
    std::deque<double> window_;
    double             sum_            {0.0};
    double             current_volume_ {0.0};
};

static_assert(Indicator<VolumeSpike>, "VolumeSpike must satisfy the Indicator concept");

} // namespace coe::strategy
