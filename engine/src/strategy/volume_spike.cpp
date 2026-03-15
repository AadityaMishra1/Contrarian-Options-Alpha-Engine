#include <coe/strategy/volume_spike.hpp>

namespace coe::strategy {

VolumeSpike::VolumeSpike(int lookback, double spike_threshold) noexcept
    : lookback_{lookback}, spike_threshold_{spike_threshold}
{}

void VolumeSpike::update(double volume) noexcept {
    current_volume_ = volume;

    window_.push_back(volume);
    sum_ += volume;

    if (static_cast<int>(window_.size()) > lookback_) {
        sum_ -= window_.front();
        window_.pop_front();
    }
}

double VolumeSpike::value() const noexcept {
    if (!ready()) return 0.0;

    const double average = sum_ / static_cast<double>(lookback_);
    if (average == 0.0) return 0.0;

    return current_volume_ / average;
}

bool VolumeSpike::ready() const noexcept {
    return static_cast<int>(window_.size()) == lookback_;
}

void VolumeSpike::reset() noexcept {
    window_.clear();
    sum_            = 0.0;
    current_volume_ = 0.0;
}

bool VolumeSpike::is_spike() const noexcept {
    return ready() && (value() >= spike_threshold_);
}

} // namespace coe::strategy
