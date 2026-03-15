#pragma once

#include <coe/strategy/concepts.hpp>

namespace coe::strategy {

/// Option-Greeks quality gate and scorer.
///
/// Evaluates whether a current option quote satisfies the parametric
/// thresholds required by the contrarian strategy and assigns a normalised
/// [0, 1] composite score based on how well each criterion is met.
///
/// Criteria:
///   1. delta_min <= |delta| <= delta_max     (target moneyness band)
///   2. iv_percentile < iv_pct_max            (not buying elevated IV)
///   3. bid-ask spread pct < spread_pct_max   (acceptable liquidity)
///
/// Score computation:
///   Each criterion contributes an equal 1/3 weight.  Within a passing
///   criterion the partial score scales linearly from the boundary towards the
///   ideal centre so that options deeper inside the bounds score higher.
class GreeksFilter {
public:
    /// @param delta_min      Minimum absolute delta (default 0.20).
    /// @param delta_max      Maximum absolute delta (default 0.40).
    /// @param iv_pct_max     Maximum IV percentile (default 50.0).
    /// @param spread_pct_max Maximum bid-ask spread as % of mid (default 20.0).
    explicit GreeksFilter(double delta_min     = 0.20,
                          double delta_max     = 0.40,
                          double iv_pct_max    = 50.0,
                          double spread_pct_max = 20.0) noexcept;

    GreeksFilter(const GreeksFilter&)            = default;
    GreeksFilter& operator=(const GreeksFilter&) = default;
    GreeksFilter(GreeksFilter&&)                 = default;
    GreeksFilter& operator=(GreeksFilter&&)      = default;
    ~GreeksFilter()                              = default;

    /// Ingest the latest Greeks and quote data for the current option.
    /// @param delta         Signed option delta.
    /// @param iv_percentile Current IV expressed as a percentile (0–100).
    /// @param bid           Best bid price.
    /// @param ask           Best ask price.
    void update(double delta, double iv_percentile, double bid, double ask) noexcept;

    /// True when all three criteria pass.  False before the first update().
    [[nodiscard]] bool passes() const noexcept;

    /// Normalised [0, 1] composite quality score.  Returns 0.0 before the
    /// first update() or when passes() == false.
    [[nodiscard]] double score() const noexcept;

private:
    // Configuration
    double delta_min_;
    double delta_max_;
    double iv_pct_max_;
    double spread_pct_max_;

    // Most recently ingested values
    double abs_delta_    {-1.0};  // -1 sentinel = no data yet
    double iv_pct_       {0.0};
    double spread_pct_   {0.0};

    bool   has_data_     {false};

    // ── Per-criterion scoring helpers ──────────────────────────────────────

    /// Partial score for the delta criterion (0 if failing, 0-1 if passing).
    [[nodiscard]] double delta_partial_score()  const noexcept;

    /// Partial score for the IV-percentile criterion.
    [[nodiscard]] double iv_partial_score()     const noexcept;

    /// Partial score for the bid-ask spread criterion.
    [[nodiscard]] double spread_partial_score() const noexcept;
};

static_assert(Filter<GreeksFilter>, "GreeksFilter must satisfy the Filter concept");

} // namespace coe::strategy
