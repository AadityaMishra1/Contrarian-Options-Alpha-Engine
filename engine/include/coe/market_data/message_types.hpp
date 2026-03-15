#pragma once

#include "coe/common/types.hpp"

#include <cstdint>
#include <variant>

namespace coe::md {

// Pull in primitive aliases from common so callers need only include this header.
using coe::common::OptionType;
using coe::common::Price;
using coe::common::Quantity;
using coe::common::Symbol;
using coe::common::Timestamp;

// ── Message structs ────────────────────────────────────────────────────────

/// Top-of-book quote update for any instrument.
struct Quote {
    Symbol   symbol;
    Price    bid;
    Price    ask;
    Quantity bid_size;
    Quantity ask_size;
    Timestamp ts;
};

/// Single print (last sale) for any instrument.
struct Trade {
    Symbol    symbol;
    Price     price;
    Quantity  volume;
    Timestamp ts;
};

/// Full NBBO quote enriched with option greeks for a listed contract.
struct OptionsQuote {
    Symbol     symbol;       ///< OSI-format option symbol, e.g. "SPY240119C00450000"
    Symbol     underlying;   ///< Underlying equity/index symbol, e.g. "SPY"
    OptionType type;         ///< Call or Put
    Price      strike;       ///< Strike price in USD
    int32_t    dte;          ///< Days-to-expiration at time of this quote
    Price      bid;
    Price      ask;
    double     delta;        ///< [−1, 1]
    double     gamma;        ///< Per unit of underlying move
    double     theta;        ///< Daily time decay in USD
    double     vega;         ///< Per 1-point IV move
    double     iv;           ///< Implied volatility (annualised, decimal)
    Timestamp  ts;
};

// ── Discriminated union ────────────────────────────────────────────────────

/// A single market-data event; visitors should use std::visit.
using MarketMessage = std::variant<Quote, Trade, OptionsQuote>;

} // namespace coe::md
