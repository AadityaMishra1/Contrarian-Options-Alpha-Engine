#pragma once

#include "coe/common/types.hpp"

namespace coe::md {

using coe::common::Symbol;

// ── DataFeed concept ───────────────────────────────────────────────────────

/// Models a live market-data feed that can be started, stopped, and
/// subscribed to.
///
/// Constraints
/// -----------
/// Any type F satisfying DataFeed must provide:
///   • void  start()                     – begin connecting and delivering data
///   • void  stop()                      – cease delivery and release resources
///   • void  subscribe(const Symbol&)    – register interest in a symbol
///   • bool  is_connected() const        – true iff the transport is live
///
/// Notes
/// -----
/// The concept intentionally leaves scheduling (threading, callbacks, polling)
/// up to the implementing class so that different transports (WebSocket,
/// multicast UDP, file replay) can each choose an appropriate model.
template <typename F>
concept DataFeed = requires(F feed, const F const_feed, const Symbol& sym) {
    { feed.start() }          -> std::same_as<void>;
    { feed.stop() }           -> std::same_as<void>;
    { feed.subscribe(sym) }   -> std::same_as<void>;
    { const_feed.is_connected() } -> std::same_as<bool>;
};

} // namespace coe::md
