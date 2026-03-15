#pragma once

#include "coe/common/error.hpp"
#include "coe/common/types.hpp"
#include "coe/market_data/data_feed.hpp"
#include "coe/market_data/message_types.hpp"
#include "coe/market_data/ring_buffer.hpp"

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

namespace coe::md {

using coe::common::ErrorCode;
using coe::common::Result;
using coe::common::Symbol;

namespace beast = boost::beast;
namespace asio  = boost::asio;
namespace ssl   = boost::asio::ssl;

// ── WebSocketClient ────────────────────────────────────────────────────────

/// Live market-data feed backed by a WebSocket connection.
///
/// The class satisfies the DataFeed concept.  A dedicated std::thread
/// drives the Boost.Asio io_context so the caller never blocks.
///
/// Lifecycle
/// ---------
///   1. Construct with a URL and a reference to an SPSCRingBuffer.
///   2. Call subscribe() for each symbol of interest.
///   3. Call start()  – spawns the I/O thread and establishes the connection.
///   4. Call stop()   – signals the I/O thread to exit and closes the socket.
///
/// Thread safety
/// -------------
/// subscribe() may be called before start().  Calling subscribe() after
/// start() appends to the subscription list and sends a subscribe message
/// on the wire if the connection is live, protected by a mutex.
/// All other methods are NOT thread-safe between themselves.
class WebSocketClient {
public:
    /// @param url     Full WebSocket URL, e.g. "wss://stream.example.com/feed"
    /// @param buffer  Ring buffer that receives parsed MarketMessages.
    explicit WebSocketClient(std::string                          url,
                             SPSCRingBuffer<MarketMessage, 65536>& buffer);

    ~WebSocketClient();

    // Non-copyable, non-movable – owns OS-level socket resources.
    WebSocketClient(const WebSocketClient&)            = delete;
    WebSocketClient& operator=(const WebSocketClient&) = delete;
    WebSocketClient(WebSocketClient&&)                 = delete;
    WebSocketClient& operator=(WebSocketClient&&)      = delete;

    // ── DataFeed interface ─────────────────────────────────────────────────

    /// Resolve, connect, TLS-handshake, WS-handshake, and begin the read
    /// loop, all on a dedicated jthread.
    void start();

    /// Request orderly shutdown: stop the jthread, send WS close frame.
    void stop();

    /// Register @p symbol for subscription.  If already connected, a
    /// subscribe JSON message is sent immediately.
    void subscribe(const Symbol& symbol);

    /// Returns true when the WebSocket handshake has completed successfully
    /// and no error has been encountered since.
    [[nodiscard]] bool is_connected() const noexcept;

    // ── Parsing ────────────────────────────────────────────────────────────

    /// Parse a raw JSON string into a MarketMessage variant.
    /// Returns ErrorCode::WebSocketError when the message cannot be parsed,
    /// or ErrorCode::InvalidParameter when the "type" field is unrecognised.
    [[nodiscard]] Result<MarketMessage> parse_message(const std::string& json) const;

private:
    // ── Internal helpers ───────────────────────────────────────────────────

    /// Entry point for the I/O thread; runs until stop_requested_ is set.
    void run_io_loop();

    /// Connect, TLS-handshake, WS-handshake, then enter the async read loop.
    void do_connect();

    /// Send a JSON subscribe payload for every symbol in subscriptions_.
    void send_subscribe_all();

    /// Send a JSON subscribe payload for a single symbol.
    void send_subscribe(const Symbol& symbol);

    /// Async read loop: read one message, dispatch, reschedule.
    void do_read();

    /// Dispatch a successfully read frame.
    void on_message(const std::string& raw);

    // ── Data members ───────────────────────────────────────────────────────

    std::string                           url_;
    SPSCRingBuffer<MarketMessage, 65536>& ring_;

    // Parsed URL components (populated in constructor).
    std::string host_;
    std::string port_;
    std::string path_;

    // Asio/Beast objects – owned by the I/O thread after start().
    std::unique_ptr<asio::io_context>                               ioc_;
    std::unique_ptr<ssl::context>                                   ssl_ctx_;
    std::unique_ptr<beast::websocket::stream<
        beast::ssl_stream<beast::tcp_stream>>>                      ws_;

    beast::flat_buffer read_buf_;

    // Subscription list – written before start() and thereafter under mutex_.
    std::vector<Symbol>   subscriptions_;
    std::mutex            subscriptions_mutex_;

    std::atomic<bool>     connected_{false};
    std::atomic<bool>     stop_requested_{false};
    std::thread           io_thread_;
};

// ── Concept verification ───────────────────────────────────────────────────

static_assert(DataFeed<WebSocketClient>,
              "WebSocketClient must satisfy the DataFeed concept");

} // namespace coe::md
