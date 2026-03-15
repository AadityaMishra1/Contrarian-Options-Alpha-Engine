#include "coe/market_data/ws_client.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <boost/asio/ssl.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>

#include <chrono>
#include <mutex>
#include <stdexcept>
#include <string>

namespace coe::md {

namespace beast     = boost::beast;
namespace http      = beast::http;
namespace websocket = beast::websocket;
namespace asio      = boost::asio;
namespace ssl       = boost::asio::ssl;
using     tcp       = asio::ip::tcp;
using     json      = nlohmann::json;

// ── URL parsing ────────────────────────────────────────────────────────────

/// Splits "wss://host:port/path" into its three components.
/// Supports wss:// (port 443) and ws:// (port 80) schemes.
static void parse_url(const std::string& url,
                      std::string& host,
                      std::string& port,
                      std::string& path) {
    std::string_view sv{url};

    bool use_tls = false;
    if (sv.starts_with("wss://")) {
        use_tls = true;
        sv.remove_prefix(6);
    } else if (sv.starts_with("ws://")) {
        sv.remove_prefix(5);
    } else {
        throw std::invalid_argument("WebSocketClient: unsupported URL scheme: " + url);
    }

    const auto slash_pos = sv.find('/');
    const std::string_view authority = (slash_pos == std::string_view::npos)
                                           ? sv
                                           : sv.substr(0, slash_pos);
    path = (slash_pos == std::string_view::npos) ? "/" : std::string(sv.substr(slash_pos));

    const auto colon_pos = authority.rfind(':');
    if (colon_pos == std::string_view::npos) {
        host = std::string(authority);
        port = use_tls ? "443" : "80";
    } else {
        host = std::string(authority.substr(0, colon_pos));
        port = std::string(authority.substr(colon_pos + 1));
    }
}

// ── Construction / destruction ─────────────────────────────────────────────

WebSocketClient::WebSocketClient(std::string                           url,
                                 SPSCRingBuffer<MarketMessage, 65536>& buffer)
    : url_(std::move(url)), ring_(buffer) {
    parse_url(url_, host_, port_, path_);
    spdlog::debug("WebSocketClient: host={} port={} path={}", host_, port_, path_);
}

WebSocketClient::~WebSocketClient() {
    stop();
}

// ── DataFeed interface ─────────────────────────────────────────────────────

void WebSocketClient::start() {
    if (io_thread_.joinable()) {
        spdlog::warn("WebSocketClient::start() called while already running");
        return;
    }

    stop_requested_.store(false, std::memory_order_release);
    io_thread_ = std::thread([this]() {
        run_io_loop();
    });
}

void WebSocketClient::stop() {
    if (!io_thread_.joinable()) {
        return;
    }

    stop_requested_.store(true, std::memory_order_release);

    // Attempt a graceful WebSocket close from the I/O thread's context.
    // If the io_context is still alive, post a close onto it.
    if (ioc_ && ws_ && connected_.load(std::memory_order_acquire)) {
        asio::post(*ioc_, [this]() {
            beast::error_code ec;
            ws_->close(websocket::close_code::normal, ec);
            if (ec && ec != websocket::error::closed) {
                spdlog::warn("WebSocketClient: close error: {}", ec.message());
            }
        });
    }

    io_thread_.join();
    connected_.store(false, std::memory_order_release);
    spdlog::info("WebSocketClient: stopped");
}

void WebSocketClient::subscribe(const Symbol& symbol) {
    {
        std::lock_guard lock{subscriptions_mutex_};
        subscriptions_.push_back(symbol);
    }

    // If already connected, send the subscribe message immediately.
    if (connected_.load(std::memory_order_acquire) && ioc_) {
        asio::post(*ioc_, [this, symbol]() {
            send_subscribe(symbol);
        });
    }
}

bool WebSocketClient::is_connected() const noexcept {
    return connected_.load(std::memory_order_acquire);
}

// ── I/O thread entry point ─────────────────────────────────────────────────

void WebSocketClient::run_io_loop() {
    spdlog::info("WebSocketClient: I/O thread started");

    constexpr int    kMaxRetries    = 10;
    constexpr auto   kRetryBaseMs   = std::chrono::milliseconds{500};
    constexpr double kBackoffFactor = 2.0;

    int  retry         = 0;
    auto retry_delay   = kRetryBaseMs;

    while (!stop_requested_.load(std::memory_order_acquire) && retry <= kMaxRetries) {
        try {
            ioc_     = std::make_unique<asio::io_context>(1);
            ssl_ctx_ = std::make_unique<ssl::context>(ssl::context::tlsv12_client);
            ssl_ctx_->set_default_verify_paths();

            ws_ = std::make_unique<websocket::stream<beast::ssl_stream<beast::tcp_stream>>>(
                asio::make_strand(*ioc_),
                *ssl_ctx_);

            do_connect();

            // Run until the I/O context runs out of work or is stopped.
            ioc_->run();

        } catch (const beast::system_error& ex) {
            spdlog::error("WebSocketClient: beast error: {}", ex.what());
        } catch (const std::exception& ex) {
            spdlog::error("WebSocketClient: error: {}", ex.what());
        }

        connected_.store(false, std::memory_order_release);

        if (stop_requested_.load(std::memory_order_acquire)) {
            break;
        }

        ++retry;
        spdlog::warn("WebSocketClient: reconnecting in {}ms (attempt {}/{})",
                     retry_delay.count(), retry, kMaxRetries);

        // Interruptible sleep: check stop every 50 ms.
        const auto deadline = std::chrono::steady_clock::now() + retry_delay;
        while (!stop_requested_.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds{50});
        }

        retry_delay = std::chrono::milliseconds{
            static_cast<long>(static_cast<double>(retry_delay.count()) * kBackoffFactor)};
    }

    spdlog::info("WebSocketClient: I/O thread exiting");
}

// ── Connection ─────────────────────────────────────────────────────────────

void WebSocketClient::do_connect() {
    // DNS resolution
    tcp::resolver resolver{*ioc_};
    const auto endpoints = resolver.resolve(host_, port_);
    spdlog::debug("WebSocketClient: resolved {} to {} endpoint(s)",
                  host_, std::distance(endpoints.begin(), endpoints.end()));

    // TCP connect
    auto& tcp_layer = beast::get_lowest_layer(*ws_);
    tcp_layer.connect(endpoints);
    tcp_layer.expires_never(); // hand over timeout responsibility to WS layer

    // TLS handshake
    ws_->next_layer().handshake(ssl::stream_base::client);

    // WebSocket handshake
    ws_->set_option(websocket::stream_base::decorator([](websocket::request_type& req) {
        req.set(http::field::user_agent, "contrarian-options-engine/0.1");
    }));
    ws_->handshake(host_, path_);

    connected_.store(true, std::memory_order_release);
    spdlog::info("WebSocketClient: connected to {}{}", host_, path_);

    // Send subscribe messages for all registered symbols.
    send_subscribe_all();

    // Begin the async read loop.
    do_read();
}

// ── Subscription helpers ───────────────────────────────────────────────────

void WebSocketClient::send_subscribe_all() {
    std::lock_guard lock{subscriptions_mutex_};
    for (const auto& sym : subscriptions_) {
        // Build and synchronously send each subscribe frame.
        // We are already on the I/O thread here.
        const json payload = {
            {"action", "subscribe"},
            {"params", sym}
        };
        const std::string text = payload.dump();
        beast::error_code ec;
        ws_->write(asio::buffer(text), ec);
        if (ec) {
            spdlog::error("WebSocketClient: subscribe write error for {}: {}", sym, ec.message());
        } else {
            spdlog::debug("WebSocketClient: subscribed to {}", sym);
        }
    }
}

void WebSocketClient::send_subscribe(const Symbol& symbol) {
    if (!connected_.load(std::memory_order_acquire)) {
        return;
    }
    const json payload = {
        {"action", "subscribe"},
        {"params", symbol}
    };
    const std::string text = payload.dump();
    beast::error_code ec;
    ws_->write(asio::buffer(text), ec);
    if (ec) {
        spdlog::error("WebSocketClient: subscribe write error for {}: {}", symbol, ec.message());
    } else {
        spdlog::debug("WebSocketClient: subscribed to {}", symbol);
    }
}

// ── Async read loop ────────────────────────────────────────────────────────

void WebSocketClient::do_read() {
    read_buf_.clear();

    ws_->async_read(read_buf_,
        [this](beast::error_code ec, std::size_t /*bytes_transferred*/) {
            if (ec) {
                if (ec == websocket::error::closed) {
                    spdlog::info("WebSocketClient: server closed connection");
                } else {
                    spdlog::error("WebSocketClient: read error: {}", ec.message());
                }
                connected_.store(false, std::memory_order_release);
                ioc_->stop();
                return;
            }

            on_message(beast::buffers_to_string(read_buf_.data()));
            read_buf_.consume(read_buf_.size());

            // Check stop token before scheduling the next read.
            // The stop flag is not checked inside async handlers, but the io_context
            // will have been stopped by WebSocketClient::stop() via asio::post.
            do_read();
        });
}

// ── Message dispatch ───────────────────────────────────────────────────────

void WebSocketClient::on_message(const std::string& raw) {
    auto result = parse_message(raw);
    if (!coe::common::is_ok(result)) {
        const auto code = coe::common::get_error(result);
        spdlog::warn("WebSocketClient: parse failure ({}) for: {}",
                     coe::common::toString(code), raw);
        return;
    }

    if (!ring_.try_push(coe::common::get_value(result))) {
        spdlog::warn("WebSocketClient: ring buffer full, dropping message");
    }
}

// ── parse_message ──────────────────────────────────────────────────────────

Result<MarketMessage> WebSocketClient::parse_message(const std::string& raw) const {
    using coe::common::ErrorCode;

    json doc;
    try {
        doc = json::parse(raw);
    } catch (const json::parse_error& ex) {
        spdlog::debug("WebSocketClient: JSON parse error: {}", ex.what());
        return ErrorCode::WebSocketError;
    }

    if (!doc.is_object()) {
        spdlog::debug("WebSocketClient: message is not a JSON object");
        return ErrorCode::WebSocketError;
    }

    const auto type_it = doc.find("type");
    if (type_it == doc.end() || !type_it->is_string()) {
        spdlog::debug("WebSocketClient: missing or non-string 'type' field");
        return ErrorCode::InvalidParameter;
    }

    const std::string msg_type = type_it->get<std::string>();

    // ── Quote ──────────────────────────────────────────────────────────────
    if (msg_type == "quote" || msg_type == "Q") {
        Quote q;
        try {
            q.symbol   = doc.at("sym").get<std::string>();
            q.bid      = doc.at("bp").get<double>();
            q.ask      = doc.at("ap").get<double>();
            q.bid_size = doc.at("bs").get<int32_t>();
            q.ask_size = doc.at("as").get<int32_t>();

            // Timestamps are nanoseconds-since-epoch integers.
            const int64_t ns = doc.at("t").get<int64_t>();
            q.ts = std::chrono::nanoseconds{ns};
        } catch (const json::exception& ex) {
            spdlog::debug("WebSocketClient: Quote field error: {}", ex.what());
            return ErrorCode::WebSocketError;
        }
        return MarketMessage{std::in_place_type<Quote>, q};
    }

    // ── Trade ──────────────────────────────────────────────────────────────
    if (msg_type == "trade" || msg_type == "T") {
        Trade t;
        try {
            t.symbol = doc.at("sym").get<std::string>();
            t.price  = doc.at("p").get<double>();
            t.volume = doc.at("s").get<int32_t>();

            const int64_t ns = doc.at("t").get<int64_t>();
            t.ts = std::chrono::nanoseconds{ns};
        } catch (const json::exception& ex) {
            spdlog::debug("WebSocketClient: Trade field error: {}", ex.what());
            return ErrorCode::WebSocketError;
        }
        return MarketMessage{std::in_place_type<Trade>, t};
    }

    // ── OptionsQuote ───────────────────────────────────────────────────────
    if (msg_type == "options_quote" || msg_type == "OQ") {
        OptionsQuote oq;
        try {
            oq.symbol     = doc.at("sym").get<std::string>();
            oq.underlying = doc.at("underlying").get<std::string>();
            oq.strike     = doc.at("strike").get<double>();
            oq.dte        = doc.at("dte").get<int32_t>();
            oq.bid        = doc.at("bp").get<double>();
            oq.ask        = doc.at("ap").get<double>();
            oq.delta      = doc.at("delta").get<double>();
            oq.gamma      = doc.at("gamma").get<double>();
            oq.theta      = doc.at("theta").get<double>();
            oq.vega       = doc.at("vega").get<double>();
            oq.iv         = doc.at("iv").get<double>();

            const std::string opt_type = doc.at("option_type").get<std::string>();
            if (opt_type == "call" || opt_type == "C") {
                oq.type = coe::common::OptionType::Call;
            } else if (opt_type == "put" || opt_type == "P") {
                oq.type = coe::common::OptionType::Put;
            } else {
                spdlog::debug("WebSocketClient: unknown option_type '{}'", opt_type);
                return ErrorCode::InvalidParameter;
            }

            const int64_t ns = doc.at("t").get<int64_t>();
            oq.ts = std::chrono::nanoseconds{ns};
        } catch (const json::exception& ex) {
            spdlog::debug("WebSocketClient: OptionsQuote field error: {}", ex.what());
            return ErrorCode::WebSocketError;
        }
        return MarketMessage{std::in_place_type<OptionsQuote>, oq};
    }

    spdlog::debug("WebSocketClient: unrecognised message type '{}'", msg_type);
    return ErrorCode::InvalidParameter;
}

} // namespace coe::md
