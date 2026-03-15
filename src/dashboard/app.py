"""Flask + Flask-SocketIO dashboard app for the Contrarian Options Alpha Engine."""
from __future__ import annotations

import logging
from typing import Any

from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO

logger = logging.getLogger(__name__)

socketio = SocketIO()

# In-memory state updated by the trading engine via the public helper functions below.
_state: dict[str, Any] = {
    "positions": [],
    "trades": [],
    "equity_curve": [],
    "signals": [],
    "metrics": {
        "daily_pnl": 0.0,
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "trade_count": 0,
        "open_positions": 0,
    },
}

_MAX_SIGNALS = 100


def create_app(config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config: Optional mapping of Flask config values to override defaults.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = "dev-only-key"
    if config:
        app.config.update(config)

    socketio.init_app(app, cors_allowed_origins="*", async_mode="threading")

    # ------------------------------------------------------------------ routes

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/api/positions")
    def api_positions():  # type: ignore[return]
        """Return current open positions as JSON."""
        return jsonify(_state["positions"])

    @app.route("/api/trades")
    def api_trades():  # type: ignore[return]
        """Return all recorded trades as JSON."""
        return jsonify(_state["trades"])

    @app.route("/api/equity")
    def api_equity():  # type: ignore[return]
        """Return equity-curve data points as JSON."""
        return jsonify(_state["equity_curve"])

    @app.route("/api/signals")
    def api_signals():  # type: ignore[return]
        """Return recent signals (capped at 100) as JSON."""
        return jsonify(_state["signals"])

    @app.route("/api/metrics")
    def api_metrics():  # type: ignore[return]
        """Return aggregate performance metrics as JSON."""
        return jsonify(_state["metrics"])

    return app


# --------------------------------------------------------------- state helpers


def update_positions(positions: list[dict[str, Any]]) -> None:
    """Replace the full positions list and broadcast the update.

    Args:
        positions: List of position dicts from the trading engine.
    """
    _state["positions"] = positions
    socketio.emit("positions_update", positions)
    logger.debug("positions_update emitted: %d positions", len(positions))


def add_trade(trade: dict[str, Any]) -> None:
    """Append a completed trade and broadcast the update.

    Args:
        trade: Trade record dict (symbol, side, qty, pnl, timestamp, …).
    """
    _state["trades"].append(trade)
    socketio.emit("trade_update", trade)
    logger.debug("trade_update emitted: %s", trade.get("symbol"))


def update_equity(point: dict[str, Any]) -> None:
    """Append an equity-curve data point and broadcast the update.

    Args:
        point: Dict with at minimum ``{"timestamp": ..., "equity": float}``.
    """
    _state["equity_curve"].append(point)
    socketio.emit("equity_update", point)


def add_signal(signal: dict[str, Any]) -> None:
    """Append a new signal, enforce the rolling window cap, and broadcast.

    Args:
        signal: Signal dict (symbol, type, score, timestamp, …).
    """
    _state["signals"].append(signal)
    if len(_state["signals"]) > _MAX_SIGNALS:
        _state["signals"] = _state["signals"][-_MAX_SIGNALS:]
    socketio.emit("signal_update", signal)
    logger.debug("signal_update emitted: %s", signal.get("type"))


def update_metrics(metrics: dict[str, Any]) -> None:
    """Merge new metric values into the running state and broadcast.

    Args:
        metrics: Partial or full metrics dict to merge. Unknown keys are
            accepted so callers can extend the metrics schema freely.
    """
    _state["metrics"].update(metrics)
    socketio.emit("metrics_update", _state["metrics"])
    logger.debug("metrics_update emitted")


def run_dashboard(
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
) -> None:
    """Create the app and start the SocketIO development server.

    Args:
        host: Interface to bind (default ``0.0.0.0`` = all interfaces).
        port: TCP port to listen on.
        debug: Enable Flask debug / auto-reload mode.
    """
    app = create_app()
    logger.info("Starting dashboard on %s:%d (debug=%s)", host, port, debug)
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
