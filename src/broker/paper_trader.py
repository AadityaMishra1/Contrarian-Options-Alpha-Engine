"""Main paper-trading orchestrator for the Contrarian Options Alpha Engine.

Wires together IBKR connectivity, signal generation, risk management, and
order execution into a single async event loop that runs during regular
market hours.  All C++ engine components are imported with a graceful
fallback to pure-Python equivalents so the module is always importable.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from .connection import IBKRConnection, IBKRConnectionError
from .order_bridge import IBKROrderBridge, OrderType, Side
from .reconciliation import PositionReconciler

# ---------------------------------------------------------------------------
# Optional C++ engine import
# ---------------------------------------------------------------------------

try:
    from coe_engine import (  # type: ignore[import]
        Config,
        KellyPositionSizer,
        OrderManager,
        PositionTracker,
        RiskLimits,
        RiskManager,
    )

    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    OrderManager = None  # type: ignore[assignment,misc]
    PositionTracker = None  # type: ignore[assignment,misc]
    RiskManager = None  # type: ignore[assignment,misc]
    KellyPositionSizer = None  # type: ignore[assignment,misc]
    RiskLimits = None  # type: ignore[assignment,misc]
    Config = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Optional signal-layer imports (may not exist yet)
# ---------------------------------------------------------------------------

try:
    from src.signals.options_chain import OptionsChainAnalyzer  # type: ignore[import]
    from src.signals.screener import OptionsScreener  # type: ignore[import]
    from src.signals.sentiment import SentimentFilter  # type: ignore[import]
    from src.signals.technicals import TechnicalAnalyzer  # type: ignore[import]

    HAS_SIGNALS = True
except ImportError:
    HAS_SIGNALS = False
    OptionsScreener = None  # type: ignore[assignment,misc]
    SentimentFilter = None  # type: ignore[assignment,misc]
    TechnicalAnalyzer = None  # type: ignore[assignment,misc]
    OptionsChainAnalyzer = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_MARKET_OPEN = dtime(9, 30)
_MARKET_CLOSE = dtime(16, 0)


# ---------------------------------------------------------------------------
# Fallback pure-Python tracking structures
# ---------------------------------------------------------------------------

@dataclass
class _PythonPosition:
    """Minimal position record used when the C++ engine is absent.

    Attributes:
        symbol: Contract symbol string.
        quantity: Number of contracts held.
        avg_cost: Average entry cost per contract.
        entry_rsi: RSI value at entry (used for exit checks).
        dte: Days-to-expiry remaining at entry.
        stop_loss_threshold: P&L level (fraction) that triggers a stop.
    """

    symbol: str
    quantity: int
    avg_cost: float
    entry_rsi: float = 0.0
    dte: int = 3
    stop_loss_threshold: float = -0.50
    bars_held: int = 0


@dataclass
class _PythonState:
    """Aggregate paper-trading state when the C++ engine is absent.

    Attributes:
        capital: Current available cash.
        positions: Open positions indexed by symbol.
        trades: Closed trade records.
        daily_pnl: Running P&L for today.
    """

    capital: float = 10_000.0
    positions: dict[str, _PythonPosition] = field(default_factory=dict)
    trades: list[dict] = field(default_factory=list)
    daily_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Default paper trading configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "ibkr": {
        "host": "127.0.0.1",
        "port": 7497,  # TWS paper trading
        "client_id": 1,
    },
    "trading": {
        "scan_interval": 60,       # seconds between full screening passes
        "exit_check_interval": 30,  # seconds between exit condition checks
        "reconcile_interval": 300,  # seconds between position reconciliations
        "initial_capital": 10_000.0,
        "max_positions": 5,
        "rsi_exit": 50.0,
        "stop_loss_pct": -0.50,
        "dte_min": 1,
        "dte_max": 5,
        "option_min_price": 0.05,
        "option_max_price": 0.30,
    },
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PaperTradingOrchestrator:
    """Async main loop for live paper trading via IBKR.

    Args:
        config_path: Path to a YAML configuration file.  Missing keys fall
                     back to :data:`_DEFAULT_CONFIG`.

    The orchestrator handles:

    * IBKR connection lifecycle.
    * Signal generation (screen → sentiment → technicals → options chain).
    * Risk gating (RiskManager / Kelly sizing).
    * Order submission and fill tracking.
    * Exit monitoring (RSI cross, DTE expiry, stop-loss).
    * Periodic position reconciliation.
    * Daily summary logging on shutdown.
    """

    def __init__(self, config_path: str = "config/paper_trading.yaml") -> None:
        self._config = self._load_config(config_path)
        self._running = False
        self._shutdown_event = asyncio.Event()

        # IBKR layer
        ibkr_cfg = self._config["ibkr"]
        self._connection = IBKRConnection(
            host=ibkr_cfg["host"],
            port=ibkr_cfg["port"],
            client_id=ibkr_cfg["client_id"],
        )

        # These are initialised inside run() once we have a live loop
        self._order_bridge: IBKROrderBridge | None = None
        self._reconciler: PositionReconciler | None = None

        # C++ engine components (None when HAS_ENGINE is False)
        self._order_manager: Any | None = None
        self._position_tracker: Any | None = None
        self._risk_manager: Any | None = None
        self._kelly_sizer: Any | None = None

        # Signal layer components
        self._screener: Any | None = None
        self._sentiment: Any | None = None
        self._technicals: Any | None = None
        self._options_chain: Any | None = None

        # Pure-Python fallback state
        self._state = _PythonState(
            capital=self._config["trading"]["initial_capital"]
        )

        self._init_engine_components()
        self._init_signal_components()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Execute the main paper-trading loop.

        Connects to IBKR, reconciles positions, then cycles through
        signal generation and exit checks until shutdown is requested.
        """
        self._install_signal_handlers()

        try:
            async with self._connection:
                logger.info("Connected to IBKR paper trading.")

                self._order_bridge = IBKROrderBridge(
                    self._connection,
                    order_manager=self._order_manager,
                    position_tracker=self._position_tracker,
                )
                self._reconciler = PositionReconciler(
                    self._connection,
                    position_tracker=self._position_tracker,
                    check_interval=self._config["trading"]["reconcile_interval"],
                )

                # Startup reconciliation
                discrepancies = await self._reconciler.startup_sync()
                if discrepancies:
                    logger.warning(
                        "Startup reconciliation found %d discrepancy(ies).",
                        len(discrepancies),
                    )

                await self._reconciler.start()
                self._running = True

                scan_interval = self._config["trading"]["scan_interval"]
                exit_interval = self._config["trading"]["exit_check_interval"]

                scan_task = asyncio.ensure_future(self._scan_loop(scan_interval))
                exit_task = asyncio.ensure_future(self._exit_loop(exit_interval))

                await self._shutdown_event.wait()

                logger.info("Shutdown requested. Cancelling open orders.")
                scan_task.cancel()
                exit_task.cancel()
                await asyncio.gather(scan_task, exit_task, return_exceptions=True)

                await self._cancel_all_open_orders()
                await self._reconciler.stop()

                summary = await self.daily_summary()
                logger.info("Daily summary: %s", summary)

        except IBKRConnectionError as exc:
            logger.critical("Fatal IBKR connection error: %s", exc)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Scan loop
    # ------------------------------------------------------------------

    async def _scan_loop(self, interval: int) -> None:
        """Periodically screen for entry candidates and submit orders.

        Args:
            interval: Seconds between full screening passes.
        """
        while True:
            await asyncio.sleep(interval)
            if not self.is_market_hours():
                logger.debug("Outside market hours — skipping scan.")
                continue
            try:
                await self._process_entries()
            except Exception as exc:
                logger.error("Entry scan error: %s", exc)

    async def _process_entries(self) -> None:
        """Run the full entry pipeline for one scan cycle."""
        if self._screener is None:
            logger.debug("No screener configured — skipping entry scan.")
            return

        trading_cfg = self._config["trading"]
        max_positions = trading_cfg["max_positions"]

        # Guard: don't open more positions than the limit
        current_count = self._open_position_count()
        if current_count >= max_positions:
            logger.info(
                "Max positions reached (%d/%d). Skipping scan.",
                current_count,
                max_positions,
            )
            return

        candidates = await self._screener.scan()
        logger.info("Screener returned %d candidate(s).", len(candidates))

        for candidate in candidates:
            symbol = getattr(candidate, "symbol", str(candidate))

            # 1. Sentiment gate
            if self._sentiment is not None:
                headlines = getattr(candidate, "headlines", [])
                sentiment = await self._sentiment.classify(symbol, headlines)
                if str(sentiment).upper() == "FUNDAMENTAL_PROBLEM":
                    logger.info("Skipping %s: FUNDAMENTAL_PROBLEM sentiment.", symbol)
                    continue

            # 2. Technical signal
            if self._technicals is not None:
                bar = getattr(candidate, "latest_bar", None)
                if bar is not None:
                    self._technicals.feed_bar(bar)
                signal = self._technicals.evaluate()
                if not signal:
                    logger.debug("Skipping %s: no technical signal.", symbol)
                    continue

            # 3. Options chain selection
            option_candidate = candidate
            if self._options_chain is not None:
                chain = await self._options_chain.get_chain(symbol)
                if chain is None:
                    logger.debug("Skipping %s: no options chain available.", symbol)
                    continue
                option_candidate = chain

            # 4. Risk gate
            if self._risk_manager is not None:
                approved = self._risk_manager.check_new_order(option_candidate)
                if not approved:
                    logger.info("Skipping %s: risk manager rejected.", symbol)
                    continue

            # 5. Position sizing
            contracts = 1
            if self._kelly_sizer is not None:
                contracts = self._kelly_sizer.calculate_size(option_candidate)
                contracts = max(1, int(contracts))

            # 6. Submit order
            await self._submit_entry(symbol, option_candidate, contracts)

            # Refresh count after submission
            if self._open_position_count() >= max_positions:
                break

    async def _submit_entry(
        self, symbol: str, option_candidate: Any, contracts: int
    ) -> None:
        """Build and submit a buy order for an option candidate.

        Args:
            symbol: Underlying ticker symbol.
            option_candidate: Object with option contract attributes.
            contracts: Number of contracts to buy.
        """
        if self._order_bridge is None:
            logger.warning("Order bridge not initialised — cannot submit entry for %s.", symbol)
            return

        order = _SimpleOrder(
            order_id=self._next_order_id(),
            symbol=symbol,
            expiry=getattr(option_candidate, "expiry", ""),
            strike=getattr(option_candidate, "strike", 0.0),
            right=getattr(option_candidate, "right", "P"),
            side=Side.Buy,
            order_type=OrderType.Market,
            quantity=contracts,
        )

        try:
            ibkr_id = await self._order_bridge.submit_order(order)
            logger.info(
                "Entry order submitted: symbol=%s contracts=%d ibkr_id=%s",
                symbol,
                contracts,
                ibkr_id,
            )

            if not HAS_ENGINE:
                # Track locally
                cost = float(getattr(option_candidate, "price", 0.10))
                self._state.positions[symbol] = _PythonPosition(
                    symbol=symbol,
                    quantity=contracts,
                    avg_cost=cost,
                    dte=int(getattr(option_candidate, "dte", 3)),
                )
                self._state.capital -= contracts * cost * 100

        except Exception as exc:
            logger.error("Failed to submit entry for %s: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Exit loop
    # ------------------------------------------------------------------

    async def _exit_loop(self, interval: int) -> None:
        """Periodically check exit conditions for all open positions.

        Args:
            interval: Seconds between exit checks.
        """
        while True:
            await asyncio.sleep(interval)
            if not self.is_market_hours():
                continue
            try:
                await self.check_exits()
            except Exception as exc:
                logger.error("Exit check error: %s", exc)

    async def check_exits(self) -> None:
        """Evaluate exit conditions for every open position.

        Exit triggers (first match wins):
          * RSI rises above the configured exit threshold.
          * DTE has reached zero (option expiry).
          * Stop-loss: current P&L fraction <= ``stop_loss_pct``.
        """
        if self._order_bridge is None:
            return

        rsi_exit = float(self._config["trading"]["rsi_exit"])
        stop_loss = float(self._config["trading"]["stop_loss_pct"])

        if HAS_ENGINE and self._position_tracker is not None:
            # Delegate to C++ tracker
            try:
                positions = self._position_tracker.get_all_positions()
                for pos in positions:
                    await self._evaluate_exit(pos, rsi_exit, stop_loss)
            except Exception as exc:
                logger.error("C++ position exit check error: %s", exc)
        else:
            # Pure-Python fallback
            symbols_to_exit: list[str] = []
            for symbol, pos in self._state.positions.items():
                pos.bars_held += 1
                exit_reason = self._check_exit_conditions_py(
                    pos, rsi_exit, stop_loss
                )
                if exit_reason:
                    symbols_to_exit.append(symbol)
                    logger.info(
                        "Exit triggered for %s: reason=%s", symbol, exit_reason
                    )

            for symbol in symbols_to_exit:
                await self._submit_exit(symbol)

    async def _evaluate_exit(
        self, pos: Any, rsi_exit: float, stop_loss: float
    ) -> None:
        """Evaluate a single C++ position for exit conditions.

        Args:
            pos: C++ position object from the PositionTracker.
            rsi_exit: RSI level above which we close the position.
            stop_loss: Fractional P&L floor (e.g. -0.50).
        """
        symbol = getattr(pos, "symbol", "")
        current_rsi = float(getattr(pos, "current_rsi", 0.0))
        dte = int(getattr(pos, "dte_remaining", 1))
        pnl_pct = float(getattr(pos, "unrealised_pnl_pct", 0.0))

        exit_reason: str | None = None
        if current_rsi > rsi_exit:
            exit_reason = "rsi_cross"
        elif dte <= 0:
            exit_reason = "dte_expire"
        elif pnl_pct <= stop_loss:
            exit_reason = "stop_loss"

        if exit_reason:
            logger.info("Exit triggered for %s: reason=%s", symbol, exit_reason)
            await self._submit_exit(symbol)

    def _check_exit_conditions_py(
        self,
        pos: _PythonPosition,
        rsi_exit: float,
        stop_loss: float,
    ) -> str | None:
        """Determine whether a Python-tracked position should be closed.

        Args:
            pos: The :class:`_PythonPosition` to evaluate.
            rsi_exit: RSI threshold for exit.
            stop_loss: P&L fraction threshold for stop-loss.

        Returns:
            A string reason ("rsi_cross", "dte_expire", "stop_loss")
            or None if no exit condition is met.
        """
        # DTE expiry: use bars_held as a proxy for time passing
        if pos.bars_held >= pos.dte:
            return "dte_expire"

        # Approximate current value as 50 % of entry (conservative mid)
        current_value = pos.avg_cost * 0.5
        pnl_pct = (current_value - pos.avg_cost) / max(pos.avg_cost, 1e-9)
        if pnl_pct <= stop_loss:
            return "stop_loss"

        return None

    async def _submit_exit(self, symbol: str) -> None:
        """Submit a market sell order to close a position.

        Args:
            symbol: The symbol of the position to close.
        """
        if self._order_bridge is None:
            return

        qty = 1
        if not HAS_ENGINE:
            pos = self._state.positions.pop(symbol, None)
            if pos is None:
                return
            qty = pos.quantity

        order = _SimpleOrder(
            order_id=self._next_order_id(),
            symbol=symbol,
            expiry="",
            strike=0.0,
            right="P",
            side=Side.Sell,
            order_type=OrderType.Market,
            quantity=qty,
        )

        try:
            await self._order_bridge.submit_order(order)
            logger.info("Exit order submitted for %s (qty=%d).", symbol, qty)
        except Exception as exc:
            logger.error("Failed to submit exit for %s: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    async def daily_summary(self) -> dict:
        """Compute and return the day's trading performance.

        Returns:
            Dictionary with keys: total_trades, winning_trades,
            losing_trades, win_rate, total_pnl, capital.
        """
        trades = self._state.trades
        total = len(trades)
        wins = sum(1 for t in trades if t.get("pnl", 0.0) > 0)
        losses = total - wins
        total_pnl = sum(t.get("pnl", 0.0) for t in trades)
        win_rate = wins / total if total else 0.0

        summary = {
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 4),
            "capital": round(self._state.capital, 2),
        }
        logger.info("Daily summary: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Market hours
    # ------------------------------------------------------------------

    def is_market_hours(self) -> bool:
        """Return True if the current ET time is within regular market hours.

        Market hours are defined as 09:30–16:00 US/Eastern, Monday–Friday.

        Returns:
            True during 09:30–16:00 ET on weekdays, False otherwise.
        """
        now_et = datetime.now(_ET)
        if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        current_time = now_et.time().replace(tzinfo=None)
        return _MARKET_OPEN <= current_time <= _MARKET_CLOSE

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_engine_components(self) -> None:
        """Initialise C++ engine objects if the extension is available."""
        if not HAS_ENGINE:
            logger.warning(
                "coe_engine C++ module not found. Using Python fallback tracking."
            )
            return

        try:
            risk_limits = RiskLimits(
                daily_loss_limit=self._config.get("risk", {}).get("daily_loss_limit", -50.0),
                max_positions=self._config["trading"]["max_positions"],
                max_single_position_pct=self._config.get("risk", {}).get(
                    "max_single_position", 20.0
                ),
            )
            engine_config = Config(
                kelly_fraction=self._config.get("sizing", {}).get("kelly_fraction", 0.5),
                max_bet_pct=self._config.get("sizing", {}).get("max_bet", 20.0),
            )
            self._order_manager = OrderManager()
            self._position_tracker = PositionTracker()
            self._risk_manager = RiskManager(risk_limits)
            self._kelly_sizer = KellyPositionSizer(engine_config)
            logger.info("C++ engine components initialised.")
        except Exception as exc:
            logger.error("Failed to initialise C++ engine: %s. Using fallback.", exc)

    def _init_signal_components(self) -> None:
        """Initialise signal-layer components if they are importable."""
        if not HAS_SIGNALS:
            logger.warning(
                "Signal modules not found (src.signals.*). Scanning will be skipped."
            )
            return

        try:
            self._screener = OptionsScreener(self._config)
            self._sentiment = SentimentFilter(self._config)
            self._technicals = TechnicalAnalyzer(self._config)
            self._options_chain = OptionsChainAnalyzer(self._config)
            logger.info("Signal layer components initialised.")
        except Exception as exc:
            logger.error("Failed to initialise signal components: %s", exc)

    def _open_position_count(self) -> int:
        """Return the number of currently open positions."""
        if HAS_ENGINE and self._position_tracker is not None:
            try:
                return len(self._position_tracker.get_all_positions())
            except Exception:
                pass
        return len(self._state.positions)

    _order_counter: int = 0

    def _next_order_id(self) -> int:
        """Return a monotonically increasing local order ID."""
        PaperTradingOrchestrator._order_counter += 1
        return PaperTradingOrchestrator._order_counter

    async def _cancel_all_open_orders(self) -> None:
        """Cancel every tracked open order via the order bridge."""
        if self._order_bridge is None:
            return
        for order_id in list(self._order_bridge._order_map.keys()):
            try:
                await self._order_bridge.cancel_order(order_id)
            except Exception as exc:
                logger.error("Error cancelling order %d: %s", order_id, exc)

    def _install_signal_handlers(self) -> None:
        """Register SIGINT / SIGTERM handlers for graceful shutdown."""

        def _handle(sig, frame) -> None:
            logger.info("Signal %s received. Initiating shutdown.", sig)
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load YAML configuration with fallback to defaults.

        Args:
            config_path: Path to the paper trading YAML config file.

        Returns:
            Merged configuration dictionary.
        """
        cfg: dict = {}
        path = Path(config_path)
        if path.exists():
            try:
                with path.open() as fh:
                    loaded = yaml.safe_load(fh) or {}
                cfg = loaded
                logger.info("Loaded paper trading config from %s.", config_path)
            except Exception as exc:
                logger.warning(
                    "Could not load config from %s: %s. Using defaults.", config_path, exc
                )
        else:
            logger.info(
                "Config file %s not found. Using default configuration.", config_path
            )

        # Deep-merge defaults (defaults fill in missing keys only)
        merged = _deep_merge(_DEFAULT_CONFIG, cfg)
        return merged


# ---------------------------------------------------------------------------
# Thin order struct used internally (mirrors C++ Order interface)
# ---------------------------------------------------------------------------

@dataclass
class _SimpleOrder:
    """Lightweight order object mirroring the C++ engine's Order struct.

    Attributes:
        order_id: Monotonic local identifier.
        symbol: Underlying ticker.
        expiry: Option expiry date as YYYYMMDD string.
        strike: Option strike price.
        right: "C" for call, "P" for put.
        side: :data:`Side.Buy` or :data:`Side.Sell`.
        order_type: :data:`OrderType.Market` or :data:`OrderType.Limit`.
        quantity: Number of contracts.
        limit_price: Limit price, used only when order_type is Limit.
    """

    order_id: int
    symbol: str
    expiry: str
    strike: float
    right: str
    side: str
    order_type: str
    quantity: int
    limit_price: float = 0.0


# ---------------------------------------------------------------------------
# Config merge helper
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Values present in ``override`` always win. Nested dicts are merged
    recursively rather than replaced wholesale.

    Args:
        base: The default/fallback dictionary.
        override: User-supplied values that take precedence.

    Returns:
        A new merged dictionary.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
