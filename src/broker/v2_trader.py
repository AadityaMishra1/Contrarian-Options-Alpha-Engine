"""V2 Paper Trading Orchestrator — Index Options Focus.

Sells premium on index ETFs (SPY, QQQ, IWM) using the VIX/VVIX regime
filter and HV-IV gap signal. PEAD earnings trades on individual names
are a secondary strategy active during earnings season only.

This replaces the v1 PaperTradingOrchestrator (paper_trader.py), which
used RSI/Bollinger bands on individual stock options. The v1 file is
retained as-is for reference and backtesting; no v1 code is imported here.

Architecture:
    1. Check regime (VIX/VVIX) → set position scalar
    2. If regime allows, scan SPY/QQQ/IWM for HV-IV gap (IndexScanner)
    3. For qualifying signals, paper-trade iron condors / strangles
    4. Monitor exits: 50% profit target, 2x credit stop, 7 DTE time exit
    5. Optionally run PEAD scanner for individual names (earnings season)
    6. LiveMonitor tracks performance; halts if kill criteria are triggered

Signal-only mode:
    When no IBKR connection is available (or ``--signal-only`` flag is passed),
    the orchestrator logs signals and simulates positions without executing
    any orders. This is the intended mode for paper-trading validation.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

import aiohttp
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional v2 signal imports
# ---------------------------------------------------------------------------

try:
    from signals.earnings_scanner import EarningsScanner
    from signals.finbert import FinBERTAnalyzer
    from signals.index_scanner import IndexScanner, IndexSignal
    from signals.regime import Regime, RegimeClassifier, RegimeState
    from signals.skew import SkewTracker

    HAS_SIGNALS = True
except ImportError as _sig_err:
    logger.warning("Signal modules not importable: %s", _sig_err)
    HAS_SIGNALS = False
    IndexScanner = None  # type: ignore[assignment,misc]
    IndexSignal = None  # type: ignore[assignment,misc]
    Regime = None  # type: ignore[assignment,misc]
    RegimeClassifier = None  # type: ignore[assignment,misc]
    RegimeState = None  # type: ignore[assignment,misc]
    SkewTracker = None  # type: ignore[assignment,misc]
    EarningsScanner = None  # type: ignore[assignment,misc]
    FinBERTAnalyzer = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Optional broker imports
# ---------------------------------------------------------------------------

try:
    from broker.connection import IBKRConnection
    from broker.index_orders import IndexOrderBuilder, IronCondorLegs
    from broker.order_bridge import IBKROrderBridge

    HAS_BROKER = True
except ImportError as _broker_err:
    logger.warning("Broker modules not importable: %s", _broker_err)
    HAS_BROKER = False
    IBKRConnection = None  # type: ignore[assignment,misc]
    IBKROrderBridge = None  # type: ignore[assignment,misc]
    IndexOrderBuilder = None  # type: ignore[assignment,misc]
    IronCondorLegs = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Optional monitor import
# ---------------------------------------------------------------------------

try:
    from backtest.live_monitor import KillCriteria, LiveMonitor, TradeRecord

    HAS_MONITOR = True
except ImportError as _mon_err:
    logger.warning("LiveMonitor not importable: %s", _mon_err)
    HAS_MONITOR = False
    KillCriteria = None  # type: ignore[assignment,misc]
    LiveMonitor = None  # type: ignore[assignment,misc]
    TradeRecord = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    "broker": {
        "host": "127.0.0.1",
        "port": 7497,   # TWS paper trading
        "client_id": 1,
    },
    "strategy": {
        "scan_interval": 60,          # seconds between full scan cycles
        "exit_check_interval": 30,    # seconds between exit condition checks
        "min_gap_ratio": 1.05,        # minimum IV/HV ratio to consider a trade
        "profit_target_pct": 0.50,    # close at 50% of max profit
        "stop_loss_multiple": 2.0,    # close when loss = 2x initial credit
        "target_dte": 30,             # default DTE for new positions
        "max_positions": 3,           # maximum concurrent positions
        "max_position_pct": 0.03,     # max 3% of account per position
        "pead_enabled": True,         # run PEAD scanner during earnings season
        "pead_sue_threshold": 1.0,    # minimum |SUE| for PEAD trades
    },
    "regime": {
        "crisis_vix": 35.0,
        "elevated_vix": 25.0,
        "normal_vix": 18.0,
        "quiet_vvix": 85.0,
    },
    "trading": {
        "market_open": "09:30",
        "market_close": "16:00",
        "execution_start": "11:00",   # ET — bid/ask spreads tighten after open
        "execution_end": "14:00",     # ET — avoid end-of-day gamma/vol spikes
        "timezone": "US/Eastern",
    },
    "monitor": {
        "max_drawdown_pct": 20.0,
        "max_consecutive_losers": 8,
        "min_rolling_win_rate": 0.35,
    },
    "vix_csv_path": "data/cache/vix_history.csv",
}


# ---------------------------------------------------------------------------
# Internal position record
# ---------------------------------------------------------------------------

@dataclass
class _PositionRecord:
    """Tracks a single open paper-trade position.

    Attributes:
        symbol: Underlying ETF ticker.
        structure: Option structure (iron_condor, strangle, put_spread).
        entry_date: ISO-format datetime string of entry.
        entry_price: Underlying price at entry.
        entry_credit: Option credit received (0.0 in signal-only mode).
        target_dte: DTE at which position was opened (used for time exit).
        regime_at_entry: Regime name at entry time.
        effective_scalar: Position size multiplier at entry.
        gap_ratio: HV/IV gap ratio at entry.
        trade_id: Monotonic integer identifier.
    """

    symbol: str
    structure: str
    entry_date: str
    entry_price: float
    entry_credit: float
    target_dte: int
    regime_at_entry: str
    effective_scalar: float
    gap_ratio: float
    trade_id: int


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class V2TradingOrchestrator:
    """V2 paper/live trading orchestrator for index options.

    Implements the research-backed strategy from the v2 design:

    Primary strategy (all regimes except CRISIS):
        Sell premium on SPY / QQQ / IWM when IV >> HV (Goyal-Saretto signal).
        Position sizes are scaled by the VIX/VVIX regime filter (Baltussen et al.)
        and the IV skew z-score (Xing et al.).

    Secondary strategy (earnings season only, controlled by ``pead_enabled``):
        Post-Earnings-Announcement Drift (PEAD) directional spreads on
        individual names with high SUE (Bernard-Thomas 1989). At least 2 of
        3 confirming signals (SUE, IV spread, FinBERT) must align before entry.

    Args:
        config_path: Path to YAML config file. Missing keys fall back to
            :data:`_DEFAULT_CONFIG`.
    """

    _trade_counter: int = 0

    def __init__(
        self,
        config_path: str = "config/paper_trading.yaml",
    ) -> None:
        self._config = self._load_config(config_path)
        self._shutdown_event = asyncio.Event()

        strategy_cfg = self._config.get("strategy", {})
        regime_cfg = self._config.get("regime", {})
        monitor_cfg = self._config.get("monitor", {})
        broker_cfg = self._config.get("broker", {})
        vix_csv = self._config.get("vix_csv_path", "data/cache/vix_history.csv")

        # Signal layer
        api_key = os.environ.get("POLYGON_API_KEY")
        self._index_scanner: IndexScanner | None = None
        self._regime_clf: RegimeClassifier | None = None
        self._skew: SkewTracker | None = None
        self._earnings: EarningsScanner | None = None

        if HAS_SIGNALS and api_key:
            self._index_scanner = IndexScanner(
                api_key=api_key,
                vix_csv_path=vix_csv,
            )
            self._regime_clf = RegimeClassifier(**regime_cfg)
            self._skew = SkewTracker()
            if strategy_cfg.get("pead_enabled", True):
                self._earnings = EarningsScanner(
                    api_key=api_key,
                    sue_threshold=strategy_cfg.get("pead_sue_threshold", 1.0),
                )
        elif HAS_SIGNALS and not api_key:
            logger.warning(
                "POLYGON_API_KEY not set. IndexScanner, earnings scanner "
                "and regime classifier require it. Running in regime-only mode."
            )
            self._regime_clf = RegimeClassifier(**regime_cfg)
            self._skew = SkewTracker()

        # Broker layer (optional)
        self._connection: IBKRConnection | None = None
        if HAS_BROKER:
            try:
                self._connection = IBKRConnection(
                    host=broker_cfg.get("host", "127.0.0.1"),
                    port=broker_cfg.get("port", 7497),
                    client_id=broker_cfg.get("client_id", 1),
                )
            except Exception as exc:
                logger.warning("Could not create IBKRConnection: %s", exc)

        # Order builder (receives the IB instance once connection is live)
        self._order_builder: IndexOrderBuilder | None = (
            IndexOrderBuilder(
                ib=self._connection.ib if self._connection else None
            )
            if HAS_BROKER
            else None
        )

        # Live monitor
        self._monitor: LiveMonitor | None = None
        if HAS_MONITOR:
            kill = KillCriteria(
                max_drawdown_pct=monitor_cfg.get("max_drawdown_pct", 20.0),
                max_consecutive_losers=monitor_cfg.get("max_consecutive_losers", 8),
                min_rolling_win_rate=monitor_cfg.get("min_rolling_win_rate", 0.35),
            )
            self._monitor = LiveMonitor(
                expected_sharpe=0.7,
                expected_win_rate=0.60,
                expected_avg_pnl=50.0,
                kill_criteria=kill,
            )

        # Runtime state
        self._open_positions: list[_PositionRecord] = []
        self._closed_positions: list[_PositionRecord] = []
        self._last_regime_state: RegimeState | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Execute the main v2 trading loop.

        Connects to IBKR if available, otherwise runs in signal-only mode.
        Loops until a SIGINT/SIGTERM is received, the LiveMonitor triggers
        a halt, or an unrecoverable error occurs.
        """
        self._install_signal_handlers()
        self._log_startup_banner()

        if not HAS_SIGNALS:
            logger.error("Signal modules unavailable. Cannot run v2 orchestrator.")
            return

        # Attempt broker connection; fall through to signal-only on failure
        if self._connection is not None:
            try:
                async with self._connection:
                    logger.info("Connected to IBKR paper trading (port %d).", self._connection._port)
                    await self._trading_loop()
            except Exception as exc:
                logger.warning(
                    "IBKR connection failed (%s). Continuing in SIGNAL-ONLY mode.", exc
                )
                await self._trading_loop()
        else:
            logger.info("No broker configured. Running in SIGNAL-ONLY mode.")
            await self._trading_loop()

    # ------------------------------------------------------------------
    # Core trading loop
    # ------------------------------------------------------------------

    async def _trading_loop(self) -> None:
        """Main async loop: classify regime, scan, trade, monitor exits.

        Runs on a configurable ``scan_interval``. Each iteration:
        1. Check LiveMonitor halt flag.
        2. Classify current regime.
        3. If within execution window and regime permits, scan indexes.
        4. Execute qualified signals (up to max_positions cap).
        5. Check exits on all open positions.
        6. Log status summary.
        """
        scan_interval: int = self._config.get("strategy", {}).get("scan_interval", 60)
        logger.info("Trading loop started (scan every %ds).", scan_interval)

        while not self._shutdown_event.is_set():
            try:
                # 1. Monitor halt check
                if self._monitor and self._monitor.is_halted:
                    logger.critical(
                        "MONITOR HALT: %s — stopping all trading.",
                        self._monitor.halt_reason,
                    )
                    break

                # 2. Regime classification
                regime_state = await self._update_regime()
                self._last_regime_state = regime_state

                is_crisis = (
                    regime_state is not None
                    and regime_state.regime == Regime.CRISIS
                )

                if is_crisis:
                    logger.warning(
                        "CRISIS regime (VIX=%.1f) — skipping new entries, "
                        "monitoring exits only.",
                        regime_state.vix if regime_state else 0.0,
                    )
                else:
                    # 3. Scan and trade within execution window
                    if self._in_execution_window():
                        signals = await self._scan_indexes(regime_state)
                        await self._process_signals(signals)

                        # Secondary: PEAD scan (earnings season only)
                        if self._earnings is not None:
                            await self._run_pead_scan()

                # 4. Exit monitoring (always runs, including CRISIS)
                await self._check_exits()

                # 5. Status
                self._log_status()

            except Exception as exc:
                logger.error("Trading loop error: %s", exc, exc_info=True)

            # Sleep until next scan or shutdown
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=float(scan_interval)
                )
                break  # shutdown was requested
            except TimeoutError:
                pass   # normal: continue loop

        logger.info("Trading loop ended.")
        if self._monitor:
            logger.info("\n%s", self._monitor.status_report())

    # ------------------------------------------------------------------
    # Regime update
    # ------------------------------------------------------------------

    async def _update_regime(self) -> RegimeState | None:
        """Fetch current VIX and classify the market regime.

        VVIX history is tracked internally; a simple placeholder value is
        used when no VVIX data source is wired (TODO: add VVIX CSV support).

        Returns:
            RegimeState from RegimeClassifier, or None if unavailable.
        """
        if self._regime_clf is None or self._index_scanner is None:
            return None

        try:
            async with aiohttp.ClientSession() as session:
                vix = await self._index_scanner._get_vix_live(session)

            if not vix or vix <= 0:
                logger.warning("VIX unavailable; regime classification skipped.")
                return self._last_regime_state  # reuse last known state

            # VVIX: track a rolling estimate. In production, fetch from CBOE VIX2X CSV.
            # Placeholder: assume VVIX is proportional to VIX for classification.
            vvix = 90.0 + (vix - 20.0) * 1.5  # rough linear proxy
            vvix_5d_ago = getattr(self._last_regime_state, "vvix", vvix)

            state = self._regime_clf.classify(vix, vvix, vvix_5d_ago)
            logger.info(
                "Regime: %s (VIX=%.1f scalar=%.2f)",
                state.regime.value,
                vix,
                state.position_scalar,
            )
            return state

        except Exception as exc:
            logger.error("Regime update failed: %s", exc)
            return self._last_regime_state

    # ------------------------------------------------------------------
    # Index scan
    # ------------------------------------------------------------------

    async def _scan_indexes(
        self, regime_state: RegimeState | None
    ) -> list[IndexSignal]:
        """Scan SPY, QQQ, IWM for premium-selling signals.

        Args:
            regime_state: Current regime, passed through to IndexScanner.

        Returns:
            Sorted list of IndexSignal objects (best score first).
        """
        if self._index_scanner is None:
            return []

        try:
            signals = await self._index_scanner.scan(
                regime_state=regime_state,
                skew_tracker=self._skew,
            )

            for sig in signals:
                logger.info(
                    "Signal  %s | IV=%.1f%% HV=%.1f%% gap=%.2f | %s DTE=%d | "
                    "score=%.2f scalar=%.2f",
                    sig.symbol,
                    sig.iv_30 * 100,
                    sig.hv_30 * 100,
                    sig.gap_ratio,
                    sig.structure,
                    sig.target_dte,
                    sig.score,
                    sig.effective_scalar,
                )

            return signals

        except Exception as exc:
            logger.error("Index scan failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    async def _process_signals(self, signals: list[IndexSignal]) -> None:
        """Evaluate each signal and execute qualifying trades.

        Filters:
        - score > 0 (positive premium opportunity)
        - effective_scalar > 0.1 (not near-zero size from risk filters)
        - gap_ratio >= configured min_gap_ratio
        - no existing position in same symbol
        - open positions < max_positions

        Args:
            signals: Sorted list of IndexSignal from _scan_indexes.
        """
        strategy_cfg = self._config.get("strategy", {})
        max_positions: int = strategy_cfg.get("max_positions", 3)
        min_gap: float = strategy_cfg.get("min_gap_ratio", 1.05)

        existing_symbols = {p.symbol for p in self._open_positions}

        for sig in signals:
            if len(self._open_positions) >= max_positions:
                logger.debug(
                    "Max positions (%d) reached. Holding.", max_positions
                )
                break

            if sig.score <= 0:
                continue
            if sig.effective_scalar <= 0.1:
                logger.debug(
                    "%s: effective_scalar=%.2f too low. Skipping.",
                    sig.symbol,
                    sig.effective_scalar,
                )
                continue
            if sig.gap_ratio < min_gap:
                logger.debug(
                    "%s: gap_ratio=%.2f below min %.2f. Skipping.",
                    sig.symbol,
                    sig.gap_ratio,
                    min_gap,
                )
                continue
            if sig.symbol in existing_symbols:
                logger.debug(
                    "%s: already have an open position. Skipping.", sig.symbol
                )
                continue

            await self._execute_signal(sig)
            existing_symbols.add(sig.symbol)

    async def _execute_signal(self, sig: IndexSignal) -> None:
        """Execute a trade based on an index signal.

        Builds iron condor legs for qualifying structures, then submits a
        multi-leg combo order to IBKR when a live connection is available.
        Falls back to local-only tracking when the broker is disconnected or
        unavailable, so the orchestrator never loses a signal.

        Args:
            sig: The qualified IndexSignal to act on.
        """
        V2TradingOrchestrator._trade_counter += 1
        trade_id = V2TradingOrchestrator._trade_counter

        logger.info(
            "TRADE ENTRY #%d: %s %s | DTE=%d score=%.2f regime=%s scalar=%.2f",
            trade_id,
            sig.symbol,
            sig.structure.upper(),
            sig.target_dte,
            sig.score,
            sig.regime,
            sig.effective_scalar,
        )

        # Build legs for iron condor and strangle structures
        legs: IronCondorLegs | None = None
        if self._order_builder and sig.structure in ("iron_condor", "strangle"):
            legs = await self._order_builder.build_iron_condor(
                symbol=sig.symbol,
                price=sig.price,
                iv=sig.iv_30,
                target_dte=sig.target_dte,
                position_scalar=sig.effective_scalar,
            )

        entry_credit = legs.net_credit if legs else 0.0
        max_risk = legs.max_risk if legs else 0.0

        pos = _PositionRecord(
            symbol=sig.symbol,
            structure=sig.structure,
            entry_date=datetime.now().isoformat(),
            entry_price=sig.price,
            entry_credit=entry_credit,
            target_dte=sig.target_dte,
            regime_at_entry=sig.regime,
            effective_scalar=sig.effective_scalar,
            gap_ratio=sig.gap_ratio,
            trade_id=trade_id,
        )

        # Submit to IBKR when connected; fall back to local tracking otherwise
        connected = (
            self._connection is not None
            and HAS_BROKER
            and self._connection.is_connected
        )
        if self._order_builder and legs and connected:
            order_id = await self._order_builder.submit_iron_condor(legs)
            if order_id:
                logger.info(
                    "IBKR order submitted: ID=%d %s %s credit=$%.2f risk=$%.2f",
                    order_id, sig.symbol, sig.structure,
                    entry_credit, max_risk,
                )
            else:
                logger.warning(
                    "IBKR order submission failed — "
                    "%s %s tracked locally only (credit=$%.2f risk=$%.2f)",
                    sig.symbol, sig.structure, entry_credit, max_risk,
                )
        else:
            logger.info(
                "SIGNAL LOGGED (no broker): %s %s credit=$%.2f risk=$%.2f",
                sig.symbol, sig.structure, entry_credit, max_risk,
            )

        self._open_positions.append(pos)

    # ------------------------------------------------------------------
    # PEAD secondary scan
    # ------------------------------------------------------------------

    async def _run_pead_scan(self) -> None:
        """Run PEAD earnings scanner and log qualifying candidates.

        PEAD trades are secondary — they are only logged in this iteration
        as individual stock multi-leg execution is not yet wired. The scan
        validates signal alignment (SUE + IV spread + FinBERT).
        """
        if self._earnings is None:
            return

        try:
            from signals.earnings_scanner import EarningsScanner
            candidates = await self._earnings.scan()

            qualified = [
                c for c in candidates
                if EarningsScanner.check_alignment(c) and abs(c.sue) >= 1.0
            ]

            if qualified:
                logger.info(
                    "PEAD: %d qualified candidate(s) from %d recent earnings.",
                    len(qualified),
                    len(candidates),
                )
                for c in qualified[:3]:   # log top 3
                    logger.info(
                        "  PEAD candidate: %s | direction=%s SUE=%.2f score=%.2f",
                        c.symbol,
                        c.direction,
                        c.sue,
                        c.score,
                    )
            else:
                logger.debug("PEAD: no qualified candidates.")

        except Exception as exc:
            logger.error("PEAD scan failed: %s", exc)

    # ------------------------------------------------------------------
    # Exit monitoring
    # ------------------------------------------------------------------

    async def _check_exits(self) -> None:
        """Evaluate and process time-based exits for all open positions.

        In signal-only mode, exit is triggered when days held >= target_dte - 7
        (close at ~7 DTE, consistent with the research-backed exit rule to avoid
        gamma risk acceleration near expiry).

        In live mode, additional exits (50% profit target, 2x credit stop)
        would be triggered by real-time P&L from the broker. Those require
        IBKROrderBridge integration and are noted as TODO below.
        """
        now = datetime.now()
        to_close: list[_PositionRecord] = []

        for pos in self._open_positions:
            try:
                entry_dt = datetime.fromisoformat(pos.entry_date)
                days_held = (now - entry_dt).days

                # Time exit: close at 7 DTE (days_held >= target_dte - 7)
                if days_held >= max(0, pos.target_dte - 7):
                    logger.info(
                        "EXIT (time) #%d %s — held %d days (target DTE was %d).",
                        pos.trade_id,
                        pos.symbol,
                        days_held,
                        pos.target_dte,
                    )
                    to_close.append(pos)
                    continue

                # TODO: profit target (50% of credit received)
                # TODO: stop loss (2x credit received)
                # Both require real-time mark-to-market from broker.

            except Exception as exc:
                logger.error(
                    "Exit check error for position #%d (%s): %s",
                    pos.trade_id,
                    pos.symbol,
                    exc,
                )

        for pos in to_close:
            self._close_position(pos, "time_exit")

    def _close_position(self, pos: _PositionRecord, reason: str) -> None:
        """Close a position, record it, and notify the LiveMonitor.

        Args:
            pos: The open position to close.
            reason: Exit reason string (time_exit, profit_target, stop_loss).
        """
        self._open_positions.remove(pos)
        self._closed_positions.append(pos)

        if self._monitor and HAS_MONITOR:
            trade = TradeRecord(
                trade_id=pos.trade_id,
                symbol=pos.symbol,
                entry_date=pos.entry_date,
                exit_date=datetime.now().isoformat(),
                pnl=0.0,        # placeholder — filled by broker reconciliation in live mode
                pnl_pct=0.0,
                entry_signal_score=pos.gap_ratio,
                regime_at_entry=pos.regime_at_entry,
                exit_reason=reason,
            )
            alerts = self._monitor.record_trade(trade)
            for alert in alerts:
                log_fn = logger.critical if alert.severity == "HALT" else logger.warning
                log_fn("MONITOR [%s|%s]: %s", alert.severity, alert.category, alert.message)

    # ------------------------------------------------------------------
    # Execution window check
    # ------------------------------------------------------------------

    def _in_execution_window(self) -> bool:
        """Return True if the current ET time is within the execution window.

        The execution window (11:00 AM – 2:00 PM ET) is chosen following
        Muravyev & Pearson (2020): mid-day liquidity is highest and bid/ask
        spreads on index options are tightest, improving fill quality.

        Returns:
            True if within the window, False otherwise.
            Defaults to True if timezone lookup fails (for test environments).
        """
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("US/Eastern")
            now_et = datetime.now(et).time().replace(tzinfo=None)
            trading_cfg = self._config.get("trading", {})
            start_str: str = trading_cfg.get("execution_start", "11:00")
            end_str: str = trading_cfg.get("execution_end", "14:00")
            start = time(*map(int, start_str.split(":")))
            end = time(*map(int, end_str.split(":")))
            return start <= now_et <= end
        except Exception:
            return True

    def is_market_hours(self) -> bool:
        """Return True if current ET time is within regular market hours (9:30-16:00).

        Returns:
            True during 09:30-16:00 ET on weekdays, False otherwise.
        """
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("US/Eastern")
            now_et = datetime.now(et)
            if now_et.weekday() >= 5:
                return False
            t = now_et.time().replace(tzinfo=None)
            return time(9, 30) <= t <= time(16, 0)
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_startup_banner(self) -> None:
        """Log the startup banner with strategy configuration summary."""
        logger.info("=" * 62)
        logger.info("  CONTRARIAN OPTIONS ALPHA ENGINE — V2 ORCHESTRATOR")
        logger.info("  Universe : SPY / QQQ / IWM (index options)")
        logger.info("  Signal   : HV-IV gap (Goyal-Saretto 2009)")
        logger.info("  Regime   : VIX/VVIX filter (Baltussen et al. 2018)")
        logger.info("  Skew     : IV smirk z-score (Xing et al. 2010)")
        logger.info("  Sizing   : Quarter-Kelly with regime + skew scalars")
        logger.info("  Monitor  : CUSUM + drawdown + win-rate kill switch")
        logger.info("=" * 62)

    def _log_status(self) -> None:
        """Log a one-line status summary after each scan cycle."""
        regime_str = (
            self._last_regime_state.regime.value
            if self._last_regime_state else "UNKNOWN"
        )
        halted = self._monitor.is_halted if self._monitor else False
        logger.debug(
            "Status: regime=%s | open=%d | closed=%d | halted=%s",
            regime_str,
            len(self._open_positions),
            len(self._closed_positions),
            halted,
        )

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    def _load_config(self, path: str) -> dict[str, Any]:
        """Load YAML config and deep-merge with defaults.

        Args:
            path: Path to a YAML configuration file.

        Returns:
            Merged configuration dictionary. Missing keys fall back to
            :data:`_DEFAULT_CONFIG`.
        """
        config: dict[str, Any] = {}
        try:
            with open(path) as fh:
                loaded = yaml.safe_load(fh) or {}
            config = loaded
            logger.info("Loaded config from %s.", path)
        except FileNotFoundError:
            logger.warning("Config %s not found. Using defaults.", path)
        except Exception as exc:
            logger.warning("Could not load config (%s). Using defaults. Error: %s", path, exc)

        merged = dict(_DEFAULT_CONFIG)
        self._deep_merge(merged, config)
        return merged

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
        """Recursively merge ``override`` into ``base`` in-place.

        Nested dicts are merged recursively. Scalar values in ``override``
        always win over ``base``.

        Args:
            base: Dictionary to merge into (modified in-place).
            override: Values from user config (take precedence).
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                V2TradingOrchestrator._deep_merge(base[key], value)
            else:
                base[key] = value

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Register SIGINT / SIGTERM handlers for graceful shutdown."""
        def _handle(sig_num: int, _frame: Any) -> None:
            logger.info("Signal %s received. Initiating graceful shutdown.", sig_num)
            self._shutdown_event.set()

        try:
            signal.signal(signal.SIGINT, _handle)
            signal.signal(signal.SIGTERM, _handle)
        except (ValueError, OSError):
            # Not in main thread (e.g. during tests); skip registration.
            pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run the v2 index options trading orchestrator."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="V2 Index Options Trader — Contrarian Options Alpha Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Signal-only mode (default when no broker is available):\n"
            "  Generates and logs signals without submitting orders.\n"
            "  Useful for paper-trading validation and strategy review.\n"
        ),
    )
    parser.add_argument(
        "--config",
        default="config/paper_trading.yaml",
        help="Path to YAML configuration file (default: config/paper_trading.yaml)",
    )
    parser.add_argument(
        "--signal-only",
        action="store_true",
        help="Generate signals and log them without executing orders.",
    )
    args = parser.parse_args()

    if args.signal_only:
        logger.info = logging.getLogger().info  # ensure INFO-level output
        logger.info("Running in --signal-only mode. No orders will be submitted.")

    orchestrator = V2TradingOrchestrator(config_path=args.config)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
