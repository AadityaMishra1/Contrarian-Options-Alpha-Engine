"""IBKR connection wrapper with auto-reconnect and heartbeat.

Wraps ib_insync.IB with retry logic, exponential backoff on initial
connect, and a background heartbeat task that reconnects on drop.
"""

from __future__ import annotations

import asyncio
import logging

try:
    from ib_insync import IB

    HAS_IB_INSYNC = True
except ImportError:
    HAS_IB_INSYNC = False
    IB = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_MAX_RETRIES: int = 3
_HEARTBEAT_INTERVAL: float = 10.0
_BACKOFF_BASE: float = 2.0  # seconds — doubled each retry


class IBKRConnectionError(Exception):
    """Raised when connection to IBKR cannot be established after max retries."""


class IBKRConnection:
    """Async context manager wrapping ib_insync.IB with reconnect logic.

    Args:
        host: TWS / IB Gateway hostname or IP.
        port: 7497 for TWS paper trading, 7496 for TWS live,
              4002 for IB Gateway paper, 4001 for IB Gateway live.
        client_id: Unique integer client identifier. Multiple
                   connections to the same TWS instance must use
                   different client IDs.
        readonly: When True the connection is opened in read-only mode
                  (no order submission).

    Example::

        async with IBKRConnection(port=7497) as conn:
            positions = conn.ib.positions()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        readonly: bool = False,
    ) -> None:
        if not HAS_IB_INSYNC:
            raise ImportError(
                "ib_insync is not installed. Run: pip install ib_insync"
            )

        self._host = host
        self._port = port
        self._client_id = client_id
        self._readonly = readonly

        self._ib: IB = IB()
        self._connected: bool = False
        self._heartbeat_task: asyncio.Task | None = None  # type: ignore[type-arg]

        # Wire the disconnect callback so we schedule a reconnect automatically.
        self._ib.disconnectedEvent += self._on_disconnected

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> IBKRConnection:
        """Connect on entry and start the heartbeat loop."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop heartbeat and cleanly disconnect on exit."""
        await self.disconnect()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to TWS / IB Gateway with exponential-backoff retries.

        Raises:
            IBKRConnectionError: If all retry attempts are exhausted.
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info(
                    "Connecting to IBKR at %s:%d (client_id=%d, attempt=%d/%d)",
                    self._host,
                    self._port,
                    self._client_id,
                    attempt,
                    _MAX_RETRIES,
                )
                await self._ib.connectAsync(
                    self._host,
                    self._port,
                    clientId=self._client_id,
                    readonly=self._readonly,
                )
                self._connected = True
                logger.info("Connected to IBKR successfully.")
                self._start_heartbeat()
                return
            except Exception as exc:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    "Connection attempt %d failed: %s. Retrying in %.1fs.",
                    attempt,
                    exc,
                    wait,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(wait)

        raise IBKRConnectionError(
            f"Failed to connect to IBKR at {self._host}:{self._port} "
            f"after {_MAX_RETRIES} attempts."
        )

    async def disconnect(self) -> None:
        """Stop heartbeat and cleanly disconnect from TWS / IB Gateway."""
        self._stop_heartbeat()
        if self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IBKR.")
        self._connected = False

    async def heartbeat(self) -> None:
        """Background loop that checks connectivity every 10 seconds.

        If the connection has dropped, a single reconnect attempt is made.
        Reconnect failures are logged but do not propagate — the next
        heartbeat tick will retry.
        """
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            if not self._ib.isConnected():
                logger.warning("Heartbeat: IBKR connection lost. Attempting reconnect.")
                try:
                    await self.connect()
                except IBKRConnectionError as exc:
                    logger.error("Heartbeat reconnect failed: %s", exc)
            else:
                logger.debug("Heartbeat: IBKR connection healthy.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ib(self) -> IB:
        """The underlying ib_insync.IB instance.

        Returns:
            The raw IB client. Callers may use this to call any
            ib_insync API directly.
        """
        return self._ib

    @property
    def is_connected(self) -> bool:
        """True if the socket is currently connected to TWS / IB Gateway."""
        return self._ib.isConnected()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _start_heartbeat(self) -> None:
        """Schedule the heartbeat coroutine as a background asyncio Task."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.ensure_future(self.heartbeat())
            logger.debug("Heartbeat task started.")

    def _stop_heartbeat(self) -> None:
        """Cancel the heartbeat Task if it is running."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            logger.debug("Heartbeat task cancelled.")
        self._heartbeat_task = None

    def _on_disconnected(self) -> None:
        """Callback registered with ib_insync disconnectedEvent.

        Schedules a reconnect coroutine on the running event loop so
        that reconnection happens transparently without blocking the
        caller.
        """
        logger.warning("IBKR disconnectedEvent fired. Scheduling reconnect.")
        self._connected = False
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self._reconnect_once())

    async def _reconnect_once(self) -> None:
        """Single reconnect attempt triggered by the disconnect callback."""
        try:
            await asyncio.sleep(1.0)  # brief pause before hammering TWS
            await self.connect()
        except IBKRConnectionError as exc:
            logger.error("Reconnect after disconnect event failed: %s", exc)
