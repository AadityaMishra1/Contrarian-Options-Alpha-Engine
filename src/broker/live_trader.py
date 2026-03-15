"""Live trading orchestrator for the Contrarian Options Alpha Engine.

Extends :class:`~src.broker.paper_trader.PaperTradingOrchestrator` with:

* Interactive startup confirmation.
* Capital exposure guard (``_check_capital_limit``).
* Position-size scaling factor (``_scale_position_size``).
* Tighter defaults read from ``config/live_trading.yaml``.
"""

from __future__ import annotations

import logging
from typing import Any

from .paper_trader import PaperTradingOrchestrator

logger = logging.getLogger(__name__)


class LiveTradingOrchestrator(PaperTradingOrchestrator):
    """Live trading with additional safety constraints.

    Overrides the base paper-trading loop to enforce:

    * A mandatory interactive ``CONFIRM`` prompt before any orders are placed.
    * A configurable ``position_scale`` factor that shrinks Kelly-sized
      position counts for live capital.
    * A hard cap on total notional exposure (``max_capital``).

    Args:
        config_path: Path to the live-trading YAML config file.
    """

    def __init__(self, config_path: str = "config/live_trading.yaml") -> None:
        super().__init__(config_path=config_path)
        self._max_capital: float = self._config.get("capital", {}).get(
            "max_capital", 500.0
        )
        self._initial_capital: float = self._config.get("capital", {}).get(
            "initial_capital", 200.0
        )
        self._position_scale: float = self._config.get("capital", {}).get(
            "position_scale", 0.5
        )
        self._require_confirmation: bool = self._config.get("trading", {}).get(
            "require_confirmation", True
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run with startup confirmation and capital constraints.

        If ``require_confirmation`` is ``true`` in config the user must type
        ``CONFIRM`` at the prompt, otherwise execution is aborted cleanly.
        """
        if self._require_confirmation and not self._confirm_startup():
            logger.info("Live trading cancelled by user.")
            return

        logger.warning(
            "LIVE TRADING ACTIVE | Capital: $%.2f | Max: $%.2f | Scale: %.1fx",
            self._initial_capital,
            self._max_capital,
            self._position_scale,
        )
        await super().run()

    # ------------------------------------------------------------------
    # Entry override — inject scale + capital guard
    # ------------------------------------------------------------------

    async def _submit_entry(
        self, symbol: str, option_candidate: Any, contracts: int
    ) -> None:
        """Scale contracts and check capital limit before submitting.

        Applies :meth:`_scale_position_size` to the raw ``contracts`` count
        and then gates on :meth:`_check_capital_limit` before delegating to
        the parent implementation.

        Args:
            symbol: Underlying ticker symbol.
            option_candidate: Object carrying option contract attributes.
            contracts: Raw contract count (pre-scale).
        """
        scaled_contracts = self._scale_position_size(contracts)

        # Estimate notional exposure for the scaled order.
        price = float(getattr(option_candidate, "price", 0.10))
        order_value = scaled_contracts * price * 100

        if not self._check_capital_limit(order_value):
            logger.warning(
                "Capital limit would be exceeded for %s "
                "(order_value=$%.2f, max_capital=$%.2f). Skipping.",
                symbol,
                order_value,
                self._max_capital,
            )
            return

        logger.debug(
            "Live entry: symbol=%s raw_contracts=%d scaled_contracts=%d order_value=$%.2f",
            symbol,
            contracts,
            scaled_contracts,
            order_value,
        )
        await super()._submit_entry(symbol, option_candidate, scaled_contracts)

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------

    def _scale_position_size(self, contracts: int) -> int:
        """Apply the configured scale factor and enforce a minimum of 1.

        Args:
            contracts: Raw Kelly-sized contract count.

        Returns:
            Scaled contract count, always >= 1.
        """
        return max(1, int(contracts * self._position_scale))

    def _check_capital_limit(self, order_value: float) -> bool:
        """Return True when adding ``order_value`` stays within ``max_capital``.

        Computes current total notional exposure across all open positions and
        checks that the proposed order would not push it past ``max_capital``.

        Args:
            order_value: Estimated notional cost of the new order in dollars.

        Returns:
            True if the order is within capital limits, False otherwise.
        """
        current_exposure = sum(
            abs(p.get("quantity", 0) * p.get("current_price", 0) * 100)
            for p in self._get_positions_list()
        )
        return (current_exposure + order_value) <= self._max_capital

    def _get_positions_list(self) -> list[dict]:
        """Return open positions as a list of plain dicts.

        Bridges between the C++ PositionTracker (when available) and the
        pure-Python :class:`~src.broker.paper_trader._PythonState` fallback,
        normalising both into a common ``{"quantity": int, "current_price": float}``
        shape used by :meth:`_check_capital_limit`.

        Returns:
            List of position dicts with at least ``quantity`` and
            ``current_price`` keys.
        """
        from .paper_trader import HAS_ENGINE  # avoid circular at module level

        if HAS_ENGINE and self._position_tracker is not None:
            try:
                raw_positions = self._position_tracker.get_all_positions()
                return [
                    {
                        "quantity": int(getattr(p, "quantity", 0)),
                        "current_price": float(getattr(p, "current_price", 0.0)),
                    }
                    for p in raw_positions
                ]
            except Exception as exc:
                logger.warning(
                    "Failed to read positions from C++ tracker: %s. Falling back.", exc
                )

        # Pure-Python fallback: use avg_cost as proxy for current_price.
        return [
            {
                "quantity": pos.quantity,
                "current_price": pos.avg_cost,
            }
            for pos in self._state.positions.values()
        ]

    # ------------------------------------------------------------------
    # Startup confirmation
    # ------------------------------------------------------------------

    def _confirm_startup(self) -> bool:
        """Prompt the operator to confirm live trading before the loop starts.

        Prints a summary of key risk parameters and requires the user to type
        ``CONFIRM`` (case-sensitive) to proceed.

        Returns:
            True if the operator typed ``CONFIRM``, False otherwise.
        """
        daily_loss_limit = abs(
            self._config.get("risk", {}).get("daily_loss_limit", -30.0)
        )
        print("\n" + "=" * 60)
        print("  LIVE TRADING CONFIRMATION")
        print("=" * 60)
        print(f"  Capital:          ${self._initial_capital:.2f}")
        print(f"  Max Capital:      ${self._max_capital:.2f}")
        print(f"  Position Scale:   {self._position_scale:.1f}x")
        print(f"  Daily Loss Limit: ${daily_loss_limit:.2f}")
        print("=" * 60)
        try:
            response = input("  Type 'CONFIRM' to start live trading: ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        return response == "CONFIRM"
