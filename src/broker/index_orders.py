"""Index Options Order Construction.

Builds multi-leg ib_insync combo orders for iron condors, strangles,
and put spreads on index ETFs (SPY, QQQ, IWM).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

logger = logging.getLogger(__name__)

try:
    from ib_insync import (
        IB,
        ComboLeg,
        Contract,
        TagValue,
    )
    from ib_insync import (
        Option as IBOption,
    )
    from ib_insync import (
        Order as IBOrder,
    )
    HAS_IB = True
except ImportError:
    HAS_IB = False
    IB = None  # type: ignore[assignment,misc]
    Contract = None  # type: ignore[assignment,misc]
    ComboLeg = None  # type: ignore[assignment,misc]
    IBOrder = None  # type: ignore[assignment,misc]
    IBOption = None  # type: ignore[assignment,misc]
    TagValue = None  # type: ignore[assignment,misc]


@dataclass
class IronCondorLegs:
    """The four legs of an iron condor.

    Attributes:
        short_put: Dict with strike, expiry, and right="P" for the short put.
        long_put: Dict with strike, expiry, and right="P" for the long put.
        short_call: Dict with strike, expiry, and right="C" for the short call.
        long_call: Dict with strike, expiry, and right="C" for the long call.
        net_credit: Estimated credit received for the full structure.
        max_risk: Maximum loss (wing width minus credit).
        symbol: Underlying ETF ticker.
        expiry: Expiry date string (YYYY-MM-DD).
    """

    short_put: dict
    long_put: dict
    short_call: dict
    long_call: dict
    net_credit: float
    max_risk: float
    symbol: str
    expiry: str


class IndexOrderBuilder:
    """Builds multi-leg options orders for index ETFs.

    Constructs and optionally submits iron condor combo (BAG) orders
    through a connected ib_insync IB instance. When ``ib`` is None or
    ib_insync is not installed, order construction still works and
    ``submit_iron_condor`` returns None without raising.

    Args:
        ib: Connected ib_insync.IB instance, or None for signal-only mode.
    """

    def __init__(self, ib: Any = None) -> None:
        self._ib = ib

    async def build_iron_condor(
        self,
        symbol: str,
        price: float,
        iv: float,
        target_dte: int = 30,
        wing_width: float = 5.0,
        position_scalar: float = 1.0,
        capital: float = 25000.0,
        max_position_pct: float = 0.03,
    ) -> IronCondorLegs | None:
        """Build an iron condor for an index ETF.

        Short strikes are placed at approximately 1 standard deviation from
        the current price. Wings are placed ``wing_width`` dollars beyond
        the short strikes.

        Args:
            symbol: ETF ticker (SPY, QQQ, IWM).
            price: Current underlying price.
            iv: Implied volatility as a decimal (e.g. 0.27 for 27%).
            target_dte: Target days to expiration.
            wing_width: Distance in dollars between short and long strikes.
            position_scalar: Combined regime and skew scalar.
            capital: Total account capital.
            max_position_pct: Maximum position as a fraction of capital.

        Returns:
            IronCondorLegs dataclass, or None if inputs are invalid.
        """
        if price <= 0 or iv <= 0:
            logger.warning(
                "build_iron_condor: invalid inputs price=%.2f iv=%.4f for %s",
                price, iv, symbol,
            )
            return None

        # Calculate expected move (1 std dev)
        expected_move = price * iv * (target_dte / 365.0) ** 0.5

        # Short strikes: ~1 std dev out
        short_put_strike = self._round_strike(price - expected_move, symbol)
        short_call_strike = self._round_strike(price + expected_move, symbol)

        # Long strikes: wing_width further out
        long_put_strike = short_put_strike - wing_width
        long_call_strike = short_call_strike + wing_width

        # Find expiry
        expiry = self._find_expiry(target_dte)

        # Estimate credit (~35% of wing width, scaled by position_scalar)
        credit_estimate = wing_width * 0.35 * position_scalar
        max_risk = wing_width - credit_estimate

        legs = IronCondorLegs(
            short_put={"strike": short_put_strike, "expiry": expiry, "right": "P"},
            long_put={"strike": long_put_strike, "expiry": expiry, "right": "P"},
            short_call={"strike": short_call_strike, "expiry": expiry, "right": "C"},
            long_call={"strike": long_call_strike, "expiry": expiry, "right": "C"},
            net_credit=credit_estimate,
            max_risk=max_risk,
            symbol=symbol,
            expiry=expiry,
        )

        logger.info(
            "Iron condor built: %s %s P%g/%g C%g/%g credit=$%.2f risk=$%.2f",
            symbol, expiry,
            long_put_strike, short_put_strike,
            short_call_strike, long_call_strike,
            credit_estimate, max_risk,
        )

        return legs

    async def submit_iron_condor(self, legs: IronCondorLegs) -> int | None:
        """Submit an iron condor combo order to IBKR.

        Qualifies all four option contracts, builds a BAG (combo) contract,
        and places a limit order at the estimated net credit.

        Args:
            legs: IronCondorLegs produced by ``build_iron_condor``.

        Returns:
            The IBKR order ID as an integer, or None if submission failed
            (no connection, qualification failure, or any exception).
        """
        if not HAS_IB or self._ib is None:
            logger.warning("IBKR not connected — cannot submit order")
            return None

        try:
            # Create and qualify the four option contracts
            # Order: long_put BUY, short_put SELL, short_call SELL, long_call BUY
            leg_specs = [
                (legs.long_put, "BUY"),
                (legs.short_put, "SELL"),
                (legs.short_call, "SELL"),
                (legs.long_call, "BUY"),
            ]

            qualified_legs: list[tuple[Any, str]] = []
            for leg_data, action in leg_specs:
                contract = IBOption(
                    symbol=legs.symbol,
                    lastTradeDateOrContractMonth=leg_data["expiry"].replace("-", ""),
                    strike=leg_data["strike"],
                    right=leg_data["right"],
                    exchange="SMART",
                    currency="USD",
                )
                qualified = await self._ib.qualifyContractsAsync(contract)
                if not qualified:
                    logger.error(
                        "Failed to qualify contract: %s strike=%g right=%s",
                        legs.symbol, leg_data["strike"], leg_data["right"],
                    )
                    return None
                qualified_legs.append((qualified[0], action))

            # Build BAG (combo) contract
            combo = Contract()
            combo.symbol = legs.symbol
            combo.secType = "BAG"
            combo.currency = "USD"
            combo.exchange = "SMART"

            combo.comboLegs = []
            for contract, action in qualified_legs:
                leg = ComboLeg()
                leg.conId = contract.conId
                leg.ratio = 1
                leg.action = action
                leg.exchange = "SMART"
                combo.comboLegs.append(leg)

            # Limit order at estimated net credit (sell the condor = collect credit)
            order = IBOrder()
            order.action = "SELL"
            order.orderType = "LMT"
            order.totalQuantity = 1
            order.lmtPrice = round(legs.net_credit, 2)
            order.transmit = True
            order.smartComboRoutingParams = [TagValue("NonGuaranteed", "1")]

            trade = self._ib.placeOrder(combo, order)
            logger.info(
                "Submitted iron condor order: %s orderId=%s",
                legs.symbol, trade.order.orderId,
            )
            return trade.order.orderId

        except Exception as exc:
            logger.error("Failed to submit iron condor: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _round_strike(strike: float, symbol: str) -> float:
        """Round a raw strike price to the nearest valid increment.

        SPY, QQQ, and IWM all trade in $1 increments on standard expirations.

        Args:
            strike: Raw computed strike value.
            symbol: Underlying ticker (currently unused; reserved for future
                per-symbol increment logic).

        Returns:
            Strike rounded to the nearest dollar.
        """
        return float(round(strike))

    @staticmethod
    def _find_expiry(target_dte: int) -> str:
        """Find the nearest Friday expiry at or beyond the target DTE.

        Args:
            target_dte: Desired days to expiration from today.

        Returns:
            Expiry date string in YYYY-MM-DD format.
        """
        target = date.today() + timedelta(days=target_dte)
        # Advance to next Friday (weekday 4) if not already on one
        days_until_friday = (4 - target.weekday()) % 7
        expiry = target + timedelta(days=days_until_friday)
        return expiry.strftime("%Y-%m-%d")
