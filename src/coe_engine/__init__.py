"""Python wrapper for the Contrarian Options Engine C++ core."""
from __future__ import annotations

try:
    from _coe_engine import (
        # Strategy
        RSI,
        BollingerBands,
        CircuitBreaker,
        CoeError,
        Config,
        DailyPnLTracker,
        ErrorCode,
        GreeksFilter,
        KellyPositionSizer,
        OptionContract,
        OptionType,
        Order,
        OrderManager,
        OrderState,
        # Execution
        OrderType,
        Position,
        PositionTracker,
        # Risk
        RiskLimits,
        RiskManager,
        # Common
        Side,
        Signal,
        SignalScorer,
        VolumeSpike,
        next_order_id,
    )

    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

__all__ = [
    "Side", "OptionType", "ErrorCode", "Config", "OptionContract", "CoeError",
    "RSI", "BollingerBands", "VolumeSpike", "GreeksFilter", "Signal", "SignalScorer",
    "OrderType", "OrderState", "Order", "OrderManager", "Position", "PositionTracker",
    "KellyPositionSizer", "next_order_id",
    "RiskLimits", "DailyPnLTracker", "CircuitBreaker", "RiskManager",
    "HAS_NATIVE",
]
