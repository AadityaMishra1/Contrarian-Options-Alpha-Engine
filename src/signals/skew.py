"""IV Skew Measurement and Scoring.

Implements the Xing, Zhang, Zhao (2010) smirk measure: the difference
between OTM put IV and ATM call IV. Steep smirk indicates informed
bearish flow; flat smirk indicates safe conditions for selling puts.

Reference: Xing, Zhang, Zhao (2010). "What Does the Individual Option
Volatility Smirk Tell Us?" JFQA, 45(3), 641-662.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SkewReading:
    """A single skew measurement for a symbol."""

    symbol: str
    iv_otm_put: float    # 25-delta put IV
    iv_atm: float        # 50-delta (ATM) IV
    smirk: float         # iv_otm_put - iv_atm
    zscore: float        # relative to rolling history
    is_steep: bool       # zscore > 1.5 (dangerous to sell puts)
    is_flat: bool        # zscore < -1.0 (safe to sell puts)


class SkewTracker:
    """Tracks IV skew for a set of symbols over time.

    Maintains a rolling window of smirk measurements per symbol
    and computes z-scores for anomaly detection.

    Args:
        lookback: Rolling window size for z-score computation.
        steep_threshold: Z-score above which skew is "steep" (avoid selling).
        flat_threshold: Z-score below which skew is "flat" (safe to sell).
    """

    def __init__(
        self,
        lookback: int = 60,
        steep_threshold: float = 1.5,
        flat_threshold: float = -1.0,
    ) -> None:
        self.lookback = lookback
        self.steep_threshold = steep_threshold
        self.flat_threshold = flat_threshold
        self._history: dict[str, deque[float]] = {}

    def update(self, symbol: str, iv_otm_put: float, iv_atm: float) -> SkewReading:
        """Record a new skew measurement and return the assessment.

        Args:
            symbol: Ticker symbol.
            iv_otm_put: Implied vol of 25-delta OTM put.
            iv_atm: Implied vol of ATM (50-delta) option.

        Returns:
            SkewReading with z-score and classification.
        """
        smirk = iv_otm_put - iv_atm

        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self.lookback)

        history = self._history[symbol]
        history.append(smirk)

        # Compute z-score
        if len(history) < 10:
            zscore = 0.0
        else:
            values = list(history)
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 0.01
            zscore = (smirk - mean) / std

        return SkewReading(
            symbol=symbol,
            iv_otm_put=iv_otm_put,
            iv_atm=iv_atm,
            smirk=smirk,
            zscore=zscore,
            is_steep=zscore > self.steep_threshold,
            is_flat=zscore < self.flat_threshold,
        )

    def get_position_scalar(self, symbol: str) -> float:
        """Get a continuous position scalar based on skew.

        Returns 1.0 for flat skew, scales down for steep skew,
        0.0 for extremely steep (z > 2.5).

        Implements the user feedback: not binary allow/avoid,
        but a continuous throttle on position size.
        """
        if symbol not in self._history or len(self._history[symbol]) < 10:
            return 0.8  # conservative default when insufficient data

        values = list(self._history[symbol])
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 0.01
        current_z = (values[-1] - mean) / std

        if current_z <= self.flat_threshold:
            return 1.0  # flat skew, safe to sell
        elif current_z <= 0.0:
            return 0.9
        elif current_z <= 1.0:
            return 0.7
        elif current_z <= self.steep_threshold:
            return 0.5
        elif current_z <= 2.5:
            return 0.25
        else:
            return 0.0  # extremely steep, do not sell

    def reset(self, symbol: str | None = None) -> None:
        """Reset history for a symbol or all symbols."""
        if symbol:
            self._history.pop(symbol, None)
        else:
            self._history.clear()
