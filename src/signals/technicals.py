"""Technical analysis layer wrapping the C++ SignalScorer via pybind11.

Provides a Python interface to feed price/volume/greeks data and evaluate
composite signals. Falls back to a pure-Python implementation replicating
the same RSI, Bollinger Band, volume-spike and greeks scoring logic used in
backtest/replay_engine.py when the native extension is unavailable.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import the native C++ extension
# ---------------------------------------------------------------------------

try:
    import _coe_engine  # type: ignore[import-not-found]
    _NATIVE_AVAILABLE = True
    logger.info("Loaded native _coe_engine extension")
except ImportError:
    _NATIVE_AVAILABLE = False
    logger.info(
        "_coe_engine extension not available — using pure-Python fallback"
    )

# ---------------------------------------------------------------------------
# Signal result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """Composite signal produced by the technical scorer.

    Attributes:
        symbol: Ticker symbol.
        composite_score: Weighted composite in [0.0, 1.0].
        rsi_score: Normalised RSI component in [0.0, 1.0].
        bb_score: Bollinger-band component in [0.0, 1.0].
        volume_score: Volume-spike component in [0.0, 1.0].
        greeks_score: Options-greeks component in [0.0, 1.0].
        triggered: True when composite_score >= min_composite threshold.
    """

    symbol: str
    composite_score: float
    rsi_score: float
    bb_score: float
    volume_score: float
    greeks_score: float
    triggered: bool


# ---------------------------------------------------------------------------
# Pure-Python SignalScorer (fallback)
# ---------------------------------------------------------------------------


class _PythonSignalScorer:
    """Pure-Python replication of the C++ SignalScorer logic.

    Replicates RSI (Wilder's EMA), Bollinger Bands, volume-spike detection,
    and Greeks scoring from backtest/replay_engine.py, with weights and
    thresholds sourced from config/strategy.yaml.

    Args:
        config: Parsed strategy configuration dict (from strategy.yaml).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})

        # RSI parameters
        rsi_cfg = strat.get("rsi", {})
        self._rsi_period: int = int(rsi_cfg.get("period", 14))
        self._rsi_oversold: float = float(rsi_cfg.get("oversold", 30.0))

        # Bollinger Band parameters
        bb_cfg = strat.get("bollinger", {})
        self._bb_period: int = int(bb_cfg.get("period", 20))
        self._bb_mult: float = float(bb_cfg.get("multiplier", 2.0))

        # Volume parameters
        vol_cfg = strat.get("volume", {})
        self._vol_lookback: int = int(vol_cfg.get("lookback", 20))
        self._vol_spike_threshold: float = float(vol_cfg.get("spike_threshold", 2.0))

        # Greeks parameters
        greeks_cfg = strat.get("greeks", {})
        self._delta_min: float = float(greeks_cfg.get("delta_min", 0.20))
        self._delta_max: float = float(greeks_cfg.get("delta_max", 0.40))
        self._iv_pct_max: float = float(greeks_cfg.get("iv_percentile_max", 50.0))
        self._spread_pct_max: float = float(greeks_cfg.get("spread_pct_max", 20.0))

        # Scoring weights and threshold
        scoring_cfg = strat.get("scoring", {})
        self._w_rsi: float = float(scoring_cfg.get("weight_rsi", 0.30))
        self._w_bb: float = float(scoring_cfg.get("weight_bollinger", 0.25))
        self._w_vol: float = float(scoring_cfg.get("weight_volume", 0.25))
        self._w_greeks: float = float(scoring_cfg.get("weight_greeks", 0.20))
        self._min_composite: float = float(scoring_cfg.get("min_composite", 0.65))

        # Per-symbol state
        self._prices: dict[str, deque[float]] = {}
        self._volumes: dict[str, deque[float]] = {}
        self._greeks: dict[str, dict[str, float]] = {}

        # Wilder EMA state for RSI
        self._rsi_avg_gain: dict[str, float] = {}
        self._rsi_avg_loss: dict[str, float] = {}
        self._rsi_seeded: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Feed methods
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float) -> None:
        """Append a new closing price and update RSI EMA state."""
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=self._bb_period + 1)
            self._rsi_avg_gain[symbol] = 0.0
            self._rsi_avg_loss[symbol] = 0.0
            self._rsi_seeded[symbol] = False

        prices = self._prices[symbol]
        if prices:
            delta = price - prices[-1]
            gain = max(0.0, delta)
            loss = max(0.0, -delta)
            alpha = 1.0 / self._rsi_period

            if not self._rsi_seeded[symbol]:
                # Accumulate until we have enough deltas for the seed average
                # We need rsi_period deltas, so rsi_period+1 prices
                prices.append(price)
                if len(prices) > self._rsi_period:
                    # Seed: simple average of first rsi_period gains/losses
                    deltas = [
                        prices[i] - prices[i - 1] for i in range(1, len(prices))
                    ]
                    gains_seed = [max(0.0, d) for d in deltas[-self._rsi_period:]]
                    losses_seed = [max(0.0, -d) for d in deltas[-self._rsi_period:]]
                    self._rsi_avg_gain[symbol] = sum(gains_seed) / self._rsi_period
                    self._rsi_avg_loss[symbol] = sum(losses_seed) / self._rsi_period
                    self._rsi_seeded[symbol] = True
            else:
                prices.append(price)
                self._rsi_avg_gain[symbol] = (
                    alpha * gain + (1 - alpha) * self._rsi_avg_gain[symbol]
                )
                self._rsi_avg_loss[symbol] = (
                    alpha * loss + (1 - alpha) * self._rsi_avg_loss[symbol]
                )
        else:
            prices.append(price)

    def update_volume(self, symbol: str, volume: float) -> None:
        """Append a new volume bar."""
        if symbol not in self._volumes:
            self._volumes[symbol] = deque(maxlen=self._vol_lookback + 1)
        self._volumes[symbol].append(volume)

    def update_greeks(
        self,
        symbol: str,
        delta: float,
        iv_pct: float,
        bid: float,
        ask: float,
    ) -> None:
        """Store the latest options greeks snapshot for *symbol*."""
        self._greeks[symbol] = {
            "delta": delta,
            "iv_pct": iv_pct,
            "bid": bid,
            "ask": ask,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, symbol: str) -> Signal | None:
        """Compute and return a composite Signal for *symbol*.

        Returns:
            Signal dataclass or None if insufficient data.
        """
        prices = self._prices.get(symbol)
        volumes = self._volumes.get(symbol)

        if prices is None or len(prices) < self._rsi_period + 1:
            logger.debug("Insufficient price history for %s", symbol)
            return None

        price_list = list(prices)
        current_price = price_list[-1]

        # -- RSI score --
        rsi_score = self._score_rsi(symbol)

        # -- Bollinger Band score --
        bb_score = self._score_bollinger(price_list, current_price)

        # -- Volume score --
        vol_score = self._score_volume(volumes)

        # -- Greeks score --
        greeks_score = self._score_greeks(symbol)

        composite = (
            self._w_rsi * rsi_score
            + self._w_bb * bb_score
            + self._w_vol * vol_score
            + self._w_greeks * greeks_score
        )

        return Signal(
            symbol=symbol,
            composite_score=round(composite, 4),
            rsi_score=round(rsi_score, 4),
            bb_score=round(bb_score, 4),
            volume_score=round(vol_score, 4),
            greeks_score=round(greeks_score, 4),
            triggered=composite >= self._min_composite,
        )

    def reset(self) -> None:
        """Clear all per-symbol state."""
        self._prices.clear()
        self._volumes.clear()
        self._greeks.clear()
        self._rsi_avg_gain.clear()
        self._rsi_avg_loss.clear()
        self._rsi_seeded.clear()

    # ------------------------------------------------------------------
    # Private scoring helpers
    # ------------------------------------------------------------------

    def _score_rsi(self, symbol: str) -> float:
        """Return RSI score in [0, 1]; higher = more oversold."""
        if not self._rsi_seeded.get(symbol, False):
            return 0.0
        avg_loss = self._rsi_avg_loss[symbol]
        if avg_loss == 0.0:
            rsi = 100.0
        else:
            rs = self._rsi_avg_gain[symbol] / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # Score is inversely proportional to RSI; max at oversold threshold
        if rsi >= self._rsi_oversold:
            return 0.0
        return 1.0 - (rsi / self._rsi_oversold)

    def _score_bollinger(self, price_list: list[float], current: float) -> float:
        """Return BB score in [0, 1]; 1.0 when price is at/below lower band."""
        n = min(len(price_list), self._bb_period)
        if n < 2:
            return 0.0
        window = price_list[-n:]
        mean = sum(window) / n
        variance = sum((p - mean) ** 2 for p in window) / n
        std = math.sqrt(variance)
        lower_band = mean - self._bb_mult * std
        if std == 0.0:
            return 0.0
        # Distance below lower band, clamped to [0, 1]
        distance = (lower_band - current) / (self._bb_mult * std)
        return max(0.0, min(1.0, 0.5 + distance))

    def _score_volume(self, volumes: deque[float] | None) -> float:
        """Return volume score in [0, 1]; 1.0 on a clear spike."""
        if volumes is None or len(volumes) < 2:
            return 0.0
        vol_list = list(volumes)
        current_vol = vol_list[-1]
        mean_vol = sum(vol_list[:-1]) / len(vol_list[:-1])
        if mean_vol == 0.0:
            return 0.0
        ratio = current_vol / mean_vol
        if ratio < self._vol_spike_threshold:
            return 0.0
        # Score proportional to spike magnitude, capped at 1.0
        return min(1.0, (ratio - self._vol_spike_threshold) / self._vol_spike_threshold)

    def _score_greeks(self, symbol: str) -> float:
        """Return greeks score in [0, 1]; 1.0 for ideal options parameters."""
        g = self._greeks.get(symbol)
        if g is None:
            return 0.0

        delta = abs(g.get("delta", 0.0))
        iv_pct = g.get("iv_pct", 100.0)
        bid = g.get("bid", 0.0)
        ask = g.get("ask", 0.0)

        # Delta score: ideal is 0.30; symmetric penalty toward delta_min/max
        delta_ideal = (self._delta_min + self._delta_max) / 2.0
        delta_range = (self._delta_max - self._delta_min) / 2.0
        delta_score = max(0.0, 1.0 - abs(delta - delta_ideal) / delta_range)

        # IV score: lower percentile is better
        iv_score = max(0.0, 1.0 - iv_pct / self._iv_pct_max) if iv_pct <= self._iv_pct_max else 0.0

        # Spread score: tighter is better
        mid = (bid + ask) / 2.0 if (bid + ask) > 0 else 1.0
        spread_pct = ((ask - bid) / mid * 100.0) if mid > 0 else 100.0
        spread_score = max(0.0, 1.0 - spread_pct / self._spread_pct_max)

        return (delta_score + iv_score + spread_score) / 3.0


# ---------------------------------------------------------------------------
# TechnicalAnalyzer — public interface
# ---------------------------------------------------------------------------


class TechnicalAnalyzer:
    """Thin wrapper that routes to the C++ extension or pure-Python fallback.

    Args:
        config_path: Path to the strategy YAML configuration file.
    """

    def __init__(self, config_path: str = "config/strategy.yaml") -> None:
        with open(config_path) as fh:
            config = yaml.safe_load(fh)

        if _NATIVE_AVAILABLE:
            logger.debug("TechnicalAnalyzer using native C++ SignalScorer")
            self._scorer = _coe_engine.SignalScorer(config)  # type: ignore[name-defined]
            self._native = True
        else:
            logger.debug("TechnicalAnalyzer using pure-Python SignalScorer")
            self._scorer: _PythonSignalScorer = _PythonSignalScorer(config)
            self._native = False

    # ------------------------------------------------------------------
    # Feed methods
    # ------------------------------------------------------------------

    def feed_bar(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp_ns: int,
    ) -> None:
        """Feed a new price/volume bar for *symbol*.

        Args:
            symbol: Ticker symbol.
            price: Closing price of the bar.
            volume: Volume of the bar.
            timestamp_ns: Bar timestamp in nanoseconds (used by native engine).
        """
        if self._native:
            self._scorer.update_price(symbol, price, timestamp_ns)
            self._scorer.update_volume(symbol, volume, timestamp_ns)
        else:
            self._scorer.update_price(symbol, price)
            self._scorer.update_volume(symbol, volume)

    def feed_greeks(
        self,
        symbol: str,
        delta: float,
        iv_pct: float,
        bid: float,
        ask: float,
    ) -> None:
        """Feed an options greeks snapshot for *symbol*.

        Args:
            symbol: Ticker symbol.
            delta: Absolute delta of the option.
            iv_pct: Implied volatility percentile (0–100).
            bid: Option bid price.
            ask: Option ask price.
        """
        self._scorer.update_greeks(symbol, delta, iv_pct, bid, ask)

    def evaluate(self, symbol: str) -> Signal | None:
        """Evaluate the composite technical signal for *symbol*.

        Args:
            symbol: Ticker symbol.

        Returns:
            ``Signal`` dataclass when sufficient data is available, else None.
        """
        if self._native:
            raw = self._scorer.evaluate(symbol)
            if raw is None:
                return None
            # Assume the native scorer returns an object with the same fields
            return Signal(
                symbol=symbol,
                composite_score=float(raw.composite_score),
                rsi_score=float(raw.rsi_score),
                bb_score=float(raw.bb_score),
                volume_score=float(raw.volume_score),
                greeks_score=float(raw.greeks_score),
                triggered=bool(raw.triggered),
            )
        return self._scorer.evaluate(symbol)

    def reset(self) -> None:
        """Clear all internal state across all symbols."""
        self._scorer.reset()
