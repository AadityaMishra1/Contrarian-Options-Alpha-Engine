"""Sentiment filter using Claude AI to classify price dips.

Classifies recent news headlines for a ticker as either a temporary,
buyable dip or a fundamental deterioration that should be avoided —
supporting the contrarian options strategy's entry filter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_CLASSIFICATION_TEMPORARY = "TEMPORARY_DIP"
_CLASSIFICATION_FUNDAMENTAL = "FUNDAMENTAL_PROBLEM"

_SYSTEM_PROMPT = """\
You are a financial analyst for a contrarian options trading strategy.
The strategy buys cheap OTM put options on beaten-down stocks only when
the decline is driven by temporary fear or overreaction — NOT when the
business itself is deteriorating.

Given recent news headlines about a stock, classify the current price dip as:
- TEMPORARY_DIP: The decline is driven by overreaction, sector rotation, or
  short-term fear. The underlying business fundamentals remain intact. The stock
  is likely to recover within days to weeks.
- FUNDAMENTAL_PROBLEM: The decline reflects genuine deterioration — earnings
  collapse, regulatory action, fraud, accounting restatement, structural
  disruption, or management crisis.

Be conservative: when uncertain, lean toward FUNDAMENTAL_PROBLEM to protect
capital.

Respond ONLY with valid JSON in exactly this format:
{"classification": "TEMPORARY_DIP"|"FUNDAMENTAL_PROBLEM", "confidence": 0.0-1.0, "reasoning": "brief explanation"}\
"""

_USER_PROMPT_TEMPLATE = """\
Ticker: {symbol}

Recent news headlines:
{headlines}

Classify this price dip.\
"""


@dataclass
class SentimentResult:
    """Result of a sentiment classification call.

    Attributes:
        classification: Either ``TEMPORARY_DIP`` or ``FUNDAMENTAL_PROBLEM``.
        confidence: Model confidence in [0.0, 1.0].
        reasoning: Brief natural-language explanation from the model.
        cached: Whether this result was served from the TTL cache.
    """

    classification: str
    confidence: float
    reasoning: str
    cached: bool = field(default=False)

    @property
    def is_buyable(self) -> bool:
        """True when classification is TEMPORARY_DIP."""
        return self.classification == _CLASSIFICATION_TEMPORARY


# ---------------------------------------------------------------------------
# Simple TTL cache (avoids cachetools dependency)
# ---------------------------------------------------------------------------


class _TTLCache:
    """Minimal TTL cache keyed by arbitrary strings.

    Args:
        ttl: Time-to-live in seconds for each entry.
        maxsize: Maximum number of entries before old items are evicted.
    """

    def __init__(self, ttl: int = 3600, maxsize: int = 256) -> None:
        self._ttl = ttl
        self._maxsize = maxsize
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any | None:
        """Return cached value or None if missing / expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Store value under key, evicting oldest entry when at capacity."""
        if len(self._store) >= self._maxsize and key not in self._store:
            # Evict the oldest entry
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        self._store[key] = (value, time.monotonic())


# ---------------------------------------------------------------------------
# Rate limiter — token-bucket capped at N calls per minute
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Simple asyncio-compatible rate limiter.

    Args:
        calls_per_minute: Maximum allowed calls per 60-second window.
    """

    def __init__(self, calls_per_minute: int = 10) -> None:
        self._interval = 60.0 / calls_per_minute
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until the rate limit allows the next call."""
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# SentimentFilter
# ---------------------------------------------------------------------------


class SentimentFilter:
    """Classifies stock news as a temporary dip vs. fundamental problem.

    Uses the Anthropic Claude API (claude-haiku model for cost efficiency)
    with a TTL cache and per-symbol rate limiting to avoid redundant calls.

    Args:
        api_key: Anthropic API key. Falls back to the ``ANTHROPIC_API_KEY``
            environment variable when not provided.
        model: Claude model identifier.
        confidence_threshold: Minimum confidence required for a result to be
            considered actionable. Results below this are logged as uncertain.
        cache_ttl: Seconds to cache results per symbol.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        confidence_threshold: float = 0.7,
        cache_ttl: int = 3600,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key is required. Pass api_key= or set the "
                "ANTHROPIC_API_KEY environment variable."
            )
        self._api_key = resolved_key
        self.model = model
        self.confidence_threshold = confidence_threshold
        self._cache: _TTLCache = _TTLCache(ttl=cache_ttl)
        self._rate_limiter = _RateLimiter(calls_per_minute=10)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def classify(
        self, symbol: str, news_headlines: list[str]
    ) -> SentimentResult:
        """Classify recent headlines for *symbol* as buyable or not.

        Args:
            symbol: Ticker symbol (e.g. ``"AAPL"``).
            news_headlines: List of recent news headline strings, newest first.

        Returns:
            A ``SentimentResult`` with classification, confidence, and reasoning.
        """
        cache_key = f"{symbol}:{hash(tuple(news_headlines[:10]))}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Returning cached sentiment for %s", symbol)
            return SentimentResult(
                classification=cached["classification"],
                confidence=cached["confidence"],
                reasoning=cached["reasoning"],
                cached=True,
            )

        await self._rate_limiter.acquire()

        try:
            raw = await self._call_claude(symbol, news_headlines)
        except Exception:
            logger.exception("Claude API call failed for %s", symbol)
            return SentimentResult(
                classification="FUNDAMENTAL_PROBLEM",
                confidence=0.0,
                reasoning="API call failed — defaulting to safe classification",
                cached=False,
            )

        result = self._parse_response(symbol, raw)

        if result.confidence < self.confidence_threshold:
            logger.warning(
                "Sentiment for %s has low confidence (%.2f < %.2f): %s",
                symbol,
                result.confidence,
                self.confidence_threshold,
                result.reasoning,
            )

        self._cache.set(cache_key, {
            "classification": result.classification,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        })
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_claude(
        self, symbol: str, headlines: list[str]
    ) -> str:
        """Make an async HTTP request to the Anthropic Messages API.

        Args:
            symbol: Ticker symbol for prompt interpolation.
            headlines: List of headline strings.

        Returns:
            Raw text content from the model's first message block.

        Raises:
            RuntimeError: If the API call fails or returns a non-200 status.
        """
        import aiohttp  # local import to keep top-level deps light

        headline_block = "\n".join(
            f"{i + 1}. {h}" for i, h in enumerate(headlines[:15])
        )
        user_message = _USER_PROMPT_TEMPLATE.format(
            symbol=symbol, headlines=headline_block
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 256,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        url = "https://api.anthropic.com/v1/messages"
        async with aiohttp.ClientSession() as session, session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"Anthropic API error {resp.status} for {symbol}: {body}"
                )
            data = await resp.json()

        # Extract text from the first content block
        content_blocks = data.get("content", [])
        if not content_blocks:
            raise RuntimeError(f"Empty content in Anthropic response for {symbol}")
        return str(content_blocks[0].get("text", ""))

    def _parse_response(self, symbol: str, raw: str) -> SentimentResult:
        """Parse Claude's JSON response into a SentimentResult.

        Falls back to FUNDAMENTAL_PROBLEM with low confidence if parsing fails,
        ensuring the strategy errs on the side of caution.

        Args:
            symbol: Ticker symbol (used only for logging).
            raw: Raw text output from Claude.

        Returns:
            Parsed ``SentimentResult``.
        """
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        try:
            parsed: dict[str, Any] = json.loads(text)
            classification = str(parsed.get("classification", _CLASSIFICATION_FUNDAMENTAL))
            if classification not in (_CLASSIFICATION_TEMPORARY, _CLASSIFICATION_FUNDAMENTAL):
                logger.warning(
                    "Unexpected classification '%s' for %s — defaulting to %s",
                    classification,
                    symbol,
                    _CLASSIFICATION_FUNDAMENTAL,
                )
                classification = _CLASSIFICATION_FUNDAMENTAL

            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            reasoning = str(parsed.get("reasoning", "No reasoning provided."))

        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.error(
                "Failed to parse sentiment JSON for %s: %s — raw: %s",
                symbol, exc, raw[:200],
            )
            return SentimentResult(
                classification=_CLASSIFICATION_FUNDAMENTAL,
                confidence=0.0,
                reasoning=f"Parse error: {exc}",
            )

        return SentimentResult(
            classification=classification,
            confidence=confidence,
            reasoning=reasoning,
        )
