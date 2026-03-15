"""Tests for SentimentFilter, SentimentResult, _TTLCache, and _RateLimiter.

The Anthropic HTTP API is fully mocked via aiohttp patches so no live
credentials or network access are required.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals.sentiment import (
    SentimentFilter,
    SentimentResult,
    _RateLimiter,
    _TTLCache,
)


# ---------------------------------------------------------------------------
# SentimentResult dataclass
# ---------------------------------------------------------------------------


class TestSentimentResult:
    def test_is_buyable_true_for_temporary_dip(self) -> None:
        result = SentimentResult(
            classification="TEMPORARY_DIP",
            confidence=0.85,
            reasoning="Sector rotation",
        )
        assert result.is_buyable is True

    def test_is_buyable_false_for_fundamental_problem(self) -> None:
        result = SentimentResult(
            classification="FUNDAMENTAL_PROBLEM",
            confidence=0.90,
            reasoning="Revenue collapse",
        )
        assert result.is_buyable is False

    def test_cached_defaults_to_false(self) -> None:
        result = SentimentResult(
            classification="TEMPORARY_DIP",
            confidence=0.8,
            reasoning="test",
        )
        assert result.cached is False

    def test_cached_can_be_set_true(self) -> None:
        result = SentimentResult(
            classification="TEMPORARY_DIP",
            confidence=0.8,
            reasoning="test",
            cached=True,
        )
        assert result.cached is True


# ---------------------------------------------------------------------------
# _TTLCache
# ---------------------------------------------------------------------------


class TestTTLCache:
    def test_get_returns_none_for_missing_key(self) -> None:
        cache = _TTLCache(ttl=60)
        assert cache.get("absent") is None

    def test_set_and_get_roundtrip(self) -> None:
        cache = _TTLCache(ttl=60)
        cache.set("key", {"val": 42})
        assert cache.get("key") == {"val": 42}

    def test_expired_entry_returns_none(self) -> None:
        cache = _TTLCache(ttl=0)  # expires immediately
        cache.set("k", "v")
        time.sleep(0.01)
        assert cache.get("k") is None

    def test_maxsize_evicts_oldest(self) -> None:
        cache = _TTLCache(ttl=3600, maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # should evict "a"
        assert len(cache._store) == 2
        # "c" must be present
        assert cache.get("c") == 3

    def test_overwrite_existing_key_does_not_grow_store(self) -> None:
        cache = _TTLCache(ttl=3600, maxsize=5)
        cache.set("x", 1)
        cache.set("x", 2)
        assert len(cache._store) == 1
        assert cache.get("x") == 2


# ---------------------------------------------------------------------------
# _RateLimiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_does_not_raise(self) -> None:
        limiter = _RateLimiter(calls_per_minute=600)  # very high — no waiting
        await limiter.acquire()  # should complete without error

    @pytest.mark.asyncio
    async def test_sequential_acquires_succeed(self) -> None:
        limiter = _RateLimiter(calls_per_minute=600)
        await limiter.acquire()
        await limiter.acquire()


# ---------------------------------------------------------------------------
# SentimentFilter — constructor
# ---------------------------------------------------------------------------


class TestSentimentFilterInit:
    def test_requires_api_key(self) -> None:
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(ValueError, match="Anthropic API key"):
            SentimentFilter(api_key=None)

    def test_accepts_explicit_api_key(self) -> None:
        sf = SentimentFilter(api_key="test_key")
        assert sf is not None

    def test_accepts_env_var(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env_key"}):
            sf = SentimentFilter()
            assert sf is not None

    def test_default_model(self) -> None:
        sf = SentimentFilter(api_key="test_key")
        assert "claude" in sf.model.lower()

    def test_custom_confidence_threshold(self) -> None:
        sf = SentimentFilter(api_key="test_key", confidence_threshold=0.9)
        assert sf.confidence_threshold == 0.9


# ---------------------------------------------------------------------------
# SentimentFilter._parse_response — pure JSON parsing, no network
# ---------------------------------------------------------------------------


class TestParseResponse:
    def _sf(self) -> SentimentFilter:
        return SentimentFilter(api_key="test_key")

    def test_parses_temporary_dip(self) -> None:
        sf = self._sf()
        raw = json.dumps({
            "classification": "TEMPORARY_DIP",
            "confidence": 0.85,
            "reasoning": "Sector rotation",
        })
        result = sf._parse_response("AAPL", raw)
        assert result.classification == "TEMPORARY_DIP"
        assert result.confidence == pytest.approx(0.85)
        assert result.reasoning == "Sector rotation"

    def test_parses_fundamental_problem(self) -> None:
        sf = self._sf()
        raw = json.dumps({
            "classification": "FUNDAMENTAL_PROBLEM",
            "confidence": 0.90,
            "reasoning": "Revenue collapse",
        })
        result = sf._parse_response("XYZ", raw)
        assert result.classification == "FUNDAMENTAL_PROBLEM"
        assert result.confidence == pytest.approx(0.90)

    def test_invalid_json_returns_safe_default(self) -> None:
        sf = self._sf()
        result = sf._parse_response("AAPL", "not json at all")
        assert result.classification == "FUNDAMENTAL_PROBLEM"
        assert result.confidence == pytest.approx(0.0)

    def test_unknown_classification_defaults_to_fundamental(self) -> None:
        sf = self._sf()
        raw = json.dumps({
            "classification": "UNKNOWN_VALUE",
            "confidence": 0.80,
            "reasoning": "test",
        })
        result = sf._parse_response("AAPL", raw)
        assert result.classification == "FUNDAMENTAL_PROBLEM"

    def test_confidence_clamped_to_1(self) -> None:
        sf = self._sf()
        raw = json.dumps({
            "classification": "TEMPORARY_DIP",
            "confidence": 1.5,  # > 1 — should be clamped
            "reasoning": "test",
        })
        result = sf._parse_response("AAPL", raw)
        assert result.confidence <= 1.0

    def test_confidence_clamped_to_0(self) -> None:
        sf = self._sf()
        raw = json.dumps({
            "classification": "TEMPORARY_DIP",
            "confidence": -0.5,  # < 0 — should be clamped
            "reasoning": "test",
        })
        result = sf._parse_response("AAPL", raw)
        assert result.confidence >= 0.0

    def test_strips_markdown_code_fences(self) -> None:
        sf = self._sf()
        payload = json.dumps({
            "classification": "TEMPORARY_DIP",
            "confidence": 0.75,
            "reasoning": "test",
        })
        raw = f"```json\n{payload}\n```"
        result = sf._parse_response("AAPL", raw)
        assert result.classification == "TEMPORARY_DIP"

    def test_missing_reasoning_key_defaults(self) -> None:
        sf = self._sf()
        raw = json.dumps({"classification": "TEMPORARY_DIP", "confidence": 0.7})
        result = sf._parse_response("AAPL", raw)
        assert isinstance(result.reasoning, str)


# ---------------------------------------------------------------------------
# SentimentFilter.classify — mocked HTTP via aiohttp patch
# ---------------------------------------------------------------------------


def _make_aiohttp_mock(response_payload: dict) -> MagicMock:
    """Return a mock for aiohttp.ClientSession that returns response_payload on post()."""
    mock_session = MagicMock()
    resp = MagicMock()
    resp.status = 200
    # Anthropic response shape: {"content": [{"text": "<json string>"}]}
    resp.json = AsyncMock(return_value={
        "content": [{"text": json.dumps(response_payload)}]
    })
    resp.raise_for_status = MagicMock()

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_session.post.return_value = ctx

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    return session_ctx


class TestSentimentFilterClassify:
    @pytest.mark.asyncio
    async def test_classify_temporary_dip(self) -> None:
        sf = SentimentFilter(api_key="test_key")
        payload = {"classification": "TEMPORARY_DIP", "confidence": 0.85, "reasoning": "test"}
        session_ctx = _make_aiohttp_mock(payload)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = await sf.classify("AAPL", ["Apple dips on sector rotation"])

        assert result.classification == "TEMPORARY_DIP"
        assert result.confidence >= 0.7
        assert result.cached is False

    @pytest.mark.asyncio
    async def test_classify_fundamental_problem(self) -> None:
        sf = SentimentFilter(api_key="test_key")
        payload = {"classification": "FUNDAMENTAL_PROBLEM", "confidence": 0.90, "reasoning": "bad"}
        session_ctx = _make_aiohttp_mock(payload)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = await sf.classify("XYZ", ["XYZ reports 80% revenue decline"])

        assert result.classification == "FUNDAMENTAL_PROBLEM"

    @pytest.mark.asyncio
    async def test_second_call_uses_cache(self) -> None:
        """The second classify() for the same symbol+headlines must be served from cache."""
        sf = SentimentFilter(api_key="test_key")
        payload = {"classification": "TEMPORARY_DIP", "confidence": 0.85, "reasoning": "test"}
        session_ctx = _make_aiohttp_mock(payload)

        with patch("aiohttp.ClientSession", return_value=session_ctx) as mock_cls:
            headlines = ["news item one"]
            result1 = await sf.classify("AAPL", headlines)
            result2 = await sf.classify("AAPL", headlines)

        # Second result must be flagged as cached
        assert result2.cached is True
        assert result2.classification == result1.classification
        # aiohttp.ClientSession() should have been instantiated only once
        assert mock_cls.call_count == 1

    @pytest.mark.asyncio
    async def test_different_headlines_bypass_cache(self) -> None:
        """Different headlines for the same symbol produce a fresh API call."""
        sf = SentimentFilter(api_key="test_key")
        payload = {"classification": "TEMPORARY_DIP", "confidence": 0.85, "reasoning": "test"}

        call_count = 0

        class _FakeSessionCtx:
            def __init__(self) -> None:
                nonlocal call_count
                call_count += 1

            async def __aenter__(self) -> MagicMock:
                mock_session = MagicMock()
                resp = MagicMock()
                resp.status = 200
                resp.json = AsyncMock(return_value={"content": [{"text": json.dumps(payload)}]})
                ctx = MagicMock()
                ctx.__aenter__ = AsyncMock(return_value=resp)
                ctx.__aexit__ = AsyncMock(return_value=False)
                mock_session.post.return_value = ctx
                return mock_session

            async def __aexit__(self, *args: object) -> bool:
                return False

        with patch("aiohttp.ClientSession", side_effect=_FakeSessionCtx):
            await sf.classify("AAPL", ["headline A"])
            result2 = await sf.classify("AAPL", ["headline B — completely different"])

        assert result2.cached is False
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_api_500_error_returns_fundamental_default(self) -> None:
        """An HTTP 500 response from the API should produce a safe FUNDAMENTAL_PROBLEM default."""
        sf = SentimentFilter(api_key="test_key")

        mock_session = MagicMock()
        resp = MagicMock()
        resp.status = 500
        resp.text = AsyncMock(return_value="Internal Server Error")

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.post.return_value = ctx

        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = await sf.classify("AAPL", ["some news"])

        assert result.classification == "FUNDAMENTAL_PROBLEM"
        assert result.confidence == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_network_exception_returns_fundamental_default(self) -> None:
        """A network-level exception should produce a safe FUNDAMENTAL_PROBLEM default."""
        sf = SentimentFilter(api_key="test_key")

        mock_session = MagicMock()
        mock_session.post.side_effect = Exception("connection refused")

        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = await sf.classify("AAPL", ["some news"])

        assert result.classification == "FUNDAMENTAL_PROBLEM"
        assert result.confidence == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_low_confidence_result_still_returned(self) -> None:
        """Low-confidence results are returned (just logged as warnings)."""
        sf = SentimentFilter(api_key="test_key", confidence_threshold=0.7)
        payload = {"classification": "TEMPORARY_DIP", "confidence": 0.50, "reasoning": "uncertain"}
        session_ctx = _make_aiohttp_mock(payload)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = await sf.classify("AAPL", ["ambiguous news"])

        # Low-confidence result still returned — strategy can choose to ignore it
        assert result.classification == "TEMPORARY_DIP"
        assert result.confidence == pytest.approx(0.50)

    @pytest.mark.asyncio
    async def test_empty_headlines_list_handled(self) -> None:
        """An empty headlines list must not raise — produces a valid result."""
        sf = SentimentFilter(api_key="test_key")
        payload = {"classification": "FUNDAMENTAL_PROBLEM", "confidence": 0.6, "reasoning": "no news"}
        session_ctx = _make_aiohttp_mock(payload)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = await sf.classify("AAPL", [])

        assert result is not None
        assert result.classification in ("TEMPORARY_DIP", "FUNDAMENTAL_PROBLEM")
