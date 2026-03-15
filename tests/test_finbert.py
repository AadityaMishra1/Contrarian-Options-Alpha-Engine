"""Tests for FinBERT Analyzer.

All tests are written to pass regardless of whether the transformers library
or ProsusAI/finbert model weights are installed.  When the model is absent,
the code returns a neutral fallback so the downstream pipeline can continue.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from signals.finbert import FinBERTAnalyzer, FinBERTResult


class TestFinBERTResult:
    def test_result_fields(self) -> None:
        r = FinBERTResult(
            text="test",
            positive=0.8,
            negative=0.1,
            neutral=0.1,
            score=0.7,
            label="positive",
        )
        assert r.score == pytest.approx(0.7)
        assert r.label == "positive"

    def test_net_score_range_positive(self) -> None:
        r = FinBERTResult(
            text="test",
            positive=1.0,
            negative=0.0,
            neutral=0.0,
            score=1.0,
            label="positive",
        )
        assert -1.0 <= r.score <= 1.0

    def test_net_score_range_negative(self) -> None:
        r = FinBERTResult(
            text="test",
            positive=0.0,
            negative=1.0,
            neutral=0.0,
            score=-1.0,
            label="negative",
        )
        assert -1.0 <= r.score <= 1.0

    def test_result_text_stored(self) -> None:
        r = FinBERTResult(
            text="hello", positive=0.5, negative=0.3, neutral=0.2,
            score=0.2, label="positive",
        )
        assert r.text == "hello"


class TestFinBERTAnalyzer:
    def test_init_succeeds(self) -> None:
        """FinBERTAnalyzer should always be constructable without raising."""
        analyzer = FinBERTAnalyzer()
        assert analyzer is not None

    def test_is_available_returns_bool(self) -> None:
        analyzer = FinBERTAnalyzer()
        assert isinstance(analyzer.is_available, bool)

    def test_analyze_returns_result(self) -> None:
        """analyze() must return a FinBERTResult even without the model."""
        analyzer = FinBERTAnalyzer()
        result = analyzer.analyze("Apple reports strong quarterly earnings")
        assert isinstance(result, FinBERTResult)
        assert -1.0 <= result.score <= 1.0

    def test_analyze_empty_text(self) -> None:
        analyzer = FinBERTAnalyzer()
        result = analyzer.analyze("")
        assert isinstance(result, FinBERTResult)
        # Empty text fallback returns neutral
        assert result.score == pytest.approx(0.0)

    def test_analyze_whitespace_only_text(self) -> None:
        analyzer = FinBERTAnalyzer()
        result = analyzer.analyze("   \n\t  ")
        assert isinstance(result, FinBERTResult)

    def test_analyze_label_is_string(self) -> None:
        analyzer = FinBERTAnalyzer()
        result = analyzer.analyze("Revenue missed estimates by 10%.")
        assert isinstance(result.label, str)
        assert result.label in {"positive", "negative", "neutral"}

    def test_analyze_probabilities_sum_to_one_when_model_loaded(self) -> None:
        """Positive + negative + neutral should sum to ~1.0 when real model used."""
        analyzer = FinBERTAnalyzer()
        if not analyzer.is_available:
            pytest.skip("FinBERT model not installed; skipping probability sum test")
        result = analyzer.analyze("Earnings beat expectations by a wide margin.")
        total = result.positive + result.negative + result.neutral
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_analyze_transcript_qa_empty(self) -> None:
        analyzer = FinBERTAnalyzer()
        score = analyzer.analyze_transcript_qa("")
        assert score == pytest.approx(0.0)

    def test_analyze_transcript_qa_whitespace(self) -> None:
        analyzer = FinBERTAnalyzer()
        score = analyzer.analyze_transcript_qa("   ")
        assert score == pytest.approx(0.0)

    def test_analyze_transcript_qa_returns_float(self) -> None:
        analyzer = FinBERTAnalyzer()
        text = (
            "Q: How are margins trending?\n\n"
            "A: Margins expanded 200bps driven by operational efficiency."
        )
        score = analyzer.analyze_transcript_qa(text)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_analyze_transcript_qa_multiple_paragraphs(self) -> None:
        """Multi-paragraph transcript should yield a single averaged float."""
        analyzer = FinBERTAnalyzer()
        text = "\n\n".join([
            "Revenue grew 15% year over year.",
            "Operating margins expanded due to cost discipline.",
            "We are cautious about macro headwinds in Q4.",
        ])
        score = analyzer.analyze_transcript_qa(text)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    @pytest.mark.skipif(
        not FinBERTAnalyzer().is_available,
        reason="transformers/FinBERT not installed",
    )
    def test_model_loaded_produces_nonzero_scores(self) -> None:
        """With the real model, clearly positive text should have pos > neg."""
        analyzer = FinBERTAnalyzer()
        result = analyzer.analyze(
            "Revenue grew 25% year over year, beating all analyst estimates"
        )
        assert result.positive > result.negative

    @pytest.mark.skipif(
        not FinBERTAnalyzer().is_available,
        reason="transformers/FinBERT not installed",
    )
    def test_negative_text_scores_negative(self) -> None:
        analyzer = FinBERTAnalyzer()
        result = analyzer.analyze(
            "The company reported a massive loss and cut its dividend entirely."
        )
        assert result.negative > result.positive
