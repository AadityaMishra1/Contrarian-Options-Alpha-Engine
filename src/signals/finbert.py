"""FinBERT Earnings Sentiment — replaces Claude API.

Uses the ProsusAI/finbert model for deterministic, free, fast sentiment
analysis on earnings call Q&A transcripts. Runs locally via HuggingFace
transformers pipeline.

Reference: Araci (2019). "FinBERT: Financial Sentiment Analysis with
Pre-trained Language Models." arXiv:1908.10063.

Why FinBERT over Claude/GPT:
- Free (open source, runs locally)
- Fast (~5ms per inference vs 500ms-2s API call)
- Deterministic (same input -> same output)
- Backtestable (frozen model weights)
- Documented predictive power in earnings context
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Lazy-load transformers (heavy import)
_pipeline = None


def _get_pipeline():
    """Lazy-initialize the FinBERT pipeline."""
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            _pipeline = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,       # return all scores
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT model loaded successfully")
        except ImportError:
            logger.warning("transformers not installed. FinBERT unavailable.")
            _pipeline = None
        except Exception as exc:
            logger.error("Failed to load FinBERT: %s", exc)
            _pipeline = None
    return _pipeline


@dataclass
class FinBERTResult:
    """Sentiment classification result."""

    text: str
    positive: float    # probability
    negative: float
    neutral: float
    score: float       # net sentiment: positive - negative, in [-1, +1]
    label: str         # "positive", "negative", or "neutral"


class FinBERTAnalyzer:
    """Analyzes earnings-related text with FinBERT.

    Designed to be applied to the Q&A section of earnings call
    transcripts, where analyst questions and management responses
    reveal more than prepared remarks.
    """

    def __init__(self) -> None:
        self._pipeline = _get_pipeline()

    @property
    def is_available(self) -> bool:
        """Whether the FinBERT model was loaded successfully."""
        return self._pipeline is not None

    def analyze(self, text: str) -> FinBERTResult:
        """Analyze a single text passage.

        Args:
            text: Financial text to classify.

        Returns:
            FinBERTResult with per-class probabilities and net score.
        """
        if not self.is_available:
            return FinBERTResult(
                text=text[:100],
                positive=0.33,
                negative=0.33,
                neutral=0.34,
                score=0.0,
                label="neutral",
            )

        # Truncate to 512 tokens max
        results = self._pipeline(text[:2000])

        scores = {r["label"]: r["score"] for r in results[0]}
        pos = scores.get("positive", 0.0)
        neg = scores.get("negative", 0.0)
        neu = scores.get("neutral", 0.0)

        net_score = pos - neg
        label = max(scores, key=scores.get)

        return FinBERTResult(
            text=text[:100],
            positive=pos,
            negative=neg,
            neutral=neu,
            score=net_score,
            label=label,
        )

    def analyze_transcript_qa(self, qa_text: str) -> float:
        """Analyze an earnings call Q&A section and return net sentiment.

        Splits the Q&A into chunks (by paragraph/sentence), analyzes each,
        and returns the average net sentiment score.

        Args:
            qa_text: Full text of the Q&A section.

        Returns:
            Net sentiment score in [-1, +1]. Positive = bullish.
        """
        if not qa_text.strip():
            return 0.0

        # Split into paragraphs/chunks
        chunks = [p.strip() for p in qa_text.split("\n\n") if len(p.strip()) > 50]
        if not chunks:
            chunks = [qa_text[:2000]]

        # Limit to 20 chunks for speed
        chunks = chunks[:20]

        scores = []
        for chunk in chunks:
            result = self.analyze(chunk)
            scores.append(result.score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)
