"""Signal generation modules for the Contrarian Options Alpha Engine v2.

v2 signals are research-backed:
- HV-IV gap scanner (Goyal-Saretto 2009)
- Regime classifier (Baltussen et al. 2018)
- Skew tracker (Xing, Zhang, Zhao 2010)
- Earnings scanner (Bernard-Thomas 1989)
- FinBERT sentiment (Araci 2019)

Legacy v1 modules (screener, sentiment, technicals, options_chain) are
retained for backward compatibility but deprecated for live trading.
"""
from __future__ import annotations

from .earnings_scanner import EarningsCandidate, EarningsScanner
from .finbert import FinBERTAnalyzer, FinBERTResult

# v2 signal modules
from .hv_iv_scanner import HVIVCandidate, HVIVScanner
from .index_scanner import IndexScanner, IndexSignal
from .options_chain import OptionsChainAnalyzer
from .regime import Regime, RegimeClassifier, RegimeState

# v1 legacy (deprecated for live use, kept for backtesting)
from .screener import OptionsScreener
from .sentiment import SentimentFilter
from .skew import SkewReading, SkewTracker
from .technicals import TechnicalAnalyzer

__all__ = [
    # v2
    "HVIVScanner",
    "HVIVCandidate",
    "IndexScanner",
    "IndexSignal",
    "RegimeClassifier",
    "Regime",
    "RegimeState",
    "SkewTracker",
    "SkewReading",
    "EarningsScanner",
    "EarningsCandidate",
    "FinBERTAnalyzer",
    "FinBERTResult",
    # v1 legacy
    "OptionsScreener",
    "SentimentFilter",
    "TechnicalAnalyzer",
    "OptionsChainAnalyzer",
]
