"""Summary metrics and optional reference-based ROUGE evaluation."""
from __future__ import annotations

import time
from dataclasses import dataclass

from rouge_score import rouge_scorer


@dataclass(frozen=True)
class SummaryMetrics:
    processing_seconds: float
    word_count: int
    sentence_count: int
    compression_ratio: float


def calculate_metrics(source: str, summary: str, elapsed_seconds: float) -> SummaryMetrics:
    source_words, summary_words = len(source.split()), len(summary.split())
    return SummaryMetrics(max(0.0, elapsed_seconds), summary_words,
                          len([s for s in summary.splitlines() if s.strip()]),
                          summary_words / source_words if source_words else 0.0)


def calculate_rouge(summary: str, reference: str) -> dict[str, dict[str, float]]:
    """Return ROUGE-1/2/L precision, recall and F1 against supplied reference."""
    if not summary.strip() or not reference.strip():
        return {}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {name: {"precision": score.precision, "recall": score.recall, "f1": score.fmeasure}
            for name, score in scores.items()}
