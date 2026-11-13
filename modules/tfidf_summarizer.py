"""Extractive summarization using sentence-level TF-IDF salience."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from modules.preprocessing import preprocess


def summarize_tfidf(text: str, sentence_count: int = 5, percentage: float | None = None) -> list[str]:
    sentences = preprocess(text)
    if not sentences:
        return []
    if percentage is not None:
        sentence_count = max(1, round(len(sentences) * percentage))
    count = min(len(sentences), max(1, int(sentence_count)))
    documents = [" ".join(sentence.normalized_tokens) for sentence in sentences]
    nonempty = [i for i, doc in enumerate(documents) if doc]
    if not nonempty:
        return [s.original for s in sentences[:count]]
    scores = [0.0] * len(sentences)
    matrix = TfidfVectorizer().fit_transform([documents[i] for i in nonempty])
    row_scores = matrix.sum(axis=1).A1
    for i, score in zip(nonempty, row_scores):
        scores[i] = float(score)
    chosen = sorted(sorted(range(len(sentences)), key=lambda i: (-scores[i], i))[:count])
    return [sentences[i].original for i in chosen]
