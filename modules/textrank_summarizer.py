"""Extractive summarization with a weighted sentence similarity graph."""
from __future__ import annotations

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.preprocessing import preprocess


def summarize_textrank(text: str, sentence_count: int = 5, percentage: float | None = None) -> list[str]:
    sentences = preprocess(text)
    if not sentences:
        return []
    if percentage is not None:
        sentence_count = max(1, round(len(sentences) * percentage))
    count = min(len(sentences), max(1, int(sentence_count)))
    docs = [" ".join(sentence.normalized_tokens) for sentence in sentences]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(sentences)))
    nonempty = [i for i, doc in enumerate(docs) if doc]
    if len(nonempty) > 1:
        matrix = TfidfVectorizer().fit_transform([docs[i] for i in nonempty])
        similarities = cosine_similarity(matrix)
        for row, i in enumerate(nonempty):
            for col in range(row + 1, len(nonempty)):
                weight = float(similarities[row, col])
                if weight > 0:
                    graph.add_edge(i, nonempty[col], weight=weight)
    scores = nx.pagerank(graph, weight="weight") if graph.number_of_nodes() > 1 else {0: 1.0}
    chosen = sorted(sorted(range(len(sentences)), key=lambda i: (-scores.get(i, 0), i))[:count])
    return [sentences[i].original for i in chosen]
