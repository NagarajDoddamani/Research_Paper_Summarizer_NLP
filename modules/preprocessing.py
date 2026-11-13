"""Corpus-independent sentence and word preprocessing with NLTK."""
from __future__ import annotations

import re
from dataclasses import dataclass

from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_SENTENCE_TOKENIZER = PunktSentenceTokenizer()
_STOP_WORDS = set(ENGLISH_STOP_WORDS)
_WORD_PATTERN = re.compile(r"(?u)\b\w+\b")


@dataclass(frozen=True)
class Sentence:
    index: int
    original: str
    normalized_tokens: tuple[str, ...]


def tokenize_words(text: str) -> list[str]:
    """Tokenize words using NLTK without requiring downloaded NLTK data."""
    if not text or not text.strip():
        return []
    # preserve_line=True avoids a runtime dependency on the punkt_tab corpus.
    return [token.casefold() for token in word_tokenize(text, preserve_line=True)
            if _WORD_PATTERN.fullmatch(token)]


def split_sentences(text: str) -> list[str]:
    """Split text into nonempty sentences while retaining source sentence text."""
    if not text or not text.strip():
        return []
    return [part.strip() for part in _SENTENCE_TOKENIZER.tokenize(text.strip()) if part.strip()]


def normalize_sentence(sentence: str) -> tuple[str, ...]:
    """Lowercase/tokenize a sentence and remove English stopwords."""
    return tuple(token for token in tokenize_words(sentence)
                 if token not in _STOP_WORDS and len(token) > 1)


def preprocess(text: str) -> list[Sentence]:
    """Return indexed sentences with original text and normalized tokens."""
    return [Sentence(i, original, normalize_sentence(original))
            for i, original in enumerate(split_sentences(text))]
