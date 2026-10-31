"""Tiny NLP toolkit written from scratch: tokenizer, TF-IDF index, extractive summarizer."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was "
    "what when where which who will with you your we our can do does me my about into than then "
    "they them their there so if not but".split()
)


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def tokenize(text: str, drop_stop: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    return [_stem(t) for t in tokens if not (drop_stop and t in STOPWORDS)]


def keywords(text: str, n: int = 6) -> list[str]:
    counts = Counter(t for t in tokenize(text) if len(t) > 2)
    order = {t: i for i, t in enumerate(dict.fromkeys(tokenize(text)))}
    return sorted(counts, key=lambda t: (-counts[t], order[t]))[:n]


class TfidfIndex:
    """Cosine-similarity retrieval over a small corpus. Rebuilt lazily after additions."""

    def __init__(self) -> None:
        self._docs: list[tuple[list[str], Any]] = []
        self._idf: dict[str, float] = {}
        self._vecs: list[dict[str, float]] = []
        self._dirty = True

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, text: str, payload: Any) -> None:
        self._docs.append((tokenize(text), payload))
        self._dirty = True

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        vec = {t: c * self._idf[t] for t, c in Counter(tokens).items() if t in self._idf}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm else {}

    def _build(self) -> None:
        n = len(self._docs)
        df: Counter[str] = Counter()
        for tokens, _ in self._docs:
            df.update(set(tokens))
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self._vecs = [self._vectorize(tokens) for tokens, _ in self._docs]
        self._dirty = False

    def query(self, text: str, k: int = 3, min_score: float = 0.0) -> list[tuple[float, Any]]:
        if not self._docs:
            return []
        if self._dirty:
            self._build()
        q = self._vectorize(tokenize(text))
        scored = []
        for vec, (_, payload) in zip(self._vecs, self._docs):
            score = sum(w * vec.get(t, 0.0) for t, w in q.items())
            if score > min_score:
                scored.append((score, payload))
        scored.sort(key=lambda s: -s[0])
        return scored[:k]


def summarize_text(text: str, n: int = 3) -> str:
    """Frequency-based extractive summary that preserves original sentence order."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    sentences = [s.strip() for s in parts if len(s.strip()) > 20]
    if not sentences:
        return text.strip()[:500]
    if len(sentences) <= n:
        return " ".join(sentences)
    freq = Counter(tokenize(text))

    def score(s: str) -> float:
        toks = tokenize(s)
        return sum(freq[t] for t in toks) / (math.sqrt(len(toks)) + 1)

    top = sorted(range(len(sentences)), key=lambda i: -score(sentences[i]))[:n]
    return " ".join(sentences[i] for i in sorted(top))
