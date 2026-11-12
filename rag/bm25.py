"""Keyword retrieval: BM25 in about 60 lines, no dependency.

Why bother when we already have vectors? Because embeddings are bad at exact
strings - scheme names, acronyms, numbers, "Rs 5 lakh", "PM-JAY", "14555".
Those are exactly the queries a citizen types. BM25 nails them. Fusing the two
is what people mean by "hybrid search", and it is the highest-value upgrade
you can make to a naive RAG system.
"""
from __future__ import annotations

import math
from collections import Counter

from .embedder import tokenize


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = documents
        self.n = len(documents)
        self.avgdl = (sum(len(d) for d in documents) / self.n) if self.n else 0.0
        self.term_freqs = [Counter(d) for d in documents]
        doc_freq: Counter[str] = Counter()
        for doc in documents:
            doc_freq.update(set(doc))
        # Lucene-style IDF: always positive, so a rare term cannot score below 0.
        self.idf = {
            term: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def score(self, query_tokens: list[str], index: int) -> float:
        freqs = self.term_freqs[index]
        length = len(self.docs[index]) or 1
        total = 0.0
        for term in query_tokens:
            if term not in freqs:
                continue
            tf = freqs[term]
            denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
            total += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / denom
        return total

    def rank(
        self,
        query: str,
        top_k: int = 20,
        allowed: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Return [(row_index, score)] best first, optionally restricted."""
        tokens = tokenize(query)
        if not tokens:
            return []
        rows = range(self.n) if allowed is None else allowed
        scored = [(i, self.score(tokens, i)) for i in rows]
        scored = [pair for pair in scored if pair[1] > 0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    @classmethod
    def from_texts(cls, texts: list[str]) -> "BM25":
        return cls([tokenize(t) for t in texts])
