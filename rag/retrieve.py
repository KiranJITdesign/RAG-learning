"""Step 1 of answering: retrieval.

Two retrievers run in parallel over the same chunks:

    dense   - cosine similarity between embeddings (means "sounds like")
    bm25    - keyword overlap with IDF weighting (means "contains the words")

Their results are merged with Reciprocal Rank Fusion, which needs no score
normalisation and no tuning: a chunk that both retrievers like wins. You then
get one ranked list with full provenance, so when an answer is wrong you can
see whether retrieval or generation was at fault.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .bm25 import BM25
from .embedder import Embedder, tokenize
from .store import StoredChunk, VectorStore

STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "being",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "may", "might", "must", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "it", "they", "them", "their", "this", "that", "these", "those",
    "for", "of", "to", "in", "on", "at", "by", "with", "from", "and", "or",
    "but", "if", "then", "than", "as", "so", "not", "no", "any", "what",
    "which", "who", "whom", "whose", "when", "where", "why", "how", "get",
    "got", "give", "given", "tell", "please", "about", "there", "here", "also",
    "much", "many", "under", "into", "does", "scheme", "schemes", "eligible",
    "eligibility", "apply", "available", "has", "have", "had", "want", "need",
    "looking", "find", "know", "there", "still", "anyone", "someone", "please",
}


@dataclass
class Hit:
    chunk: StoredChunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float = 0.0
    bm25_score: float = 0.0

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


def content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2}


def answerability(
    query: str, texts: list[str], idf: dict[str, float], n_docs: int
) -> tuple[float, list[str], list[str]]:
    """Should we answer this at all? Returns (coverage, matched, missing).

    Coverage is IDF-weighted, so a rare word counts for more than a common one:
    matching "mgnrega" is evidence, matching "account" is not. Rare words that
    are absent from the corpus at all get the maximum IDF, which is why an
    off-topic question scores badly even when one common word overlaps.

    This is a heuristic, and it is worth knowing what it cannot do. Because it
    is lexical, it fails on questions whose words differ from the document's
    words ("no PF" instead of "unorganised sector"). In extractive mode that is
    harmless - the answer is a quotation, so it cannot be fabricated. With an
    LLM in the loop, the prompt's NOT_FOUND rule and the citation check are the
    real defence, and this gate is the cheap first pass.
    """
    terms = content_terms(query)
    if not terms:
        return 0.0, [], []
    vocabulary: set[str] = set()
    for text in texts:
        vocabulary.update(tokenize(text))
    max_idf = math.log(1.0 + (n_docs + 0.5) / 0.5)  # IDF of a term seen nowhere

    def weight(term: str) -> float:
        return idf.get(term, max_idf)

    matched = sorted(t for t in terms if t in vocabulary)
    missing = sorted(t for t in terms if t not in vocabulary)
    total = sum(weight(t) for t in terms) or 1.0
    coverage = sum(weight(t) for t in matched) / total
    return coverage, matched, missing


def title_evidence(query: str, titles: list[str]) -> list[str]:
    """Query words that appear in a document *title*.

    Matching "pension" against the title "Atal Pension Yojana" is far stronger
    evidence than matching "capital" against the phrase "working capital loan"
    buried in one paragraph. This is what lets a single strong word carry a
    question on its own.
    """
    vocabulary: set[str] = set()
    for title in titles:
        vocabulary.update(tokenize(title))
    return sorted(t for t in content_terms(query) if t in vocabulary)


def should_abstain(
    coverage: float,
    matched: list[str],
    title_hits: list[str],
    min_coverage: float,
) -> tuple[bool, str]:
    """The "do I know enough to answer?" decision, in one readable place."""
    if not matched:
        return True, "none of the question's words appear in the retrieved text"
    if coverage < min_coverage and not title_hits:
        return True, f"weak lexical overlap ({coverage:.0%})"
    if len(matched) < 2 and not title_hits:
        return True, "only one incidental word matched, and not a scheme name"
    return False, ""


def _title_boost(query: str, chunk: StoredChunk) -> float:
    """Small, explainable rerank: a query term in the heading is strong signal."""
    terms = content_terms(query)
    if not terms:
        return 1.0
    heading = f"{chunk.title} {chunk.section}".lower()
    hits = sum(1 for t in terms if t in heading)
    return 1.0 + 0.15 * hits


def hybrid_search(
    store: VectorStore,
    embedder: Embedder,
    bm25: BM25,
    query: str,
    top_k: int = 5,
    candidates: int = 20,
    filters: dict[str, str] | None = None,
    mode: str = "hybrid",
    rrf_k: int = 60,
) -> list[Hit]:
    allowed = store.allowed_rows(filters)
    if not allowed:
        return []

    dense_ranks: dict[int, tuple[int, float]] = {}
    bm25_ranks: dict[int, tuple[int, float]] = {}

    if mode in {"hybrid", "dense"}:
        query_vector = embedder.embed_one(query)
        for rank, (row, score) in enumerate(
            store.search_dense(query_vector, candidates, allowed), start=1
        ):
            dense_ranks[row] = (rank, score)
    if mode in {"hybrid", "bm25"}:
        for rank, (row, score) in enumerate(
            bm25.rank(query, candidates, allowed), start=1
        ):
            bm25_ranks[row] = (rank, score)

    hits: list[Hit] = []
    for row in set(dense_ranks) | set(bm25_ranks):
        score = 0.0
        if row in dense_ranks:
            score += 1.0 / (rrf_k + dense_ranks[row][0])
        if row in bm25_ranks:
            score += 1.0 / (rrf_k + bm25_ranks[row][0])
        chunk = store.chunks[row]
        hits.append(
            Hit(
                chunk=chunk,
                score=score * _title_boost(query, chunk),
                dense_rank=dense_ranks[row][0] if row in dense_ranks else None,
                bm25_rank=bm25_ranks[row][0] if row in bm25_ranks else None,
                dense_score=dense_ranks[row][1] if row in dense_ranks else 0.0,
                bm25_score=bm25_ranks[row][1] if row in bm25_ranks else 0.0,
            )
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]
