"""The two public entry points of the whole project.

    build_index()  - ingestion:  files -> chunks -> vectors -> sqlite
    RAG.ask()      - answering:  question -> hits -> prompt -> checked answer

Everything else in rag/ is a step that these two functions call in order. If
you understand these forty lines, you understand the pipeline.
"""
from __future__ import annotations

import time
from pathlib import Path

import config
from . import generate
from .bm25 import BM25
from .chunker import chunk_corpus
from .embedder import Embedder, get_embedder
from .llm import BaseLLM, describe, get_llm
from .loaders import load_corpus
from .retrieve import (
    Hit,
    answerability,
    hybrid_search,
    should_abstain,
    title_evidence,
)
from .store import VectorStore


def build_index(
    data_dir: Path | None = None,
    db_path: Path | None = None,
    size: int | None = None,
    overlap: int | None = None,
    backend: str | None = None,
    verbose: bool = True,
) -> dict:
    data_dir = Path(data_dir or config.DATA_DIR)
    db_path = Path(db_path or config.INDEX_DB)
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    started = time.time()
    docs = load_corpus(data_dir)
    chunks = chunk_corpus(docs, size, overlap)
    if verbose:
        print(f"[1/4] loaded {len(docs)} documents from {data_dir}")
        print(f"[2/4] split into {len(chunks)} chunks "
              f"(size={size}, overlap={overlap})")

    embedder = get_embedder(backend)
    if verbose:
        print(f"[3/4] embedding with {embedder.name} ...")
    vectors = embedder.embed([c.embed_text for c in chunks])

    store = VectorStore.build(db_path, docs, chunks, vectors, embedder.name)
    elapsed = time.time() - started
    if verbose:
        print(f"[4/4] wrote {len(store)} vectors to {db_path}")
        print(f"done in {elapsed:.1f}s | dim={vectors.shape[1]} | "
              f"embedder={embedder.name}")
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "embedder": embedder.name,
        "seconds": round(elapsed, 2),
        "db": str(db_path),
    }


class RAG:
    def __init__(
        self,
        db_path: Path | None = None,
        backend: str | None = None,
        llm: BaseLLM | None | str = "auto",
        top_k: int | None = None,
    ) -> None:
        self.store = VectorStore.load(Path(db_path or config.INDEX_DB))
        self.embedder: Embedder = get_embedder(backend)
        if self.store.embedder_name and self.store.embedder_name != self.embedder.name:
            raise SystemExit(
                f"The index was built with '{self.store.embedder_name}' but you are "
                f"querying with '{self.embedder.name}'.\n"
                f"Rebuild the index:  python ingest.py --reset"
            )
        self.bm25 = BM25.from_texts(self.store.texts)
        self.llm = get_llm() if llm == "auto" else llm
        self.top_k = top_k or config.TOP_K

    @property
    def provider(self) -> str:
        return describe(self.llm)

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
        mode: str = "hybrid",
    ) -> list[Hit]:
        return hybrid_search(
            self.store,
            self.embedder,
            self.bm25,
            question,
            top_k=top_k or self.top_k,
            candidates=config.CANDIDATES,
            filters=filters,
            mode=mode,
            rrf_k=config.RRF_K,
        )

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
        mode: str = "hybrid",
        min_coverage: float | None = None,
    ) -> generate.Answer:
        question = question.strip()
        if not question:
            raise ValueError("empty question")
        min_coverage = config.COVERAGE_MIN if min_coverage is None else min_coverage

        hits = self.retrieve(question, top_k=top_k, filters=filters, mode=mode)
        coverage, matched, missing = answerability(
            question,
            [h.chunk.text for h in hits],
            self.bm25.idf,
            len(self.store),
        )

        title_hits = title_evidence(
            question, list(dict.fromkeys(c.title for c in self.store.chunks))
        )
        abstain, reason = should_abstain(coverage, matched, title_hits, min_coverage)

        if not hits or abstain:
            titles = ", ".join(dict.fromkeys(h.chunk.title for h in hits)) or "nothing"
            text = (
                "I could not answer that from the documents I have. "
                f"The closest material I found is: {titles}. "
                "This assistant only answers from its own scheme summaries, so it "
                "will not guess."
            )
            return generate.Answer(
                question=question,
                text=text,
                hits=hits,
                provider=self.provider,
                coverage=coverage,
                missing_terms=missing,
                abstained=True,
                notes=[
                    f"abstained because {reason}",
                    f"coverage {coverage:.0%}, matched terms {matched or 'none'}, "
                    f"title hits {title_hits or 'none'}"
                ],
            )

        text, provider, notes = generate.generate(question, hits, self.llm)

        if generate.is_not_found(text):
            return generate.Answer(
                question=question,
                text="The documents retrieved do not contain that answer. "
                + text.split(":", 1)[-1].strip(),
                hits=hits,
                provider=provider,
                coverage=coverage,
                missing_terms=missing,
                abstained=True,
                notes=notes + ["model reported NOT_FOUND"],
            )

        citations, ratio, cite_notes = generate.validate_citations(text, len(hits))
        notes += cite_notes
        grounded = bool(citations) and ratio >= 0.5
        if not citations:
            notes.append("no [S#] citations found - treat this answer as unverified")
        elif ratio < 0.5:
            notes.append(f"only {ratio:.0%} of sentences carry a citation")

        return generate.Answer(
            question=question,
            text=text,
            hits=hits,
            provider=provider,
            grounded=grounded,
            citations=citations,
            coverage=coverage,
            missing_terms=missing,
            notes=notes,
        )
