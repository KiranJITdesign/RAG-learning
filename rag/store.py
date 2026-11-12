"""The vector store: SQLite + numpy, no external service.

A production system would use pgvector, Qdrant, Weaviate or FAISS. For a
corpus of a few thousand chunks, loading every vector into one numpy matrix
and doing a single matmul is faster, simpler and completely inspectable - and
you can open index.sqlite3 with any SQLite browser and read your own chunks.
Swapping this file for a real vector DB is a contained change, because nothing
else in the project knows how storage works.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunker import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title  TEXT NOT NULL,
    meta   TEXT NOT NULL,
    path   TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    title      TEXT NOT NULL,
    section    TEXT,
    text       TEXT NOT NULL,
    embed_text TEXT NOT NULL,
    meta       TEXT NOT NULL,
    ordinal    INTEGER,
    dim        INTEGER,
    vector     BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


@dataclass
class StoredChunk:
    row: int
    doc_id: str
    title: str
    section: str
    text: str
    meta: dict[str, str]

    @property
    def label(self) -> str:
        return f"{self.title} > {self.section}" if self.section else self.title


class VectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.chunks: list[StoredChunk] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.embedder_name = ""

    # ------------------------------------------------------------- writing --
    @classmethod
    def build(
        cls,
        path: Path,
        docs: list,
        chunks: list[Chunk],
        vectors: np.ndarray,
        embedder_name: str,
    ) -> "VectorStore":
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        conn = sqlite3.connect(path)
        try:
            conn.executescript(SCHEMA)
            conn.executemany(
                "INSERT INTO documents (doc_id, title, meta, path) VALUES (?,?,?,?)",
                [(d.doc_id, d.title, json.dumps(d.meta), d.path) for d in docs],
            )
            conn.executemany(
                """INSERT INTO chunks
                   (doc_id, title, section, text, embed_text, meta, ordinal, dim, vector)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        c.doc_id,
                        c.title,
                        c.section,
                        c.text,
                        c.embed_text,
                        json.dumps(c.meta),
                        c.ordinal,
                        int(vectors[i].shape[0]),
                        vectors[i].astype(np.float32).tobytes(),
                    )
                    for i, c in enumerate(chunks)
                ],
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('embedder', ?)",
                (embedder_name,),
            )
            conn.commit()
        finally:
            conn.close()
        return cls.load(path)

    # ------------------------------------------------------------- reading --
    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        if not path.exists():
            raise SystemExit(
                f"No index at {path}.\nBuild it first:  python ingest.py"
            )
        store = cls(path)
        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                """SELECT id, doc_id, title, section, text, meta
                   FROM chunks ORDER BY id"""
            ).fetchall()
            vectors = [
                np.frombuffer(row[0], dtype=np.float32)
                for row in conn.execute("SELECT vector FROM chunks ORDER BY id")
            ]
            meta_row = conn.execute(
                "SELECT value FROM index_meta WHERE key = 'embedder'"
            ).fetchone()
            store.embedder_name = meta_row[0] if meta_row else ""
        finally:
            conn.close()
        store.chunks = [
            StoredChunk(
                row=i,
                doc_id=r[1],
                title=r[2],
                section=r[3] or "",
                text=r[4],
                meta=json.loads(r[5]),
            )
            for i, r in enumerate(rows)
        ]
        store.matrix = (
            np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
        )
        return store

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.chunks]

    def allowed_rows(self, filters: dict[str, str] | None) -> set[int]:
        """Metadata filtering, done in Python because the corpus is small.

        A filter matches when the stored value equals the requested value, or
        when the stored value is "all". That one rule makes "who: all" schemes
        visible to a "rural" query without any special-casing upstream.
        """
        if not filters:
            return set(range(len(self.chunks)))
        allowed: set[int] = set()
        for chunk in self.chunks:
            for key, wanted in filters.items():
                have = (chunk.meta.get(key) or "").strip().lower()
                if have == "all" or have == wanted.strip().lower():
                    continue
                break
            else:
                allowed.add(chunk.row)
        return allowed

    def search_dense(
        self, query_vector: np.ndarray, top_k: int, allowed: set[int] | None = None
    ) -> list[tuple[int, float]]:
        if not len(self.chunks):
            return []
        scores = self.matrix @ query_vector.astype(np.float32)
        rows = range(len(self.chunks)) if allowed is None else sorted(allowed)
        ranked = [(i, float(scores[i])) for i in rows]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]
