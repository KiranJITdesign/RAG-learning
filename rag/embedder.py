"""Step 3 of ingestion: text -> vector.

Three interchangeable backends, same interface, so you can see exactly what
the embedding step buys you and what it costs:

    hashing              0 download, 0 rupees, offline, weaker
    sentence-transformers  ~90 MB download, CPU is fine, much better
    openai                needs a key and money, best of the three

Start with `hashing` so the project runs on any laptop with no network. Then
switch to sentence-transformers and run eval/run_eval.py again - the jump in
recall@5 is the single most convincing demo of why embeddings matter.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

import config

WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Used by BM25 and by the hashing embedder."""
    return WORD.findall(text.lower())


class Embedder:
    name = "base"
    dim = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """Hashed bag of words + bigrams, sublinear TF, L2 normalised.

    This is the classic "hashing trick": instead of building a vocabulary, hash
    each feature into one of `dim` buckets. Cheap, stateless, deterministic,
    and good enough that hybrid retrieval works. It is *not* semantic - "farmer"
    and "agriculturalist" land in different buckets - which is exactly the gap
    a real embedding model closes.
    """

    name = "hashing"

    def __init__(self, dim: int = config.HASH_DIM) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        sign = 1.0 if (value >> 63) & 1 else -1.0  # signed hashing kills collisions bias
        return value % self.dim, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            counts: dict[str, int] = {}
            for feature in features:
                counts[feature] = counts.get(feature, 0) + 1
            for feature, count in counts.items():
                bucket, sign = self._bucket(feature)
                out[row, bucket] += sign * (1.0 + np.log(count))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str = config.EMBED_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import

        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, model_name: str = "") -> None:
        import os

        import requests  # noqa: F401  (checked early so failure is clear)

        self.model = model_name or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self.key = os.getenv("OPENAI_API_KEY", "")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.dim = 1536

    def embed(self, texts: list[str]) -> np.ndarray:
        import requests

        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()["data"]
        vectors = np.array([item["embedding"] for item in data], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def get_embedder(backend: str | None = None) -> Embedder:
    """Factory with a graceful fallback, so a missing dependency never blocks
    you from running the pipeline."""
    backend = (backend or config.EMBED_BACKEND).lower()
    if backend in {"hashing", "hash", "tfidf"}:
        return HashingEmbedder()
    if backend in {"sentence-transformers", "st", "minilm"}:
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:
            print(f"[warn] sentence-transformers unavailable ({exc}); using hashing.")
            return HashingEmbedder()
    if backend == "openai":
        try:
            return OpenAIEmbedder()
        except Exception as exc:
            print(f"[warn] OpenAI embeddings unavailable ({exc}); using hashing.")
            return HashingEmbedder()
    raise SystemExit(f"Unknown EMBED_BACKEND: {backend}")
