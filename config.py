"""Every knob of the pipeline lives here, so you never have to hunt for a
magic number inside a module. Change a value, re-run, and watch what happens -
that is the fastest way to build intuition about RAG.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "policies"
INDEX_DIR = ROOT / "data" / "index"
INDEX_DB = INDEX_DIR / "index.sqlite3"

# Load a .env file if python-dotenv is installed. Never required.
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # pragma: no cover
    pass

# ---------------------------------------------------------------- chunking ---
# Characters, not tokens, on purpose: characters are what you can count while
# reading the source document.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# -------------------------------------------------------------- embeddings ---
# hashing            -> no download, no API key, works offline (default)
# sentence-transformers -> better quality, downloads ~90 MB model
# openai             -> best quality, needs OPENAI_API_KEY
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "hashing")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HASH_DIM = 512

# --------------------------------------------------------------- retrieval ---
TOP_K = 5           # how many chunks reach the LLM
CANDIDATES = 20     # how many each retriever returns before fusion
RRF_K = 60          # reciprocal rank fusion constant (standard value)
COVERAGE_MIN = 0.25  # IDF-weighted share of the question's words that must be
                     # present in the retrieved text, else we abstain.
                     # Calibrated on eval/questions.jsonl - re-run the eval after
                     # changing it. Lower = answers more, and more wrongly.

# --------------------------------------------------------------------- LLM ---
# extractive -> no LLM at all, quotes the best sentences (always works)
# ollama     -> free, local, private. Needs `ollama serve` running.
# openai | gemini | anthropic -> needs an API key in .env
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "extractive")
LLM_MODEL = os.getenv("LLM_MODEL", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TEMPERATURE = 0.0
MAX_ANSWER_TOKENS = 700
