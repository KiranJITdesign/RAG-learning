#!/usr/bin/env python3
"""Build the index. Run this once, and again whenever your documents change.

    python ingest.py                 # build with the settings in config.py
    python ingest.py --reset         # delete and rebuild
    python ingest.py --size 400 --overlap 60    # experiment with chunking
    python ingest.py --backend sentence-transformers
"""
from __future__ import annotations

import argparse
from pathlib import Path

import config
from rag.pipeline import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SchemeSaathi index")
    parser.add_argument("--data", default=str(config.DATA_DIR), help="folder of .md/.txt/.pdf")
    parser.add_argument("--db", default=str(config.INDEX_DB), help="sqlite file to write")
    parser.add_argument("--size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--backend", default=config.EMBED_BACKEND,
                        choices=["hashing", "sentence-transformers", "openai"])
    parser.add_argument("--reset", action="store_true", help="delete the existing index first")
    args = parser.parse_args()

    db = Path(args.db)
    if db.exists() and not args.reset:
        print(f"Replacing the existing index at {db}")
    if args.reset and db.exists():
        db.unlink()
        print(f"Deleted {db}")

    stats = build_index(
        data_dir=Path(args.data),
        db_path=db,
        size=args.size,
        overlap=args.overlap,
        backend=args.backend,
    )
    print("\nIndex ready. Ask a question:")
    print('  python ask.py "I am a small farmer in Maharashtra, what money can I get?"')


if __name__ == "__main__":
    main()
