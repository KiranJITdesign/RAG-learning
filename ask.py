#!/usr/bin/env python3
"""Ask a question from the terminal.

    python ask.py "I am 19, SC, family income 1.8 lakh, doing B.Com - any scholarship?"
    python ask.py "what is the Ayushman cover for a family of six?"
    python ask.py --debug "how many days of work under MGNREGA?"
    python ask.py --mode bm25 "PM-JAY 5 lakh"
    python ask.py --filter who=rural "money for a farmer"
    python ask.py            # interactive mode
    python ask.py --json "..."   # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import sys

import config
from rag.pipeline import RAG

DIVIDER = "-" * 72


def render(answer, debug: bool) -> None:
    print()
    print(answer.text.strip())
    print()
    if answer.abstained:
        print("(abstained - the knowledge base does not cover this)")
    sources = answer.sources()
    if sources:
        print("Sources")
        for source in sources:
            print(f"  [{source['marker']}] {source['title']} > {source['section']}")
            if source["source_url"]:
                print(f"        {source['source_url']}")
    if debug:
        print(DIVIDER)
        print(f"provider      : {answer.provider}")
        print(f"coverage      : {answer.coverage:.0%}"
              + (f"  missing: {', '.join(answer.missing_terms)}" if answer.missing_terms else ""))
        print(f"grounded      : {answer.grounded}")
        print(f"citations     : {answer.citations or 'none'}")
        for note in answer.notes:
            print(f"note          : {note}")
        print("retrieved chunks")
        for index, hit in enumerate(answer.hits, start=1):
            print(f"  S{index}  score={hit.score:.4f}  "
                  f"dense_rank={hit.dense_rank}  bm25_rank={hit.bm25_rank}  "
                  f"{hit.chunk.doc_id}")
            print(f"       {hit.chunk.label}")
    print()


def parse_filters(pairs: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--filter needs key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        filters[key.strip()] = value.strip()
    return filters


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask SchemeSaathi a question")
    parser.add_argument("question", nargs="*", help="the question, in quotes")
    parser.add_argument("--debug", action="store_true", help="show retrieval internals")
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="chunks to retrieve")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    parser.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE",
                        help="metadata filter, e.g. --filter who=rural")
    args = parser.parse_args()

    rag = RAG(top_k=args.k)
    filters = parse_filters(args.filter)

    if not args.json:
        print(f"index: {len(rag.store)} chunks | embedder: {rag.embedder.name} | "
              f"llm: {rag.provider}")

    def handle(question: str) -> None:
        answer = rag.ask(question, top_k=args.k, filters=filters, mode=args.mode)
        if args.json:
            print(json.dumps(
                {
                    "question": answer.question,
                    "answer": answer.text,
                    "abstained": answer.abstained,
                    "grounded": answer.grounded,
                    "provider": answer.provider,
                    "coverage": round(answer.coverage, 3),
                    "citations": answer.citations,
                    "sources": answer.sources(),
                    "notes": answer.notes,
                },
                indent=2, ensure_ascii=False,
            ))
        else:
            render(answer, args.debug)

    if args.question:
        handle(" ".join(args.question))
        return

    print("Interactive mode. Ctrl-D or 'exit' to quit.")
    while True:
        try:
            question = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"exit", "quit", ""}:
            return
        handle(question)


if __name__ == "__main__":
    sys.exit(main())
