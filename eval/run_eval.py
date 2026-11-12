#!/usr/bin/env python3
"""Measure the system instead of admiring it.

Three families of metric, because they fail for different reasons:

  retrieval  hit@k and MRR    - did the right document come back at all?
  answering  abstain accuracy - did it answer when it should, and refuse when
                                it should?
  grounding  keyword and citation rates - is the answer actually supported by
                                the retrieved text, or just plausible?

Run it after every change. This is the habit that turns a RAG demo into a RAG
system.

    python eval/run_eval.py
    python eval/run_eval.py --k 3 --mode bm25
    python eval/run_eval.py --show-failures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from rag.pipeline import RAG  # noqa: E402

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"


def load_questions(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def first_relevant_rank(answer, expected: list[str]) -> int | None:
    for rank, hit in enumerate(answer.hits, start=1):
        if hit.chunk.doc_id in expected:
            return rank
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    parser.add_argument("--questions", default=str(QUESTIONS))
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument("--json-out", default=str(Path(__file__).parent / "results.json"))
    args = parser.parse_args()

    rag = RAG(top_k=args.k)
    rows = load_questions(Path(args.questions))

    retrieval_total = retrieval_hits = 0
    reciprocal_ranks: list[float] = []
    answered = answerable = 0
    abstain_ok = 0
    contain_total = contain_ok = 0
    grounded_ok = 0
    failures: list[str] = []

    print(f"index : {len(rag.store)} chunks | embedder {rag.embedder.name} | "
          f"llm {rag.provider} | k={args.k} | mode={args.mode}\n")
    print(f"{'id':<5}{'retr':<7}{'rank':<6}{'abstain':<9}{'contains':<10}{'grounded':<9} question")
    print("-" * 96)

    for row in rows:
        answer = rag.ask(row["question"], top_k=args.k, mode=args.mode)
        expected = row.get("expected_docs") or []
        wants_abstain = bool(row.get("should_abstain"))

        if wants_abstain:
            ok = answer.abstained
            abstain_ok += ok
            answered += 1
            rank = None
            retr = "-"
        else:
            answerable += 1
            rank = first_relevant_rank(answer, expected)
            if expected:
                retrieval_total += 1
                retrieval_hits += bool(rank)
                if rank:
                    reciprocal_ranks.append(1.0 / rank)
            retr = "ok" if rank else "MISS"
            if answer.abstained:
                abstain_ok += 0
            else:
                abstain_ok += 1
            contains = "-"
            if row.get("must_contain"):
                contain_total += 1
                text = answer.text.lower()
                passed = all(str(needle).lower() in text for needle in row["must_contain"])
                contain_ok += passed
                contains = "ok" if passed else "MISS"
            else:
                contains = "-"
            if answer.grounded:
                grounded_ok += 1
            answered += 1
            if not rank or contains == "MISS":
                failures.append(f"{row['id']}: {row['question']}")

        flag = "ok" if (answer.abstained == wants_abstain) else "WRONG"
        print(f"{row['id']:<5}{retr:<7}{str(rank or '-'):<6}{flag:<9}{contains:<10}"
              f"{str(answer.grounded):<9} {row['question'][:44]}")

    def pct(num: int, den: int) -> str:
        return f"{num / den:.0%}" if den else "n/a"

    summary = {
        "questions": len(rows),
        "hit_rate_at_k": round(retrieval_hits / retrieval_total, 3) if retrieval_total else None,
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3) if reciprocal_ranks else None,
        "abstain_accuracy": round(abstain_ok / answered, 3) if answered else None,
        "keyword_pass_rate": round(contain_ok / contain_total, 3) if contain_total else None,
        "grounded_rate": round(grounded_ok / answerable, 3) if answerable else None,
        "k": args.k,
        "mode": args.mode,
        "embedder": rag.embedder.name,
        "provider": rag.provider,
    }

    print("\n" + "=" * 96)
    print(f"hit@{args.k}            {pct(retrieval_hits, retrieval_total):>6}"
          f"   ({retrieval_hits}/{retrieval_total} answerable questions)")
    print(f"MRR                {summary['mrr'] or 0:>6.3f}   (1.0 = right doc always first)")
    print(f"answer/abstain     {pct(abstain_ok, answered):>6}"
          f"   ({abstain_ok}/{answered} decisions correct)")
    print(f"key fact present   {pct(contain_ok, contain_total):>6}"
          f"   ({contain_ok}/{contain_total} answers contain the expected figure)")
    print(f"citation grounded  {pct(grounded_ok, answerable):>6}"
          f"   ({grounded_ok}/{answerable} answers carry valid [S#] citations)")
    print("=" * 96)

    Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.json_out}")

    if failures:
        print(f"\n{len(failures)} question(s) worth a look:")
        for failure in failures[:12]:
            print("  -", failure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
