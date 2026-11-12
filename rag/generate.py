"""Step 2 of answering: prompt building, the extractive fallback, and the
citation check that decides whether an answer is trustworthy.

Two things here are what separate a demo from something you would put in front
of a real user:

* the prompt refuses to answer without sources, and
* the code *checks* the model's answer afterwards instead of trusting it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import config
from .llm import BaseLLM, LLMError
from .retrieve import Hit, content_terms

CITATION = re.compile(r"\[S(\d+)\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+")

SYSTEM_PROMPT = """You are SchemeSaathi, an assistant that explains Indian government \
welfare schemes to ordinary citizens.

You will be given numbered source passages. Follow these rules exactly.

1. Use ONLY the numbered sources. Do not use any outside knowledge, and do not
   guess. If the sources do not contain the answer, reply with one line:
   NOT_FOUND: <what information is missing>.
2. Put a citation marker after every factual claim, using the source number:
   [S1], [S2]. A paragraph with no citation is a defect.
3. Be concrete. Give the exact amount, age limit, income ceiling, number of
   days, or document name when the sources state it. Never round or invent a
   number.
4. Never invent a scheme name, portal, helpline number or rupee figure.
5. If the sources disagree or are silent on a detail the user needs, say so
   plainly in one sentence.
6. Keep the answer under 180 words. Use short paragraphs, or a short list when
   you are enumerating eligibility conditions or documents.
7. Write plainly for someone reading on a phone. No marketing tone.
"""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    provider: str
    grounded: bool = False
    citations: list[int] = field(default_factory=list)
    coverage: float = 0.0
    missing_terms: list[str] = field(default_factory=list)
    abstained: bool = False
    notes: list[str] = field(default_factory=list)

    def sources(self) -> list[dict]:
        """The passages actually cited, in citation order."""
        used = self.citations or list(range(1, len(self.hits) + 1))
        out = []
        for number in sorted(set(used)):
            if 1 <= number <= len(self.hits):
                hit = self.hits[number - 1]
                out.append(
                    {
                        "marker": f"S{number}",
                        "title": hit.chunk.title,
                        "section": hit.chunk.section,
                        "doc_id": hit.chunk.doc_id,
                        "source_url": hit.chunk.meta.get("source", ""),
                        "excerpt": hit.chunk.text[:400],
                    }
                )
        return out


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(
            f"[S{number}] {hit.chunk.title} - {hit.chunk.section}\n{hit.chunk.text}"
        )
    context = "\n\n".join(blocks)
    return (
        f"SOURCE PASSAGES\n{context}\n\n"
        f"QUESTION\n{question}\n\n"
        "Answer using only the passages above, with [S#] citations."
    )


def validate_citations(text: str, n_sources: int) -> tuple[list[int], float, list[str]]:
    """Return (citations used, share of sentences that carry a citation, notes)."""
    notes: list[str] = []
    found = [int(m) for m in CITATION.findall(text)]
    valid = sorted({c for c in found if 1 <= c <= n_sources})
    out_of_range = sorted({c for c in found if not 1 <= c <= n_sources})
    if out_of_range:
        notes.append(f"model cited nonexistent sources {out_of_range}; stripped")
    sentences = [s for s in SENTENCE.split(text.strip()) if len(s.split()) > 3]
    cited = sum(1 for s in sentences if CITATION.search(s))
    ratio = cited / len(sentences) if sentences else 0.0
    return valid, ratio, notes


def _sentences(text: str) -> list[str]:
    """Sentences worth quoting: long enough to mean something, on one line."""
    out = []
    for raw in SENTENCE.split(text):
        sentence = " ".join(raw.split()).lstrip("-*\u2022 ").strip()
        if len(sentence.split()) > 6:
            out.append(sentence)
    return out


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 4) -> str:
    """No LLM: pick the sentences that best answer the question and quote them.

    The scoring is deliberately simple and readable: word overlap with the
    question, a bonus for a sentence that actually contains a number when the
    question asks for a quantity, and a bonus for coming from a higher-ranked
    chunk. You can see every term of it, which is the point - this is the
    "generation" step in its crudest form.

    It is not a substitute for an LLM. It reads like a stack of quotes, and it
    will miss a figure that sits in a sentence sharing no words with the
    question. Run eval/run_eval.py before and after switching LLM_PROVIDER to
    see exactly what the LLM buys you.
    """
    terms = content_terms(question)
    if not terms:
        return "NOT_FOUND: the question has no searchable content words."
    wants_number = bool(
        re.search(r"how (much|many|long|old)|rate|limit|amount|ceiling|age", question, re.I)
    )
    candidates: list[tuple[float, int, str]] = []
    for index, hit in enumerate(hits, start=1):
        rank_bonus = 0.05 * (len(hits) - index)
        for position, sentence in enumerate(_sentences(hit.chunk.text)):
            overlap = len(terms & content_terms(sentence)) / len(terms)
            score = overlap + rank_bonus - 0.01 * position
            if wants_number and re.search(r"\d", sentence):
                score += 0.45
            if score <= 0.05:
                continue
            candidates.append((score, index, sentence))
    if not candidates:
        return "NOT_FOUND: nothing in the retrieved passages matches the question."
    candidates.sort(key=lambda item: item[0], reverse=True)
    picked: list[tuple[int, str]] = []
    seen: set[str] = set()
    for _, index, sentence in candidates:
        key = sentence.lower()[:60]
        if key in seen:
            continue
        seen.add(key)
        picked.append((index, sentence))
        if len(picked) >= max_sentences:
            break
    picked.sort(key=lambda item: item[0])
    body = " ".join(f"{sentence} [S{index}]" for index, sentence in picked)
    return (
        "Here is what the documents say, quoted directly "
        "(extractive mode - set LLM_PROVIDER for a written answer):\n\n" + body
    )


def generate(
    question: str, hits: list[Hit], llm: BaseLLM | None
) -> tuple[str, str, list[str]]:
    """Return (answer_text, provider_name, notes)."""
    notes: list[str] = []
    if llm is None:
        return extractive_answer(question, hits), "extractive", notes
    try:
        text = llm.complete(SYSTEM_PROMPT, build_prompt(question, hits))
    except LLMError as exc:
        notes.append(f"{exc} - fell back to extractive mode")
        return extractive_answer(question, hits), "extractive (fallback)", notes
    if not text:
        notes.append("model returned an empty answer - fell back to extractive mode")
        return extractive_answer(question, hits), "extractive (fallback)", notes
    return text, llm.name, notes


def is_not_found(text: str) -> bool:
    return text.strip().upper().startswith("NOT_FOUND")
