"""Step 2 of answering: prompt building, the extractive fallback, and the
citation check that decides whether an answer is trustworthy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import config
from .llm import BaseLLM, LLMError
from .retrieve import Hit, content_terms

CITATION = re.compile(r"\[S(\d+)\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+")

SYSTEM_PROMPT = """You are Internal Ops Co-Pilot, an intelligent corporate assistant for employees and new hires. \
You answer IT support, HR policy, hardware setup, security, and onboarding questions grounded strictly in internal company documentation.

You will be given numbered source passages. Follow these rules exactly:

1. Always provide a structured, beautifully formatted answer:
   - Begin with a clear, direct summary line.
   - Use distinct bullet points (• or -) with bold headers for each policy rule, entitlement, or eligibility criterion.
   - For IT troubleshooting instructions, use a clean numbered sequence (1., 2., 3.).
2. Put a citation marker after every factual claim, using the source number: [S1], [S2].
3. For IT failures, note that an automated IT support ticket can be logged if the steps do not resolve the issue.
4. Keep the answer professional, concise, and easy to scan.
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
        "Provide a structured, formatted answer using only the passages above, with [S#] citations."
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
        if len(sentence.split()) > 5:
            out.append(sentence)
    return out


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 5) -> str:
    """Formatted extraction of key policy points with citations."""
    terms = content_terms(question)
    if not terms:
        return "NOT_FOUND: the question has no searchable content words."
    wants_number = bool(
        re.search(r"how (much|many|long|old)|rate|limit|amount|ceiling|age|days", question, re.I)
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

    bullets = []
    for index, sentence in picked:
        clean = re.sub(r"^\s*[-*#]+\s*", "", sentence).strip()
        if clean.count("**") % 2 != 0:
            clean = clean.replace("**", "")
        bullets.append(f"• {clean} [S{index}]")

    return "**Key Policy Points & Details:**\n\n" + "\n".join(bullets)


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
