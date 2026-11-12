"""Tests for the parts that are easy to get subtly wrong.

    pytest -q

None of these call an LLM and none need a network connection, so they run in a
second. Write a test for every bug you fix - the chunker and the citation
validator are the two places where a silent bug costs you an afternoon.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.bm25 import BM25  # noqa: E402
from rag.chunker import Chunk, chunk_document, split_long, split_sections  # noqa: E402
from rag.embedder import HashingEmbedder, tokenize  # noqa: E402
from rag.generate import validate_citations  # noqa: E402
from rag.loaders import Document, parse_front_matter  # noqa: E402
from rag.retrieve import answerability, should_abstain  # noqa: E402


# ------------------------------------------------------------------ loaders --
def test_front_matter_is_parsed():
    meta, body = parse_front_matter("---\nid: x\ntitle: Hello World\n---\nBody text\n")
    assert meta["id"] == "x"
    assert meta["title"] == "Hello World"
    assert body.startswith("Body")


def test_no_front_matter_is_fine():
    meta, body = parse_front_matter("just text")
    assert meta == {} and body == "just text"


# ------------------------------------------------------------------ chunker --
def test_sections_keep_heading_path():
    sections = split_sections("# A\ntext one\n## B\ntext two\n")
    assert ("A", "text one") in sections
    assert ("A > B", "text two") in sections


def test_long_text_is_split_with_overlap():
    body = ". ".join(f"Sentence number {i} about welfare schemes" for i in range(40))
    pieces = split_long(body, size=200, overlap=50)
    assert len(pieces) > 1
    assert all(len(p) <= 320 for p in pieces)
    # the overlap means the end of one chunk reappears in the next
    assert pieces[0][-40:].split()[-2] in pieces[1]


def test_chunk_embed_text_carries_title_and_section():
    doc = Document(doc_id="d", title="Big Scheme", body="# Who is eligible\nA farmer with land.")
    chunks = chunk_document(doc, size=800, overlap=100)
    assert chunks[0].section == "Who is eligible"
    assert "Big Scheme" in chunks[0].embed_text
    assert "Who is eligible" in chunks[0].embed_text


def test_empty_document_produces_no_chunks():
    assert chunk_document(Document(doc_id="d", title="t", body="   "), 800, 100) == []


# ----------------------------------------------------------------- embedder --
def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dim=256)
    a = embedder.embed(["farmer land support"])
    b = embedder.embed(["farmer land support"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)


def test_similar_text_scores_higher_than_unrelated():
    embedder = HashingEmbedder(dim=1024)
    query, near, far = embedder.embed(
        ["mgnrega wage employment", "mgnrega job card wage payment", "sukanya girl savings"]
    )
    assert float(query @ near) > float(query @ far)


# --------------------------------------------------------------------- bm25 --
def test_bm25_prefers_the_document_with_the_rare_term():
    bm25 = BM25.from_texts(
        [
            "aadhaar bank account for the farmer",
            "mgnrega guarantees one hundred days of wage employment",
            "the account must be linked to aadhaar",
        ]
    )
    ranked = bm25.rank("mgnrega employment", top_k=2)
    assert ranked[0][0] == 1


def test_bm25_respects_the_allowed_set():
    bm25 = BM25.from_texts(["alpha beta", "alpha gamma", "beta gamma"])
    ranked = bm25.rank("alpha", top_k=3, allowed={2})
    assert all(row == 2 for row, _ in ranked)


# ------------------------------------------------------------------- gate ----
def test_off_topic_question_is_flagged():
    idf = {"capital": 3.0, "pension": 1.2}
    coverage, matched, missing = answerability(
        "what is the capital of france", ["working capital loan"], idf, 60
    )
    assert "capital" in matched and "france" in missing
    abstain, _ = should_abstain(coverage, matched, [], 0.25)
    assert abstain, "one incidental word must not be enough to answer"


def test_scheme_name_in_title_is_enough():
    idf = {"pension": 1.2}
    coverage, matched, _ = answerability("pension for my father", ["pension scheme"], idf, 60)
    abstain, _ = should_abstain(coverage, matched, ["pension"], 0.25)
    assert not abstain


def test_nothing_matched_always_abstains():
    abstain, reason = should_abstain(0.0, [], [], 0.25)
    assert abstain and "none" in reason


# --------------------------------------------------------------- citations ---
def test_valid_citations_are_extracted():
    citations, ratio, notes = validate_citations("You get 6,000 rupees [S1]. It is paid yearly [S2].", 3)
    assert citations == [1, 2]
    assert ratio == 1.0
    assert notes == []


def test_out_of_range_citation_is_stripped_and_noted():
    citations, _, notes = validate_citations("Something [S9].", 3)
    assert citations == []
    assert any("nonexistent" in note for note in notes)


def test_uncited_answer_scores_zero():
    _, ratio, _ = validate_citations("A confident sentence with no source at all.", 3)
    assert ratio == 0.0


# ---------------------------------------------------------------- tokenizer --
def test_tokenizer_drops_punctuation_and_case():
    assert tokenize("Rs 6,000/- per YEAR!") == ["rs", "6", "000", "per", "year"]
