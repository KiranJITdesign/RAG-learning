"""Step 2 of ingestion: cut a document into retrievable chunks.

Chunking is where most beginner RAG projects quietly break. Two decisions
matter more than the chunk size itself:

1. Split on structure first (headings), size second. A chunk that begins
   mid-argument retrieves badly.
2. Put the document title and section heading *inside* the text you embed.
   A chunk that says "The applicant must be a landholding farmer family"
   is useless on its own; the same chunk prefixed with
   "PM-KISAN Samman Nidhi > Who is eligible" is findable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Document

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    doc_id: str
    title: str
    section: str          # e.g. "Who is eligible"
    text: str             # the passage as a human reads it
    embed_text: str       # title + section + passage, what we actually embed
    meta: dict[str, str]
    ordinal: int          # position within the document

    @property
    def label(self) -> str:
        return f"{self.title} > {self.section}" if self.section else self.title


def split_sections(body: str) -> list[tuple[str, str]]:
    """Return [(heading_path, section_body), ...] using markdown headings."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append((" > ".join(name for _, name in stack), text))

    for line in body.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            buffer.clear()
            level, name = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, name))
        else:
            buffer.append(line)
    flush()
    return sections


def split_long(text: str, size: int, overlap: int) -> list[str]:
    """Sentence-aware splitter with a sliding overlap window."""
    sentences = [s for s in SENTENCE_END.split(text) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= size:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            chunks.append(current)
        tail = current[-overlap:] if (overlap and current) else ""
        current = f"{tail} {sentence}".strip()
        # A single monster sentence: hard-cut it, keeping the overlap.
        while len(current) > size * 1.5:
            chunks.append(current[:size].strip())
            current = current[size - overlap :].strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading_path, section_body in split_sections(doc.body):
        # Keep only the leaf heading in the section label; the full path is
        # noisy in the prompt but useful to have around.
        leaf = heading_path.split(" > ")[-1] if heading_path else ""
        for piece in split_long(section_body, size, overlap):
            embed_text = f"{doc.title} | {leaf}\n{piece}" if leaf else f"{doc.title}\n{piece}"
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=leaf,
                    text=piece,
                    embed_text=embed_text,
                    meta=doc.meta,
                    ordinal=len(chunks),
                )
            )
    return chunks


def chunk_corpus(docs: list[Document], size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, size, overlap))
    return out
