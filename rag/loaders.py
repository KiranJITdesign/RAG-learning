"""Step 1 of ingestion: turn files on disk into Document objects.

A "document" here is one scheme, with metadata that we can later filter on.
The metadata is not decoration - it is what lets a user ask "only health
schemes for rural households" and get a sane answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf"}


@dataclass
class Document:
    doc_id: str
    title: str
    body: str
    meta: dict[str, str] = field(default_factory=dict)
    path: str = ""


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Read the tiny YAML-ish header at the top of a .md file.

    We deliberately do not depend on PyYAML: the header is only flat
    `key: value` pairs, and a 10-line parser you can read beats a dependency.
    """
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    header, body = raw[3:end], raw[end + 4 :]
    meta: dict[str, str] = {}
    for line in header.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, body.lstrip("\n")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"{path.name} is a PDF but pypdf is not installed.\n"
            "Run: pip install pypdf"
        ) from exc
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_document(path: Path) -> Document:
    if path.suffix.lower() == ".pdf":
        body, meta = _read_pdf(path), {}
    else:
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw)
    doc_id = meta.get("id") or path.stem
    title = meta.get("title") or path.stem.replace("-", " ").title()
    return Document(doc_id=doc_id, title=title, body=body, meta=meta, path=str(path))


def load_corpus(directory: Path) -> list[Document]:
    """Load every supported file in a directory, sorted for reproducible ids."""
    if not directory.exists():
        raise SystemExit(f"Data directory not found: {directory}")
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED)
    if not paths:
        raise SystemExit(f"No .md/.txt/.pdf files in {directory}")
    return [load_document(p) for p in paths]
