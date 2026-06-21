from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".pdf"}


@dataclass(slots=True)
class TextChunk:
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def iter_document_files(path: str | Path) -> Iterable[Path]:
    root = Path(path)
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield root
        return

    for file_path in sorted(root.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield file_path


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\ud800-\udfff]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(
    text: str,
    max_chars: int = 900,
    overlap_chars: int = 120,
) -> list[str]:
    """Split text into compact chunks while preserving paragraph boundaries."""
    text = normalize_text(text)
    if not text:
        return []

    max_chars = max(200, int(max_chars))
    overlap_chars = max(0, min(int(overlap_chars), max_chars // 2))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_paragraph(paragraph, max_chars, overlap_chars))
            continue

        candidate = paragraph if not current else current + "\n\n" + paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current.strip())
            current = _overlap_tail(current, overlap_chars)
            current = paragraph if not current else current + "\n\n" + paragraph

    if current.strip():
        chunks.append(current.strip())
    return chunks


def _split_long_paragraph(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = sentence if not current else current + " " + sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
                current = _overlap_tail(current, overlap_chars)
            while len(sentence) > max_chars:
                head = sentence[:max_chars].strip()
                chunks.append(head)
                sentence = sentence[max(0, max_chars - overlap_chars) :].strip()
            current = sentence if not current else current + " " + sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _overlap_tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    tail = text[-overlap_chars:].strip()
    first_space = tail.find(" ")
    if first_space > 0:
        tail = tail[first_space + 1 :].strip()
    return tail


def load_chunks(
    input_path: str | Path,
    max_chars: int = 900,
    overlap_chars: int = 120,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for file_path in iter_document_files(input_path):
        chunks.extend(_load_file_chunks(file_path, max_chars, overlap_chars))
    return chunks


def _load_file_chunks(
    file_path: Path,
    max_chars: int,
    overlap_chars: int,
) -> list[TextChunk]:
    if file_path.suffix.lower() == ".pdf":
        return _load_pdf_chunks(file_path, max_chars, overlap_chars)
    return _load_text_chunks(file_path, max_chars, overlap_chars)


def _load_text_chunks(
    file_path: Path,
    max_chars: int,
    overlap_chars: int,
) -> list[TextChunk]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    paper = _paper_title(file_path)
    output: list[TextChunk] = []
    for idx, chunk_text in enumerate(split_text(text, max_chars, overlap_chars)):
        chunk_id = f"{file_path.as_posix()}::{idx}"
        output.append(
            TextChunk(
                id=chunk_id,
                text=chunk_text,
                source=file_path.as_posix(),
                chunk_index=idx,
                metadata={
                    "paper": paper,
                    "source_name": file_path.name,
                    "section": _infer_section(chunk_text),
                    "page": _infer_page(chunk_text),
                },
            )
        )
    return output


def _load_pdf_chunks(
    file_path: Path,
    max_chars: int,
    overlap_chars: int,
) -> list[TextChunk]:
    if file_path.stat().st_size == 0:
        print(f"Warning: skipping empty PDF: {file_path}", file=sys.stderr)
        return []

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF indexing requires pypdf. Install with: pip install pypdf"
        ) from exc

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:
        print(f"Warning: skipping unreadable PDF {file_path}: {exc}", file=sys.stderr)
        return []
    paper = _paper_title(file_path)
    pages: list[tuple[int, str, int | None]] = []
    offset_votes: list[int] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = normalize_text(page_text)
        if not page_text:
            continue
        raw_page = _infer_page(page_text)
        pages.append((page_idx, page_text, raw_page))
        if _is_plausible_page_label(page_idx, raw_page):
            assert raw_page is not None
            offset_votes.append(page_idx - raw_page)

    page_offset: int | None = None
    if offset_votes:
        page_offset, _ = Counter(offset_votes).most_common(1)[0]

    output: list[TextChunk] = []
    chunk_index = 0
    current_section: str | None = None
    for page_idx, page_text, raw_page in pages:
        printed_page = _resolve_printed_page(page_idx, raw_page, page_offset)
        page_section = _infer_section(page_text) or current_section
        if page_section:
            current_section = page_section
        for chunk_text in split_text(page_text, max_chars, overlap_chars):
            chunk_id = f"{file_path.as_posix()}::page={page_idx}::chunk={chunk_index}"
            output.append(
                TextChunk(
                    id=chunk_id,
                    text=chunk_text,
                    source=file_path.as_posix(),
                    chunk_index=chunk_index,
                    metadata={
                        "paper": paper,
                        "source_name": file_path.name,
                        "section": _infer_section(chunk_text) or current_section,
                        "page": printed_page if printed_page is not None else page_idx,
                        "pdf_page": page_idx,
                    },
                )
            )
            chunk_index += 1
    return output


def _paper_title(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip()


def _is_plausible_page_label(page_idx: int, page: int | None) -> bool:
    if page is None or page <= 0:
        return False
    offset = page_idx - page
    return -10 <= offset <= 120


def _resolve_printed_page(
    page_idx: int,
    raw_page: int | None,
    page_offset: int | None,
) -> int:
    if _is_plausible_page_label(page_idx, raw_page):
        assert raw_page is not None
        if page_offset is None or abs((page_idx - raw_page) - page_offset) <= 2:
            return raw_page
    if page_offset is not None:
        inferred = page_idx - page_offset
        if inferred > 0:
            return inferred
    return page_idx


def _infer_section(text: str) -> str | None:
    head = text[:2000]
    patterns = [
        r"(?m)^#{1,6}\s+(.+?)\s*$",
        r"(?m)^(?:Part|PART|Phần|PHẦN)\s+[\wIVXLC\d .:-]+(.+?)?\s*$",
        r"(?m)^(?:Chapter|CHAPTER|Chương|CHƯƠNG)\s+[\wIVXLC\d .:-]+(.+?)?\s*$",
        r"(?m)^(?:Section|SECTION|Mục|MỤC)\s+[\wIVXLC\d .:-]+(.+?)?\s*$",
        r"(?m)^(\d+(?:\.\d+){1,4}\s+[^\n]{3,140})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, head)
        if match:
            value = match.group(0).strip("# ").strip()
            return value[:180]
    return None


def _infer_page(text: str) -> int | None:
    search_text = text[:700]
    if len(text) > 700:
        search_text += "\n" + text[-700:]
    patterns = [
        r"(?i)\bpage\s+(\d{1,5})\b",
        r"(?i)\btrang\s+(\d{1,5})\b",
        r"(?m)^[-= ]*(\d{1,5})[-= ]*$",
        r"^\s*(\d{1,5})(?=[A-Za-zÀ-ỹ])",
        r"(?m)^[^\n]{0,100}\b(\d{1,5})\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, search_text)
        if match:
            try:
                return int(match.group(1))
            except (IndexError, ValueError):
                return None
    return None
