from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from retrieval_system.indexing.chunker import TextChunk


@dataclass(slots=True)
class SearchResult:
    chunk: TextChunk
    vector_score: float
    lexical_score: float | None = None
    cross_score: float | None = None
    final_score: float | None = None
    llm_score: float | None = None
    reason: str | None = None


class FileVectorStore:
    """Small persistent vector store for local experiments."""

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.vectors_path = self.index_dir / "vectors.npy"
        self.chunks_path = self.index_dir / "chunks.jsonl"
        self.meta_path = self.index_dir / "index_meta.json"
        self.vectors: np.ndarray | None = None
        self.chunks: list[TextChunk] = []

    def save(self, chunks: list[TextChunk], vectors: np.ndarray, metadata: dict[str, Any]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks/vectors length mismatch: {len(chunks)} != {len(vectors)}")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        vectors = _normalize(vectors.astype(np.float32, copy=False))
        np.save(self.vectors_path, vectors)
        with self.chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        with self.meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        self.vectors = vectors
        self.chunks = chunks

    def load(self) -> "FileVectorStore":
        if not self.vectors_path.exists() or not self.chunks_path.exists():
            raise FileNotFoundError(f"Missing vector index files in {self.index_dir}")
        self.vectors = np.load(self.vectors_path)
        self.chunks = []
        with self.chunks_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                self.chunks.append(TextChunk(**data))
        if len(self.chunks) != len(self.vectors):
            raise ValueError(
                f"Corrupt index: chunks/vectors mismatch {len(self.chunks)} != {len(self.vectors)}"
            )
        return self

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[SearchResult]:
        if self.vectors is None:
            self.load()
        assert self.vectors is not None
        if len(self.chunks) == 0:
            return []

        query = _normalize(query_vector.reshape(1, -1).astype(np.float32, copy=False))[0]
        scores = self.vectors @ query
        top_k = max(1, min(int(top_k), len(scores)))
        indices = np.argpartition(-scores, top_k - 1)[:top_k]
        indices = indices[np.argsort(-scores[indices])]
        return [
            SearchResult(chunk=self.chunks[int(idx)], vector_score=float(scores[int(idx)]))
            for idx in indices
        ]

    def context_window_chunks(
        self,
        chunk: TextChunk,
        before: int = 1,
        after: int = 1,
    ) -> list[TextChunk]:
        by_source = [
            item
            for item in self.chunks
            if item.source == chunk.source
            and chunk.chunk_index - before <= item.chunk_index <= chunk.chunk_index + after
        ]
        by_source.sort(key=lambda item: item.chunk_index)
        return by_source

    def context_window(self, chunk: TextChunk, before: int = 1, after: int = 1) -> str:
        return "\n\n".join(
            item.text for item in self.context_window_chunks(chunk, before=before, after=after)
        )

    def context_window_with_citations(
        self,
        chunk: TextChunk,
        before: int = 1,
        after: int = 1,
    ) -> str:
        blocks = []
        for item in self.context_window_chunks(chunk, before=before, after=after):
            blocks.append(f"[citation: {format_citation(item)}]\n{item.text}")
        return "\n\n".join(blocks)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vectors, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return vectors / denom


def format_citation(chunk: TextChunk) -> str:
    metadata = chunk.metadata or {}
    paper = _clean_metadata(metadata.get("paper")) or _clean_metadata(metadata.get("book")) or _clean_metadata(metadata.get("source_name"))
    if not paper:
        paper = Path(chunk.source).stem
    section = _clean_metadata(metadata.get("section")) or "section unknown"
    page = metadata.get("page")
    if page in (None, ""):
        page_text = "page unknown"
    else:
        page_text = f"page {page}"
    pdf_page = metadata.get("pdf_page")
    pdf_page_text = ""
    if pdf_page not in (None, "", page):
        pdf_page_text = f"; pdf_page {pdf_page}"
    return (
        f"paper={paper}; section={section}; {page_text}{pdf_page_text}; "
        f"source={chunk.source}; chunk={chunk.chunk_index}"
    )


def _clean_metadata(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
