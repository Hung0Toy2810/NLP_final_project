from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

from retrieval_system.vectordb.store import SearchResult


if TYPE_CHECKING:
    import numpy as np


class CrossEncoderReranker:
    """Rerank retrieved chunks with a pretrained MS MARCO cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Cross-encoder reranking requires sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, int(max_length))
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=self.max_length,
        )

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        if not results:
            return []
        query = query.strip()
        if not query:
            raise ValueError("Cannot rerank an empty query")

        pairs = [(query, _passage_text(result)) for result in results]
        raw_scores: np.ndarray = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = raw_scores.reshape(-1).tolist()
        if len(scores) != len(results):
            raise RuntimeError(
                f"Cross-encoder returned {len(scores)} scores for {len(results)} candidates"
            )

        reranked = list(results)
        for result, raw_score in zip(reranked, scores):
            result.cross_score = _sigmoid(float(raw_score))
            result.final_score = result.cross_score
        reranked.sort(
            key=lambda item: item.cross_score if item.cross_score is not None else float("-inf"),
            reverse=True,
        )
        return reranked[: max(1, min(int(top_k), len(reranked)))]


def _passage_text(result: SearchResult) -> str:
    metadata = result.chunk.metadata or {}
    prefix = " ".join(
        str(value).strip()
        for value in (metadata.get("paper"), metadata.get("section"))
        if value and str(value).strip()
    )
    text = result.chunk.text.strip()
    return f"{prefix}\n{text}" if prefix else text


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)
