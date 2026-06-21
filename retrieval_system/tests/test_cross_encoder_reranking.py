from __future__ import annotations

import unittest

import numpy as np

from retrieval_system.indexing.chunker import TextChunk
from retrieval_system.reranking.cross_encoder import CrossEncoderReranker
from retrieval_system.vectordb.store import SearchResult


class FakeCrossEncoderModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = np.asarray(scores, dtype=np.float32)
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs, **_):
        self.pairs = list(pairs)
        return self.scores


def _result(index: int, text: str, vector_score: float) -> SearchResult:
    return SearchResult(
        chunk=TextChunk(
            id=f"doc::{index}",
            text=text,
            source="paper.pdf",
            chunk_index=index,
            metadata={"paper": "Paper", "section": f"Section {index}"},
        ),
        vector_score=vector_score,
    )


class CrossEncoderRerankerTest(unittest.TestCase):
    def test_reranks_by_cross_score_and_keeps_top_k(self) -> None:
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "fake"
        reranker.batch_size = 8
        reranker.max_length = 128
        reranker.model = FakeCrossEncoderModel([-2.0, 3.0, 0.0])
        results = [
            _result(0, "weak", 0.9),
            _result(1, "strong", 0.5),
            _result(2, "medium", 0.7),
        ]

        output = reranker.rerank("test query", results, top_k=2)

        self.assertEqual([item.chunk.chunk_index for item in output], [1, 2])
        self.assertGreater(output[0].cross_score or 0.0, output[1].cross_score or 0.0)
        self.assertEqual(output[0].final_score, output[0].cross_score)

    def test_pair_contains_metadata_and_chunk_text(self) -> None:
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "fake"
        reranker.batch_size = 8
        reranker.max_length = 128
        fake_model = FakeCrossEncoderModel([1.0])
        reranker.model = fake_model

        reranker.rerank("query", [_result(3, "passage body", 0.4)], top_k=1)

        self.assertEqual(fake_model.pairs[0][0], "query")
        self.assertIn("Paper", fake_model.pairs[0][1])
        self.assertIn("Section 3", fake_model.pairs[0][1])
        self.assertIn("passage body", fake_model.pairs[0][1])

    def test_empty_query_is_rejected(self) -> None:
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.batch_size = 8
        reranker.model = FakeCrossEncoderModel([1.0])
        with self.assertRaisesRegex(ValueError, "empty query"):
            reranker.rerank("  ", [_result(0, "text", 0.1)])


if __name__ == "__main__":
    unittest.main()
