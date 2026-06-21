from __future__ import annotations

import unittest
from typing import cast

import numpy as np

from retrieval_system.indexing.chunker import TextChunk
from retrieval_system.indexing.encoder import SFTBEEncoder
from retrieval_system.reranking.cross_encoder import CrossEncoderReranker
from retrieval_system.search.pipeline import RetrievalPipeline
from retrieval_system.vectordb.store import FileVectorStore, SearchResult


class FakeEncoder:
    def encode_one(self, query: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class FakeStore:
    def __init__(self) -> None:
        self.requested_top_k: int | None = None

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        self.requested_top_k = top_k
        return [
            SearchResult(
                chunk=TextChunk(
                    id=f"doc::{index}",
                    text=f"passage {index}",
                    source="paper.pdf",
                    chunk_index=index,
                ),
                vector_score=1.0 - index / 100.0,
            )
            for index in range(top_k)
        ]


class FakeReranker:
    def __init__(self) -> None:
        self.input_size = 0
        self.requested_top_k = 0

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        self.input_size = len(results)
        self.requested_top_k = top_k
        output = list(reversed(results))[:top_k]
        for rank, result in enumerate(output):
            result.cross_score = 1.0 - rank / max(1, top_k)
            result.final_score = result.cross_score
        return output


class RetrievalPipelineRerankingTest(unittest.TestCase):
    def test_vector_top_100_is_reduced_to_cross_top_10(self) -> None:
        store = FakeStore()
        reranker = FakeReranker()
        pipeline = RetrievalPipeline(
            encoder=cast(SFTBEEncoder, FakeEncoder()),
            store=cast(FileVectorStore, store),
            cross_encoder=cast(CrossEncoderReranker, reranker),
            llm_client=None,
        )

        results = pipeline.retrieve(
            "query",
            vector_top_k=100,
            cross_rerank_k=10,
            llm_rerank_k=8,
            final_k=5,
            use_llm=False,
        )

        self.assertEqual(store.requested_top_k, 100)
        self.assertEqual(reranker.input_size, 100)
        self.assertEqual(reranker.requested_top_k, 10)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(result.cross_score is not None for result in results))


if __name__ == "__main__":
    unittest.main()
