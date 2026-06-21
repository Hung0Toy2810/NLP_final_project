from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from retrieval_system.indexing.encoder import SFTBEEncoder
from retrieval_system.llm.ollama_client import OllamaClient
from retrieval_system.llm.verifier import rerank_with_llm
from retrieval_system.reranking.cross_encoder import CrossEncoderReranker
from retrieval_system.vectordb.store import FileVectorStore, SearchResult, format_citation


ANSWER_SYSTEM_PROMPT = """You answer using only the provided retrieved context.
If the context contradicts the user's assumption, say so clearly.
If the context is insufficient, say that the evidence is insufficient.
Every factual claim must cite the supporting context id, for example [1].
Use only context ids that are shown in the retrieved context.
The first sentence must name the source paper, section, and page for the main
evidence. If a page or section is unknown, say page unknown or section unknown.
Do not invent sources, pages, or sections.
Keep the answer concise.
Use this format:
Answer:
- <claim with citation>

Evidence:
- [1] <paper>, <section>, <page>
"""


@dataclass(slots=True)
class RetrievalPipeline:
    encoder: SFTBEEncoder
    store: FileVectorStore
    cross_encoder: CrossEncoderReranker | None = None
    llm_client: OllamaClient | None = None
    context_before: int = 1
    context_after: int = 1

    @classmethod
    def from_paths(
        cls,
        index_dir: str | Path,
        checkpoint_path: str | Path,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "gemma3:4b",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        batch_size: int = 32,
        cross_encoder_batch_size: int = 32,
    ) -> "RetrievalPipeline":
        encoder = SFTBEEncoder(checkpoint_path=checkpoint_path, batch_size=batch_size)
        store = FileVectorStore(index_dir).load()
        cross_encoder = CrossEncoderReranker(
            model_name=cross_encoder_model,
            batch_size=cross_encoder_batch_size,
        )
        client = OllamaClient(host=ollama_host, model=ollama_model)
        return cls(
            encoder=encoder,
            store=store,
            cross_encoder=cross_encoder,
            llm_client=client,
        )

    def retrieve(
        self,
        query: str,
        vector_top_k: int = 100,
        cross_rerank_k: int = 10,
        llm_rerank_k: int = 8,
        final_k: int = 5,
        use_llm: bool = True,
    ) -> list[SearchResult]:
        query_vector = self.encoder.encode_one(query)
        candidates = self.store.search(query_vector, top_k=vector_top_k)
        if not candidates:
            return []
        _apply_lexical_boost(query, candidates)
        candidates.sort(
            key=lambda item: item.final_score if item.final_score is not None else item.vector_score,
            reverse=True,
        )

        if self.cross_encoder is not None:
            candidates = self.cross_encoder.rerank(
                query,
                candidates,
                top_k=cross_rerank_k,
            )
        else:
            candidates = candidates[: max(1, min(cross_rerank_k, len(candidates)))]

        rerank_input = candidates[: max(1, min(llm_rerank_k, len(candidates)))]
        if use_llm and self.llm_client is not None:
            try:
                reranked = rerank_with_llm(
                    query,
                    rerank_input,
                    context_provider=self.context_for_result,
                    client=self.llm_client,
                )
                remaining = candidates[len(rerank_input) :]
                results = reranked + remaining
            except RuntimeError as exc:
                results = candidates
                for item in results:
                    item.reason = f"LLM verifier unavailable: {exc}"
        else:
            results = candidates

        results.sort(
            key=lambda item: item.final_score if item.final_score is not None else item.vector_score,
            reverse=True,
        )
        return results[: max(1, final_k)]

    def answer(self, query: str, results: list[SearchResult]) -> str:
        if self.llm_client is None:
            raise RuntimeError("LLM client is not configured")
        context_blocks = []
        evidence = self.answer_evidence(results)
        for idx, result in enumerate(evidence, start=1):
            context_blocks.append(
                f"[{idx}] citation={format_citation(result.chunk)}\n"
                f"score={_score_text(result)}\n"
                f"{self.context_for_result(result)}"
            )
        prompt = (
            "/no_think\n"
            "User query:\n"
            f"{query}\n\n"
            "Retrieved context:\n"
            + "\n\n---\n\n".join(context_blocks)
            + "\n\nAnswer the user using the context above."
        )
        answer = self.llm_client.generate(
            prompt,
            system=ANSWER_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=900,
        )
        if answer.strip():
            return answer
        return _fallback_answer(evidence)

    def answer_evidence(
        self,
        results: list[SearchResult],
        max_items: int = 4,
        min_score_ratio: float = 0.75,
    ) -> list[SearchResult]:
        if not results:
            return []

        top = results[0]
        top_score = top.final_score if top.final_score is not None else top.vector_score
        top_paper = _paper_id(top)
        selected: list[SearchResult] = []
        for result in results:
            score = result.final_score if result.final_score is not None else result.vector_score
            same_paper = _paper_id(result) == top_paper
            close_enough = score >= top_score * min_score_ratio
            if same_paper and close_enough:
                selected.append(result)
            if len(selected) >= max_items:
                break

        if selected:
            return selected
        return results[:1]

    def context_for_result(self, result: SearchResult) -> str:
        return self.store.context_window_with_citations(
            result.chunk,
            before=self.context_before,
            after=self.context_after,
        )

    def citation_for_result(self, result: SearchResult) -> str:
        return format_citation(result.chunk)


def _score_text(result: SearchResult) -> str:
    parts = [f"vector={result.vector_score:.4f}"]
    if result.cross_score is not None:
        parts.append(f"cross={result.cross_score:.4f}")
    if result.llm_score is not None:
        parts.append(f"llm={result.llm_score:.4f}")
    return ", ".join(parts)


def _paper_id(result: SearchResult) -> str:
    metadata = result.chunk.metadata or {}
    paper = metadata.get("paper") or metadata.get("book") or metadata.get("source_name")
    return str(paper or result.chunk.source).strip()


def _apply_lexical_boost(query: str, results: list[SearchResult]) -> None:
    query_terms = _content_terms(query)
    if not query_terms:
        for result in results:
            result.lexical_score = 0.0
            result.final_score = result.vector_score
        return

    for result in results:
        text_terms = _content_terms(_searchable_text(result))
        overlap = len(query_terms & text_terms) / len(query_terms)
        result.lexical_score = overlap
        result.final_score = 0.82 * result.vector_score + 0.18 * overlap


def _searchable_text(result: SearchResult) -> str:
    metadata = result.chunk.metadata or {}
    return " ".join(
        str(part)
        for part in (
            metadata.get("paper"),
            metadata.get("section"),
            result.chunk.text,
        )
        if part
    )


_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "how",
    "into",
    "that",
    "the",
    "their",
    "this",
    "toward",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def _content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _fallback_answer(results: list[SearchResult]) -> str:
    if not results:
        return "Không có bằng chứng phù hợp trong index."
    lines = [
        "LLM không trả về câu trả lời, dưới đây là các bằng chứng truy xuất được:"
    ]
    for idx, result in enumerate(results, start=1):
        text = result.chunk.text.replace("\n", " ").strip()
        if len(text) > 420:
            text = text[:420].rstrip() + "..."
        lines.append(f"[{idx}] {format_citation(result.chunk)}\n{text}")
    return "\n\n".join(lines)
