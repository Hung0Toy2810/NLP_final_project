from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Callable

from retrieval_system.llm.ollama_client import OllamaClient
from retrieval_system.vectordb.store import SearchResult, format_citation


SYSTEM_PROMPT = """You are a strict semantic retrieval verifier.
Score whether each candidate contains evidence relevant to the user query.
High score means the candidate is useful evidence, even if it contradicts the
query assumption. Penalize unrelated text. Pay attention to negation, numbers,
locations, entity names, and subject-object reversal.
Return only valid JSON."""


def rerank_with_llm(
    query: str,
    results: list[SearchResult],
    context_provider: Callable[[SearchResult], str],
    client: OllamaClient,
) -> list[SearchResult]:
    if not results:
        return []

    prompt = _build_prompt(query, results, context_provider)
    response = client.generate(
        prompt,
        system=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=1600,
    )
    scores = _parse_scores(response)
    if not scores:
        return [
            replace(
                item,
                llm_score=None,
                final_score=item.cross_score if item.cross_score is not None else item.vector_score,
                reason="LLM parse failed",
            )
            for item in results
        ]

    output: list[SearchResult] = []
    for idx, item in enumerate(results, start=1):
        info = scores.get(idx, {})
        llm_score = _clamp_score(info.get("score"))
        reason = str(info.get("reason", "")).strip() or None
        base_score = item.cross_score if item.cross_score is not None else item.vector_score
        final_score = (
            0.25 * base_score + 0.75 * llm_score
            if llm_score is not None
            else item.vector_score
        )
        output.append(
            replace(
                item,
                llm_score=llm_score,
                final_score=final_score,
                reason=reason,
            )
        )
    output.sort(key=lambda item: item.final_score if item.final_score is not None else item.vector_score, reverse=True)
    return output


def _build_prompt(
    query: str,
    results: list[SearchResult],
    context_provider: Callable[[SearchResult], str],
) -> str:
    blocks = []
    for idx, result in enumerate(results, start=1):
        context = context_provider(result).strip()
        blocks.append(
            f"Candidate {idx}\n"
            f"citation: {format_citation(result.chunk)}\n"
            f"source: {result.chunk.source}\n"
            f"vector_score: {result.vector_score:.4f}\n"
            f"cross_score: {result.cross_score if result.cross_score is not None else 'n/a'}\n"
            f"text:\n{context}\n"
        )

    return (
        "/no_think\n"
        "User query:\n"
        f"{query}\n\n"
        "Candidates:\n"
        + "\n---\n".join(blocks)
        + "\n\nReturn JSON with this exact shape:\n"
        '[{"id": 1, "score": 0.0, "reason": "short reason"}, ...]\n'
        "Use score from 0 to 1. Score useful contradictory evidence high."
    )


def _parse_scores(text: str) -> dict[int, dict]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(parsed, dict):
                parsed = parsed.get("results", parsed.get("candidates", []))
            if isinstance(parsed, list):
                out = {}
                for item in parsed:
                    if isinstance(item, dict) and "id" in item:
                        out[int(item["id"])] = item
                return out
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return {}


def _clamp_score(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))
