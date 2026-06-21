from __future__ import annotations

import argparse
import sys
import time

from retrieval_system.settings import load_config


def main() -> None:
    start = time.time()
    config = load_config()
    parser = argparse.ArgumentParser(description="Query SFT-BE vector DB with optional Ollama verification")
    parser.add_argument("query", help="User query")
    parser.add_argument("--index-dir", default=config["index_dir"])
    parser.add_argument("--checkpoint", default=config["checkpoint_path"])
    parser.add_argument("--ollama-host", default=config["ollama_host"])
    parser.add_argument("--ollama-model", default=config["ollama_model"])
    parser.add_argument("--cross-encoder-model", default=config["cross_encoder_model"])
    parser.add_argument(
        "--cross-encoder-batch-size",
        type=int,
        default=int(config["cross_encoder_batch_size"]),
    )
    parser.add_argument("--vector-top-k", type=int, default=int(config["vector_top_k"]))
    parser.add_argument("--cross-rerank-k", type=int, default=int(config["cross_rerank_k"]))
    parser.add_argument("--llm-rerank-k", type=int, default=int(config["llm_rerank_k"]))
    parser.add_argument("--final-context-k", type=int, default=int(config["final_context_k"]))
    parser.add_argument("--llm-rerank", dest="use_llm_rerank", action="store_true")
    parser.add_argument("--no-llm-rerank", dest="use_llm_rerank", action="store_false")
    parser.set_defaults(use_llm_rerank=False)
    parser.add_argument("--answer", action="store_true", help="Ask Ollama to answer from final context")
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()

    _progress(args, "Importing retrieval pipeline")
    from retrieval_system.search.pipeline import RetrievalPipeline

    _progress(args, f"Imports ready in {time.time() - start:.1f}s")
    _progress(args, "Loading retrieval pipeline")
    pipeline = RetrievalPipeline.from_paths(
        index_dir=args.index_dir,
        checkpoint_path=args.checkpoint,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        cross_encoder_model=args.cross_encoder_model,
        cross_encoder_batch_size=args.cross_encoder_batch_size,
    )
    _progress(args, f"Pipeline ready in {time.time() - start:.1f}s")
    _progress(args, "Retrieving top results")
    results = pipeline.retrieve(
        args.query,
        vector_top_k=args.vector_top_k,
        cross_rerank_k=args.cross_rerank_k,
        llm_rerank_k=args.llm_rerank_k,
        final_k=args.final_context_k,
        use_llm=args.use_llm_rerank,
    )
    _progress(args, f"Retrieved {len(results)} results in {time.time() - start:.1f}s")

    print("\nTop results")
    print("=" * 80)
    for idx, result in enumerate(results, start=1):
        score = result.final_score if result.final_score is not None else result.vector_score
        print(
            f"{idx}. score={score:.4f} vector={result.vector_score:.4f} "
            f"cross={result.cross_score} llm={result.llm_score}"
        )
        print(f"   citation={pipeline.citation_for_result(result)}")
        if result.reason:
            print(f"   reason={result.reason}")
        preview = result.chunk.text.replace("\n", " ")
        print(f"   text={preview[:240]}")
        if args.show_context:
            print("   context:")
            print(pipeline.context_for_result(result))
        print("-" * 80)

    if args.answer:
        print("\nAnswer")
        print("=" * 80)
        sys.stdout.flush()
        _progress(args, "Generating answer with Ollama")
        print(pipeline.answer(args.query, results))
        _progress(args, f"Done in {time.time() - start:.1f}s")


def _progress(args: argparse.Namespace, message: str) -> None:
    if not args.quiet_progress:
        print(f"[retrieval] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
