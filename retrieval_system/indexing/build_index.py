from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

from retrieval_system.settings import load_config
from retrieval_system.indexing.chunker import load_chunks
from retrieval_system.indexing.encoder import SFTBEEncoder
from retrieval_system.vectordb.store import FileVectorStore


def _normalize_input_paths(input_path: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(input_path, (str, Path)):
        return [Path(input_path)]
    return [Path(path) for path in input_path]


def _sanitize_chunks(chunks):
    output = []
    for chunk in chunks:
        text = "" if chunk.text is None else str(chunk.text).strip()
        if not text:
            continue
        chunk.text = text
        output.append(chunk)
    return output


def build_index(
    input_path: str | Path | Sequence[str | Path],
    index_dir: str | Path,
    checkpoint_path: str | Path,
    chunk_max_chars: int = 900,
    chunk_overlap_chars: int = 120,
    batch_size: int = 32,
) -> None:
    start = time.time()
    input_paths = _normalize_input_paths(input_path)
    chunks = []
    for path in input_paths:
        chunks.extend(load_chunks(path, chunk_max_chars, chunk_overlap_chars))
    chunks = _sanitize_chunks(chunks)
    if not chunks:
        raise RuntimeError(f"No supported document chunks found under {input_paths}")

    encoder = SFTBEEncoder(checkpoint_path=checkpoint_path, batch_size=batch_size)
    vectors = encoder.encode(chunk.text for chunk in chunks)
    store = FileVectorStore(index_dir)
    store.save(
        chunks,
        vectors,
        metadata={
            "input_path": str(input_path),
            "input_paths": [str(path) for path in input_paths],
            "checkpoint_path": str(checkpoint_path),
            "num_chunks": len(chunks),
            "embedding_dim": encoder.dim,
            "chunk_max_chars": chunk_max_chars,
            "chunk_overlap_chars": chunk_overlap_chars,
            "created_at_unix": time.time(),
            "elapsed_sec": time.time() - start,
        },
    )
    print(f"Indexed {len(chunks)} chunks into {index_dir}")
    print(f"Elapsed: {time.time() - start:.1f}s")


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Build local SFT-BE vector index")
    parser.add_argument(
        "input_path",
        nargs="+",
        help="One or more files/directories containing .txt/.md/.rst/.pdf documents",
    )
    parser.add_argument("--index-dir", default=config["index_dir"])
    parser.add_argument("--checkpoint", default=config["checkpoint_path"])
    parser.add_argument("--chunk-max-chars", type=int, default=int(config["chunk_max_chars"]))
    parser.add_argument("--chunk-overlap-chars", type=int, default=int(config["chunk_overlap_chars"]))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    build_index(
        input_path=args.input_path,
        index_dir=args.index_dir,
        checkpoint_path=args.checkpoint,
        chunk_max_chars=args.chunk_max_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
