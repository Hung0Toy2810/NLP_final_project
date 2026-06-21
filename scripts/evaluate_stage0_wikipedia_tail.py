import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, get_device  # noqa: E402
from dataset import get_tokenizer  # noqa: E402
from model.encoder import create_sftbe_model  # noqa: E402


def extract_sentences(text: str) -> list[str]:
    sentences = []
    for sent in text.split(". "):
        sent = sent.strip()
        if 20 <= len(sent) <= 500:
            sentences.append(sent)
    return sentences


def collect_tail_sentences(target_mb: float) -> tuple[list[str], dict[str, Any]]:
    dataset_name, dataset_config = DATA_CONFIG["wikipedia_dataset"]
    dataset = cast(
        Dataset,
        load_dataset(dataset_name, dataset_config, split="train"),
    )

    target_bytes = int(target_mb * 1024 * 1024)
    total_bytes = 0
    article_count = 0
    sentences: list[str] = []

    for idx in range(len(dataset) - 1, -1, -1):
        item = dataset[idx]
        text = str(item.get("text", ""))
        encoded_len = len(text.encode("utf-8"))
        total_bytes += encoded_len
        article_count += 1
        sentences.extend(extract_sentences(text))
        if total_bytes >= target_bytes:
            first_index = idx
            break
    else:
        first_index = 0

    meta = {
        "dataset": f"{dataset_name}/{dataset_config}",
        "split": "train",
        "direction": "tail_to_head",
        "target_mb": target_mb,
        "actual_text_bytes": total_bytes,
        "actual_text_mb": total_bytes / (1024 * 1024),
        "article_count": article_count,
        "first_index_included": first_index,
        "last_index_included": len(dataset) - 1,
        "sentence_count_after_filter": len(sentences),
        "sentence_filter": "20 <= len(sentence) <= 500, split by '. '",
    }
    return sentences, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-mb", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "stage0_final.pt"))
    parser.add_argument("--output", default=str(ROOT / "checkpoints" / "stage0_tail10mb_eval.json"))
    args = parser.parse_args()

    start = time.time()
    device = get_device()
    tokenizer = get_tokenizer(DATA_CONFIG["tokenizer_name"])
    sentences, data_meta = collect_tail_sentences(args.target_mb)

    model = create_sftbe_model(MODEL_CONFIG).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    teacher = SentenceTransformer(TRAIN_CONFIG["teacher_model"], device=str(device))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    total_loss = 0.0
    total_cosine = 0.0
    total_samples = 0
    min_cosine = float("inf")
    max_cosine = float("-inf")

    with torch.no_grad():
        for start_idx in range(0, len(sentences), args.batch_size):
            batch_texts = sentences[start_idx:start_idx + args.batch_size]
            encoded = tokenizer(
                batch_texts,
                max_length=MODEL_CONFIG["max_seq_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            student = model(input_ids, attention_mask)
            teacher_embeddings = teacher.encode(
                batch_texts,
                batch_size=args.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).to(device=device, dtype=torch.float32)

            student_norm = F.normalize(student.float(), p=2, dim=-1)
            teacher_norm = F.normalize(teacher_embeddings.float(), p=2, dim=-1)
            cosine = (student_norm * teacher_norm).sum(dim=-1)
            loss = 1.0 - cosine

            total_loss += loss.sum().item()
            total_cosine += cosine.sum().item()
            total_samples += len(batch_texts)
            min_cosine = min(min_cosine, cosine.min().item())
            max_cosine = max(max_cosine, cosine.max().item())

            if (start_idx // args.batch_size + 1) % 50 == 0:
                print(
                    json.dumps(
                        {
                            "processed": total_samples,
                            "total": len(sentences),
                            "loss": total_loss / max(1, total_samples),
                            "cosine": total_cosine / max(1, total_samples),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    result = {
        **data_meta,
        "checkpoint": args.checkpoint,
        "teacher_model": TRAIN_CONFIG["teacher_model"],
        "tokenizer": DATA_CONFIG["tokenizer_name"],
        "device": str(device),
        "batch_size": args.batch_size,
        "test_samples": total_samples,
        "test_loss": total_loss / max(1, total_samples),
        "test_cosine": total_cosine / max(1, total_samples),
        "min_cosine": min_cosine,
        "max_cosine": max_cosine,
        "elapsed_sec": time.time() - start,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
