from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
import re

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DATA_CONFIG, MODEL_CONFIG, get_device  # noqa: E402

if TYPE_CHECKING:
    import torch


class SFTBEEncoder:
    """Load SFT-BE checkpoint and produce normalized sentence embeddings."""

    def __init__(
        self,
        checkpoint_path: str | Path = "checkpoints/stage0_final.pt",
        device: "str | torch.device | None" = None,
        batch_size: int = 32,
    ) -> None:
        import torch
        from transformers import AutoTokenizer
        from model.encoder import create_sftbe_model

        self.torch = torch
        self.functional = __import__("torch.nn.functional", fromlist=[""])
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SFT-BE checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(device) if device is not None else get_device()
        self.batch_size = max(1, int(batch_size))
        self.tokenizer = AutoTokenizer.from_pretrained(
            DATA_CONFIG["tokenizer_name"],
            use_fast=True,
        )

        self.model = create_sftbe_model(MODEL_CONFIG).to(self.device)
        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

    @property
    def dim(self) -> int:
        return int(MODEL_CONFIG["hidden_size"])

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        items = []
        for text in texts:
            if text is None:
                continue
            item = _clean_for_tokenizer(str(text))
            if item:
                items.append(item)
        if not items:
            return np.empty((0, self.dim), dtype=np.float32)

        torch = self.torch
        vectors = []
        with torch.no_grad():
            for start in range(0, len(items), self.batch_size):
                batch = items[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=MODEL_CONFIG["max_seq_length"],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                output = self.model(input_ids, attention_mask).float()
                output = self.functional.normalize(output, p=2, dim=-1)
                vectors.append(output.cpu())

        return torch.cat(vectors, dim=0).numpy().astype(np.float32, copy=False)

    def encode_one(self, text: str) -> np.ndarray:
        vectors = self.encode([text])
        if len(vectors) == 0:
            raise ValueError("Cannot encode an empty query")
        return vectors[0]


def _clean_for_tokenizer(text: str) -> str:
    text = re.sub(r"[\ud800-\udfff]", "", text)
    return text.strip()
