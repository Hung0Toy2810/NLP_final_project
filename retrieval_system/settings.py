from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    config["checkpoint_path"] = os.environ.get("SFTBE_RETRIEVAL_CHECKPOINT", config["checkpoint_path"])
    config["index_dir"] = os.environ.get("SFTBE_RETRIEVAL_INDEX_DIR", config["index_dir"])
    config["ollama_host"] = os.environ.get("OLLAMA_HOST", config["ollama_host"])
    config["ollama_model"] = os.environ.get("OLLAMA_MODEL", config["ollama_model"])
    config["cross_encoder_model"] = os.environ.get(
        "CROSS_ENCODER_MODEL", config["cross_encoder_model"]
    )
    return config
