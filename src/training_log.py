import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional


def _json_safe(value: Any) -> Any:
    """Convert common numeric/path values to JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


class TrainingJsonLogger:
    """
    Append-only JSONL logger for training metrics.

    JSONL is intentionally used instead of one large JSON array so the monitor
    can read partial logs while training is still running.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, event: str, **payload: Any) -> None:
        record = {
            "event": event,
            "time": datetime.now(timezone.utc).isoformat(),
            "time_unix": time.time(),
            "host": socket.gethostname(),
        }
        record.update({key: _json_safe(value) for key, value in payload.items()})

        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


_LOGGER: Optional[TrainingJsonLogger] = None


def init_training_logger(path: str) -> TrainingJsonLogger:
    global _LOGGER
    _LOGGER = TrainingJsonLogger(path)
    return _LOGGER


def get_training_logger() -> Optional[TrainingJsonLogger]:
    return _LOGGER


def log_event(event: str, **payload: Any) -> None:
    logger = get_training_logger()
    if logger is not None:
        logger.log(event, **payload)
