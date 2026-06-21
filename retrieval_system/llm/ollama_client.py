from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class OllamaClient:
    host: str = "http://localhost:11434"
    model: str = "gemma3:4b"
    timeout_sec: float | None = None
    think: bool | None = False

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        url = self.host.rstrip("/") + "/api/generate"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if self.think is not None:
            payload["think"] = self.think
        if system:
            payload["system"] = system

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            if self.timeout_sec is None:
                response_ctx = urllib.request.urlopen(request)
            else:
                response_ctx = urllib.request.urlopen(request, timeout=self.timeout_sec)
            with response_ctx as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        return str(data.get("response", "")).strip()
