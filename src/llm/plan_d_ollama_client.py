from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"


@dataclass(frozen=True)
class OllamaGenerateResult:
    """Result from one local Ollama generate request."""

    response_received: bool
    response_text: str
    latency_seconds: float
    raw_response: dict[str, Any] | None
    error: str | None


class OllamaClient:
    """Minimal local-only Ollama generate client for Plan D smoke tests."""

    def __init__(self, endpoint: str = DEFAULT_OLLAMA_GENERATE_URL, timeout_seconds: float = 45.0):
        if endpoint != DEFAULT_OLLAMA_GENERATE_URL:
            raise ValueError("Plan D smoke tests only allow the local Ollama generate endpoint.")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float = 0.0,
    ) -> OllamaGenerateResult:
        """Run exactly one local Ollama generate request configured for JSON output."""

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            latency = time.perf_counter() - started
            error = f"HTTP {exc.code}: {exc.reason}"
            return OllamaGenerateResult(False, "", latency, None, error)
        except urllib.error.URLError as exc:
            latency = time.perf_counter() - started
            return OllamaGenerateResult(False, "", latency, None, str(exc.reason))
        except TimeoutError:
            latency = time.perf_counter() - started
            return OllamaGenerateResult(False, "", latency, None, "timed out")
        except Exception as exc:  # pragma: no cover - depends on local Ollama state
            latency = time.perf_counter() - started
            return OllamaGenerateResult(False, "", latency, None, str(exc))

        latency = time.perf_counter() - started
        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            return OllamaGenerateResult(False, "", latency, None, f"invalid Ollama JSON: {exc}")

        response_text = parsed.get("response")
        if not isinstance(response_text, str):
            return OllamaGenerateResult(False, "", latency, parsed, "Ollama response field missing")

        return OllamaGenerateResult(True, response_text.strip(), latency, parsed, None)
