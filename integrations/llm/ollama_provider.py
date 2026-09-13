"""Local model provider via Ollama.

Uses the standard library rather than a client package, so running a local
model adds no dependency to the project. Ollama exposes ``/api/chat`` on
localhost by default.

Two things differ from the hosted providers and are worth knowing:

* ``format: "json"`` constrains the model to emit valid JSON, which makes the
  repair retry in :class:`TextLLMProvider` mostly redundant here. It stays in
  place because "valid JSON" is not the same as "matches our schema".
* Local models are slower and the default timeout is correspondingly generous.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from core.exceptions import EvaluationError
from integrations.llm.base import TextLLMProvider


class OllamaProvider(TextLLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 180.0,
        num_predict: int = 1500,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_predict = num_predict

    def _post(self, path: str, payload: dict) -> dict:
        """POST JSON and return the decoded body. Overridden in tests."""
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise EvaluationError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise EvaluationError(
                f"could not reach Ollama at {self.base_url}: {exc.reason}. "
                "Is `ollama serve` running?"
            ) from exc
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"Ollama returned non-JSON: {exc}") from exc

    def complete(self, system: str, prompt: str) -> str:
        body = self._post(
            "/api/chat",
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"num_predict": self.num_predict, "temperature": 0},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        if "error" in body:
            raise EvaluationError(f"Ollama error: {body['error']}")
        content = (body.get("message") or {}).get("content")
        if not content:
            raise EvaluationError("Ollama returned no message content")
        return content
