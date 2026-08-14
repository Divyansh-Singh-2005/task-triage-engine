"""Anthropic chat provider.

The SDK is imported lazily so the rest of the application runs, and the test
suite passes, without ``anthropic`` installed or an API key present.

The key is read once at construction and never logged, never written to an
event payload, and never included in an exception message.
"""

from __future__ import annotations

from core.exceptions import EvaluationError
from integrations.llm.base import TextLLMProvider


class AnthropicProvider(TextLLMProvider):
    """Calls the Anthropic Messages API for task evaluation."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_tokens: int = 1500,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise EvaluationError(
                "ANTHROPIC_API_KEY is not set; set it in .env or switch "
                "LLM_PROVIDER to 'heuristic'"
            )
        self._api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover - env dependent
                raise EvaluationError(
                    "the 'anthropic' package is not installed; run "
                    "pip install anthropic"
                ) from exc
            self._client = Anthropic(api_key=self._api_key, timeout=self.timeout)
        return self._client

    def complete(self, system: str, prompt: str) -> str:
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # network, auth, rate limit, overload
            # str(exc) from the SDK does not contain the key, but the type name
            # alone is enough for diagnosis, so keep the surface small.
            raise EvaluationError(
                f"Anthropic request failed ({type(exc).__name__}): {exc}"
            ) from exc

        parts = [
            block.text
            for block in getattr(message, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        if not parts:
            raise EvaluationError("Anthropic returned no text content")
        return "".join(parts)
