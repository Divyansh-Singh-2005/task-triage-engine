"""OpenAI chat provider.

The SDK is imported lazily, so the app runs and the tests pass without
``openai`` installed. The key is read once at construction and never logged.

Note the shape difference from Anthropic: there is no separate ``system``
parameter, so the system prompt becomes a message with role "system". That
translation is exactly what this layer exists to absorb.
"""

from __future__ import annotations

from core.exceptions import EvaluationError
from integrations.llm.base import TextLLMProvider


class OpenAIProvider(TextLLMProvider):
    name = "openai"

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
                "OPENAI_API_KEY is not set; set it in .env or switch "
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
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - env dependent
                raise EvaluationError(
                    "the 'openai' package is not installed; run pip install openai"
                ) from exc
            self._client = OpenAI(api_key=self._api_key, timeout=self.timeout)
        return self._client

    def complete(self, system: str, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                # JSON mode still needs the word "json" somewhere in the
                # messages, which the system prompt already supplies.
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            raise EvaluationError(
                f"OpenAI request failed ({type(exc).__name__}): {exc}"
            ) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise EvaluationError("OpenAI returned no choices")
        content = getattr(choices[0].message, "content", None)
        if not content:
            raise EvaluationError("OpenAI returned empty content")
        return content
