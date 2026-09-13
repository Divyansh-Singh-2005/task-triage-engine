"""Provider selection.

One place decides which provider is in use, so switching between a hosted
model, a local model and the offline baseline is a single environment variable
rather than a code change.
"""

from __future__ import annotations

from core.exceptions import ConfigError
from integrations.llm.base import LLMProvider
from integrations.llm.heuristic import HeuristicProvider

KNOWN_PROVIDERS = ("heuristic", "anthropic", "openai", "ollama")


def get_provider(settings) -> LLMProvider:
    """Build the provider named by ``settings.llm_provider``.

    A hosted provider with no key configured falls back to the heuristic
    baseline rather than crashing the monitoring loop: a scored baseline plus a
    visible warning beats no scores at all. Ollama needs no key, so a missing
    local server surfaces as an evaluation error instead - that is a real
    misconfiguration, not an absent credential.
    """
    name = (settings.llm_provider or "heuristic").strip().casefold()

    if name == "heuristic":
        return HeuristicProvider()

    if name == "anthropic":
        if not settings.anthropic_api_key:
            return HeuristicProvider()
        from integrations.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
        )

    if name == "openai":
        if not settings.openai_api_key:
            return HeuristicProvider()
        from integrations.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_tokens=settings.llm_max_tokens,
        )

    if name == "ollama":
        from integrations.llm.ollama_provider import OllamaProvider

        return OllamaProvider(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            num_predict=settings.llm_max_tokens,
        )

    raise ConfigError(
        f"unknown LLM provider {name!r} (known: {', '.join(KNOWN_PROVIDERS)})"
    )
