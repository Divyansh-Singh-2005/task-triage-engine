"""Provider selection.

One place decides which provider is in use, so switching between a real model
and the offline baseline is a single environment variable rather than a code
change.
"""

from __future__ import annotations

from core.exceptions import ConfigError
from integrations.llm.base import LLMProvider
from integrations.llm.heuristic import HeuristicProvider

KNOWN_PROVIDERS = ("heuristic", "anthropic")


def get_provider(settings) -> LLMProvider:
    """Build the provider named by ``settings.llm_provider``.

    Falls back to the heuristic provider when a model provider is selected but
    has no key configured, rather than crashing the monitoring loop: a scored
    baseline plus a warning is more useful than no scores at all.
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

    raise ConfigError(
        f"unknown LLM provider {name!r} (known: {', '.join(KNOWN_PROVIDERS)})"
    )
