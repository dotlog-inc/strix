"""Model-subscription routes, provider-agnostic.

Two ``STRIX_LLM`` prefixes run on a consumer plan instead of a metered API key:

- ``chatgpt/<model>`` — a ChatGPT Plus/Pro plan (:mod:`strix.config.codex`)
- ``claude/<model>`` — a Claude Pro/Max/Team plan (:mod:`strix.config.claude_code`)

Code that only needs to know "is this run metered?" or "what do we call the
plan?" asks here rather than each provider module.
"""

from __future__ import annotations

from strix.config import claude_code, codex


def subscription_provider(model_name: str | None) -> str | None:
    """``codex.PROVIDER`` / ``claude_code.PROVIDER`` for a subscription model, else None."""
    if codex.subscription_model(model_name):
        return codex.PROVIDER
    if claude_code.subscription_model(model_name):
        return claude_code.PROVIDER
    return None


def is_subscription_model(model_name: str | None) -> bool:
    return subscription_provider(model_name) is not None


def auth_mode(model_name: str | None) -> str:
    return "subscription" if is_subscription_model(model_name) else "api_key"


def subscription_label(model_name: str | None) -> str:
    """Human-readable plan name for the interface, e.g. ``ChatGPT subscription``."""
    provider = subscription_provider(model_name)
    if provider == claude_code.PROVIDER:
        return "Claude subscription"
    return "ChatGPT subscription"
