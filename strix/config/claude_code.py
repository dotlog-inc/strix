"""Claude subscription route: ``STRIX_LLM=claude/<model>``.

Runs the agents on the user's own Claude Code sign-in (a Claude Pro/Max/Team
plan) through the Claude Agent SDK, instead of a metered Anthropic API key.

The SDK launches the unmodified Claude Code binary, which authenticates with
the user's own ``claude auth login`` session. Strix never touches OAuth tokens
itself: that is the sanctioned path (Anthropic allows the Agent SDK and
``claude -p`` to draw from a subscription's limits), as opposed to lifting the
OAuth token out of Claude Code and calling the API directly, which Anthropic
blocks and forbids.

Only the model name and sign-in state live here; the agent loop that drives
the SDK is :mod:`strix.core.claude_execution`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess  # nosec B404 - runs the Claude Code CLI the user installed
from typing import Any


logger = logging.getLogger(__name__)


PROVIDER = "claude"
SUBSCRIPTION_PREFIX = "claude/"

# ``claude auth status`` takes a moment to load the CLI; a sign-in check must
# never hang a launch.
_AUTH_STATUS_TIMEOUT_S = 20

# The default model slug when ``STRIX_LLM=claude/`` names no model. Claude Code
# resolves its aliases (``sonnet``, ``opus``, ``haiku``) to the current release.
DEFAULT_MODEL = "sonnet"


def subscription_model(model_name: str | None) -> str | None:
    """The Claude Code model slug behind a ``claude/<model>`` STRIX_LLM, or None.

    ``claude/`` with nothing after it selects :data:`DEFAULT_MODEL`.
    """
    name = (model_name or "").strip()
    if not name.lower().startswith(SUBSCRIPTION_PREFIX):
        return None
    return name[len(SUBSCRIPTION_PREFIX) :].strip() or DEFAULT_MODEL


def auth_mode(model_name: str | None) -> str:
    return "subscription" if subscription_model(model_name) else "api_key"


def cli_path() -> str | None:
    """The Claude Code binary the run uses: ``CLAUDE_CODE_PATH``, then ``claude`` on PATH.

    Returns None to let the Agent SDK fall back to the CLI bundled with it. A
    user-installed binary is preferred because it is the one ``claude auth
    login`` signed in with.
    """
    configured = os.environ.get("CLAUDE_CODE_PATH")
    if configured:
        return configured
    return shutil.which("claude")


def auth_status() -> dict[str, Any] | None:
    """Parsed ``claude auth status --json``, or None when it cannot be read.

    None means "unknown", not "signed out": an old CLI without the subcommand,
    a missing binary, or a slow start all land here, and the run then finds out
    on its first model call.
    """
    binary = cli_path()
    if binary is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv, no shell
            [binary, "auth", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=_AUTH_STATUS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("claude auth status failed", exc_info=True)
        return None
    try:
        data = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def is_authenticated() -> bool | None:
    """Whether Claude Code is signed in: True/False, or None when it cannot tell."""
    status = auth_status()
    if status is None:
        return None
    logged_in = status.get("loggedIn")
    return bool(logged_in) if isinstance(logged_in, bool) else None


def uses_api_key_billing() -> bool:
    """Whether an ``ANTHROPIC_API_KEY`` in the environment would override the sign-in.

    Claude Code prefers an API key over the subscription login when both are
    present, so a run meant for the subscription would be silently metered.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def sign_in_hint() -> str:
    return "Run `claude auth login` (or just `claude`) and sign in with your Claude account."


def login(argv: list[str] | None = None) -> int:
    """Run ``claude auth login`` interactively; returns its exit code."""
    binary = cli_path()
    if binary is None:
        logger.error("Claude Code is not installed (no `claude` on PATH)")
        return 127
    try:
        return subprocess.call([binary, "auth", "login", *(argv or [])])  # noqa: S603  # nosec B603
    except OSError:
        logger.exception("could not start claude auth login")
        return 1


def logout() -> int:
    binary = cli_path()
    if binary is None:
        return 127
    try:
        return subprocess.call([binary, "auth", "logout"])  # noqa: S603  # nosec B603
    except OSError:
        logger.exception("could not start claude auth logout")
        return 1
