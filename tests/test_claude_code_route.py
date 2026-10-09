"""Tests for the Claude subscription route (``STRIX_LLM=claude/<model>``)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest
from agents import RunConfig
from agents.tool import FunctionTool, ToolOutputImage
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from strix.config import claude_code, codex, subscription
from strix.config.models import (
    configure_sdk_model_defaults,
    routes_through_litellm,
    uses_chat_completions_tool_schema,
)
from strix.core import claude_execution, execution
from strix.core.agents import AgentCoordinator
from strix.interface import auth_cli, environment
from strix.tools.finish.tool import finish_scan
from strix.tools.thinking.tool import think


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _Agent:
    """A stand-in for ``SandboxAgent``: run items hold a weak reference to it."""

    def __init__(self, name: str, instructions: str, tools: list[Any]) -> None:
        self.name = name
        self.instructions = instructions
        self.tools = tools


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude/sonnet", "sonnet"),
        ("CLAUDE/claude-opus-4-6", "claude-opus-4-6"),
        ("claude/", claude_code.DEFAULT_MODEL),
        ("  claude/haiku  ", "haiku"),
        ("anthropic/claude-sonnet-4-6", None),
        ("chatgpt/gpt-5.4", None),
        ("claude-sonnet-4-6", None),
        (None, None),
    ],
)
def test_subscription_model_parses_the_claude_prefix(
    model: str | None, expected: str | None
) -> None:
    assert claude_code.subscription_model(model) == expected


def test_auth_mode_is_subscription_for_the_claude_prefix_only() -> None:
    assert claude_code.auth_mode("claude/sonnet") == "subscription"
    assert claude_code.auth_mode("anthropic/claude-sonnet-4-6") == "api_key"


def test_subscription_helpers_cover_both_plans() -> None:
    assert subscription.subscription_provider("claude/sonnet") == claude_code.PROVIDER
    assert subscription.subscription_provider("chatgpt/gpt-5.4") == codex.PROVIDER
    assert subscription.subscription_provider("openai/gpt-5.4") is None
    assert subscription.auth_mode("claude/sonnet") == "subscription"
    assert subscription.auth_mode("openai/gpt-5.4") == "api_key"
    assert subscription.subscription_label("claude/sonnet") == "Claude subscription"
    assert subscription.subscription_label("chatgpt/gpt-5.4") == "ChatGPT subscription"


def test_is_authenticated_reads_claude_auth_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_code, "auth_status", lambda: {"loggedIn": True})
    assert claude_code.is_authenticated() is True
    monkeypatch.setattr(claude_code, "auth_status", lambda: {"loggedIn": False})
    assert claude_code.is_authenticated() is False
    monkeypatch.setattr(claude_code, "auth_status", lambda: None)
    assert claude_code.is_authenticated() is None


def test_claude_route_skips_the_openai_sdk_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(
        llm=SimpleNamespace(model="claude/sonnet", api_key="k", api_base=None)
    )
    touched: list[str] = []
    monkeypatch.setattr(
        "strix.config.models._configure_litellm_compatibility", lambda: touched.append("litellm")
    )
    monkeypatch.setattr("strix.config.models.request_log.install", lambda: None)
    configure_sdk_model_defaults(cast("Any", settings))
    assert touched == []


def test_claude_route_wants_json_function_tools_and_no_litellm() -> None:
    settings = SimpleNamespace(llm=SimpleNamespace(api_type=None))
    assert uses_chat_completions_tool_schema("claude/sonnet", cast("Any", settings)) is True
    assert routes_through_litellm("claude/sonnet") is False


# --------------------------------------------------------------------------- tool bridge


def test_tool_result_to_mcp_text_image_and_json() -> None:
    assert claude_execution.tool_result_to_mcp("hello") == {
        "content": [{"type": "text", "text": "hello"}]
    }
    image = ToolOutputImage(image_url="data:image/png;base64,aGk=")
    assert claude_execution.tool_result_to_mcp(image) == {
        "content": [{"type": "image", "data": "aGk=", "mimeType": "image/png"}]
    }
    mixed = claude_execution.tool_result_to_mcp(["a", {"k": 1}])
    assert mixed["content"][0] == {"type": "text", "text": "a"}
    assert json.loads(mixed["content"][1]["text"]) == {"k": 1}


def test_tool_input_schema_drops_additional_properties_guard() -> None:
    tool = FunctionTool(
        name="t",
        description="d",
        params_json_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        on_invoke_tool=cast("Any", None),
    )
    schema = claude_execution._tool_input_schema(tool)
    assert schema["properties"] == {"x": {"type": "string"}}
    assert "additionalProperties" not in schema


@pytest.mark.parametrize(
    ("tool_name", "output", "interactive", "expected"),
    [
        ("finish_scan", json.dumps({"success": True, "scan_completed": True}), False, True),
        ("finish_scan", json.dumps({"success": False, "scan_completed": False}), False, False),
        ("agent_finish", json.dumps({"success": True, "agent_completed": True}), False, True),
        ("wait_for_user", json.dumps({"success": True, "wait_outcome": "waiting"}), True, True),
        ("wait_for_user", json.dumps({"success": True, "wait_outcome": "waiting"}), False, False),
        ("think", json.dumps({"success": True}), True, False),
        ("finish_scan", "not json", False, False),
    ],
)
def test_lifecycle_settled(tool_name: str, output: str, interactive: bool, expected: bool) -> None:
    assert (
        claude_execution._lifecycle_settled(tool_name, output, interactive=interactive) is expected
    )


@pytest.mark.asyncio
async def test_bridge_invokes_strix_tool_and_flags_lifecycle() -> None:
    coordinator = AgentCoordinator()
    await coordinator.register("root", "Root Agent", parent_id=None)
    context: dict[str, Any] = {"coordinator": coordinator, "agent_id": "root", "parent_id": None}
    agent = _Agent("Root Agent", "sys", [think, finish_scan])
    bridge = claude_execution._ToolBridge(
        agent=agent,
        tools=claude_execution.agent_function_tools(agent, None),
        context=context,
        agent_id="root",
        interactive=False,
        session=None,
        event_sink=None,
    )
    interrupts: list[bool] = []
    bridge.interrupt = lambda: interrupts.append(True)
    tools = {tool.name: tool for tool in bridge.sdk_tools()}
    assert set(tools) == {"think", "finish_scan"}

    thought = await tools["think"].handler({"thought": "plan"})
    assert thought["content"][0]["type"] == "text"
    assert not bridge.turn.settled

    fields = ("executive_summary", "methodology", "technical_analysis", "recommendations")
    result = await tools["finish_scan"].handler(dict.fromkeys(fields, "x"))
    assert json.loads(result["content"][0]["text"])["scan_completed"] is True
    assert bridge.turn.settled
    assert interrupts == [True]
    async with coordinator._lock:
        assert coordinator.statuses["root"] == "completed"


@pytest.mark.asyncio
async def test_bridge_reports_tool_exceptions_as_errors() -> None:
    async def _raise(_ctx: Any, _raw: str) -> str:
        raise RuntimeError("kaboom")

    boom = FunctionTool(
        name="boom",
        description="Always fails.",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=_raise,
    )
    agent = _Agent("a", "", [boom])
    bridge = claude_execution._ToolBridge(
        agent=agent,
        tools=claude_execution.agent_function_tools(agent, None),
        context={},
        agent_id="a",
        interactive=False,
        session=None,
        event_sink=None,
    )
    result = await bridge.sdk_tools()[0].handler({})
    assert result["is_error"] is True
    assert "kaboom" in result["content"][0]["text"]


def test_usage_from_assistant_counts_cached_input() -> None:
    usage = claude_execution._usage_from_assistant(
        {
            "input_tokens": 10,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 5,
            "output_tokens": 7,
        }
    )
    assert usage is not None
    assert usage.requests == 1
    assert usage.input_tokens == 115
    assert usage.input_tokens_details.cached_tokens == 100
    assert usage.output_tokens == 7
    assert usage.total_tokens == 122
    assert claude_execution._usage_from_assistant(None) is None
    assert claude_execution._usage_from_assistant({"input_tokens": 0}) is None


# --------------------------------------------------------------------------- dispatch


@pytest.mark.asyncio
async def test_run_agent_loop_dispatches_claude_models(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _fake_loop(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "strix.core.claude_execution.run_claude_agent_loop", _fake_loop, raising=True
    )
    coordinator = AgentCoordinator()
    sandbox = SimpleNamespace(session="sandbox-session")
    run_config = RunConfig(model="claude/opus")
    run_config.sandbox = cast("Any", sandbox)
    result = await execution.run_agent_loop(
        agent=SimpleNamespace(name="Root Agent"),
        initial_input="go",
        run_config=run_config,
        context={"run_dir": "runs/scan-x"},
        max_turns=7,
        coordinator=coordinator,
        agent_id="root",
        interactive=False,
    )
    assert result is None
    assert captured["model"] == "opus"
    assert captured["sandbox_session"] == "sandbox-session"
    assert captured["run_dir"] == "runs/scan-x"
    assert captured["max_turns"] == 7


# --------------------------------------------------------------------------- driver


@dataclass
class _Script:
    """What the fake Claude Code does for each ``query``: call tools, then end."""

    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    text: str = "working"
    result_kwargs: dict[str, Any] = field(default_factory=dict)


class _FakeClient:
    scripts: ClassVar[list[_Script]] = []
    tools: ClassVar[dict[str, Any]] = {}
    prompts: ClassVar[list[str]] = []
    interrupts: ClassVar[int] = 0

    def __init__(self, options: Any) -> None:
        self.options = options

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def interrupt(self) -> None:
        type(self).interrupts += 1

    async def query(self, prompt: str, session_id: str = "default") -> None:  # noqa: ARG002
        type(self).prompts.append(prompt)

    async def receive_response(self) -> AsyncIterator[Any]:
        script = type(self).scripts.pop(0) if type(self).scripts else _Script()
        blocks: list[Any] = [TextBlock(text=script.text)]
        blocks.extend(
            ToolUseBlock(id=f"tu{i}", name=f"mcp__strix__{name}", input=args)
            for i, (name, args) in enumerate(script.tool_calls)
        )
        yield AssistantMessage(
            content=blocks,
            model="fake",
            usage={"input_tokens": 3, "output_tokens": 2},
            session_id="sess-1",
        )
        for i, (name, args) in enumerate(script.tool_calls):
            handler = type(self).tools[name].handler
            result = await handler(args)
            yield UserMessage(
                content=[ToolResultBlock(tool_use_id=f"tu{i}", content=result["content"])]
            )
        result_kwargs: dict[str, Any] = {
            "subtype": "success",
            "duration_ms": 1,
            "duration_api_ms": 1,
            "is_error": False,
            "num_turns": 1,
            "session_id": "sess-1",
            **script.result_kwargs,
        }
        yield ResultMessage(**result_kwargs)


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    _FakeClient.scripts = []
    _FakeClient.tools = {}
    _FakeClient.prompts = []
    _FakeClient.interrupts = 0

    def _fake_server(name: str, tools: Any = None, **_: Any) -> dict[str, Any]:
        _FakeClient.tools = {tool.name: tool for tool in tools or []}
        return {"type": "sdk", "name": name, "instance": cast("Any", None)}

    monkeypatch.setattr(claude_execution, "create_sdk_mcp_server", _fake_server)
    monkeypatch.setattr(claude_execution, "ClaudeSDKClient", _FakeClient)
    return _FakeClient


def _root_agent() -> Any:
    return _Agent("Root Agent", "You are the root agent.", [think, finish_scan])


@pytest.mark.asyncio
async def test_loop_finishes_when_finish_scan_succeeds(fake_client: type[_FakeClient]) -> None:
    fields = ("executive_summary", "methodology", "technical_analysis", "recommendations")
    fake_client.scripts = [
        _Script(tool_calls=[("think", {"thought": "plan"})]),
        _Script(tool_calls=[("finish_scan", dict.fromkeys(fields, "done"))]),
    ]
    coordinator = AgentCoordinator()
    await coordinator.register("root", "Root Agent", parent_id=None)
    context: dict[str, Any] = {"coordinator": coordinator, "agent_id": "root", "parent_id": None}

    await claude_execution.run_claude_agent_loop(
        agent=_root_agent(),
        initial_input="Scan the target.",
        context=context,
        max_turns=10,
        coordinator=coordinator,
        agent_id="root",
        interactive=False,
        model="sonnet",
    )

    async with coordinator._lock:
        assert coordinator.statuses["root"] == "completed"
        assert coordinator.metadata["root"]["claude_session_id"] == "sess-1"
    # First turn ended without a lifecycle tool, so it was nudged back once.
    assert len(fake_client.prompts) == 2
    assert fake_client.prompts[0] == "Scan the target."
    assert "recovery attempt 1/" in fake_client.prompts[1]
    # The settled finish_scan result reached Claude Code, then the turn was cut.
    assert fake_client.interrupts == 1
    assert context[claude_execution.LLM_TURN_KEY] == 2


@pytest.mark.asyncio
async def test_options_expose_only_strix_tools(fake_client: type[_FakeClient]) -> None:
    fields = ("executive_summary", "methodology", "technical_analysis", "recommendations")
    fake_client.scripts = [_Script(tool_calls=[("finish_scan", dict.fromkeys(fields, "done"))])]
    coordinator = AgentCoordinator()
    await coordinator.register("root", "Root Agent", parent_id=None)
    seen: list[Any] = []
    original_init = fake_client.__init__

    def _capture(self: Any, options: Any) -> None:
        seen.append(options)
        original_init(self, options)

    fake_client.__init__ = _capture  # type: ignore[method-assign]
    try:
        await claude_execution.run_claude_agent_loop(
            agent=_root_agent(),
            initial_input="go",
            context={"coordinator": coordinator, "agent_id": "root", "parent_id": None},
            max_turns=2,
            coordinator=coordinator,
            agent_id="root",
            interactive=False,
            model="sonnet",
        )
    finally:
        fake_client.__init__ = original_init  # type: ignore[method-assign]
    options = seen[0]
    assert options.tools == []
    assert options.model == "sonnet"
    assert set(options.allowed_tools) == {"mcp__strix__think", "mcp__strix__finish_scan"}
    assert options.setting_sources == []
    assert "You are the root agent." in options.system_prompt
    assert "mcp__strix__" in options.system_prompt


@pytest.mark.asyncio
async def test_loop_fails_on_non_transient_error(fake_client: type[_FakeClient]) -> None:
    fake_client.scripts = [
        _Script(result_kwargs={"is_error": True, "api_error_status": 401, "errors": ["nope"]})
    ]
    coordinator = AgentCoordinator()
    await coordinator.register("root", "Root Agent", parent_id=None)
    with pytest.raises(claude_execution.ClaudeRouteError, match="claude auth login"):
        await claude_execution.run_claude_agent_loop(
            agent=_root_agent(),
            initial_input="go",
            context={"coordinator": coordinator, "agent_id": "root", "parent_id": None},
            max_turns=2,
            coordinator=coordinator,
            agent_id="root",
            interactive=False,
            model="sonnet",
        )
    async with coordinator._lock:
        assert coordinator.statuses["root"] == "failed"
        assert "nope" in coordinator.errors["root"]


# --------------------------------------------------------------------------- interface


def test_validate_environment_requires_claude_sign_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRIX_LLM", "claude/sonnet")
    monkeypatch.setattr("strix.config.loader._cached", None)
    monkeypatch.setattr(claude_code, "cli_path", lambda: "/usr/bin/claude")
    monkeypatch.setattr(claude_code, "is_authenticated", lambda: False)
    with pytest.raises(SystemExit):
        environment.validate_environment()

    monkeypatch.setattr(claude_code, "is_authenticated", lambda: True)
    environment.validate_environment()  # no API key needed

    monkeypatch.setattr(claude_code, "cli_path", lambda: None)
    with pytest.raises(SystemExit):
        environment.validate_environment()


def test_auth_cli_accepts_claude_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "claude" in auth_cli._ACCEPTED_PROVIDERS
    calls: list[str] = []
    monkeypatch.setattr(claude_code, "cli_path", lambda: "/usr/bin/claude")
    monkeypatch.setattr(claude_code, "login", lambda *_args: calls.append("login") or 0)
    monkeypatch.setattr(claude_code, "logout", lambda: calls.append("logout") or 0)
    assert auth_cli.run_auth(["login", "claude"]) == 0
    assert auth_cli.run_auth(["logout", "claude"]) == 0
    assert calls == ["login", "logout"]


def test_auth_status_reports_claude_sign_in(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(claude_code, "is_authenticated", lambda: True)
    assert auth_cli.run_auth(["status"]) == 0
    assert "Claude subscription" in capsys.readouterr().out
