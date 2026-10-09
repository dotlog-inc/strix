"""Agent loop for the Claude subscription route (``STRIX_LLM=claude/<model>``).

The Claude Agent SDK is an agent loop, not a model: it runs the Claude Code
binary, which plans, calls tools, and keeps the conversation itself. So on this
route the ``agents`` SDK ``Runner`` is not used. Instead each Strix agent owns
one ``ClaudeSDKClient`` session, and every Strix tool (the host-side function
tools plus the sandbox-bound shell/filesystem tools) is handed to Claude Code as
an in-process MCP tool whose handler calls the very same ``on_invoke_tool`` the
``Runner`` would have called.

What stays the same: the system prompt, the tools and their behaviour, the
:class:`~strix.core.agents.AgentCoordinator` lifecycle (``finish_scan`` /
``agent_finish`` / ``wait_for_user`` / ``wait_for_agents`` set the agent's
status exactly as before), child agents, the report state, and the SQLite
session transcript the viewer reads.

What changes: Claude Code owns context management (compaction) and the model
call retry policy, cost is $0 (the plan is not metered), and a turn that ends
without a lifecycle tool is nudged back to work with a fresh user message
rather than a session replay.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem
from agents.stream_events import RunItemStreamEvent
from agents.tool import FunctionTool, ToolOutputImage
from agents.tool_context import ToolContext
from agents.usage import Usage
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    UserMessage,
    create_sdk_mcp_server,
)
from claude_agent_sdk import tool as sdk_tool
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from strix.config import claude_code, load_settings
from strix.core.execution import notify_parent_on_terminal
from strix.core.hooks import (
    LLM_TURN_KEY,
    BudgetExceededError,
    SubagentBudgetReservedError,
)
from strix.core.sessions import session_write_lock
from strix.report.state import get_global_report_state


if TYPE_CHECKING:
    from collections.abc import Sequence

    from agents.items import TResponseInputItem
    from agents.memory import Session
    from agents.sandbox.session.base_sandbox_session import BaseSandboxSession
    from claude_agent_sdk import SdkMcpTool

    from strix.core.agents import AgentCoordinator, Status


logger = logging.getLogger(__name__)

StreamEventSink = Callable[[str, Any], None]

MCP_SERVER_NAME = "strix"
_TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"

# Tools that settle the agent's status: once one of them succeeds the turn must
# end, so the handler asks the driver to interrupt Claude Code.
_LIFECYCLE_TOOLS = frozenset({"finish_scan", "agent_finish", "finish_pr_review"})
_PARKING_TOOLS = frozenset({"wait_for_user", "wait_for_agents"})

_INTERACTIVE_RECOVERY_LIMIT = 3
_MAX_TRANSIENT_RETRIES = 5
_TRANSIENT_RETRY_BASE_DELAY_S = 2.0
_TRANSIENT_RETRY_MAX_DELAY_S = 90.0
_TRANSIENT_API_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_WAITING_AUTO_RESUME_TIMEOUT_S = 300.0
_MAX_IDLE_AUTO_RESUMES = 3

_EFFORT_BY_REASONING = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

_ROUTE_INSTRUCTIONS = """
<claude_code_adapter>
You are running inside Claude Code's agent loop. Every Strix tool is exposed as an
MCP tool named `mcp__strix__<tool>`; call it by that name, with the arguments the
tool's own schema declares. No Claude Code built-in tools (Bash, Read, Edit, ...)
are available: the sandbox shell is `exec_command`, files are read and written
through it (and `apply_patch`), screenshots are viewed with `view_image`.

A turn that ends in plain text does not end the run and does not reach a user who
can act on it: finish with the lifecycle tool the instructions require
(`finish_scan` for the root agent, `agent_finish` for a sub-agent, `wait_for_user`
/ `wait_for_agents` to pause). After a successful lifecycle tool call the run stops
automatically; do not keep working after it.
</claude_code_adapter>
""".strip()


class ClaudeRouteError(RuntimeError):
    """A failure specific to driving Claude Code (not installed, signed out, ...)."""


@dataclass
class _TurnState:
    """What the tool handlers learn during one Claude Code response."""

    stop_after_call_ids: set[str] = field(default_factory=set)
    settled: bool = False
    tool_calls: int = 0


@dataclass
class _InterruptHandle:
    """What the coordinator cancels when a message lands mid-turn.

    Mirrors the ``cancel(mode=...)`` the ``Runner`` stream exposes, so
    ``AgentCoordinator.send`` can interrupt a Claude Code turn the same way.
    """

    client: ClaudeSDKClient
    loop: asyncio.AbstractEventLoop

    def cancel(self, mode: str = "immediate") -> None:
        _ = mode
        self.loop.call_soon_threadsafe(self._schedule)

    def _schedule(self) -> None:
        task = self.loop.create_task(self._interrupt())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    async def _interrupt(self) -> None:
        with contextlib.suppress(Exception):
            await self.client.interrupt()


_background_tasks: set[asyncio.Task[Any]] = set()


# --------------------------------------------------------------------------- tools


def _sandbox_tools(agent: Any, sandbox_session: BaseSandboxSession | None) -> list[Any]:
    """The sandbox capability tools (shell, filesystem) bound to the live session.

    ``Runner`` does this inside ``prepare_sandbox_agent``; here it is done by
    hand because nothing else prepares the agent.
    """
    if sandbox_session is None:
        return []
    tools: list[Any] = []
    for capability in getattr(agent, "capabilities", None) or ():
        bound = capability.clone()
        bound.bind(sandbox_session)
        tools.extend(bound.tools())
    return tools


async def _capability_instructions(agent: Any) -> str:
    fragments: list[str] = []
    manifest = getattr(agent, "manifest", None)
    for capability in getattr(agent, "capabilities", None) or ():
        try:
            fragment = await capability.instructions(manifest)
        except Exception:  # noqa: BLE001 - a missing fragment never blocks a run
            logger.debug("capability instructions failed", exc_info=True)
            continue
        if fragment:
            fragments.append(str(fragment))
    return "\n\n".join(fragments)


def _is_enabled(tool: Any) -> bool:
    enabled = getattr(tool, "is_enabled", True)
    return enabled if isinstance(enabled, bool) else True


def agent_function_tools(
    agent: Any, sandbox_session: BaseSandboxSession | None
) -> list[FunctionTool]:
    """Every tool the agent can call, as ``FunctionTool`` objects with a JSON schema.

    The agent is built with ``chat_completions_tools=True`` on this route, so
    the sandbox ``CustomTool`` (``apply_patch``) already arrives as a function
    tool with a one-field schema.
    """
    tools: list[FunctionTool] = []
    seen: set[str] = set()
    for tool in [*(getattr(agent, "tools", None) or []), *_sandbox_tools(agent, sandbox_session)]:
        if not isinstance(tool, FunctionTool) or not _is_enabled(tool):
            continue
        if tool.name in seen:
            continue
        seen.add(tool.name)
        tools.append(tool)
    return tools


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image_block(image: ToolOutputImage) -> dict[str, Any] | None:
    url = image.image_url or ""
    if not url.startswith("data:"):
        return None
    header, _, payload = url.partition(",")
    mime = header[len("data:") :].split(";", 1)[0] or "image/png"
    if not payload:
        return None
    try:
        base64.b64decode(payload, validate=True)
    except (ValueError, TypeError):
        return None
    return {"type": "image", "data": payload, "mimeType": mime}


def tool_result_to_mcp(result: Any) -> dict[str, Any]:
    """Shape a Strix tool's return value as an MCP ``CallToolResult`` payload."""
    outputs = result if isinstance(result, list) else [result]
    blocks: list[dict[str, Any]] = []
    for output in outputs:
        if isinstance(output, ToolOutputImage):
            block = _image_block(output)
            blocks.append(block or _text_block("[image could not be delivered]"))
        elif isinstance(output, dict) and output.get("type") == "image":
            block = _image_block(ToolOutputImage(image_url=output.get("image_url")))
            blocks.append(block or _text_block("[image could not be delivered]"))
        elif isinstance(output, str):
            blocks.append(_text_block(output))
        elif output is None:
            blocks.append(_text_block(""))
        else:
            blocks.append(_text_block(json.dumps(output, ensure_ascii=False, default=str)))
    return {"content": blocks}


def _tool_input_schema(tool: FunctionTool) -> dict[str, Any]:
    schema = dict(tool.params_json_schema or {})
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    # Strict mode forbids extra keys; Claude Code validates input against the
    # schema before the handler runs, so be permissive about what it sends.
    schema.pop("additionalProperties", None)
    return schema


def _tool_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, ToolOutputImage):
        return "[image]"
    if isinstance(result, list):
        return "\n".join(_tool_text(item) for item in result)
    return json.dumps(result, ensure_ascii=False, default=str)


def _lifecycle_settled(tool_name: str, output: Any, *, interactive: bool) -> bool:
    """Whether ``tool_name``'s result ended or parked the agent (same test as the factory's)."""
    if not isinstance(output, str):
        return False
    try:
        parsed = json.loads(output)
    except (TypeError, ValueError):
        return False
    if not isinstance(parsed, dict) or not parsed.get("success"):
        return False
    if tool_name in _LIFECYCLE_TOOLS:
        return bool(
            parsed.get("scan_completed")
            or parsed.get("agent_completed")
            or parsed.get("review_completed")
        )
    if tool_name in _PARKING_TOOLS:
        return bool(interactive and parsed.get("wait_outcome") == "waiting")
    return False


class _ToolBridge:
    """Builds the MCP tools and remembers what they did during a turn."""

    def __init__(
        self,
        *,
        agent: Any,
        tools: Sequence[FunctionTool],
        context: dict[str, Any],
        agent_id: str,
        interactive: bool,
        session: Session | None,
        event_sink: StreamEventSink | None,
    ) -> None:
        self._agent = agent
        self._tools = list(tools)
        self._context = context
        self._agent_id = agent_id
        self._interactive = interactive
        self._session = session
        self._event_sink = event_sink
        self.turn = _TurnState()
        self.interrupt: Callable[[], Any] | None = None

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self._tools]

    def sdk_tools(self) -> list[SdkMcpTool[Any]]:
        return [self._wrap(tool) for tool in self._tools]

    def _wrap(self, tool: FunctionTool) -> SdkMcpTool[Any]:
        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            return await self._invoke(tool, args)

        return sdk_tool(tool.name, tool.description or tool.name, _tool_input_schema(tool))(handler)

    async def _invoke(self, tool: FunctionTool, args: dict[str, Any]) -> dict[str, Any]:
        call_id = f"call_{uuid.uuid4().hex[:16]}"
        raw_input = json.dumps(args or {}, ensure_ascii=False)
        self.turn.tool_calls += 1
        self._emit_tool_call(tool.name, call_id, raw_input)
        ctx = ToolContext(
            context=self._context,
            tool_name=tool.name,
            tool_call_id=call_id,
            tool_arguments=raw_input,
            agent=self._agent,
        )
        try:
            result = await tool.on_invoke_tool(ctx, raw_input)
        except Exception as exc:  # noqa: BLE001 - tool errors are model-visible results
            logger.debug("tool %s failed on the Claude route", tool.name, exc_info=True)
            message = str(exc) or exc.__class__.__name__
            self._emit_tool_output(tool.name, call_id, message)
            await self._persist_tool_exchange(tool.name, call_id, raw_input, message)
            return {"content": [_text_block(message)], "is_error": True}
        text = _tool_text(result)
        self._emit_tool_output(tool.name, call_id, result)
        await self._persist_tool_exchange(tool.name, call_id, raw_input, text)
        if _lifecycle_settled(tool.name, result, interactive=self._interactive):
            self.turn.settled = True
            self.turn.stop_after_call_ids.add(call_id)
            if self.interrupt is not None:
                # Let the result reach Claude Code first; the driver interrupts
                # once it sees the tool result echoed back.
                self.interrupt()
        return tool_result_to_mcp(result)

    # -- projections: TUI events and the SQLite transcript ----------------------

    def _emit(self, name: str, build: Callable[[], Any]) -> None:
        """Hand a synthetic run item to the TUI sink; a projection failure never ends a turn."""
        if self._event_sink is None:
            return
        try:
            self._event_sink(self._agent_id, RunItemStreamEvent(name=name, item=build()))  # type: ignore[arg-type]
        except Exception:
            logger.exception("stream event sink failed for %s", self._agent_id)

    def _emit_tool_call(self, name: str, call_id: str, raw_input: str) -> None:
        self._emit(
            "tool_called",
            lambda: ToolCallItem(
                agent=self._agent,
                raw_item=ResponseFunctionToolCall(
                    arguments=raw_input, call_id=call_id, name=name, type="function_call"
                ),
            ),
        )

    def _emit_tool_output(self, name: str, call_id: str, output: Any) -> None:
        self._emit(
            "tool_output",
            lambda: ToolCallOutputItem(
                agent=self._agent,
                raw_item=cast(
                    "Any",
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "name": name,
                        "output": _tool_text(output),
                    },
                ),
                output=output,
            ),
        )

    def emit_assistant_text(self, text: str) -> None:
        if not text.strip():
            return
        self._emit(
            "message_output_created",
            lambda: MessageOutputItem(
                agent=self._agent,
                raw_item=ResponseOutputMessage(
                    id=f"msg_{uuid.uuid4().hex[:16]}",
                    content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
            ),
        )

    async def _persist_tool_exchange(
        self, name: str, call_id: str, raw_input: str, output: str
    ) -> None:
        await self.persist(
            [
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": raw_input,
                },
                {"type": "function_call_output", "call_id": call_id, "output": output},
            ]
        )

    async def persist(self, items: list[dict[str, Any]]) -> None:
        """Append transcript items to the agent's SQLite session (viewer/history only)."""
        if self._session is None or not items:
            return
        try:
            async with session_write_lock(self._session):
                await self._session.add_items(cast("list[TResponseInputItem]", items))
        except Exception:
            logger.exception("failed to persist Claude route transcript for %s", self._agent_id)


# --------------------------------------------------------------------------- usage


def _usage_from_assistant(usage: dict[str, Any] | None) -> Usage | None:
    if not isinstance(usage, dict):
        return None
    input_tokens = int(usage.get("input_tokens") or 0)
    cached = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_input = input_tokens + cached + cache_write
    if not (total_input or output_tokens):
        return None
    return Usage(
        requests=1,
        input_tokens=total_input,
        input_tokens_details=InputTokensDetails(
            cached_tokens=cached, cache_write_tokens=cache_write
        ),
        output_tokens=output_tokens,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=total_input + output_tokens,
    )


def _record_usage(agent: Any, agent_id: str, model: str, usage: Usage | None) -> None:
    report_state = get_global_report_state()
    if report_state is None or usage is None:
        return
    name = getattr(agent, "name", None)
    try:
        report_state.record_sdk_usage(
            agent_id=agent_id,
            agent_name=name if isinstance(name, str) else None,
            model=model,
            usage=usage,
        )
    except Exception:
        logger.exception("failed to record Claude route usage for %s", agent_id)


# --------------------------------------------------------------------------- prompts


def _input_to_text(initial_input: Any) -> str:
    """Flatten an ``agents`` input (string or item list) into one user message."""
    if isinstance(initial_input, str):
        return initial_input
    parts: list[str] = []
    for item in initial_input or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                block["text"]
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            )
    return "\n\n".join(part for part in parts if part)


def _tool_required_message(
    *, context: dict[str, Any], root_finish_tool: str, attempt: int, limit: int, interactive: bool
) -> str:
    finish_tool = root_finish_tool if context.get("parent_id") is None else "agent_finish"
    if interactive:
        return (
            "Your previous message ended a turn without a lifecycle tool call. Plain text "
            "never ends execution and never hands control to the user. Continue immediately "
            "and call a tool. If you have nothing to do until the user replies, call "
            "wait_for_user. If you are blocked waiting for another agent, call "
            f"wait_for_agents. If the whole engagement is complete, call {finish_tool}. "
            f"This is recovery attempt {attempt}/{limit}."
        )
    return (
        "Your previous response ended the autonomous run without a lifecycle tool call. "
        "That is invalid in non-interactive mode; plain text final answers are ignored. "
        f"Continue immediately and call a tool. If your work is complete, call {finish_tool}. "
        "If you are blocked waiting for another agent, call wait_for_agents. "
        f"This is recovery attempt {attempt}/{limit}."
    )


def _mailbox_to_text(items: list[Any]) -> str:
    return _input_to_text(items)


# --------------------------------------------------------------------------- driver


def build_options(
    *,
    model: str,
    system_prompt: str,
    bridge: _ToolBridge,
    max_turns: int,
    run_dir: Any,
    resume_session_id: str | None,
) -> Any:
    """The ``ClaudeAgentOptions`` for one agent."""
    settings = load_settings()
    server = create_sdk_mcp_server(name=MCP_SERVER_NAME, version="1.0.0", tools=bridge.sdk_tools())
    allowed = [f"{_TOOL_PREFIX}{name}" for name in bridge.tool_names]
    env: dict[str, str] = {
        # Nothing but the scan matters to Claude Code here.
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
        "DISABLE_TELEMETRY": "1",
    }
    return ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        tools=[],  # no built-ins: the sandbox is the only execution environment
        mcp_servers={MCP_SERVER_NAME: server},
        allowed_tools=allowed,
        strict_mcp_config=True,
        setting_sources=[],  # ignore the user's CLAUDE.md / skills / hooks
        max_turns=max(1, int(max_turns)),
        cwd=str(run_dir) if run_dir is not None else None,
        cli_path=claude_code.cli_path(),
        env=env,
        include_partial_messages=False,
        effort=cast("Any", _EFFORT_BY_REASONING.get(settings.llm.reasoning_effort, "high")),
        resume=resume_session_id,
        stderr=lambda line: logger.debug("claude-code: %s", line.rstrip()),
    )


class _ClaudeAgentDriver:
    """One Claude Code session for one Strix agent."""

    def __init__(
        self,
        *,
        agent: Any,
        context: dict[str, Any],
        coordinator: AgentCoordinator,
        agent_id: str,
        model: str,
        max_turns: int,
        interactive: bool,
        session: Session | None,
        event_sink: StreamEventSink | None,
        sandbox_session: BaseSandboxSession | None,
        run_dir: Any,
    ) -> None:
        self._agent = agent
        self._context = context
        self._coordinator = coordinator
        self._agent_id = agent_id
        self._model = model
        self._max_turns = max_turns
        self._interactive = interactive
        self._session = session
        self._event_sink = event_sink
        self._sandbox_session = sandbox_session
        self._run_dir = run_dir
        self._client: ClaudeSDKClient | None = None
        self._handle: _InterruptHandle | None = None
        self._bridge = _ToolBridge(
            agent=agent,
            tools=agent_function_tools(agent, sandbox_session),
            context=context,
            agent_id=agent_id,
            interactive=interactive,
            session=session,
            event_sink=event_sink,
        )
        self._interrupt_requested = False
        self._claude_session_id: str | None = None

    @property
    def max_turns(self) -> int:
        return self._max_turns

    # -- session ---------------------------------------------------------------

    async def _system_prompt(self) -> str:
        base = getattr(self._agent, "instructions", "")
        parts = [base if isinstance(base, str) else ""]
        capability_text = await _capability_instructions(self._agent)
        if capability_text:
            parts.append(
                f"<sandbox_capability_instructions>\n{capability_text}\n</sandbox_capability_instructions>"
            )
        parts.append(_ROUTE_INSTRUCTIONS)
        return "\n\n".join(part for part in parts if part)

    async def _stored_session_id(self) -> str | None:
        async with self._coordinator._lock:
            metadata = self._coordinator.metadata.get(self._agent_id) or {}
        value = metadata.get("claude_session_id")
        return value if isinstance(value, str) and value else None

    async def _store_session_id(self, session_id: str | None) -> None:
        if not session_id or session_id == self._claude_session_id:
            return
        self._claude_session_id = session_id
        async with self._coordinator._lock:
            self._coordinator.metadata.setdefault(self._agent_id, {})["claude_session_id"] = (
                session_id
            )
        await self._coordinator._maybe_snapshot()

    async def connect(self, *, resume: bool) -> None:
        resume_id = await self._stored_session_id() if resume else None
        options = build_options(
            model=self._model,
            system_prompt=await self._system_prompt(),
            bridge=self._bridge,
            max_turns=self._max_turns,
            run_dir=self._run_dir,
            resume_session_id=resume_id,
        )
        client = ClaudeSDKClient(options=options)
        try:
            await client.connect()
        except Exception as exc:
            raise ClaudeRouteError(_describe_sdk_error(exc)) from exc
        self._client = client
        self._handle = _InterruptHandle(client=client, loop=asyncio.get_running_loop())
        self._bridge.interrupt = self._request_interrupt
        await self._coordinator.attach_stream(self._agent_id, cast("Any", self._handle))

    def _request_interrupt(self) -> None:
        self._interrupt_requested = True

    async def close(self) -> None:
        client, self._client = self._client, None
        handle, self._handle = self._handle, None
        if handle is not None:
            with contextlib.suppress(Exception):
                await self._coordinator.detach_stream(self._agent_id, cast("Any", handle))
        if client is not None:
            with contextlib.suppress(Exception):
                await client.disconnect()

    # -- one Claude Code response ------------------------------------------------

    async def respond(self, prompt: str) -> Any:
        """Send ``prompt`` and consume the whole response. Returns the ``ResultMessage``."""
        client = self._client
        if client is None:
            raise ClaudeRouteError("Claude Code session is not connected")

        self._bridge.turn = _TurnState()
        self._interrupt_requested = False
        await self._bridge.persist([{"role": "user", "content": prompt}])
        await client.query(prompt)

        result: Any = None
        interrupted = False
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                self._context[LLM_TURN_KEY] = int(self._context.get(LLM_TURN_KEY, 0)) + 1
                _record_usage(
                    self._agent, self._agent_id, self._model, _usage_from_assistant(message.usage)
                )
                await self._store_session_id(message.session_id)
                text = "\n".join(
                    block.text for block in message.content if isinstance(block, TextBlock)
                )
                if text.strip():
                    self._bridge.emit_assistant_text(text)
                    await self._bridge.persist(
                        [{"role": "assistant", "content": [{"type": "output_text", "text": text}]}]
                    )
            elif isinstance(message, UserMessage):
                # Tool results come back as user messages; a settled lifecycle
                # tool's result has now reached Claude Code, so stop the turn.
                if self._interrupt_requested and not interrupted and _has_tool_result(message):
                    interrupted = True
                    with contextlib.suppress(Exception):
                        await client.interrupt()
            elif isinstance(message, ResultMessage):
                result = message
                await self._store_session_id(message.session_id)
                if message.total_cost_usd:
                    report_state = get_global_report_state()
                    if report_state is not None:
                        report_state.record_observed_llm_cost(float(message.total_cost_usd))
        return result


def _has_tool_result(message: Any) -> bool:
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return False
    return any(isinstance(block, ToolResultBlock) for block in content)


def _describe_sdk_error(exc: BaseException) -> str:
    name = exc.__class__.__name__
    text = str(exc)
    if name == "CLINotFoundError":
        return (
            "Claude Code is not installed or not on PATH. Install it "
            "(https://docs.claude.com/en/docs/claude-code) and sign in with `claude auth login`."
        )
    return f"{name}: {text}" if text else name


def _result_error(result: Any) -> tuple[bool, int | None, str]:
    """``(is_error, api_status, message)`` for a ``ResultMessage``."""
    if result is None:
        return True, None, "Claude Code ended the turn without a result"
    is_error = bool(getattr(result, "is_error", False))
    status = getattr(result, "api_error_status", None)
    errors = getattr(result, "errors", None) or []
    message = "; ".join(str(e) for e in errors) or str(getattr(result, "result", "") or "")
    subtype = str(getattr(result, "subtype", "") or "")
    if subtype.startswith("error") and not is_error:
        is_error = True
    return is_error, status if isinstance(status, int) else None, message or subtype


def _retry_delay(attempt: int) -> float:
    return min(
        _TRANSIENT_RETRY_BASE_DELAY_S * float(2 ** (attempt - 1)), _TRANSIENT_RETRY_MAX_DELAY_S
    )


async def _agent_status(coordinator: AgentCoordinator, agent_id: str) -> Status | None:
    async with coordinator._lock:
        return coordinator.statuses.get(agent_id)


async def _settle(
    coordinator: AgentCoordinator, agent_id: str, status: Status, error: str | None = None
) -> None:
    await coordinator.set_status(agent_id, status, error=error)
    await notify_parent_on_terminal(coordinator, agent_id, status)


async def run_claude_agent_loop(
    *,
    agent: Any,
    initial_input: Any,
    context: dict[str, Any],
    max_turns: int,
    coordinator: AgentCoordinator,
    agent_id: str,
    interactive: bool,
    model: str,
    session: Session | None = None,
    start_parked: bool = False,
    event_sink: StreamEventSink | None = None,
    sandbox_session: BaseSandboxSession | None = None,
    run_dir: Any = None,
) -> None:
    """Drive one Strix agent on Claude Code until a lifecycle tool settles it.

    The contract matches :func:`strix.core.execution.run_agent_loop`: the
    agent's status on the coordinator is the result, child agents are spawned
    through the ``create_agent`` tool exactly as on the other routes, and in
    interactive runs the loop parks on ``wait_for_user`` and resumes when the
    coordinator delivers a message.
    """
    await coordinator.attach_runtime(
        agent_id, session=session, interrupt_on_message=interactive, resumable=interactive
    )
    if coordinator.budget_stopped:
        await coordinator.set_status(agent_id, "stopped")
        raise BudgetExceededError("scan budget reached")
    if coordinator.reserve_stopped and context.get("parent_id") is not None:
        await coordinator.set_status(agent_id, "stopped")
        raise SubagentBudgetReservedError("scan reached the sub-agent budget reserve")

    driver = _ClaudeAgentDriver(
        agent=agent,
        context=context,
        coordinator=coordinator,
        agent_id=agent_id,
        model=model,
        max_turns=max_turns,
        interactive=interactive,
        session=session,
        event_sink=event_sink,
        sandbox_session=sandbox_session,
        run_dir=run_dir,
    )
    is_resume = not _input_to_text(initial_input).strip()
    try:
        await driver.connect(resume=is_resume)
    except ClaudeRouteError as exc:
        logger.exception("agent %s could not start Claude Code", agent_id)
        await _settle(coordinator, agent_id, "failed", error=str(exc))
        if not interactive:
            raise
        return

    try:
        prompt = _input_to_text(initial_input)
        if is_resume:
            prompt = "Resume where you left off and continue the engagement."
        if not (start_parked and interactive):
            await _run_until_lifecycle(
                driver, coordinator, agent_id, context, prompt, interactive=interactive
            )
        if not interactive:
            return
        await _interactive_park_loop(driver, coordinator, agent_id, context)
    finally:
        await driver.close()


async def _run_until_lifecycle(
    driver: _ClaudeAgentDriver,
    coordinator: AgentCoordinator,
    agent_id: str,
    context: dict[str, Any],
    prompt: str,
    *,
    interactive: bool,
) -> None:
    recovery_limit = _INTERACTIVE_RECOVERY_LIMIT if interactive else max(1, driver.max_turns)
    transient_retries = 0
    while True:
        if coordinator.budget_stopped:
            await coordinator.set_status(agent_id, "stopped")
            raise BudgetExceededError("scan budget reached")
        if coordinator.reserve_stopped and context.get("parent_id") is not None:
            await coordinator.set_status(agent_id, "stopped")
            raise SubagentBudgetReservedError("scan reached the sub-agent budget reserve")

        await coordinator.mark_running(agent_id)
        try:
            result = await driver.respond(prompt)
        except Exception as exc:
            message = _describe_sdk_error(exc)
            logger.exception("agent %s: Claude Code turn failed", agent_id)
            await _settle(coordinator, agent_id, "failed", error=message)
            if not interactive:
                raise ClaudeRouteError(message) from exc
            return

        status = await _agent_status(coordinator, agent_id)
        if status != "running":
            await coordinator.reset_recovery(agent_id)
            return

        is_error, api_status, error_text = _result_error(result)
        if is_error:
            transient = api_status in _TRANSIENT_API_STATUSES or api_status is None
            if transient and transient_retries < _MAX_TRANSIENT_RETRIES:
                transient_retries += 1
                delay = _retry_delay(transient_retries)
                logger.warning(
                    "agent %s: transient Claude Code error (%s); retrying in %.1fs (%d/%d): %s",
                    agent_id,
                    api_status,
                    delay,
                    transient_retries,
                    _MAX_TRANSIENT_RETRIES,
                    error_text,
                )
                await asyncio.sleep(delay)
                prompt = (
                    "The previous model call failed with a transient error. Continue exactly "
                    "where you left off."
                )
                continue
            hint = _auth_hint(api_status)
            text = f"{error_text or 'Claude Code reported an error'}{hint}"
            logger.error("agent %s: Claude Code error: %s", agent_id, text)
            await _settle(coordinator, agent_id, "failed", error=text)
            if not interactive:
                raise ClaudeRouteError(text)
            return

        terminal = str(getattr(result, "terminal_reason", "") or "")
        if terminal == "max_turns":
            logger.warning("agent %s reached max_turns on Claude Code", agent_id)
            await _settle(coordinator, agent_id, "stopped")
            return

        recoveries = await coordinator.record_recovery(agent_id)
        logger.warning(
            "agent %s ended a turn without a lifecycle tool call (interactive=%s); "
            "forcing tool continuation (%d/%d)",
            agent_id,
            interactive,
            recoveries,
            recovery_limit,
        )
        if recoveries >= recovery_limit:
            if not interactive:
                await _settle(coordinator, agent_id, "crashed")
                raise ClaudeRouteError(
                    "Agent exhausted recovery attempts without calling finish_scan or agent_finish."
                )
            await coordinator.park_waiting(agent_id, wait_kind="stalled")
            return
        prompt = _tool_required_message(
            context=context,
            root_finish_tool=coordinator.root_finish_tool,
            attempt=recoveries,
            limit=recovery_limit,
            interactive=interactive,
        )


def _auth_hint(api_status: int | None) -> str:
    if api_status in (401, 403):
        return f" — {claude_code.sign_in_hint()}"
    return ""


async def _interactive_park_loop(
    driver: _ClaudeAgentDriver,
    coordinator: AgentCoordinator,
    agent_id: str,
    context: dict[str, Any],
) -> None:
    """Mirror ``execution._run_agent_loop``'s interactive wait/resume cycle."""
    while True:
        timeout = await _waiting_timeout(coordinator, agent_id)
        try:
            woke = await coordinator.wait_for_message(agent_id, timeout=timeout)
        except asyncio.CancelledError:
            return
        if coordinator.budget_stopped:
            await coordinator.set_status(agent_id, "stopped")
            raise BudgetExceededError("scan budget reached")
        if coordinator.reserve_stopped and context.get("parent_id") is not None:
            await coordinator.set_status(agent_id, "stopped")
            raise SubagentBudgetReservedError("scan reached the sub-agent budget reserve")
        if woke:
            await coordinator.reset_recovery(agent_id)
            await coordinator.reset_idle_resumes(agent_id)
            _, items = await coordinator.consume_pending(agent_id, include_items=True)
            prompt = _mailbox_to_text(items) or "A message arrived; continue."
        else:
            idle_resumes = await coordinator.record_idle_resume(agent_id)
            if idle_resumes >= _MAX_IDLE_AUTO_RESUMES:
                await coordinator.park_waiting(agent_id, wait_kind="stalled")
                continue
            prompt = "Waiting timeout reached. Resuming execution."
        await _run_until_lifecycle(driver, coordinator, agent_id, context, prompt, interactive=True)


async def _waiting_timeout(coordinator: AgentCoordinator, agent_id: str) -> float | None:
    async with coordinator._lock:
        status = coordinator.statuses.get(agent_id)
        has_error = agent_id in coordinator.errors
        runtime = coordinator.runtimes.get(agent_id)
        gated = runtime.user_wake_required if runtime is not None else False
        wait_kind = coordinator.wait_kinds.get(agent_id)
        idle_resumes = coordinator.idle_resume_counts.get(agent_id, 0)
    if status != "waiting" or has_error or gated:
        return None
    if wait_kind != "agents" or idle_resumes >= _MAX_IDLE_AUTO_RESUMES:
        return None
    return _WAITING_AUTO_RESUME_TIMEOUT_S
