from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

from agent_framework import AgentSession, ContextProvider, FunctionTool, SessionContext

from monty_bridge import DurableCodeBridge, InlineCodeBridge, generate_type_stubs


def _build_tool_summaries(tools: Sequence[FunctionTool]) -> str:
    if not tools:
        return "- No tools are currently registered."
    lines: list[str] = []
    for tool_obj in tools:
        params = tool_obj.parameters().get("properties", {})
        param_names = [n for n in params if isinstance(n, str)]
        param_summary = ", ".join(param_names) if param_names else "none"
        desc = str(tool_obj.description or "").strip() or "No description."
        lines.append(f"- `{tool_obj.name}`: {desc} Parameters: {param_summary}.")
    return "\n".join(lines)


def _build_codeact_instructions(tools: Sequence[FunctionTool], *, durable: bool) -> str:
    tool_summaries = _build_tool_summaries(tools)

    durable_note = ""
    if durable:
        durable_note = """

`execute_code` runs durably. It returns a JSON object with an `instance_id`.
After calling `execute_code`, call `check_execution(instance_id=...)` to poll for the result.
If the status is `Running`, call `check_execution` again with `wait_seconds` to wait before checking.

IMPORTANT: Code must be deterministic. Do not use `random`, `time.time()`, `uuid4()`, or any other
non-deterministic operations. The code may be replayed multiple times, and each replay must produce
the same sequence of tool calls. All non-deterministic work should happen inside tool functions.

Additional durable primitives available inside `execute_code`:
- `await wait_for_external_event('EventName')` — pauses execution until an external event is raised. Returns the event payload.
- `await when_any([{"tool": "name", "kwargs": {...}}, ...])` — races multiple tool calls, returns `{"index": <winner_index>, "result": <winner_result>}`.
- `asyncio.gather(...)` — fan-out: `results = await asyncio.gather(tool_a(...), tool_b(...))` runs tool calls in parallel.
"""

    return f"""You have a primary tool: `execute_code`.

Inside `execute_code`, call registered tools directly as async functions:
`result = await tool_name(param=value)`. Always use `await` and keyword arguments.
Your code is type-checked — argument types must match the tool signatures below.
For fan-out, use `asyncio.gather`: `results = await asyncio.gather(tool_a(...), tool_b(...))`.

To surface results, end the code with `print(...)`. The sandbox does not return the value of the last expression.

Registered tools:
{tool_summaries}

Prefer a single `execute_code` call when possible, combining multiple tool calls with Python control flow.{durable_note}
"""


def _build_execute_code_description(tools: Sequence[FunctionTool]) -> str:
    tool_summaries = _build_tool_summaries(tools)
    return f"""Execute Python code in a sandboxed environment.

Inside the sandbox, call registered tools directly as async functions:
`result = await tool_name(param=value)`. Always use `await` and keyword arguments.
Code is type-checked against tool signatures before execution.
For fan-out, use `asyncio.gather`: `results = await asyncio.gather(tool_a(...), tool_b(...))`.

Registered tools:
{tool_summaries}

Use `print(...)` to surface results."""


class CodeActProvider(ContextProvider):
    """Inject a CodeAct surface using Monty for code execution."""

    DEFAULT_SOURCE_ID = "codeact"

    def __init__(
        self,
        source_id: str = DEFAULT_SOURCE_ID,
        *,
        tools: Sequence[FunctionTool | Callable[..., Any]] | None = None,
        durable: bool = False,
        durable_client: Any = None,
    ) -> None:
        super().__init__(source_id)
        self._raw_tools = list(tools or [])
        self._durable = durable
        self._durable_client = durable_client

    def _normalize_tools(self) -> list[FunctionTool]:
        from agent_framework._tools import normalize_tools
        return [t for t in normalize_tools(self._raw_tools) if isinstance(t, FunctionTool)]

    def _build_tool_map(self) -> dict[str, Callable[..., Any]]:
        """Build name -> callable map from raw tools (for inline bridge)."""
        tool_map: dict[str, Callable[..., Any]] = {}
        for raw_tool in self._raw_tools:
            if callable(raw_tool) and not isinstance(raw_tool, FunctionTool):
                tool_map[raw_tool.__name__] = raw_tool
            elif isinstance(raw_tool, FunctionTool):
                tool_map[raw_tool.name] = raw_tool.invoke
        return tool_map

    def _build_raw_tool_map(self) -> dict[str, Callable[..., Any]]:
        """Build name -> original callable map (for type stub generation)."""
        tool_map: dict[str, Callable[..., Any]] = {}
        for raw_tool in self._raw_tools:
            if callable(raw_tool) and not isinstance(raw_tool, FunctionTool):
                tool_map[raw_tool.__name__] = raw_tool
        return tool_map

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        normalized_tools = self._normalize_tools()
        approval_required = {t.name for t in normalized_tools if getattr(t, "approval_mode", None) == "always_require"}

        # Generate type stubs from tool signatures
        raw_tool_map = self._build_raw_tool_map()
        type_stubs = generate_type_stubs(raw_tool_map) if raw_tool_map else None

        # Build execute_code tool
        if self._durable:
            execute_code = _make_durable_execute_code_tool(
                tools=normalized_tools,
                durable_client=self._durable_client,
            )
        else:
            execute_code = _make_inline_execute_code_tool(
                tools=normalized_tools,
                tool_map=self._build_tool_map(),
                type_stubs=type_stubs,
            )

        # Pre-execution approval: if any tool requires approval, gate execute_code itself
        if approval_required:
            execute_code.approval_mode = "always_require"

        tools_to_inject: list[FunctionTool] = [execute_code]

        if self._durable:
            check_exec = _make_check_execution_tool(self._durable_client)
            tools_to_inject.append(check_exec)

        instructions = _build_codeact_instructions(normalized_tools, durable=self._durable)
        context.extend_instructions(self.source_id, instructions)
        context.extend_tools(self.source_id, tools_to_inject)


def _make_inline_execute_code_tool(
    *,
    tools: Sequence[FunctionTool],
    tool_map: dict[str, Callable[..., Any]],
    type_stubs: str | None = None,
) -> FunctionTool:
    description = _build_execute_code_description(tools)

    async def execute_code(*, code: str) -> str:
        """Execute Python code in a sandboxed environment."""
        bridge = InlineCodeBridge(tool_map, type_stubs=type_stubs)
        result = await bridge.run(code)
        return json.dumps(result)

    return FunctionTool(
        name="execute_code",
        description=description,
        func=execute_code,
    )


def _make_durable_execute_code_tool(
    *,
    tools: Sequence[FunctionTool],
    durable_client: Any,
) -> FunctionTool:
    description = _build_execute_code_description(tools) + (
        "\n\nThis tool runs code durably. It returns a JSON object with an `instance_id` "
        "that you can use with `check_execution` to poll for the result."
    )

    async def execute_code(*, code: str) -> str:
        """Execute Python code durably via an orchestration."""
        instance_id = await durable_client.start_new("codeact_orchestrator", None, code)
        return json.dumps({"instance_id": instance_id})

    return FunctionTool(
        name="execute_code",
        description=description,
        func=execute_code,
    )


def _make_check_execution_tool(durable_client: Any) -> FunctionTool:
    async def check_execution(*, instance_id: str, wait_seconds: int = 0) -> str:
        """Check the status of a code execution and retrieve its result."""
        if wait_seconds > 0:
            await asyncio.sleep(min(wait_seconds, 60))

        status = await durable_client.get_status(instance_id)
        if status is None:
            return json.dumps({"status": "NotFound"})

        runtime_status = str(status.runtime_status)
        result: dict[str, Any] = {"status": runtime_status}

        if runtime_status in ("Completed",):
            result["output"] = status.output
        elif runtime_status in ("Failed",):
            result["error"] = str(status.output) if status.output else "Unknown error"

        return json.dumps(result)

    return FunctionTool(
        name="check_execution",
        description="Check the status of a code execution and retrieve its result. Use wait_seconds to delay before checking.",
        func=check_execution,
    )


def register_durable_codeact(
    app: Any,
    *,
    tools: Sequence[Callable[..., Any]] | None = None,
    approval_required_tools: set[str] | None = None,
) -> None:
    """Register hidden durable infrastructure (orchestrator + activity) on the app.

    Call this once at module level. The orchestrator and activity are internal
    and not meant to be called directly by users.

    Args:
        app: The DFApp instance.
        tools: List of tool callables to make available as activities.
        approval_required_tools: Set of tool names that require per-call approval
            via external events before the activity is scheduled.
    """
    tool_list = list(tools or [])
    tool_map: dict[str, Callable[..., Any]] = {}
    tool_names: set[str] = set()
    _approval_required = approval_required_tools or set()
    for t in tool_list:
        name = t.__name__ if callable(t) and not isinstance(t, FunctionTool) else getattr(t, "name", str(t))
        tool_map[name] = t
        tool_names.add(name)

    # Generate type stubs from tool signatures
    _type_stubs = generate_type_stubs(tool_map) if tool_map else None

    # Register orchestrator
    @app.orchestration_trigger(context_name="context")
    def codeact_orchestrator(context):
        code = context.get_input()
        bridge = DurableCodeBridge(
            context, tool_names=tool_names,
            approval_required_tools=_approval_required,
            type_stubs=_type_stubs,
        )
        return (yield from bridge.run(code))

    # Register activity
    @app.activity_trigger(input_name="params")
    def codeact_call_tool(params):
        name = params["name"]
        kwargs = params.get("kwargs", {})
        if name not in tool_map:
            raise ValueError(f"Tool {name!r} is not registered.")
        return tool_map[name](**kwargs)
