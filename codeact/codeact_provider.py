from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

from agent_framework import AgentSession, ContextProvider, FunctionTool, SessionContext

from codeact.monty_bridge import DurableCodeBridge, InlineCodeBridge, generate_type_stubs


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

Use `start_execute_code` to run code durably. It returns an instance_id but NOT the result.
After calling `start_execute_code`, you MUST call `get_execution_result(instance_id=..., wait_seconds=3)` to get the result.
- If status is `Running`, call `get_execution_result` again with `wait_seconds=3`.
- If status is `Completed`, the `result` field contains the output — use it as your answer.
- If status is `Failed`, the `error` field explains what went wrong.
- Keep calling `get_execution_result` until you get `Completed` or `Failed`.

IMPORTANT: Code must be deterministic. Do not use `random`, `time.time()`, `uuid4()`, or any other
non-deterministic operations. The code may be replayed multiple times.

Additional durable primitives available inside `start_execute_code`:
- `await wait_for_external_event('EventName')` — pauses execution until an external event is raised. Returns the event payload.
- `await when_any([{"tool": "name", "kwargs": {...}}, ...])` — races multiple tool calls, returns `{"index": <winner_index>, "result": <winner_result>}`.
- `asyncio.gather(...)` — fan-out: `results = await asyncio.gather(tool_a(...), tool_b(...))` runs tool calls in parallel.
"""

    primary_tool = "start_execute_code" if durable else "execute_code"

    return f"""You have a primary tool: `{primary_tool}`.

Inside `{primary_tool}`, call registered tools directly as async functions:
`result = await tool_name(param=value)`. Always use `await` and keyword arguments.
Your code is type-checked — argument types must match the tool signatures below.
For fan-out, use `asyncio.gather`: `results = await asyncio.gather(tool_a(...), tool_b(...))`.

To surface results, end the code with `print(...)`. The sandbox does not return the value of the last expression.

Registered tools:
{tool_summaries}

Prefer a single `{primary_tool}` call when possible, combining multiple tool calls with Python control flow.{durable_note}
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
        type_stubs = generate_type_stubs(raw_tool_map, durable=self._durable) if raw_tool_map else None

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
            check_exec = _make_get_execution_result_tool(self._durable_client)
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
        "\n\nThis tool starts a durable code execution. It returns an instance_id but NOT the result. "
        "After calling this, you MUST call `get_execution_result(instance_id=..., wait_seconds=3)` "
        "to retrieve the actual output."
    )

    async def start_execute_code(*, code: str) -> str:
        """Start a durable Python code execution. Returns an instance_id — call get_execution_result next."""
        instance_id = await durable_client.start_new("codeact_orchestrator", None, code)
        return (
            f"Execution started (instance_id: {instance_id}). "
            f"Call get_execution_result(instance_id=\"{instance_id}\", wait_seconds=3) to get the output."
        )

    return FunctionTool(
        name="start_execute_code",
        description=description,
        func=start_execute_code,
    )


def _extract_durable_result(instance_id: str, raw_output: Any) -> str:
    """Extract a clean result string from the orchestration output.

    Combines stdout (from print) and output (last expression value) like Jupyter.
    """
    stdout = ""
    output_val = None

    if isinstance(raw_output, dict):
        stdout = raw_output.get("stdout", "")
        output_val = raw_output.get("output")
    elif isinstance(raw_output, str):
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, dict):
                stdout = parsed.get("stdout", "")
                output_val = parsed.get("output")
            else:
                stdout = raw_output
        except json.JSONDecodeError:
            stdout = raw_output

    # Combine: stdout first, then output value (if not None)
    parts = []
    if stdout and stdout.strip():
        parts.append(stdout.strip())
    if output_val is not None:
        parts.append(json.dumps(output_val) if not isinstance(output_val, str) else output_val)

    result = "\n".join(parts) if parts else "(no output)"

    return json.dumps({
        "instance_id": instance_id,
        "status": "Completed",
        "result": result,
    })


def _make_get_execution_result_tool(durable_client: Any) -> FunctionTool:
    async def get_execution_result(*, instance_id: str, wait_seconds: int = 0) -> str:
        """Get the result of a durable code execution. You MUST call this after start_execute_code."""
        print(f"[get_execution_result] instance_id={instance_id}, wait_seconds={wait_seconds}")
        if wait_seconds > 0:
            await asyncio.sleep(min(wait_seconds, 60))

        status = await durable_client.get_status(instance_id)
        if status is None:
            print(f"[get_execution_result] status is None")
            return json.dumps({"status": "NotFound"})

        runtime_status = status.runtime_status.name
        print(f"[get_execution_result] runtime_status={runtime_status}, output={status.output}")

        if runtime_status == "Completed":
            result = _extract_durable_result(instance_id, status.output)
            print(f"[get_execution_result] returning: {result}")
            return result
        elif runtime_status == "Failed":
            error_msg = str(status.output) if status.output else "Unknown error"
            return json.dumps({"status": "Failed", "error": error_msg})
        else:
            return json.dumps({"status": runtime_status})

    return FunctionTool(
        name="get_execution_result",
        description=(
            "Get the result of a durable code execution. You MUST call this after every start_execute_code call. "
            "Returns JSON with a `status` field: 'Running', 'Completed', or 'Failed'. "
            "When status is 'Completed', the `result` field contains the output. "
            "When status is 'Running', call again with `wait_seconds=3` to wait before re-checking. "
            "Keep calling until status is 'Completed' or 'Failed'."
        ),
        func=get_execution_result,
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
    _type_stubs = generate_type_stubs(tool_map, durable=True) if tool_map else None

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
