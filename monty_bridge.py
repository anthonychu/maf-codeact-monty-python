from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_monty import FunctionSnapshot, FutureSnapshot, Monty, MontyComplete, NameLookupSnapshot


MAX_PRINT_OUTPUT_CHARS = 8192


@dataclass(frozen=True)
class _ToolCallWork:
    name: str
    kwargs: dict[str, Any]
    task: Any


def _ensure_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Non-finite floating point values are not JSON-safe.")
        return value
    if isinstance(value, (list, tuple)):
        return [_ensure_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _ensure_json_value(v) for k, v in value.items()}
    raise ValueError(f"Value of type {type(value).__name__} is not JSON-safe.")


def _external_error(exc: Exception) -> dict[str, str]:
    return {"exc_type": type(exc).__name__, "message": str(exc)}


def _parse_call_tool(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Parse call_tool(name, **kwargs) arguments."""
    if not args:
        raise ValueError("call_tool requires a tool name as the first argument.")
    name = args[0]
    if not isinstance(name, str) or not name:
        raise ValueError("Tool name must be a non-empty string.")
    # Any remaining positional args beyond the name are not allowed
    if len(args) > 1:
        raise ValueError("call_tool accepts only the tool name as a positional argument. Use keyword arguments for parameters.")
    return name, dict(kwargs)


class _PrintCollector:
    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.truncated = False

    def __call__(self, stream: str, text: str) -> None:
        if self.truncated:
            return
        current_size = sum(len(c) for c in self.chunks)
        remaining = MAX_PRINT_OUTPUT_CHARS - current_size
        if remaining <= 0:
            self.truncated = True
            return
        text_value = str(text)
        if len(text_value) > remaining:
            self.chunks.append(text_value[:remaining])
            self.truncated = True
        else:
            self.chunks.append(text_value)

    @property
    def output(self) -> str:
        return "".join(self.chunks)


class InlineCodeBridge:
    """Execute Monty code inline (non-durable). Tool calls invoke functions directly.

    call_tool returns a future so the LLM code uses `await call_tool(...)`.
    When Monty yields a FutureSnapshot, the bridge invokes the tool and resumes.
    """

    def __init__(self, tool_map: dict[str, Callable[..., Any]]) -> None:
        self.tool_map = tool_map
        self.pending_calls: dict[int, tuple[str, dict[str, Any]]] = {}

    async def run(self, code: str) -> dict[str, Any]:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Code must be a non-empty string.")

        printer = _PrintCollector()
        monty = Monty(code, script_name="codeact.py")
        progress = monty.start(print_callback=printer)

        while True:
            if isinstance(progress, MontyComplete):
                return {
                    "output": _ensure_json_value(progress.output),
                    "stdout": printer.output,
                }
            if isinstance(progress, FunctionSnapshot):
                progress = self._handle_function(progress)
                continue
            if isinstance(progress, FutureSnapshot):
                progress = await self._handle_future(progress)
                continue
            if isinstance(progress, NameLookupSnapshot):
                raise RuntimeError(f"Name lookup not supported: {progress.variable_name!r}")
            raise RuntimeError(f"Unsupported Monty progress type: {type(progress).__name__}")

    def _handle_function(self, snapshot: FunctionSnapshot) -> Any:
        if snapshot.is_os_function:
            return snapshot.resume({
                "exc_type": "PermissionError",
                "message": "OS and filesystem calls are not available.",
            })

        function_name = str(snapshot.function_name)
        if function_name != "call_tool":
            return snapshot.resume({
                "exc_type": "NameError",
                "message": f"Function {function_name!r} is not available. Use call_tool(name, **kwargs) to call tools.",
            })

        try:
            name, kwargs = _parse_call_tool(snapshot.args, snapshot.kwargs)
            if name not in self.tool_map:
                allowed = ", ".join(sorted(self.tool_map.keys()))
                raise ValueError(f"Tool {name!r} is not registered. Available tools: {allowed}")

            self.pending_calls[int(snapshot.call_id)] = (name, kwargs)
        except Exception as exc:
            return snapshot.resume(_external_error(exc))

        return snapshot.resume({"future": ...})

    async def _handle_future(self, snapshot: FutureSnapshot) -> Any:
        pending_call_ids = [int(cid) for cid in snapshot.pending_call_ids]
        if not pending_call_ids:
            return snapshot.resume({})

        resume_results: dict[int, Any] = {}
        for cid in pending_call_ids:
            if cid not in self.pending_calls:
                raise RuntimeError(f"Unknown future call ID: {cid}")
            name, kwargs = self.pending_calls.pop(cid)
            try:
                tool_func = self.tool_map[name]
                if inspect.iscoroutinefunction(tool_func):
                    result = await tool_func(**kwargs)
                else:
                    result = tool_func(**kwargs)
                resume_results[cid] = {"return_value": _ensure_json_value(result)}
            except Exception as exc:
                resume_results[cid] = _external_error(exc)

        return snapshot.resume(resume_results)


class DurableCodeBridge:
    """Execute Monty code inside a Durable Functions orchestrator.

    Tool calls become activity invocations via context.call_activity().
    This is a generator-based bridge (yields Durable tasks).
    """

    ACTIVITY_NAME = "codeact_call_tool"

    def __init__(self, context: Any, tool_names: set[str]) -> None:
        self.context = context
        self.tool_names = tool_names
        self.pending_work: dict[int, _ToolCallWork] = {}
        self.printer = _PrintCollector()

    def run(self, code: str):
        if not isinstance(code, str) or not code.strip():
            raise ValueError("The orchestrator input must be a non-empty Python code string.")

        monty = Monty(code, script_name="codeact.py")
        progress = monty.start(print_callback=self.printer)

        while True:
            if isinstance(progress, MontyComplete):
                return {
                    "output": _ensure_json_value(progress.output),
                    "stdout": self.printer.output,
                }
            if isinstance(progress, FunctionSnapshot):
                progress = self._handle_function(progress)
                continue
            if isinstance(progress, FutureSnapshot):
                progress = yield from self._handle_future(progress)
                continue
            if isinstance(progress, NameLookupSnapshot):
                raise RuntimeError(f"Name lookup not supported: {progress.variable_name!r}")
            raise RuntimeError(f"Unsupported Monty progress type: {type(progress).__name__}")

    def _handle_function(self, snapshot: FunctionSnapshot) -> Any:
        if snapshot.is_os_function:
            return snapshot.resume({
                "exc_type": "PermissionError",
                "message": "OS and filesystem calls are not available.",
            })

        function_name = str(snapshot.function_name)
        if function_name != "call_tool":
            return snapshot.resume({
                "exc_type": "NameError",
                "message": f"Function {function_name!r} is not available. Use call_tool(name, **kwargs) to call tools.",
            })

        try:
            name, kwargs = _parse_call_tool(snapshot.args, snapshot.kwargs)
            if name not in self.tool_names:
                allowed = ", ".join(sorted(self.tool_names))
                raise ValueError(f"Tool {name!r} is not registered. Available tools: {allowed}")

            payload = {"name": name, "kwargs": _ensure_json_value(kwargs)}
            task = self.context.call_activity(self.ACTIVITY_NAME, payload)
            self.pending_work[int(snapshot.call_id)] = _ToolCallWork(name=name, kwargs=kwargs, task=task)
        except Exception as exc:
            return snapshot.resume(_external_error(exc))

        return snapshot.resume({"future": ...})

    def _handle_future(self, snapshot: FutureSnapshot):
        pending_call_ids = [int(cid) for cid in snapshot.pending_call_ids]
        if not pending_call_ids:
            return snapshot.resume({})

        missing = [cid for cid in pending_call_ids if cid not in self.pending_work]
        if missing:
            raise RuntimeError(f"Unknown future call IDs: {missing}")

        work_items = [self.pending_work[cid] for cid in pending_call_ids]
        tasks = [w.task for w in work_items]

        try:
            if len(tasks) == 1:
                results = [( yield tasks[0] )]
            else:
                results = yield self.context.task_all(tasks)
        except Exception as exc:
            resume_results = {cid: _external_error(exc) for cid in pending_call_ids}
        else:
            resume_results = {
                cid: {"return_value": _ensure_json_value(result)}
                for cid, result in zip(pending_call_ids, results, strict=True)
            }

        for cid in pending_call_ids:
            self.pending_work.pop(cid, None)
        return snapshot.resume(resume_results)
