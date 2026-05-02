from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_monty import FunctionSnapshot, FutureSnapshot, Monty, MontyComplete, NameLookupSnapshot


MAX_PRINT_OUTPUT_CHARS = 8192

# Prelude injected into all Monty code so asyncio.gather works for fan-out.
CODEACT_PRELUDE = """\
import asyncio
"""


@dataclass(frozen=True)
class _ToolCallWork:
    name: str
    kwargs: dict[str, Any]
    task: Any


@dataclass(frozen=True)
class _ExternalEventWork:
    event_name: str
    task: Any


@dataclass(frozen=True)
class _WhenAnyWork:
    specs: list[tuple[str, dict[str, Any]]]
    tasks: list[Any]


@dataclass(frozen=True)
class _ApprovalThenToolWork:
    name: str
    kwargs: dict[str, Any]
    approval_task: Any


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
    if len(args) > 1:
        raise ValueError("call_tool accepts only the tool name as a positional argument. Use keyword arguments for parameters.")
    return name, dict(kwargs)


def _parse_event_name(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Parse wait_for_external_event(name) arguments."""
    if len(args) > 1:
        raise ValueError("wait_for_external_event accepts one positional argument: name.")
    unexpected = set(kwargs) - {"name"}
    if unexpected:
        raise ValueError(f"Unsupported wait_for_external_event keyword arguments: {sorted(unexpected)}")
    if args:
        if "name" in kwargs:
            raise ValueError("wait_for_external_event name was provided twice.")
        event_name = args[0]
    elif "name" in kwargs:
        event_name = kwargs["name"]
    else:
        raise ValueError("wait_for_external_event requires an event name.")
    if not isinstance(event_name, str) or not event_name:
        raise ValueError("Event name must be a non-empty string.")
    return event_name


def _parse_when_any_specs(
    args: tuple[Any, ...], kwargs: dict[str, Any], allowed_names: set[str],
) -> list[tuple[str, dict[str, Any]]]:
    """Parse when_any([{tool, kwargs}, ...]) arguments."""
    if kwargs:
        raise ValueError("when_any does not accept keyword arguments.")
    if len(args) != 1 or not isinstance(args[0], list):
        raise ValueError("when_any requires one list of tool specs.")
    raw_specs = args[0]
    if not raw_specs:
        raise ValueError("when_any requires a non-empty list of tool specs.")

    specs: list[tuple[str, dict[str, Any]]] = []
    for spec in raw_specs:
        if not isinstance(spec, dict):
            raise ValueError("Each when_any spec must be a dict with 'tool' and optional 'kwargs'.")
        name = spec.get("tool")
        if not isinstance(name, str) or not name:
            raise ValueError("Each when_any spec must have a 'tool' string.")
        if name not in allowed_names:
            raise ValueError(f"Tool {name!r} is not registered.")
        tool_kwargs = spec.get("kwargs", {})
        if not isinstance(tool_kwargs, dict):
            raise ValueError("'kwargs' in when_any spec must be a dict.")
        specs.append((name, tool_kwargs))
    return specs


def _build_code(code: str) -> str:
    return f"{CODEACT_PRELUDE}\n{code}"


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
        monty = Monty(_build_code(code), script_name="codeact.py")
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
        if function_name == "call_tool":
            return self._schedule_call_tool(snapshot)
        if function_name == "wait_for_external_event":
            return snapshot.resume({
                "exc_type": "RuntimeError",
                "message": "wait_for_external_event is only available in durable mode.",
            })
        if function_name == "when_any":
            return self._schedule_when_any(snapshot)

        return snapshot.resume({
            "exc_type": "NameError",
            "message": f"Function {function_name!r} is not available. Use call_tool(name, **kwargs) to call tools.",
        })

    def _schedule_call_tool(self, snapshot: FunctionSnapshot) -> Any:
        try:
            name, kwargs = _parse_call_tool(snapshot.args, snapshot.kwargs)
            if name not in self.tool_map:
                allowed = ", ".join(sorted(self.tool_map.keys()))
                raise ValueError(f"Tool {name!r} is not registered. Available tools: {allowed}")
            self.pending_calls[int(snapshot.call_id)] = ("call_tool", name, kwargs)
        except Exception as exc:
            return snapshot.resume(_external_error(exc))
        return snapshot.resume({"future": ...})

    def _schedule_when_any(self, snapshot: FunctionSnapshot) -> Any:
        try:
            specs = _parse_when_any_specs(snapshot.args, snapshot.kwargs, set(self.tool_map.keys()))
            self.pending_calls[int(snapshot.call_id)] = ("when_any", specs, None)
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
            entry = self.pending_calls.pop(cid)
            kind = entry[0]

            if kind == "call_tool":
                _, name, kwargs = entry
                try:
                    tool_func = self.tool_map[name]
                    if inspect.iscoroutinefunction(tool_func):
                        result = await tool_func(**kwargs)
                    else:
                        result = tool_func(**kwargs)
                    resume_results[cid] = {"return_value": _ensure_json_value(result)}
                except Exception as exc:
                    resume_results[cid] = _external_error(exc)

            elif kind == "when_any":
                _, specs, _ = entry
                try:
                    # In inline mode, run all tools and return the first result
                    first_result = None
                    first_index = 0
                    for i, (name, kwargs) in enumerate(specs):
                        tool_func = self.tool_map[name]
                        if inspect.iscoroutinefunction(tool_func):
                            r = await tool_func(**kwargs)
                        else:
                            r = tool_func(**kwargs)
                        if first_result is None:
                            first_result = r
                            first_index = i
                    resume_results[cid] = {"return_value": _ensure_json_value({
                        "index": first_index,
                        "result": _ensure_json_value(first_result),
                    })}
                except Exception as exc:
                    resume_results[cid] = _external_error(exc)

        return snapshot.resume(resume_results)


class DurableCodeBridge:
    """Execute Monty code inside a Durable Functions orchestrator.

    Tool calls become activity invocations via context.call_activity().
    This is a generator-based bridge (yields Durable tasks).
    """

    ACTIVITY_NAME = "codeact_call_tool"

    def __init__(self, context: Any, tool_names: set[str], approval_required_tools: set[str] | None = None) -> None:
        self.context = context
        self.tool_names = tool_names
        self.approval_required_tools = approval_required_tools or set()
        self.pending_work: dict[int, _ToolCallWork | _ExternalEventWork | _WhenAnyWork | _ApprovalThenToolWork] = {}
        self.printer = _PrintCollector()

    def run(self, code: str):
        if not isinstance(code, str) or not code.strip():
            raise ValueError("The orchestrator input must be a non-empty Python code string.")

        monty = Monty(_build_code(code), script_name="codeact.py")
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
        if function_name == "call_tool":
            return self._schedule_call_tool(snapshot)
        if function_name == "wait_for_external_event":
            return self._schedule_wait_for_external_event(snapshot)
        if function_name == "when_any":
            return self._schedule_when_any(snapshot)

        return snapshot.resume({
            "exc_type": "NameError",
            "message": f"Function {function_name!r} is not available. Use call_tool(name, **kwargs) to call tools.",
        })

    def _schedule_call_tool(self, snapshot: FunctionSnapshot) -> Any:
        try:
            name, kwargs = _parse_call_tool(snapshot.args, snapshot.kwargs)
            if name not in self.tool_names:
                allowed = ", ".join(sorted(self.tool_names))
                raise ValueError(f"Tool {name!r} is not registered. Available tools: {allowed}")

            if name in self.approval_required_tools:
                event_name = f"approve:{name}:{snapshot.call_id}"
                approval_task = self.context.wait_for_external_event(event_name)
                self.pending_work[int(snapshot.call_id)] = _ApprovalThenToolWork(
                    name=name, kwargs=kwargs, approval_task=approval_task,
                )
            else:
                payload = {"name": name, "kwargs": _ensure_json_value(kwargs)}
                task = self.context.call_activity(self.ACTIVITY_NAME, payload)
                self.pending_work[int(snapshot.call_id)] = _ToolCallWork(name=name, kwargs=kwargs, task=task)
        except Exception as exc:
            return snapshot.resume(_external_error(exc))
        return snapshot.resume({"future": ...})

    def _schedule_wait_for_external_event(self, snapshot: FunctionSnapshot) -> Any:
        try:
            event_name = _parse_event_name(snapshot.args, snapshot.kwargs)
            task = self.context.wait_for_external_event(event_name)
            self.pending_work[int(snapshot.call_id)] = _ExternalEventWork(event_name=event_name, task=task)
        except Exception as exc:
            return snapshot.resume(_external_error(exc))
        return snapshot.resume({"future": ...})

    def _schedule_when_any(self, snapshot: FunctionSnapshot) -> Any:
        try:
            specs = _parse_when_any_specs(snapshot.args, snapshot.kwargs, self.tool_names)
            tasks = [
                self.context.call_activity(self.ACTIVITY_NAME, {"name": name, "kwargs": _ensure_json_value(kw)})
                for name, kw in specs
            ]
            self.pending_work[int(snapshot.call_id)] = _WhenAnyWork(specs=specs, tasks=tasks)
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

        # If all pending items are single-task work (call_tool or external event),
        # batch them with task_all for efficiency.
        if all(isinstance(w, (_ToolCallWork, _ExternalEventWork)) for w in work_items):
            return (yield from self._resume_single_task_batch(snapshot, pending_call_ids, work_items))

        # Approval or when_any items must be resolved one at a time.
        return (yield from self._resume_single(snapshot, pending_call_ids[0]))

    def _resume_single_task_batch(
        self,
        snapshot: FutureSnapshot,
        call_ids: list[int],
        work_items: list[_ToolCallWork | _ExternalEventWork | _WhenAnyWork],
    ):
        tasks = [w.task for w in work_items if isinstance(w, (_ToolCallWork, _ExternalEventWork))]
        try:
            if len(tasks) == 1:
                results = [(yield tasks[0])]
            else:
                results = yield self.context.task_all(tasks)
        except Exception as exc:
            resume_results = {cid: _external_error(exc) for cid in call_ids}
        else:
            resume_results = {}
            for cid, work_item, result in zip(call_ids, work_items, results, strict=True):
                if isinstance(work_item, _ExternalEventWork):
                    resume_results[cid] = {"return_value": _ensure_json_value(_normalize_event_payload(result))}
                else:
                    resume_results[cid] = {"return_value": _ensure_json_value(result)}

        for cid in call_ids:
            self.pending_work.pop(cid, None)
        return snapshot.resume(resume_results)

    def _resume_single(self, snapshot: FutureSnapshot, call_id: int):
        work_item = self.pending_work[call_id]

        if isinstance(work_item, (_ToolCallWork, _ExternalEventWork)):
            try:
                result = yield work_item.task
            except Exception as exc:
                resume_result = _external_error(exc)
            else:
                if isinstance(work_item, _ExternalEventWork):
                    resume_result = {"return_value": _ensure_json_value(_normalize_event_payload(result))}
                else:
                    resume_result = {"return_value": _ensure_json_value(result)}

        elif isinstance(work_item, _ApprovalThenToolWork):
            try:
                approval = yield work_item.approval_task
                approval = _normalize_event_payload(approval)
                if not (isinstance(approval, dict) and approval.get("approved") is True):
                    resume_result = {
                        "exc_type": "PermissionError",
                        "message": f"Tool call '{work_item.name}' was denied by approver.",
                    }
                else:
                    payload = {"name": work_item.name, "kwargs": _ensure_json_value(work_item.kwargs)}
                    task = self.context.call_activity(self.ACTIVITY_NAME, payload)
                    result = yield task
                    resume_result = {"return_value": _ensure_json_value(result)}
            except Exception as exc:
                resume_result = _external_error(exc)

        elif isinstance(work_item, _WhenAnyWork):
            try:
                winner_task = yield self.context.task_any(work_item.tasks)
                result = yield winner_task
            except Exception as exc:
                resume_result = _external_error(exc)
            else:
                winner_index = _find_task_index(work_item.tasks, winner_task)
                resume_result = {
                    "return_value": {
                        "index": winner_index,
                        "result": _ensure_json_value(result),
                    }
                }
        else:
            raise RuntimeError(f"Unsupported work item type: {type(work_item).__name__}")

        self.pending_work.pop(call_id, None)
        return snapshot.resume({call_id: resume_result})


def _normalize_event_payload(value: Any) -> Any:
    """Durable external event payloads may arrive as JSON strings; parse them."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _find_task_index(tasks: list[Any], winner_task: Any) -> int:
    for i, task in enumerate(tasks):
        if task == winner_task:
            return i
    return -1
