# Monty CodeAct for Microsoft Agent Framework

A variation of [MAF's CodeAct](https://devblogs.microsoft.com/agent-framework/codeact-with-hyperlight/) that replaces the Hyperlight WASM sandbox with [pydantic-monty](https://github.com/pydantic/monty), a Rust-based Python interpreter. Two execution modes: **non-durable** (inline, ephemeral) and **durable** (backed by Durable Functions with replay, fan-out, external events, and human-in-the-loop).

## Quick Start

### Prerequisites

- Python 3.13+
- [Azure Functions Core Tools](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local) v4+
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) (`az login` for Foundry auth)
- Docker (for Azurite and DTS emulator)
- An Azure AI Foundry project with a deployed chat model

### Local Infrastructure

Start the emulators:

```bash
# Azurite (Azure Storage emulator)
docker run -d \
  --name azurite \
  -p 10000:10000 -p 10001:10001 -p 10002:10002 \
  mcr.microsoft.com/azure-storage/azurite \
  azurite --skipApiVersionCheck --blobHost 0.0.0.0 --queueHost 0.0.0.0 --tableHost 0.0.0.0

# Durable Task Scheduler emulator (for durable mode)
docker run -d \
  --name dts-emulator \
  -p 8080:8080 -p 8082:8082 \
  mcr.microsoft.com/dts/dts-emulator:latest
```

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
az login
```

Edit `local.settings.json` to set your Foundry project endpoint and model.

### Run

```bash
func start
```

### Test

```bash
# Non-durable (uses hardcoded benchmark prompt when body is empty)
curl -s -X POST http://localhost:7071/api/run -d ''

# Durable
curl -s -X POST http://localhost:7071/api/run-durable -d ''

# Custom prompt
curl -s -X POST http://localhost:7071/api/run \
  -H "Content-Type: text/plain" \
  -d 'List all users and their discount rates.'

# Raise an external event (durable mode)
curl -s -X POST http://localhost:7071/api/orchestrations/{instanceId}/events/{eventName} \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'
```

The DTS dashboard is at `http://localhost:8082`.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  function_app.py                                         │
│  ┌──────────────┐  ┌───────────────────┐                 │
│  │ POST /run    │  │ POST /run-durable │                 │
│  └──────┬───────┘  └────────┬──────────┘                 │
│         │                   │                            │
│         ▼                   ▼                            │
│  ┌──────────────────────────────────────┐                │
│  │  CodeActProvider (ContextProvider)   │                │
│  │  - injects execute_code tool (non-   │                │
│  │    durable) or start_execute_code +  │                │
│  │    get_execution_result (durable)    │                │
│  │  - injects CodeAct instructions      │                │
│  └──────────────┬───────────────────────┘                │
│                 │                                        │
│    ┌────────────┴────────────┐                           │
│    ▼                         ▼                           │
│  InlineCodeBridge       DurableCodeBridge                │
│  (monty_bridge.py)      (monty_bridge.py)                │
│  - Monty runs code      - Monty runs code                │
│  - Type-checked via ty  - Type-checked via ty            │
│  - Direct tool calls    - Direct tool calls → activities │
│  - asyncio.gather       - asyncio.gather → task_all      │
│    (concurrent)         - when_any → task_any            │
│  - Ephemeral            - Durable replay                 │
│                         - Per-tool approval via events   │
│                                                          │
│  register_durable_codeact(app, tools)                    │
│  - codeact_orchestrator (hidden)                         │
│  - codeact_call_tool activity (hidden)                   │
└──────────────────────────────────────────────────────────┘
```

## DSL Available Inside `execute_code`

The LLM generates Python code that calls tools directly as typed async functions. Code is type-checked against tool signatures before execution using [ty](https://docs.astral.sh/ty/) (included in Monty).

| Primitive | Non-durable | Durable |
|---|---|---|
| `await list_users()` | Direct function call | Durable activity invocation |
| `await get_orders_for_user(user_id=1)` | Direct function call (type-checked) | Durable activity (type-checked) |
| `asyncio.gather(tool_a(...), tool_b(...))` | Concurrent execution via `asyncio.gather` | Parallel via `task_all` |
| `await when_any([{"tool": "n", "kwargs": {...}}, ...])` | Concurrent, returns first | True race via `task_any` |
| `await wait_for_external_event('EventName')` | Rejected (not available) | Waits for external event |
| `print(...)` | Captured in stdout | Captured in stdout |

`await call_tool('name', key=val)` is also supported as a fallback but is not type-checked.

## Feature Comparison: Hyperlight CodeAct vs. Monty CodeAct

### Execution Environment

| | Hyperlight CodeAct | Monty (non-durable) | Monty (durable) |
|---|---|---|---|
| **Runtime** | Hyperlight WASM sandbox (`hyperlight-sandbox`) | pydantic-monty (Rust-based Python interpreter) | pydantic-monty inside Durable Functions orchestrator |
| **Isolation model** | In-process WASM micro-VM on a dedicated OS thread (not a separate process). Snapshot/restore between calls for clean state. | In-process Monty interpreter. Fresh instance per `execute_code` call. | Same Monty sandbox + Durable replay isolation. |
| **Code delivery** | Code string passed directly to `sandbox.run(code=code)`. Workspace files and file mounts are copied to temp directories via `shutil.copy2`. Code itself is not copied as a file. | Code string passed to `Monty(code)`. No file copying. | Same — code passed as orchestration input string. |
| **Thread model** | `_SandboxWorker`: single-thread `ThreadPoolExecutor` per cached sandbox entry. PyO3 `unsendable` constraint requires all sandbox access from the creating thread. | Runs on the async event loop thread. | Orchestrator generator runs on the Functions worker thread. |
| **Platform** | Linux and Windows today. macOS support announced as "on the way". Default backend is WASM (`hyperlight-sandbox-backend-wasm`), which may have wider platform support than the hypervisor backends (KVM/WHP). | Any platform (pure Rust/Python, no hypervisor needed). | Any platform + DTS emulator (local) or Azure DTS (cloud). |
| **Dependencies** | `hyperlight-sandbox`, `hyperlight-sandbox-python-guest`, `hyperlight-sandbox-backend-wasm` | `pydantic-monty` | `pydantic-monty`, `azure-functions-durable` |

### Tool Calling

| | Hyperlight | Monty (non-durable) | Monty (durable) |
|---|---|---|---|
| **DSL** | `call_tool(name, **kwargs)` — synchronous, stringly-typed, via FFI host callback | `await tool_name(**kwargs)` — async, directly typed. `call_tool` also supported as fallback. | Same — `await tool_name(**kwargs)`, dispatched as Durable activity |
| **Type safety** | None. `call_tool` is stringly-typed; argument errors discovered at runtime. | **Type-checked before execution.** Type stubs auto-generated from tool signatures; [ty](https://docs.astral.sh/ty/) validates argument types, counts, and return types at parse time. Wrong types → `MontyTypingError` before any tool runs. | Same type checking. Errors caught before any activities fire. |
| **How host tools are invoked** | FFI bridge: WASM guest calls out to host Python. The callback runs `asyncio.run()` on a new `threading.Thread` to handle async `FunctionTool.invoke()`. | Monty (Rust) pauses at each tool call, producing a `FunctionSnapshot`. The Python bridge reads the function name and kwargs, invokes the host-side Python function directly, and resumes Monty with the result. Supports sync and async tools via `inspect.iscoroutinefunction()`. | Same Monty pause/resume pattern, but instead of invoking the tool directly, the bridge yields a `context.call_activity("codeact_call_tool", {name, kwargs})` Durable task — the tool runs in a separate activity function invocation. |
| **Approval mode** | Pre-execution only. `always_require` / `never_require` per tool or per `execute_code` call. If any tool requires approval, the entire code block is gated before execution. No per-tool mid-execution approval. | Pre-execution gating: if any tool has `approval_mode="always_require"`, the entire `execute_code` call is gated. | **Both.** Pre-execution gating (same as non-durable) + **per-tool mid-execution approval**: individual tool calls pause the orchestration for an external event (`approve:{tool}:{id}`) before the activity runs. Approve or deny each call granularly. |

### Fan-out / Concurrency

| | Hyperlight | Monty (non-durable) | Monty (durable) |
|---|---|---|---|
| **Fan-out** | Sequential `call_tool` only (single-threaded WASM guest, one FFI call at a time). | `asyncio.gather(tool_a(...), tool_b(...))` — tool calls dispatched concurrently via `asyncio.gather` + `asyncio.to_thread` for sync tools. | `asyncio.gather(tool_a(...), tool_b(...))` — maps to `context.task_all()`, **true parallel execution** as separate activity invocations. |
| **Race** | Not supported. | `await when_any([...])` — runs all concurrently, returns first result. | `await when_any([...])` — maps to `context.task_any()`, true race with Durable scheduling. |
| **True parallelism** | No. Single-threaded sandbox. | **Concurrent.** Multiple tool calls run concurrently on the event loop (async tools) or via `asyncio.to_thread` (sync tools). | **Yes.** Activities are separate function invocations scheduled in parallel by the Durable runtime. |

### Durability & Reliability

| | Hyperlight | Monty (non-durable) | Monty (durable) |
|---|---|---|---|
| **Durable execution** | No. Ephemeral sandbox, state lost on crash. | No. Ephemeral, in-process. | **Yes.** Durable Functions orchestrator with history-backed replay. |
| **Survives process restart** | No. | No. | **Yes.** |
| **External events** | Not supported. | Rejected with clear error message. | `await wait_for_external_event('name')` → `context.wait_for_external_event()`. |
| **Human-in-the-loop** | Not supported (approval is pre-execution gating, not mid-execution). | Not supported. | **Yes** — via external events + `POST /api/orchestrations/{id}/events/{name}`. |
| **Polling / status** | N/A (synchronous return to agent). | N/A (synchronous return). | `get_execution_result(instance_id, wait_seconds)` tool. LLM polls for result. |

### Sandbox Capabilities

| | Hyperlight | Monty (non-durable) | Monty (durable) |
|---|---|---|---|
| **Filesystem** | Opt-in: `/input` (read), `/output` (write). Host paths copied to temp dirs. Files written to `/output` are attached to tool result as `Content`. | Blocked in generated code (`PermissionError`). Host-side tool functions have full access. | Blocked in generated code (`PermissionError`). Host-side tool functions have full access. |
| **Python compatibility** | WASM Python guest (`python_guest.path`). Likely compiled CPython with most stdlib, but exact coverage undocumented. | Monty: Rust-based Python interpreter. Subset of Python — not all features/stdlib supported. Supported: `sys`, `os`, `typing`, `asyncio`, `re`, `datetime`, `json`. | Same as non-durable. |
| **Type checking** | None. | **Yes.** Type stubs auto-generated from tool `Annotated` signatures. Monty runs [ty](https://docs.astral.sh/ty/) before execution — catches wrong argument types, missing parameters, return type mismatches. | Same. Errors caught before any Durable activities fire. |
| **OS calls** | Blocked by default (no host access except registered tools, mounts, and allowed domains). | Blocked (`PermissionError` via `is_os_function`). | Blocked. |
| **Stdout capture** | `result.stdout` captured, returned as `Content.from_text()`. | Captured via `print_callback`, returned in result JSON. | Same. |
| **State management** | Snapshot/restore between calls (`sandbox.snapshot()` + `sandbox.restore()`). Each `execute_code` gets a clean sandbox. Sandboxes cached by config key. | Fresh `Monty()` instance per call. | Fresh Monty instance per **replay** (not just per orchestration). Each Durable replay re-creates `DurableCodeBridge` and `Monty(code)` from scratch; Durable history makes already-completed tasks resolve immediately so the code reaches the next pending point quickly. |

### Integration

| | Hyperlight | Monty (non-durable) | Monty (durable) |
|---|---|---|---|
| **MAF integration** | `HyperlightCodeActProvider(ContextProvider)` | `CodeActProvider(ContextProvider)` | Same `CodeActProvider(durable=True, durable_client=client)` |
| **Tools visible to LLM** | `execute_code` only. Provider-owned tools hidden inside sandbox description. | `execute_code` only. | `start_execute_code` + `get_execution_result`. |
| **Durable boilerplate** | N/A. | N/A. | Hidden: `register_durable_codeact(app, tools=TOOLS)` — one line to register orchestrator + activity. |
| **Infrastructure** | None beyond compatible platform + backend packages. | None. | Azurite + DTS emulator (local) or Azure Storage + Azure DTS (cloud). |

### Key Limitations

**Hyperlight CodeAct:**
- macOS not yet supported (announced as "on the way"). WASM backend may have narrower platform availability than expected.
- Single-threaded sandbox — no parallel tool execution within a single `execute_code` call.
- Heavy dependency chain (3 packages: sandbox + guest + backend).
- No durability, external events, or human-in-the-loop during execution.
- Approval is all-or-nothing per `execute_code` call, not per `call_tool()`.
- Alpha status ("evolving API and no guaranteed support").

**Monty CodeAct (non-durable):**
- Monty interprets a Python **subset** — not all Python features or stdlib modules are available.
- Ephemeral — no durability, no external events.
- Per-tool mid-execution approval not available (only pre-execution gating).

**Monty CodeAct (durable):**
- Same Monty Python subset limitation.
- `get_execution_result` polling relies on the LLM following instructions to poll — quality varies by model.
- Not yet exposed in DSL: durable timers, sub-orchestrations, entities, `continue_as_new`, retry policies, custom status.
- Requires DTS infrastructure (emulator locally, Azure DTS in cloud).

### Summary

| Dimension | Advantage |
|---|---|
| **Ease of setup** | Monty non-durable (zero infra, any platform) |
| **Cross-platform** | Monty (both modes) |
| **Sandbox security** | Hyperlight (WASM-level isolation with snapshot/restore) |
| **Type safety** | Monty (both modes — ty validates tool call types before execution) |
| **Concurrent tool execution** | Monty non-durable (`asyncio.gather`) and durable (`task_all`) |
| **Durability / reliability** | Monty durable (Durable Functions replay) |
| **Human-in-the-loop** | Monty durable (external events) |
| **Per-tool approval** | Monty durable (mid-execution approval via external events) |
| **Python compatibility** | Hyperlight (WASM Python guest, likely fuller stdlib) |
| **Code portability (durable ↔ non-durable)** | Monty (same code both modes, flip a flag) |

## Files

| File | Purpose |
|---|---|
| `function_app.py` | HTTP triggers (`/api/run`, `/api/run-durable`, `/api/orchestrations/.../events/...`), sample tools, LLM client |
| `codeact_provider.py` | `CodeActProvider`, `execute_code`/`start_execute_code`/`get_execution_result` tools, instructions builder, `register_durable_codeact()` |
| `monty_bridge.py` | `InlineCodeBridge` (non-durable) and `DurableCodeBridge` (durable) — Monty execution engines, type stub generator |
| `requirements.txt` | Python dependencies |
| `host.json` | Functions host config + DTS durableTask extension |
| `local.settings.json` | Foundry, DTS, and storage config |
