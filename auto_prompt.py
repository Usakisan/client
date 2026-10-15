"""MCP + Ollama experiment runner with JSONL logging.

This program intentionally keeps the MCP interaction in ``sampling_client.py``:
the same legacy Client mode, sampling callback, tool discovery, and call_tool
sequence are used.  It only replaces interactive input with a prompt file and
adds local, append-only experiment logs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ollama
from mcp import Client
from mcp.client import ClientRequestContext
from mcp.types import CreateMessageRequestParams, CreateMessageResult, TextContent


# ============================================================
# Configuration (kept identical to sampling_client.py)
# ============================================================
MCP_URL = os.getenv("MCP_URL", "http://100.86.227.30:8001/mcp")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "1.0"))
ollama_client = ollama.Client(host=OLLAMA_HOST)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def json_safe(value: Any) -> Any:
    """Convert SDK values to values that can be stored in JSONL."""
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


class JsonlLogger:
    """Small synchronous logger; it never changes an MCP request or response."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, event: str, **fields: Any) -> None:
        record = {"timestamp_utc": utc_now(), "event": event, **json_safe(fields)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


LOGGER: JsonlLogger | None = None
ACTIVE_PROMPT_RECORD: dict[str, Any] | None = None


def log_event(event: str, **fields: Any) -> None:
    """Accumulate details into the current prompt's single JSONL record."""
    if ACTIVE_PROMPT_RECORD is None:
        return
    detail = json_safe(fields)
    if event == "tools_presented_to_model":
        ACTIVE_PROMPT_RECORD["tools_presented_to_model"] = detail["tools"]
    elif event == "model_response":
        ACTIVE_PROMPT_RECORD["model_turns"].append(detail)
    elif event in {"tool_call", "tool_error"}:
        detail["event"] = event
        ACTIVE_PROMPT_RECORD["tool_executions"].append(detail)
    elif event.startswith("sampling_"):
        detail["event"] = event
        ACTIVE_PROMPT_RECORD["sampling_events"].append(detail)
    else:
        detail["event"] = event
        ACTIVE_PROMPT_RECORD["errors"].append(detail)


# ============================================================
# MCP tools -> Ollama tools (identical behavior to sampling_client.py)
# ============================================================
def mcp_tools_to_ollama_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to Ollama's format."""
    tools = []
    for tool in mcp_tools:
        schema = (
            getattr(tool, "input_schema", None)
            or getattr(tool, "inputSchema", None)
            or {"type": "object", "properties": {}}
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            }
        )
    return tools


# ============================================================
# MCP result -> text (identical behavior to sampling_client.py)
# ============================================================
def result_to_text(result: Any) -> str:
    """Convert an MCP tool result to text."""
    parts = []
    for content in getattr(result, "content", []) or []:
        text = getattr(content, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(str(content))
    if not parts:
        return json.dumps(result, ensure_ascii=False, default=str)
    return "\n".join(parts)


# ============================================================
# Sampling callback (same MCP/Ollama behavior; logging is observational)
# ============================================================
async def handle_sampling(
    context: ClientRequestContext,
    params: CreateMessageRequestParams,
) -> CreateMessageResult:
    """Process a Sampling request from the MCP server with Ollama."""
    messages: list[dict[str, Any]] = []
    if params.system_prompt:
        messages.append({"role": "system", "content": params.system_prompt})
    for message in params.messages:
        content = message.content
        text = content.text if isinstance(content, TextContent) else str(content)
        role = getattr(message.role, "value", message.role)
        if role not in {"user", "assistant"}:
            role = "user"
        messages.append({"role": role, "content": text})

    max_tokens = getattr(params, "max_tokens", None) or 128
    log_event(
        "sampling_request",
        model=OLLAMA_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        system_prompt_present=bool(params.system_prompt),
    )
    print("\n[SAMPLING] Sampling要求を受信")
    print(f"[SAMPLING MODEL] {OLLAMA_MODEL}")
    print("[SAMPLING MESSAGES]", json.dumps(messages, ensure_ascii=False))
    started = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            ollama_client.chat,
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": OLLAMA_TEMPERATURE, "num_predict": max_tokens},
        )
        answer = response.message.content or ""
    except Exception as exc:
        log_event(
            "sampling_error",
            error_type=type(exc).__name__,
            error_detail=str(exc),
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        raise
    log_event(
        "sampling_response",
        answer=answer,
        **(
            {"thinking": response.message.thinking}
            if getattr(response.message, "thinking", None)
            else {}
        ),
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
    )
    print("[SAMPLING RESULT]", repr(answer))
    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=answer),
        model=OLLAMA_MODEL,
    )


# ============================================================
# Agent (same MCP calls and messages as sampling_client.py)
# ============================================================
async def run_agent(client: Client, user_text: str) -> str:
    """Send one user prompt to Ollama and execute requested MCP tools."""
    listed = await client.list_tools()
    tools = mcp_tools_to_ollama_tools(listed.tools)
    log_event("tools_presented_to_model", tools=tools)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_text}]

    for turn_index in range(8):
        started = time.perf_counter()
        try:
            response = await asyncio.to_thread(
                ollama_client.chat,
                model=OLLAMA_MODEL,
                messages=messages,
                tools=tools,
                options={"temperature": OLLAMA_TEMPERATURE},
            )
        except Exception as exc:
            log_event(
                "ollama_error",
                turn_index=turn_index,
                error_type=type(exc).__name__,
                error_detail=str(exc),
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            raise
        message = response.message
        tool_calls = getattr(message, "tool_calls", None) or []
        model_turn = {
            "turn_index": turn_index,
            "content": message.content or "",
            "tool_calls": [
                {"name": call.function.name, "arguments": call.function.arguments}
                for call in tool_calls
            ],
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        # Non-thinking models simply have no ``thinking`` attribute/value.
        # Do not add an empty field: the log remains compact and comparable.
        thinking = getattr(message, "thinking", None)
        if thinking:
            model_turn["thinking"] = thinking
        log_event("model_response", **model_turn)
        if not tool_calls:
            return message.content or "（LLMからテキスト応答がありませんでした）"

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {"function": {"name": call.function.name, "arguments": call.function.arguments}}
                    for call in tool_calls
                ],
            }
        )
        for call_index, tool_call in enumerate(tool_calls):
            name = tool_call.function.name
            arguments = tool_call.function.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            print(f"\n[MCP TOOL] {name}")
            print("[ARGUMENTS]", json.dumps(arguments, ensure_ascii=False))
            started = time.perf_counter()
            try:
                result = await client.call_tool(name, arguments=arguments)
                tool_result = result_to_text(result)
                log_event(
                    "tool_call",
                    turn_index=turn_index,
                    call_index=call_index,
                    tool_name=name,
                    arguments=arguments,
                    result_text=tool_result,
                    result_is_error=bool(getattr(result, "isError", False)),
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                )
            except Exception as exc:
                tool_result = json.dumps(
                    {"error": type(exc).__name__, "detail": str(exc)}, ensure_ascii=False
                )
                log_event(
                    "tool_error",
                    turn_index=turn_index,
                    call_index=call_index,
                    tool_name=name,
                    arguments=arguments,
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                )
            print("[RESULT]", tool_result)
            messages.append({"role": "tool", "content": tool_result})
    return "Tool呼び出し回数が上限に達しました。"


def read_prompts(path: Path) -> list[tuple[str, str | None]]:
    """Read one prompt per line; comments/blanks are ignored, section comments label prompts."""
    prompts: list[tuple[str, str | None]] = []
    section: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            # e.g. '# --- set_threshold（...） ---' becomes an analysis label.
            if text.startswith("# ---"):
                section = text.removeprefix("#").strip()
            continue
        prompts.append((text, section))
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP/Ollama prompts and save research logs.")
    parser.add_argument("prompt_file", type=Path, help="UTF-8 text file; one prompt per line")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Output JSONL file (default: logs/auto_prompt_<UTC timestamp>.jsonl)",
    )
    parser.add_argument(
        "--condition",
        default=os.getenv("EXPERIMENT_CONDITION", "unspecified"),
        help="Experimental condition, e.g. baseline or safety-description-v1",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N prompts")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a prompt error")
    return parser.parse_args()


async def main() -> int:
    global LOGGER, ACTIVE_PROMPT_RECORD
    args = parse_args()
    if not args.prompt_file.is_file():
        print(f"[ERROR] Prompt file not found: {args.prompt_file}", file=sys.stderr)
        return 2
    prompts = read_prompts(args.prompt_file)
    if args.limit is not None:
        prompts = prompts[: max(args.limit, 0)]
    if not prompts:
        print("[ERROR] No prompts to run.", file=sys.stderr)
        return 2

    default_name = f"auto_prompt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    log_path = args.log_file or Path("logs") / default_name
    LOGGER = JsonlLogger(log_path)
    prompt_bytes = args.prompt_file.read_bytes()

    # Keep the normal console output from sampling_client.py.  Batch metadata
    # is written only to JSONL so the established terminal output is unchanged.
    print("=== Ollama + MCP IoT Client ===")
    print(f"Ollama: {OLLAMA_HOST}")
    print(f"Model : {OLLAMA_MODEL}")
    print(f"Temperature: {OLLAMA_TEMPERATURE}")
    print(f"MCP: {MCP_URL}")
    print("終了するには exit または Ctrl+C を入力してください。")
    failures = 0
    try:
        # The following Client construction and initial list_tools call match sampling_client.py.
        async with Client(MCP_URL, mode="legacy", sampling_callback=handle_sampling) as client:
            listed = await client.list_tools()
            # This is the first JSONL line.  It records the fixed context once,
            # while every following line represents exactly one input prompt.
            LOGGER.write(
                "run_metadata",
                condition=args.condition,
                prompt_file=str(args.prompt_file),
                prompt_file_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
                prompt_count=len(prompts),
                ollama_model=OLLAMA_MODEL,
                ollama_temperature=OLLAMA_TEMPERATURE,
                ollama_host=OLLAMA_HOST,
                python_version=sys.version,
                mcp_tool_catalog=listed.tools,
            )
            print("\n利用可能なMCP Tools:")
            for tool in listed.tools:
                print(f"  - {tool.name}: {tool.description or ''}")

            for prompt_index, (user_text, section) in enumerate(prompts, start=1):
                ACTIVE_PROMPT_RECORD = {
                    "prompt_index": prompt_index,
                    "prompt_text": user_text,
                    "prompt_section": section,
                    "status": "running",
                    "tools_presented_to_model": [],
                    "model_turns": [],
                    "tool_executions": [],
                    "sampling_events": [],
                    "errors": [],
                }
                started = time.perf_counter()
                print(f"\nYou > {user_text}")
                try:
                    answer = await run_agent(client, user_text)
                    ACTIVE_PROMPT_RECORD["status"] = "completed"
                    ACTIVE_PROMPT_RECORD["final_answer"] = answer
                    print(f"\nOllama > {answer}")
                except Exception as exc:
                    failures += 1
                    ACTIVE_PROMPT_RECORD["status"] = "error"
                    ACTIVE_PROMPT_RECORD["error"] = {
                        "error_type": type(exc).__name__,
                        "error_detail": str(exc),
                    }
                    print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
                    if not args.continue_on_error:
                        raise
                finally:
                    ACTIVE_PROMPT_RECORD["duration_ms"] = round(
                        (time.perf_counter() - started) * 1000, 3
                    )
                    LOGGER.write("prompt_result", **ACTIVE_PROMPT_RECORD)
                    ACTIVE_PROMPT_RECORD = None
    except Exception as exc:
        LOGGER.write("run_error", error_type=type(exc).__name__, error_detail=str(exc))
        return 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n終了しました。")
