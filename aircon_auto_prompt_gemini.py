"""MCP + Google AI Studio (Gemini) experiment runner with JSONL logging.

This is the cloud-LLM counterpart of ``aircon_auto_prompt.py``.  Prompt-file
handling, MCP interaction, air-conditioner reset, and JSONL records are kept
the same; only the LLM provider is Google AI Studio's Gemini API.

Set ``GEMINI_API_KEY`` before running.  The key is deliberately read from the
environment and is never written to logs or source code.
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
from urllib.request import Request, urlopen

from google import genai
from google.genai import types
from mcp import Client
from mcp.client import ClientRequestContext
from mcp.types import CreateMessageRequestParams, CreateMessageResult, TextContent


# ============================================================
# Configuration (MCP and experiment behavior match aircon_auto_prompt.py)
# ============================================================
MCP_URL = os.getenv("MCP_URL", "http://100.86.227.30:8001/mcp")
AIRCON_RESET_URL = "http://100.101.199.62:8000/reset"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "1.0"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_CLIENT: genai.Client | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def get_gemini_client() -> genai.Client:
    """Return the single Gemini client shared by the entire experiment run."""
    global GEMINI_CLIENT
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create an API key in Google AI Studio and "
            "export GEMINI_API_KEY before running this program."
        )
    if GEMINI_CLIENT is None:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    return GEMINI_CLIENT


async def reset_aircon_state() -> None:
    """Restore the virtual air conditioner to its initial state."""
    def post_reset() -> None:
        request = Request(AIRCON_RESET_URL, method="POST")
        with urlopen(request) as response:
            response.read()

    await asyncio.to_thread(post_reset)


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
# MCP tools -> Gemini function declarations
# ============================================================
def mcp_tools_to_gemini_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to Gemini function declarations."""
    declarations = []
    for tool in mcp_tools:
        schema = (
            getattr(tool, "input_schema", None)
            or getattr(tool, "inputSchema", None)
            or {"type": "object", "properties": {}}
        )
        declarations.append(
            {"name": tool.name, "description": tool.description or "", "parameters": schema}
        )
    return declarations


def response_text(response: Any) -> str:
    """Get plain text without failing if Gemini returned only a function call."""
    try:
        return response.text or ""
    except (AttributeError, ValueError):
        return ""


def response_function_calls(response: Any) -> list[Any]:
    """Return all Gemini function calls, including SDK-version-compatible fallbacks."""
    calls = getattr(response, "function_calls", None)
    if calls:
        return list(calls)
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []
    parts = getattr(getattr(candidates[0], "content", None), "parts", None) or []
    return [part.function_call for part in parts if getattr(part, "function_call", None)]


def response_content(response: Any) -> Any:
    """Return the model turn that must be retained before a tool response."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates or not getattr(candidates[0], "content", None):
        raise RuntimeError("Gemini returned neither content nor a tool call.")
    return candidates[0].content


def make_content(role: str, text: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part.from_text(text=text)])


def generate_config(*, tools: list[dict[str, Any]] | None = None, max_tokens: int | None = None,
                    system_instruction: str | None = None) -> types.GenerateContentConfig:
    """Build a Gemini request configuration for both normal and sampling calls."""
    tool_config = [types.Tool(function_declarations=tools)] if tools else None
    return types.GenerateContentConfig(
        tools=tool_config,
        temperature=GEMINI_TEMPERATURE,
        max_output_tokens=max_tokens,
        system_instruction=system_instruction,
    )


# ============================================================
# MCP result -> text (identical behavior to aircon_auto_prompt.py)
# ============================================================
def result_to_text(result: Any) -> str:
    """Convert an MCP tool result to text."""
    parts = []
    for content in getattr(result, "content", []) or []:
        text = getattr(content, "text", None)
        parts.append(text if text is not None else str(content))
    if not parts:
        return json.dumps(result, ensure_ascii=False, default=str)
    return "\n".join(parts)


# ============================================================
# Sampling callback (same MCP behavior, Gemini for generation)
# ============================================================
async def handle_sampling(context: ClientRequestContext, params: CreateMessageRequestParams) -> CreateMessageResult:
    """Process an MCP Sampling request with Gemini."""
    contents = []
    log_messages = []
    for message in params.messages:
        content = message.content
        text = content.text if isinstance(content, TextContent) else str(content)
        role = getattr(message.role, "value", message.role)
        role = role if role in {"user", "model"} else ("model" if role == "assistant" else "user")
        contents.append(make_content(role, text))
        log_messages.append({"role": role, "content": text})

    max_tokens = getattr(params, "max_tokens", None) or 128
    log_event("sampling_request", model=GEMINI_MODEL, messages=log_messages,
              max_tokens=max_tokens, system_prompt_present=bool(params.system_prompt))
    print("\n[SAMPLING] Sampling要求を受信")
    print(f"[SAMPLING MODEL] {GEMINI_MODEL}")
    print("[SAMPLING MESSAGES]", json.dumps(log_messages, ensure_ascii=False))
    started = time.perf_counter()
    try:
        gemini = get_gemini_client()
        response = await gemini.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=generate_config(max_tokens=max_tokens, system_instruction=params.system_prompt),
        )
        answer = response_text(response)
    except Exception as exc:
        log_event("sampling_error", error_type=type(exc).__name__, error_detail=str(exc),
                  duration_ms=round((time.perf_counter() - started) * 1000, 3))
        raise
    log_event("sampling_response", answer=answer,
              duration_ms=round((time.perf_counter() - started) * 1000, 3))
    print("[SAMPLING RESULT]", repr(answer))
    return CreateMessageResult(role="assistant", content=TextContent(type="text", text=answer), model=GEMINI_MODEL)


# ============================================================
# Agent (same MCP calls and messages as aircon_auto_prompt.py)
# ============================================================
async def run_agent(client: Client, user_text: str) -> str:
    """Send one user prompt to Gemini and execute requested MCP tools."""
    listed = await client.list_tools()
    tools = mcp_tools_to_gemini_tools(listed.tools)
    log_event("tools_presented_to_model", tools=tools)
    contents: list[Any] = [make_content("user", user_text)]

    for turn_index in range(8):
        started = time.perf_counter()
        try:
            gemini = get_gemini_client()
            response = await gemini.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=generate_config(tools=tools),
            )
        except Exception as exc:
            log_event("gemini_error", turn_index=turn_index, error_type=type(exc).__name__,
                      error_detail=str(exc), duration_ms=round((time.perf_counter() - started) * 1000, 3))
            raise

        calls = response_function_calls(response)
        text = response_text(response)
        log_event(
            "model_response", turn_index=turn_index, content=text,
            tool_calls=[{"name": call.name, "arguments": json_safe(call.args or {})} for call in calls],
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        if not calls:
            return text or "（LLMからテキスト応答がありませんでした）"

        # Gemini requires the original model turn immediately before function responses.
        contents.append(response_content(response))
        response_parts = []
        for call_index, call in enumerate(calls):
            name = call.name
            arguments = dict(call.args or {})
            print(f"\n[MCP TOOL] {name}")
            print("[ARGUMENTS]", json.dumps(arguments, ensure_ascii=False))
            started = time.perf_counter()
            try:
                result = await client.call_tool(name, arguments=arguments)
                tool_result = result_to_text(result)
                log_event("tool_call", turn_index=turn_index, call_index=call_index, tool_name=name,
                          arguments=arguments, result_text=tool_result,
                          result_is_error=bool(getattr(result, "isError", False)),
                          duration_ms=round((time.perf_counter() - started) * 1000, 3))
            except Exception as exc:
                tool_result = json.dumps({"error": type(exc).__name__, "detail": str(exc)}, ensure_ascii=False)
                log_event("tool_error", turn_index=turn_index, call_index=call_index, tool_name=name,
                          arguments=arguments, error_type=type(exc).__name__, error_detail=str(exc),
                          duration_ms=round((time.perf_counter() - started) * 1000, 3))
            print("[RESULT]", tool_result)
            response_parts.append(types.Part.from_function_response(name=name, response={"result": tool_result}))
        contents.append(types.Content(role="tool", parts=response_parts))
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
            if text.startswith("# ---"):
                section = text.removeprefix("#").strip()
            continue
        prompts.append((text, section))
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP/Gemini prompts and save research logs.")
    parser.add_argument("prompt_file", type=Path, help="UTF-8 text file; one prompt per line")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Output JSONL file (default: aircon_logs/<condition>.jsonl)")
    parser.add_argument("--condition", default=os.getenv("EXPERIMENT_CONDITION", "unspecified"),
                        help="Experimental condition, e.g. baseline or safety-description-v1")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N prompts")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a prompt error")
    return parser.parse_args()


async def main() -> int:
    global LOGGER, ACTIVE_PROMPT_RECORD
    args = parse_args()
    if not args.prompt_file.is_file():
        print(f"[ERROR] Prompt file not found: {args.prompt_file}", file=sys.stderr)
        return 2
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY is not set.", file=sys.stderr)
        return 2
    # Initialize once, before entering the MCP context.  This avoids a closed
    # SDK HTTP client being reused across model turns or sampling callbacks.
    get_gemini_client()
    prompts = read_prompts(args.prompt_file)
    if args.limit is not None:
        prompts = prompts[:max(args.limit, 0)]
    if not prompts:
        print("[ERROR] No prompts to run.", file=sys.stderr)
        return 2

    log_path = args.log_file or Path("aircon_logs") / f"{args.condition}.jsonl"
    LOGGER = JsonlLogger(log_path)
    prompt_bytes = args.prompt_file.read_bytes()
    print("=== Google AI Studio (Gemini) + MCP IoT Client ===")
    print(f"Model : {GEMINI_MODEL}")
    print(f"Temperature: {GEMINI_TEMPERATURE}")
    print(f"MCP: {MCP_URL}")
    print("終了するには exit または Ctrl+C を入力してください。")
    failures = 0
    try:
        async with Client(MCP_URL, mode="legacy", sampling_callback=handle_sampling) as client:
            listed = await client.list_tools()
            LOGGER.write("run_metadata", condition=args.condition, prompt_file=str(args.prompt_file),
                         prompt_file_sha256=hashlib.sha256(prompt_bytes).hexdigest(), prompt_count=len(prompts),
                         gemini_model=GEMINI_MODEL, gemini_temperature=GEMINI_TEMPERATURE,
                         python_version=sys.version, mcp_tool_catalog=listed.tools)
            print("\n利用可能なMCP Tools:")
            for tool in listed.tools:
                print(f"  - {tool.name}: {tool.description or ''}")
            for prompt_index, (user_text, section) in enumerate(prompts, start=1):
                ACTIVE_PROMPT_RECORD = {"prompt_index": prompt_index, "prompt_text": user_text,
                                        "prompt_section": section, "status": "running", "tools_presented_to_model": [],
                                        "model_turns": [], "tool_executions": [], "sampling_events": [], "errors": []}
                started = time.perf_counter()
                print(f"\nYou > {user_text}")
                try:
                    answer = await run_agent(client, user_text)
                    ACTIVE_PROMPT_RECORD["status"] = "completed"
                    ACTIVE_PROMPT_RECORD["final_answer"] = answer
                    print(f"\nGemini > {answer}")
                except Exception as exc:
                    failures += 1
                    ACTIVE_PROMPT_RECORD["status"] = "error"
                    ACTIVE_PROMPT_RECORD["error"] = {"error_type": type(exc).__name__, "error_detail": str(exc)}
                    print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
                    if not args.continue_on_error:
                        raise
                finally:
                    ACTIVE_PROMPT_RECORD["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
                    LOGGER.write("prompt_result", **ACTIVE_PROMPT_RECORD)
                    ACTIVE_PROMPT_RECORD = None
                    await reset_aircon_state()
    except Exception as exc:
        LOGGER.write("run_error", error_type=type(exc).__name__, error_detail=str(exc))
        return 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n終了しました。")
