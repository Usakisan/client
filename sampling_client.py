
import asyncio
import json
import os
from typing import Any

import ollama
from mcp import Client
from mcp.client import ClientRequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    TextContent,
)

# ============================================================
# Configuration
# ============================================================

MCP_URL = os.getenv(
    "MCP_URL",
    "http://100.86.227.30:8001/mcp",
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434",
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_TEMPERATURE = float(
    os.getenv("OLLAMA_TEMPERATURE", "1.0")
)

ollama_client = ollama.Client(host=OLLAMA_HOST)


# ============================================================
# MCP tools -> Ollama tools
# ============================================================

def mcp_tools_to_ollama_tools(
    mcp_tools: list[Any],
) -> list[dict[str, Any]]:
    """MCPのツール定義をOllamaの形式に変換する。"""

    tools = []

    for tool in mcp_tools:
        schema = (
            getattr(tool, "input_schema", None)
            or getattr(tool, "inputSchema", None)
            or {"type": "object", "properties": {}}
        )

        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
            },
        })

    return tools


# ============================================================
# MCP result -> text
# ============================================================

def result_to_text(result: Any) -> str:
    """MCPツールの結果をテキストに変換する。"""

    parts = []

    for content in getattr(result, "content", []) or []:
        text = getattr(content, "text", None)

        if text is not None:
            parts.append(text)
        else:
            parts.append(str(content))

    if not parts:
        return json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

    return "\n".join(parts)


# ============================================================
# Sampling callback
# ============================================================

async def handle_sampling(
    context: ClientRequestContext,
    params: CreateMessageRequestParams,
) -> CreateMessageResult:
    """MCPサーバーからのSampling要求をOllamaで処理する。"""

    messages: list[dict[str, Any]] = []

    # サーバーからsystem_promptが指定された場合のみ使用する。
    # クライアント独自の安全性判定用System Promptは追加しない。
    if params.system_prompt:
        messages.append({
            "role": "system",
            "content": params.system_prompt,
        })

    # サーバーから届いたSamplingメッセージをそのまま渡す。
    for message in params.messages:
        content = message.content

        if isinstance(content, TextContent):
            text = content.text
        else:
            text = str(content)

        role = getattr(message.role, "value", message.role)

        if role not in {"user", "assistant"}:
            role = "user"

        messages.append({
            "role": role,
            "content": text,
        })

    max_tokens = getattr(params, "max_tokens", None) or 128

    print("\n[SAMPLING] Sampling要求を受信")
    print(f"[SAMPLING MODEL] {OLLAMA_MODEL}")
    print(
        "[SAMPLING MESSAGES]",
        json.dumps(messages, ensure_ascii=False),
    )

    response = await asyncio.to_thread(
        ollama_client.chat,
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": OLLAMA_TEMPERATURE,
            "num_predict": max_tokens,
        },
    )

    answer = response.message.content or ""

    print("[SAMPLING RESULT]", repr(answer))

    return CreateMessageResult(
        role="assistant",
        content=TextContent(
            type="text",
            text=answer,
        ),
        model=OLLAMA_MODEL,
    )


# ============================================================
# Agent
# ============================================================

async def run_agent(
    client: Client,
    user_text: str,
) -> str:
    """ユーザー入力をOllamaに渡し、必要に応じてMCPツールを実行する。"""

    listed = await client.list_tools()
    tools = mcp_tools_to_ollama_tools(listed.tools)

    # 通常の会話用System Promptにも安全性判定の指示は追加しない。
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": user_text,
        },
    ]

    for _ in range(8):
        response = await asyncio.to_thread(
            ollama_client.chat,
            model=OLLAMA_MODEL,
            messages=messages,
            tools=tools,
            options={
                "temperature": OLLAMA_TEMPERATURE,
            },
        )

        message = response.message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            return (
                message.content
                or "（LLMからテキスト応答がありませんでした）"
            )

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                }
                for tool_call in tool_calls
            ],
        })

        for tool_call in tool_calls:
            name = tool_call.function.name
            arguments = tool_call.function.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            print(f"\n[MCP TOOL] {name}")
            print(
                "[ARGUMENTS]",
                json.dumps(arguments, ensure_ascii=False),
            )

            try:
                result = await client.call_tool(
                    name,
                    arguments=arguments,
                )
                tool_result = result_to_text(result)

            except Exception as exc:
                tool_result = json.dumps(
                    {
                        "error": type(exc).__name__,
                        "detail": str(exc),
                    },
                    ensure_ascii=False,
                )

            print("[RESULT]", tool_result)

            messages.append({
                "role": "tool",
                "content": tool_result,
            })

    return "Tool呼び出し回数が上限に達しました。"


# ============================================================
# Main
# ============================================================

async def main() -> None:
    print("=== Ollama + MCP IoT Client ===")
    print(f"Ollama: {OLLAMA_HOST}")
    print(f"Model : {OLLAMA_MODEL}")
    print(f"Temperature: {OLLAMA_TEMPERATURE}")
    print(f"MCP: {MCP_URL}")
    print("終了するには exit または Ctrl+C を入力してください。")

    # Sampling callbackを登録すると、SDKがSampling capabilityを宣言する。
    async with Client(
        MCP_URL,
        mode="legacy",
        sampling_callback=handle_sampling,
    ) as client:

        listed = await client.list_tools()

        print("\n利用可能なMCP Tools:")
        for tool in listed.tools:
            print(f"  - {tool.name}: {tool.description or ''}")

        while True:
            try:
                user_text = await asyncio.to_thread(
                    input,
                    "\nYou > ",
                )
            except (EOFError, KeyboardInterrupt):
                print()
                break

            user_text = user_text.strip()

            if not user_text:
                continue

            if user_text.lower() in {"exit", "quit"}:
                break

            try:
                answer = await run_agent(client, user_text)
                print(f"\nOllama > {answer}")

            except Exception as exc:
                print(
                    f"\n[ERROR] {type(exc).__name__}: {exc}"
                )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n終了しました。")
