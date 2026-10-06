import asyncio
import json
import os
from typing import Any

import ollama
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = os.getenv("MCP_URL", "http://100.86.227.30:8001/mcp")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")

ollama_client = ollama.Client(host=OLLAMA_HOST)

SYSTEM_PROMPT = """
あなたはIoT温湿度センサーを操作するローカルLLMです。
利用可能なMCPツールを使用してIoTデバイスを操作できます。
""".strip()


def mcp_tools_to_ollama_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    tools = []
    for tool in mcp_tools:
        schema = getattr(tool, "inputSchema", None) or {
            "type": "object",
            "properties": {},
        }
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


def result_to_text(result: Any) -> str:
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


async def run_agent(session: ClientSession, user_text: str) -> str:
    listed = await session.list_tools()
    tools = mcp_tools_to_ollama_tools(listed.tools)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    # Tool callingを必要な回数だけ繰り返す。
    for _ in range(8):
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=tools,
        )

        message = response.message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            return message.content or "（LLMからテキスト応答がありませんでした）"

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        }
                    }
                    for call in tool_calls
                ],
            }
        )

        for call in tool_calls:
            name = call.function.name
            arguments = call.function.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            print(f"\n[MCP TOOL] {name}")
            print(f"[ARGUMENTS] {json.dumps(arguments, ensure_ascii=False)}")

            try:
                result = await session.call_tool(name, arguments)
                tool_result = result_to_text(result)
            except Exception as exc:
                tool_result = json.dumps(
                    {"error": str(exc)},
                    ensure_ascii=False,
                )

            print(f"[RESULT] {tool_result}")

            messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                }
            )

    return "Tool呼び出し回数が上限に達しました。"


async def main() -> None:
    print("=== Ollama + MCP IoT Client ===")
    print(f"Ollama: {OLLAMA_HOST}")
    print(f"Model : {OLLAMA_MODEL}")
    print(f"MCP   : {MCP_URL}")
    print("終了するには exit または Ctrl+C を入力してください。")

    async with streamable_http_client(MCP_URL) as (
    read_stream,
    write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            listed = await session.list_tools()
            print("\n利用可能なMCP Tools:")
            for tool in listed.tools:
                print(f"  - {tool.name}: {tool.description or ''}")

            while True:
                try:
                    user_text = input("\nYou > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if not user_text:
                    continue
                if user_text.lower() in {"exit", "quit"}:
                    break

                try:
                    answer = await run_agent(session, user_text)
                    print(f"\nOllama > {answer}")
                except Exception as exc:
                    print(f"\n[ERROR] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
