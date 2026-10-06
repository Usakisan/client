
import asyncio
import json
import os
from typing import Any

import gradio as gr
import ollama

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


# ============================================================
# 設定
# ============================================================

MCP_URL = os.getenv(
    "MCP_URL",
    "http://100.86.227.30:8001/mcp"
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:14b"
)


# ============================================================
# Ollama Client
# ============================================================

ollama_client = ollama.Client(
    host=OLLAMA_HOST
)


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
あなたはIoT温湿度センサーを操作するローカルLLMです。

利用可能なMCPツールを使用してIoTデバイスを操作できます。

ユーザーの要求に応じて、必要な場合のみMCP Toolを使用してください。

Toolを使用した場合は、その実行結果を確認してから
ユーザーに分かりやすく結果を説明してください。
""".strip()


# ============================================================
# MCP Tool → Ollama Tool形式
# ============================================================

def mcp_tools_to_ollama_tools(
    mcp_tools: list[Any],
) -> list[dict[str, Any]]:

    tools = []

    for tool in mcp_tools:

        schema = getattr(
            tool,
            "inputSchema",
            None,
        )

        if not schema:
            schema = {
                "type": "object",
                "properties": {},
            }

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": (
                        tool.description or ""
                    ),
                    "parameters": schema,
                },
            }
        )

    return tools


# ============================================================
# MCP Result → Text
# ============================================================

def result_to_text(result: Any) -> str:

    parts = []

    for content in (
        getattr(result, "content", [])
        or []
    ):

        text = getattr(
            content,
            "text",
            None,
        )

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
# Agent
# ============================================================

async def run_agent(
    session: ClientSession,
    user_text: str,
    history: list[dict[str, Any]],
    log_callback=None,
) -> str:

    # --------------------------------------------------------
    # MCP Tools取得
    # --------------------------------------------------------

    listed = await session.list_tools()

    tools = mcp_tools_to_ollama_tools(
        listed.tools
    )

    # --------------------------------------------------------
    # Messages
    # --------------------------------------------------------

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # --------------------------------------------------------
    # 過去の会話
    # --------------------------------------------------------

    for item in history:

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in {
            "user",
            "assistant",
        }:
            continue

        if not isinstance(content, str):
            continue

        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    # --------------------------------------------------------
    # 現在のユーザー入力
    # --------------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    # ========================================================
    # Tool Calling Loop
    # ========================================================

    for iteration in range(8):

        print(
            f"\n[LLM] Tool loop: {iteration + 1}"
        )

        # ----------------------------------------------------
        # Ollama
        # ----------------------------------------------------

        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=tools,
        )

        message = response.message

        tool_calls = (
            getattr(
                message,
                "tool_calls",
                None,
            )
            or []
        )

        # ----------------------------------------------------
        # Toolを使用しなかった
        # ----------------------------------------------------

        if not tool_calls:

            answer = (
                message.content
                or
                "（LLMから応答がありませんでした）"
            )

            print(
                f"[LLM] {answer}"
            )

            return answer

        # ----------------------------------------------------
        # Assistant message
        # ----------------------------------------------------

        assistant_message = {
            "role": "assistant",
            "content": (
                message.content or ""
            ),
            "tool_calls": [],
        }

        for call in tool_calls:

            assistant_message[
                "tool_calls"
            ].append(
                {
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                }
            )

        messages.append(
            assistant_message
        )

        # ====================================================
        # Tool実行
        # ====================================================

        for call in tool_calls:

            tool_name = call.function.name

            arguments = call.function.arguments

            # ------------------------------------------------
            # ArgumentsをJSONへ変換
            # ------------------------------------------------

            if isinstance(
                arguments,
                str,
            ):

                try:

                    arguments = json.loads(
                        arguments
                    )

                except json.JSONDecodeError:

                    arguments = {}

            # ------------------------------------------------
            # コンソールログ
            # ------------------------------------------------

            print(
                "\n[MCP TOOL]"
            )

            print(
                f"Name: {tool_name}"
            )

            print(
                "Arguments:"
            )

            print(
                json.dumps(
                    arguments,
                    ensure_ascii=False,
                    indent=2,
                )
            )

            # ------------------------------------------------
            # GUIログ
            # ------------------------------------------------

            if log_callback:

                log_callback(
                    "\n### 🔧 MCP ToolCall\n\n"
                    f"**Tool:** `{tool_name}`\n\n"
                    "**Arguments:**\n"
                    "```json\n"
                    f"{json.dumps(arguments, ensure_ascii=False, indent=2)}\n"
                    "```\n"
                )

            # ------------------------------------------------
            # MCP Tool実行
            # ------------------------------------------------

            try:

                result = await session.call_tool(
                    tool_name,
                    arguments,
                )

                tool_result = result_to_text(
                    result
                )

                success = True

            except Exception as exc:

                tool_result = json.dumps(
                    {
                        "error": str(exc)
                    },
                    ensure_ascii=False,
                )

                success = False

            # ------------------------------------------------
            # コンソール
            # ------------------------------------------------

            print(
                "\n[RESULT]"
            )

            print(
                tool_result
            )

            # ------------------------------------------------
            # GUIログ
            # ------------------------------------------------

            if log_callback:

                if success:

                    log_callback(
                        "\n**Result:**\n"
                        "```text\n"
                        f"{tool_result}\n"
                        "```\n"
                    )

                else:

                    log_callback(
                        "\n### ❌ Tool Error\n"
                        "```text\n"
                        f"{tool_result}\n"
                        "```\n"
                    )

            # ------------------------------------------------
            # Tool結果をOllamaへ返す
            # ------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                }
            )

    # --------------------------------------------------------
    # Loop上限
    # --------------------------------------------------------

    return (
        "Tool呼び出し回数の上限に達しました。"
    )


# ============================================================
# MCP接続してAgentを実行
#
# ★ 動作確認済みCLI版と同じ接続方法
# ============================================================

async def execute_agent(
    user_text: str,
    history: list[dict[str, Any]],
    log_callback=None,
) -> str:

    print(
        "\n========================================"
    )

    print(
        "MCP CONNECTION"
    )

    print(
        "========================================"
    )

    print(
        f"MCP: {MCP_URL}"
    )

    # --------------------------------------------------------
    # MCP Streamable HTTP
    #
    # 元の動作確認済みコードと同じ2要素
    # --------------------------------------------------------

    async with streamable_http_client(
        MCP_URL
    ) as (
        read_stream,
        write_stream,
    ):

        # ----------------------------------------------------
        # ClientSession
        # ----------------------------------------------------

        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:

            # ------------------------------------------------
            # Initialize
            # ------------------------------------------------

            await session.initialize()

            print(
                "MCP initialized."
            )

            # ------------------------------------------------
            # Agent
            # ------------------------------------------------

            answer = await run_agent(
                session=session,
                user_text=user_text,
                history=history,
                log_callback=log_callback,
            )

            return answer


# ============================================================
# Chat GUI
# ============================================================

async def chat(
    user_text,
    history,
    debug_log,
):

    # --------------------------------------------------------
    # 空入力
    # --------------------------------------------------------

    if (
        user_text is None
        or not user_text.strip()
    ):

        return (
            history,
            debug_log,
            "",
        )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    if history is None:
        history = []

    history = list(history)

    # --------------------------------------------------------
    # Debug Log
    # --------------------------------------------------------

    logs = []

    def log_callback(text):

        logs.append(text)

    try:

        # ----------------------------------------------------
        # MCP + Ollama
        # ----------------------------------------------------

        answer = await execute_agent(
            user_text=user_text,
            history=history,
            log_callback=log_callback,
        )

        # ----------------------------------------------------
        # Chat履歴
        # ----------------------------------------------------

        history.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        # ----------------------------------------------------
        # Debug Log
        # ----------------------------------------------------

        if logs:

            new_log = (
                "\n\n---\n\n"
                + "\n\n".join(logs)
            )

            debug_log = (
                (debug_log or "")
                + new_log
            )

        return (
            history,
            debug_log,
            "",
        )

    except Exception as exc:

        print(
            "\n[ERROR]"
        )

        print(
            type(exc).__name__,
            exc,
        )

        error_log = (
            "\n\n---\n\n"
            "### ❌ ERROR\n\n"
            f"**Type:** `{type(exc).__name__}`\n\n"
            f"**Message:** `{exc}`"
        )

        debug_log = (
            (debug_log or "")
            + error_log
        )

        # ----------------------------------------------------
        # GUIにもエラー表示
        # ----------------------------------------------------

        history.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": (
                    "❌ エラーが発生しました。\n\n"
                    f"`{type(exc).__name__}: {exc}`"
                ),
            }
        )

        return (
            history,
            debug_log,
            "",
        )


# ============================================================
# MCP Tools一覧取得
# ============================================================

async def get_tools():

    try:

        # ----------------------------------------------------
        # 動作確認済みCLI版と同じMCP接続
        # ----------------------------------------------------

        async with streamable_http_client(
            MCP_URL
        ) as (
            read_stream,
            write_stream,
        ):

            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:

                await session.initialize()

                listed = await session.list_tools()

                # ------------------------------------------------
                # Toolなし
                # ------------------------------------------------

                if not listed.tools:

                    return (
                        "## 🔧 MCP Tools\n\n"
                        "利用可能なToolはありません。"
                    )

                # ------------------------------------------------
                # Tool一覧
                # ------------------------------------------------

                text = (
                    "## 🔧 MCP Tools\n\n"
                )

                for tool in listed.tools:

                    description = (
                        tool.description
                        or "説明なし"
                    )

                    text += (
                        f"### `{tool.name}`\n\n"
                        f"{description}\n\n"
                        "---\n\n"
                    )

                return text

    except Exception as exc:

        return (
            "## ❌ MCP Connection Error\n\n"
            f"**Type:** `{type(exc).__name__}`\n\n"
            f"**Message:** `{exc}`"
        )


# ============================================================
# Clear
# ============================================================

def clear_chat():

    return (
        [],
        "",
        "",
    )


# ============================================================
# GUI
# ============================================================

with gr.Blocks(
    title="Ollama + MCP IoT Client"
) as demo:

    # ========================================================
    # Header
    # ========================================================

    gr.Markdown(
        """
# 🤖 Ollama + MCP IoT Client

OllamaのローカルLLMからMCP経由でIoTデバイスを操作します。
"""
    )

    # ========================================================
    # Main
    # ========================================================

    with gr.Row():

        # ====================================================
        # Chat
        # ====================================================

        with gr.Column(
            scale=2
        ):

            chatbot = gr.Chatbot(
                label="Chat",
                height=600,
            )

            user_input = gr.Textbox(
                label="メッセージ",
                placeholder=(
                    "例：現在の温度を確認してください"
                ),
                lines=2,
            )

            with gr.Row():

                send_button = gr.Button(
                    "送信",
                    variant="primary",
                )

                clear_button = gr.Button(
                    "Clear",
                )

        # ====================================================
        # Side Panel
        # ====================================================

        with gr.Column(
            scale=1
        ):

            # ------------------------------------------------
            # Connection
            # ------------------------------------------------

            gr.Markdown(
                "## ⚙️ 接続情報"
            )

            gr.Markdown(
                f"""
**Ollama**

`{OLLAMA_HOST}`

**Model**

`{OLLAMA_MODEL}`

**MCP Server**

`{MCP_URL}`
"""
            )

            # ------------------------------------------------
            # Tools
            # ------------------------------------------------

            gr.Markdown(
                "## 🔧 MCP Tools"
            )

            tools_button = gr.Button(
                "MCP Toolsを取得"
            )

            tools_output = gr.Markdown(
                "まだ取得していません。"
            )

            # ------------------------------------------------
            # Debug
            # ------------------------------------------------

            gr.Markdown(
                "## 📝 MCP Debug Log"
            )

            debug_output = gr.Markdown(
                ""
            )

    # ========================================================
    # Events
    # ========================================================

    # --------------------------------------------------------
    # Send Button
    # --------------------------------------------------------

    send_button.click(
        fn=chat,
        inputs=[
            user_input,
            chatbot,
            debug_output,
        ],
        outputs=[
            chatbot,
            debug_output,
            user_input,
        ],
    )

    # --------------------------------------------------------
    # Enter Key
    # --------------------------------------------------------

    user_input.submit(
        fn=chat,
        inputs=[
            user_input,
            chatbot,
            debug_output,
        ],
        outputs=[
            chatbot,
            debug_output,
            user_input,
        ],
    )

    # --------------------------------------------------------
    # MCP Tools
    # --------------------------------------------------------

    tools_button.click(
        fn=get_tools,
        inputs=[],
        outputs=[
            tools_output,
        ],
    )

    # --------------------------------------------------------
    # Clear
    # --------------------------------------------------------

    clear_button.click(
        fn=clear_chat,
        inputs=[],
        outputs=[
            chatbot,
            debug_output,
            user_input,
        ],
    )


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":

    print(
        "========================================"
    )

    print(
        "     Ollama + MCP IoT GUI Client"
    )

    print(
        "========================================"
    )

    print(
        f"Ollama : {OLLAMA_HOST}"
    )

    print(
        f"Model  : {OLLAMA_MODEL}"
    )

    print(
        f"MCP    : {MCP_URL}"
    )

    print(
        "========================================"
    )

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
    )

