
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
#os.getenvは環境変数から値を取得。MCP_URLに値が設定されていないかチェックし、設定されていなければ、「http://100.86.227.30:8001/mcp」を使用
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
#Ollamaと通信するためのクライアントオブジェクトを作成している。
#今回はこいつを使って、LLMにメッセージを送り、応答を取得している。
ollama_client = ollama.Client(host=OLLAMA_HOST)


# ============================================================
# MCP tools -> Ollama tools
# ============================================================

def mcp_tools_to_ollama_tools(
    mcp_tools: list[Any],#list[Any]は型注釈で、任意の型要素を持つリスト
) -> list[dict[str, Any]]:#矢印は戻りの型を示す。省略しても良い
    """MCPのツール定義をOllamaの形式に変換する。"""

    tools = []

    for tool in mcp_tools:
        #toolから引数の使用を取得し、取得できなかった場合はデフォルト値を設定する処理
        schema = (
            #toolにinput_schemaが存在すればその値を取得する
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
        #getattr()は、オブジェクトから指定した属性の値を取得する関数
        #getattr(オブジェクト, "属性名", デフォルト値)
        text = getattr(content, "text", None)

        if text is not None:
            parts.append(text)
        else:
            parts.append(str(content))

    if not parts:
        #json.dumpsはpythonのデータをjson形式文字列に変換する関数
        return json.dumps(
            result,#json形式の対象
            ensure_ascii=False,
            default=str,
        )
    #.join()は、リストなどに含まれる文字列を、指定した文字列で繋げるメソッド
    return "\n".join(parts)


# ============================================================
# Sampling callback
# ============================================================
#async非同期関数として定義する。通信待ちがある処理を、プログラムを止めずに扱うために使用する。
async def handle_sampling(
    context: ClientRequestContext,
    #paramsにはサーバから送られたsampling要求の情報が入る
    params: CreateMessageRequestParams,
) -> CreateMessageResult:
    """MCPサーバーからのSampling要求をOllamaで処理する。"""
    #ollamaに渡すメッセージを格納するリストを作成
    #messagesという空リストを用意し、型注釈をつける
    messages: list[dict[str, Any]] = []

    # サーバーからsystem_promptが指定された場合のみ使用する。
    # これあるか確認。
    if params.system_prompt:
        messages.append({
            "role": "system",
            "content": params.system_prompt,
        })

    # サーバーから届いたSamplingメッセージをそのまま渡す。
    for message in params.messages:
        content = message.content
        #isinstance(content, TextContent)は、contentがTextContentならtrueを返す
        if isinstance(content, TextContent):
            text = content.text
        else:
            text = str(content)
        #roleを確認message.roleオブ弱とのvalue属性を取得できるならその値を使う
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
    #asyncio.to_thread()に、関数と、引数を渡し、awaitで処理を待つ。
    response = await asyncio.to_thread(
        ollama_client.chat,
        #以下はollama_client.chatの引数
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": OLLAMA_TEMPERATURE,
            "num_predict": max_tokens,
        },
    )
    #ollamaからのレスポンスを取り出す。
    answer = response.message.content or ""
    #repr()は¥n等の特殊文字もそのまま出してくれる
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
    client: Client,#mcpクライアントを受け取る
    user_text: str,
) -> str:
    """ユーザー入力をOllamaに渡し、必要に応じてMCPツールを実行する。"""
    #mcpサーバに問い合わせて、利用可能なツールの一覧を取得（毎回toolsをロードしてそう）
    listed = await client.list_tools()
    tools = mcp_tools_to_ollama_tools(listed.tools)
    #messagesっていうからリストを作成し、user_textを入れる
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": user_text,
        },
    ]

    for _ in range(8):#「_」は繰り替えしの回数が使われない時に使用する
        response = await asyncio.to_thread(
            ollama_client.chat,
            model=OLLAMA_MODEL,
            messages=messages,
            tools=tools,
            options={
                "temperature": OLLAMA_TEMPERATURE,
            },
        )
        #ollamaからの返答を記録
        message = response.message
        #tool情報を取得
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            return (
                message.content
                or "（LLMからテキスト応答がありませんでした）"
            )
        messages.append({
            "role": "assistant",
            #LLMが生成した文書が入る
            "content": message.content or "",
            "tool_calls": [
                {
                    "function": {
                        #呼び出すツールの名前
                        "name": tool_call.function.name,
                        #呼び出すツールの引数
                        "arguments": tool_call.function.arguments,
                    }
                }
                #リスト内包表記のfor
                for tool_call in tool_calls
            ],
        })
        #ここで、実際にtool_callを実行
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
            #ここでツールを実際に実行する
            try:
                #mcpクライアントを通して、指定したツールの実行を依頼
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
    #MCPサーバに接続
    async with Client(
        MCP_URL,
        #通信モードの設定
        mode="legacy",
        #samplingを受け取った場合に呼び出す関数
        sampling_callback=handle_sampling,
    ) as client:
        #利用可能ツールの取得
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
            #.strip()は文字列の先頭と末尾にある空白や改行を取り除く
            user_text = user_text.strip()

            if not user_text:
                continue
            #文字列に含まれる大文字を全て小文字に.lower()
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
