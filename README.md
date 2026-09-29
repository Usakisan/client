# IoT MCP Proxy + Ollama 実験環境

卒業研究「MCPサーバーによる仮想パッチ（Virtual Patching）の有効性検証」用の実験環境です。

既存の `server.py` は変更せず、Ollamaで動作するローカルLLMとPython製MCP Clientを追加します。

## 構成

```text
                    Raspberry Pi / PC

┌──────────────────────────────────────────────┐
│                                              │
│  Ollama                                      │
│  └─ qwen2.5:7b                               │
│          ↑                                   │
│          │ Ollama API                        │
│          ↓                                   │
│  ollama_mcp_client.py                        │
│          │                                   │
│          │ MCP / Streamable HTTP             │
│          ↓                                   │
│  MCP Proxy (既存 server.py)                  │
│  100.86.227.30:8000/mcp                      │
│          │                                   │
│          │ HTTP                              │
│          ↓                                   │
│  脆弱なIoTデバイスAPI                         │
│  100.101.199.62:8000                         │
│                                              │
└──────────────────────────────────────────────┘
```

役割を分離しています。

| コンポーネント | 役割 |
|---|---|
| Ollama | ローカルLLMを実行 |
| `ollama_mcp_client.py` | LLMとMCPサーバーを接続 |
| 既存 `server.py` | MCP Toolを提供し、IoT APIへ転送 |
| IoT Device API | 実験対象の脆弱なデバイスAPI |

## ディレクトリ

```text
ollama_mcp_client.py
requirements.txt
.env.example
README.md
```

このフォルダを、既存の `server.py` がある `proxy/` ディレクトリに置いても構いません。

その場合の推奨構成は次の通りです。

```text
proxy/
├── server.py              # 既存。変更しない
├── ollama_mcp_client.py   # 今回追加
├── requirements.txt
└── README.md
```

## 1. Ollamaの準備

OllamaをインストールしたPC/Raspberry Piで、使用するモデルを取得します。

例:

```bash
ollama pull qwen2.5:7b
```

確認:

```bash
ollama list
```

Ollama APIが動作していることを確認します。

```bash
curl http://127.0.0.1:11434/api/tags
```

## 2. Python環境

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windowsの場合:

```powershell
venv\Scripts\activate
pip install -r requirements.txt
```

## 3. MCP Proxyを起動

既存の `server.py` をそのまま使用します。

```bash
python server.py
```

現在の設定では、MCPエンドポイントは:

```text
http://100.86.227.30:8000/mcp
```

です。

## 4. Ollama MCP Clientを起動

```bash
python ollama_mcp_client.py
```

起動すると、MCPサーバーからTool一覧を取得します。

例:

```text
=== Ollama + MCP IoT Client ===
Ollama: http://127.0.0.1:11434
Model : qwen2.5:7b
MCP   : http://100.86.227.30:8000/mcp

利用可能なMCP Tools:
  - get_status
  - set_threshold
  - set_location
  - get_logs
  - get_history
  - calibrate
```

## 5. 実験する

例えば:

```text
You > センサーの現在の温度と湿度を取得してください
```

LLMが `get_status` をToolとして選択すると、以下のように処理されます。

```text
ユーザー
   ↓
Ollama
   ↓ tool_call: get_status
Python MCP Client
   ↓ MCP
MCP Proxy
   ↓ GET /status
IoT Device
```

実行時にはTool呼び出しと引数を表示します。

```text
[MCP TOOL] get_status
[ARGUMENTS] {}
[RESULT] ...
```

## 6. 引数を持つTool

例えば、次のような要求ができます。

```text
You > 警告温度を30度に設定してください
```

LLMが次のようなTool呼び出しを生成します。

```text
set_threshold
{
  "value": 30
}
```

Python MCP ClientはこれをMCP Proxyへ渡します。

```text
Ollama
  ↓
set_threshold(value=30)
  ↓
MCP Proxy
  ↓
POST /set_threshold
  ↓
IoT Device
```

## 7. 環境変数

デフォルト設定:

```text
MCP_URL=http://100.86.227.30:8000/mcp
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b
```

変更する場合:

```bash
export MCP_URL=http://100.86.227.30:8000/mcp
export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_MODEL=qwen2.5:7b
python ollama_mcp_client.py
```

別のモデルを使う場合:

```bash
ollama pull qwen2.5:3b
export OLLAMA_MODEL=qwen2.5:3b
python ollama_mcp_client.py
```

## 8. 現在のTool

既存 `server.py` が提供するToolを、そのままLLMへ渡します。

| Tool | IoT API |
|---|---|
| `get_status` | `GET /status` |
| `set_threshold(value)` | `POST /set_threshold` |
| `set_location(label)` | `POST /set_location` |
| `get_logs(file)` | `GET /logs?file=` |
| `get_history(keyword)` | `GET /history?keyword=` |
| `calibrate(offset)` | `POST /calibrate` |

このクライアント自身では、これらの引数をセキュリティ検証しません。

そのため、現在は「Ollama + 素朴なMCP Proxy」の実験環境です。

## 9. 研究実験の構成

### 条件① 直接アクセス

```text
Client / curl
     ↓ HTTP
IoT Device
```

### 条件② 素朴なMCP Proxy

```text
User
 ↓
Ollama
 ↓
Python MCP Client
 ↓ MCP
Naive MCP Proxy
 ↓ HTTP
IoT Device
```

現在このフォルダで実装しているのは、この条件です。

### 条件③ 対策済みMCP Proxy

将来的には `server.py` をコピーして、対策版を作ります。

```text
User
 ↓
Ollama
 ↓
Python MCP Client
 ↓ MCP
Virtual Patching MCP Proxy
 ├─ 入力検証
 ├─ Allowlist
 ├─ 認証
 └─ リクエスト制御
 ↓
IoT Device
```

IoT Device API側は変更しません。

## 10. Virtual Patchingの評価

同一のユーザー要求またはTool引数を、以下の条件で比較します。

```text
① 直接アクセス
② 素朴なMCP Proxy
③ 対策済みMCP Proxy
```

記録する項目の例:

- Tool呼び出し結果
- IoT HTTPステータス
- 攻撃リクエストの成功/拒否
- MCP Proxyでの拒否数
- 正常リクエストの成功率
- 処理時間

LLMを固定してProxyだけを変更すれば、MCP ProxyによるVirtual Patchingの効果を比較できます。

## 11. 重要な注意

この実験では、IoTデバイスAPIに意図的な脆弱性が存在します。

そのため、実験ネットワーク内だけで使用してください。

```text
Ollama
 ↓
MCP Proxy
 ↓
脆弱なIoT API
```

この構成をインターネットへ公開しないでください。

## 12. デバッグ

### Ollamaが動作しているか

```bash
curl http://127.0.0.1:11434/api/tags
```

### MCP Proxyへ到達できるか

```bash
curl -i http://100.86.227.30:8000/mcp
```

### IoT Deviceへ直接到達できるか

```bash
curl http://100.101.199.62:8000/status
```

### MCP Clientを起動

```bash
python ollama_mcp_client.py
```

起動時にTool一覧が表示されれば、MCP Proxyとの接続は確立しています。

## 13. 実験全体

最終的には次の構成で実験できます。

```text
                  ┌──────────────────┐
                  │     Ollama       │
                  │  Local LLM       │
                  └────────┬─────────┘
                           │
                           │ Tool Calling
                           ▼
                  ┌──────────────────┐
                  │ Python MCP       │
                  │ Client           │
                  └────────┬─────────┘
                           │
                           │ MCP
                           ▼
                  ┌──────────────────┐
                  │ MCP Proxy        │
                  │                  │
                  │ ① Naive          │
                  │ ② Virtual Patch  │
                  └────────┬─────────┘
                           │
                           │ HTTP
                           ▼
                  ┌──────────────────┐
                  │ Vulnerable IoT   │
                  │ Device API       │
                  └──────────────────┘
```

この構成により、LLMをローカルに置いたまま、MCP Proxyを境界としてVirtual Patchingの有効性を評価できます。
