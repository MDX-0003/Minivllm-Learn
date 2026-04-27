# mcp_server/ 代码详解

> 面向新手的完整解读。建议配合源码对照阅读。

---

## 目录

1. [总体架构](#1-总体架构)
2. [pyproject.toml — 入口点定义](#2-pyprojecttoml--入口点定义)
3. [config.py — 配置读取](#3-configpy--配置读取)
4. [server.py — MCP 层与工具定义](#4-serverpy--mcp-层与工具定义)
5. [bridge/client.py — 跨进程通信客户端](#5-bridgeclientpy--跨进程通信客户端)
6. [与外部进程（RenderDoc 扩展）的通信全流程](#6-与外部进程renderdoc-扩展的通信全流程)
7. [附录 A：客户端 / 服务端基础概念](#附录-a客户端--服务端基础概念)
8. [附录 B：MCP 协议报文格式](#附录-bmcp-协议报文格式)

---

## 1. 总体架构

### 1.1 文件结构

```
mcp_server/
├── __init__.py          # 空文件，声明这是一个 Python 包
├── config.py            # 从环境变量读取配置
├── server.py            # 核心：工具定义 + 启动入口
└── bridge/
    ├── __init__.py      # 空文件，声明子包
    └── client.py        # 跨进程通信客户端（文件 IPC）
```

### 1.2 三个"角色"与两条通信链路

```
┌─────────────────────────────────────────────────────────────────┐
│  角色 A：AI 客户端（Claude Desktop / Claude Code）               │
│  运行在：你的电脑上                                              │
│  职责：把你的自然语言翻译成工具调用，展示结果                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                  通信链路 ①：stdio（标准输入输出）
                  协议：MCP 协议（JSON-RPC 2.0 变体）
                  谁负责：FastMCP 框架自动处理
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  角色 B：MCP Server（本目录，mcp_server/）                       │
│  运行在：你的电脑上，独立进程                                     │
│  职责：①向 Claude 暴露工具；②把调用转发给 RenderDoc             │
│  文件：server.py + bridge/client.py                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                  通信链路 ②：文件 IPC
                  路径：%TEMP%/renderdoc_mcp/
                  协议：手写 JSON（request.json / response.json）
                  谁负责：bridge/client.py（写） + socket_server.py（读）
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  角色 C：RenderDoc 扩展（renderdoc_extension/）                  │
│  运行在：RenderDoc 进程内部                                       │
│  职责：执行真实的 RenderDoc API 调用，返回图形数据               │
└─────────────────────────────────────────────────────────────────┘
```

**关键认知**：`mcp_server/` 自己完全不懂 RenderDoc，它只是一个"翻译 + 传话"的中间层。

### 1.3 数据流向（一次完整的工具调用）

```
你对 Claude 说："查一下事件 42 的像素着色器"
    │
    ▼ Claude 判断：应该调用 get_shader_info(event_id=42, stage="pixel")
    │
    ▼ MCP 协议报文（JSON）通过 stdin 发到 MCP Server 进程
    │
    ▼ server.py: 装饰器mcp.tool的get_shader_info(42, "pixel") 函数被调用
    │
    ▼ bridge.call("get_shader_info", {"event_id": 42, "stage": "pixel"})
    │
    ▼ 写文件: %TEMP%/renderdoc_mcp/request.json
    │
    │ ←（等待，最多 30 秒）
    │
    ▼ RenderDoc 扩展每 100ms 轮询到文件，执行 API，写 response.json
    │
    ▼ bridge.call() 读 response.json，删除文件，返回结果
    │
    ▼ get_shader_info() 把结果返回给 FastMCP
    │
    ▼ FastMCP 把结果序列化成 MCP 协议报文，通过 stdout 发回 Claude
    │
    ▼ Claude 读到数据，用自然语言告诉你结果
```
我说“分析第42个drawcall”，llm会帮我理解，翻译其为每个mcp.tool函数的参数，并发送给bridge，bridge负责填写请求到json。至于用哪个tool，写什么参数，是云端大模型推理得到的。
```
你说："分析第 42 个 drawcall"
         │
         ▼  ← 这一步完全在 Anthropic 云端，你看不到
    Claude 大模型推理：
    "用户想分析一个 drawcall，我有这些工具：
     - get_draw_call_details(event_id)     ← 适合
     - get_shader_info(event_id, stage)    ← 也许也要用
     - get_pipeline_state(event_id)        ← 也许也要用
    我先调用 get_draw_call_details(event_id=42)"
         │
         ▼  Claude 生成一条 MCP 协议消息，发给本地的 MCP Server：
    {"method": "tools/call", "params": {"name": "get_draw_call_details", "arguments": {"event_id": 42}}}
         │
         ▼  FastMCP 框架解包，调用 Python 函数：
    server.py: get_draw_call_details(event_id=42)
         │
         ▼  函数体只有一行：
    bridge.call("get_draw_call_details", {"event_id": 42})
         │
         ▼  client.py 把字典序列化写文件：
    request.json = {"id": "uuid...", "method": "get_draw_call_details", "params": {"event_id": 42}}
         │
         ▼  RenderDoc 扩展读到文件，执行 API，写 response.json
         │
         ▼  结果回传给 Claude
    Claude 看到结果后，可能再调用 get_shader_info(42, "pixel")、get_pipeline_state(42)...
    最终 Claude 把所有结果整理成自然语言告诉你
```

---

## 2. `pyproject.toml` — 入口点定义

> 文件路径：`pyproject.toml`（项目根目录，不在 mcp_server/ 内）

这个文件决定了 `renderdoc-mcp` 命令是怎么来的，是理解整个启动机制的起点。

```toml
[project.scripts]
renderdoc-mcp = "mcp_server.server:main"
#     ↑               ↑         ↑
#  命令名称        Python 模块路径  模块内的函数名
```

**解读**：

`uv tool install .` 执行后，这一行让包管理工具在系统 PATH 里生成了一个叫 `renderdoc-mcp` 的可执行脚本。它的实际效果等价于：

```bash
python -c "from mcp_server.server import main; main()"
```

Claude Desktop 在 `claude_desktop_config.json` 里读到 `"command": "renderdoc-mcp"` 后，就在后台执行这个命令来拉起 MCP Server 进程，并接管它的 stdin/stdout 作为通信管道。

**整个命令链**：
```
claude_desktop_config.json
  └─▶ "command": "renderdoc-mcp"
        └─▶ pyproject.toml [project.scripts]
              └─▶ mcp_server.server:main (server.py里的main)
                    └─▶ mcp.run()  ← 进程在此阻塞，等待 Claude 发指令
```

---

## 3. `config.py` — 配置读取

> 文件路径：`mcp_server/config.py`，共 14 行

```python
class Settings:
    def __init__(self):
        self.renderdoc_host = os.environ.get("RENDERDOC_MCP_HOST", "127.0.0.1")
        self.renderdoc_port = int(os.environ.get("RENDERDOC_MCP_PORT", "19876"))

settings = Settings()
```

**逐行解读**：

| 字段 | 默认值 | 来源 | 实际用途 |
|------|--------|------|---------|
| `renderdoc_host` | `"127.0.0.1"` | 环境变量 `RENDERDOC_MCP_HOST` | 传给 `RenderDocBridge.__init__()` |
| `renderdoc_port` | `19876` | 环境变量 `RENDERDOC_MCP_PORT` | 传给 `RenderDocBridge.__init__()` |

**注意**：根据 `bridge/client.py` 第 32 行的注释，`host` 和 `port` 实际上**没有被使用**（历史遗留，最初打算用 socket 通信）：

```python
# client.py 第 30-32 行
def __init__(self, host: str = "127.0.0.1", port: int = 19876):
    # host/port are kept for API compatibility but not used
    self.host = host
```

`settings` 是模块级单例，在 `server.py` 第 11 行被导入：

```python
from .config import settings
```

这个文件的存在价值：将来如果需要切换通信方式（比如换回 socket），只需改这里，不用动 `server.py`。

---

## 4. `server.py` — MCP 层与工具定义

> 文件路径：`mcp_server/server.py`，共 321 行

这是整个 `mcp_server/` 最重要的文件，承担两个职责：
1. **向 Claude 暴露工具**（16 个 `@mcp.tool` 函数）
2. **提供进程入口点**（`main()` 函数）

### 4.1 初始化阶段（第 1–19 行）

```python
from fastmcp import FastMCP                              # 第 8 行
from .bridge.client import RenderDocBridge, ...          # 第 10 行
from .config import settings                             # 第 11 行

mcp = FastMCP(name="RenderDoc MCP Server")               # 第 14-16 行
bridge = RenderDocBridge(host=settings.renderdoc_host,   # 第 19 行
                         port=settings.renderdoc_port)
```

这三行建立了两个模块级全局对象：

- **`mcp`**：FastMCP 实例，相当于整个 MCP Server 的"主机"。它管理工具注册表，处理 Claude 发来的 JSON 报文，调用对应的 Python 函数。
- **`bridge`**：`RenderDocBridge` 实例，跨进程通信的客户端。16 个工具函数全部通过它与 RenderDoc 通信。这个class定义在bridge.client里

### 4.2 `@mcp.tool` 装饰器机制

每个用 `@mcp.tool` 装饰的函数都会被 FastMCP 完成以下处理：

**步骤 1**：读取函数签名的**类型注解**，生成 JSON Schema：

```python
# 这个函数签名：
def get_shader_info(
    event_id: int,
    stage: Literal["vertex", "hull", "domain", "geometry", "pixel", "compute"],
) -> dict:

# 被转换成这样的 JSON Schema（Claude 用这个来填参数）：
{
  "name": "get_shader_info",
  "inputSchema": {
    "type": "object",
    "properties": {
      "event_id": {"type": "integer"},
      "stage": {"type": "string", "enum": ["vertex", "hull", ...]}
    },
    "required": ["event_id", "stage"]
  }
}
```

**步骤 2**：读取**文档字符串（docstring）**，作为工具描述发给 Claude：

```python
def get_shader_info(...):
    """
    Get shader information for a specific stage at a given event.
    ...         ↑ 这整段文字会原封不动发给 Claude，Claude 靠它决定何时调用此工具
    """
```

**步骤 3**：MCP Server 启动时，把所有工具的描述通过 MCP 协议发给 Claude，完成"工具注册"。

### 4.3 16 个工具函数详解

所有工具函数的结构是统一的：**①收集参数 → ②过滤 None 值 → ③调用 `bridge.call()`**

#### 分组一：捕获管理类（2 个）

| 函数名 | 行号 | 参数 | bridge.call() 的方法名 |
|--------|------|------|----------------------|
| `list_captures` | 第 272 行 | `directory: str` | `"list_captures"` |
| `open_capture` | 第 289 行 | `capture_path: str` | `"open_capture"` |

**`list_captures`**（第 272–285 行）：列出指定目录下所有 `.rdc` 文件，无特殊参数处理，直接转发：
```python
return bridge.call("list_captures", {"directory": directory})
```

**`open_capture`**（第 289–303 行）：打开一个 `.rdc` 文件，注意文档中写明"会关闭当前已打开的捕获"，这是 RenderDoc API 的行为，MCP Server 层不做任何处理：
```python
return bridge.call("open_capture", {"capture_path": capture_path})
```

#### 分组二：帧数据查询类（4 个）

| 函数名 | 行号 | 说明 |
|--------|------|------|
| `get_capture_status` | 第 22 行 | 无参数，最简单的工具 |
| `get_frame_summary` | 第 75 行 | 无参数，返回全帧统计 |
| `get_draw_calls` | 第 32 行 | 7 个可选过滤参数 |
| `get_draw_call_details` | 第 130 行 | 必填 `event_id` |

**`get_draw_calls`**（第 32–68 行）是参数最复杂的工具，展示了 None 值过滤模式：

```python
# 第 56–68 行
params: dict[str, object] = {"include_children": include_children}
if marker_filter is not None:           # 只把非 None 的参数放进字典
    params["marker_filter"] = marker_filter
if exclude_markers is not None:
    params["exclude_markers"] = exclude_markers
# ... 依此类推
return bridge.call("get_draw_calls", params)
```

这样做的原因：接收端（`request_handler.py`）用 `params.get("marker_filter")` 取值，如果键不存在返回 `None`。发送端不传 `None` 值，等价于"使用默认值"，避免歧义。

#### 分组三：逆向搜索类（3 个）

| 函数名 | 行号 | 搜索方式 |
|--------|------|---------|
| `find_draws_by_shader` | 第 84 行 | 着色器名称（部分匹配）|
| `find_draws_by_texture` | 第 101 行 | 纹理名称（部分匹配）|
| `find_draws_by_resource` | 第 114 行 | 资源 ID（精确匹配）|

**`find_draws_by_shader`**（第 84–97 行）有一个可选的 `stage` 参数，使用 `Literal` 枚举限制合法值：

```python
stage: Literal["vertex", "hull", "domain", "geometry", "pixel", "compute"] | None = None
```

`Literal` 的作用：FastMCP 会把它转成 JSON Schema 的 `enum` 字段，Claude 只能填这几个字符串之一，避免拼写错误传到后端。

#### 分组四：性能分析类（1 个）

| 函数名 | 行号 | 说明 |
|--------|------|------|
| `get_action_timings` | 第 140 行 | 获取 GPU 执行耗时 |

**`get_action_timings`**（第 140–176 行）文档字符串里有一个重要提示：

```python
"""
Note: GPU timing counters may not be available on all hardware/drivers.
"""
```

MCP Server 层不处理这个情况，直接转发，具体的"不支持"错误由 RenderDoc 扩展层返回。

#### 分组五：资源数据类（5 个）

| 函数名 | 行号 | 返回数据类型 |
|--------|------|-------------|
| `get_shader_info` | 第 179 行 | 着色器反汇编、常量缓冲区 |
| `get_buffer_contents` | 第 196 行 | Base64 编码的原始字节 |
| `get_texture_info` | 第 215 行 | 纹理元数据（不含像素数据）|
| `get_texture_data` | 第 229 行 | Base64 编码的像素数据 |
| `get_pipeline_state` | 第 257 行 | 完整管线状态 |

**`get_texture_data`**（第 229–256 行）是参数最多的资源类工具：

```python
def get_texture_data(
    resource_id: str,
    mip: int = 0,           # Mip 层级
    slice: int = 0,         # 数组切片或 Cube 面
    sample: int = 0,        # MSAA 采样索引
    depth_slice: int | None = None,  # 仅 3D 纹理用
) -> dict:
```

注意 `depth_slice` 的处理方式（第 252–254 行）：只有非 `None` 时才加入参数字典，`None` 表示"返回完整 3D 体积数据"。

### 4.4 `main()` 函数（第 307–309 行）

```python
def main():
    """Run the MCP server"""
    mcp.run()
```

这是整个进程的入口，三件事由 `mcp.run()` 自动完成：

1. 把所有已注册的 `@mcp.tool` 工具描述序列化，通过 stdout 发给 Claude（"工具注册握手"）
2. 进入事件循环，阻塞读取 stdin
3. 收到 Claude 的工具调用请求 → 找到对应函数 → 执行 → 把返回值序列化 → 写回 stdout

进程在 `mcp.run()` 这里永久阻塞，直到 Claude 关闭连接（stdin 关闭）才退出。

---

## 5. `bridge/client.py` — 跨进程通信客户端

> 文件路径：`mcp_server/bridge/client.py`，共 95 行

这个文件是 MCP Server 与 RenderDoc 之间通信的全部实现，和 MCP 协议完全无关。

### 5.1 IPC 路径常量（第 14–18 行）

```python
IPC_DIR       = os.path.join(tempfile.gettempdir(), "renderdoc_mcp")
REQUEST_FILE  = os.path.join(IPC_DIR, "request.json")
RESPONSE_FILE = os.path.join(IPC_DIR, "response.json")
LOCK_FILE     = os.path.join(IPC_DIR, "lock")
```

在 Windows 上，`tempfile.gettempdir()` 通常返回 `C:\Users\你的用户名\AppData\Local\Temp`。

> **为什么用临时目录？** 系统保证当前用户有读写权限，且重启后自动清理。

这四个路径在 `renderdoc_extension/socket_server.py` 里有完全相同的定义（第 14–18 行），这是两个进程"约定好的暗语"——通信协议的核心就是这个共享目录。

### 5.2 `RenderDocBridgeError`（第 21–23 行）

```python
class RenderDocBridgeError(Exception):
    """Error communicating with RenderDoc bridge"""
    pass
```

自定义异常类，唯一目的：让调用方可以精确地 `except RenderDocBridgeError` 捕获通信错误，而不是捕获所有 `Exception`。在 `server.py` 里它被导入但没有被显式捕获——FastMCP 框架会自动把未捕获的异常转换成 MCP 协议的错误响应发回 Claude。

### 5.3 `RenderDocBridge.__init__()`（第 28–32 行）

```python
def __init__(self, host: str = "127.0.0.1", port: int = 19876):
    # host/port are kept for API compatibility but not used
    self.host = host
    self.port = port
    self.timeout = 30.0  # seconds
```

- `host` / `port`：保留字段，仅在错误信息里使用（第 41 行），实际通信不依赖它们
- `self.timeout = 30.0`：等待响应的最大秒数，硬编码，不可配置

### 5.4 `call()` 方法全流程（第 34–95 行）

这是文件的核心，完整实现了一次请求-响应的全过程：

#### 步骤 0：检查 RenderDoc 是否在运行（第 36–42 行）

```python
if not os.path.exists(IPC_DIR):
    raise RenderDocBridgeError(
        f"Cannot connect to RenderDoc MCP Bridge at {self.host}:{self.port}. "
        "Make sure RenderDoc is running with the MCP Bridge extension loaded."
    )
```

IPC 目录由 RenderDoc 扩展在启动时创建（`socket_server.py` 第 31–32 行）。目录不存在 = RenderDoc 没运行，或扩展没启用。此时直接报错，不进行后续操作。

#### 步骤 1：构造请求 JSON（第 44–48 行）

```python
request = {
    "id": str(uuid.uuid4()),   # 唯一请求 ID，e.g. "550e8400-e29b-41d4-a716..."
    "method": method,           # e.g. "get_shader_info"
    "params": params or {},     # e.g. {"event_id": 42, "stage": "pixel"}
}
```

`uuid.uuid4()` 生成随机 UUID，确保每次请求 ID 不重复。理论上响应的 `id` 字段应该和请求匹配，但当前代码没有校验这一点（因为同时只有一个请求在飞行，文件系统本身提供了串行化）。

#### 步骤 2：清理残留响应文件（第 52–53 行）

```python
if os.path.exists(RESPONSE_FILE):
    os.remove(RESPONSE_FILE)
```

如果上一次请求的响应文件还没被清理（异常退出等情况），先删掉，防止读到过期数据。

#### 步骤 3：写锁 + 写请求（第 55–65 行）

```python
# 先写锁文件，告诉 RenderDoc 扩展"我正在写，别读"
with open(LOCK_FILE, "w") as f:
    f.write("lock")

# 写请求文件
with open(REQUEST_FILE, "w", encoding="utf-8") as f:
    json.dump(request, f)

# 删锁文件，告诉 RenderDoc 扩展"写完了，可以读了"
os.remove(LOCK_FILE)
```

**为什么需要锁文件？** 文件写入不是原子操作——写到一半时，对方如果读到不完整的 JSON 会解析失败。锁文件是一个信号量：它存在 = 正在写 = 禁止读；它消失 = 写完 = 可以读。

这是一种极简的互斥机制，适用于"同一时刻只有一个写方"的场景。

#### 步骤 4：轮询等待响应（第 67–90 行）

```python
start_time = time.time()
while True:
    if os.path.exists(RESPONSE_FILE):
        time.sleep(0.01)          # 额外等 10ms，确保文件写完（对方没有锁机制）

        with open(RESPONSE_FILE, "r", encoding="utf-8") as f:
            response = json.load(f)

        os.remove(RESPONSE_FILE)  # 读完即删，保持目录干净

        if "error" in response:
            error = response["error"]
            raise RenderDocBridgeError(f"[{error['code']}] {error['message']}")

        return response.get("result")

    if time.time() - start_time > self.timeout:   # 超时检查
        raise RenderDocBridgeError("Request timed out")

    time.sleep(0.05)   # 每 50ms 检查一次
```

**轮询间隔 50ms** vs **RenderDoc 扩展轮询间隔 100ms**：两端轮询频率不同，因为 MCP Server 端是普通 Python 线程，可以更频繁；RenderDoc 端用 Qt 主线程定时器，100ms 是为了不影响 UI 响应。

**额外等待 10ms（第 69 行）的原因**：响应端没有写锁机制（`socket_server.py` 直接 `json.dump` 写文件），10ms 的缓冲是为了应对文件系统的写入延迟，确保文件完全落盘后再读。这是一个经验值，非严格保证。

**错误响应格式**（来自 `request_handler.py` 的 `_error_response()`）：

```json
{
  "id": "550e8400-...",
  "error": {
    "code": -32000,
    "message": "No capture loaded"
  }
}
```

错误码约定来自 JSON-RPC 2.0 规范（`-32601` = 方法不存在，`-32602` = 参数无效，`-32000` = 应用层错误）。

---

## 6. 与外部进程（RenderDoc 扩展）的通信全流程

下面是一次完整调用从发出到返回的**时序图**，精确到函数级别：

```
MCP Server 进程                          RenderDoc 进程
(mcp_server/)                           (renderdoc_extension/)
     │                                          │
     │  [Claude 发来工具调用请求]               │
     │                                          │  [QTimer 每 100ms 触发]
     ▼                                          ▼
server.py: get_shader_info(42, "pixel")   socket_server.py: _poll_request()
     │                                          │
     ▼                                          │  os.path.exists(REQUEST_FILE) → False
bridge.call("get_shader_info", {...})           │  return（本轮没有请求）
     │                                          │
     ▼                                          │
client.py: os.path.exists(IPC_DIR) ✓           │
     │                                          │
     ▼                                          │
client.py: open(LOCK_FILE, "w")                │  [下一轮 QTimer 触发]
     │                                          ▼
     ▼                              socket_server.py: _poll_request()
client.py: json.dump(request,                   │
           open(REQUEST_FILE, "w"))             │  os.path.exists(REQUEST_FILE) → False
     │                              (lock存在，跳过)
     ▼                                          │
client.py: os.remove(LOCK_FILE)                │  [又一轮 QTimer 触发]
     │                                          ▼
     │                              socket_server.py: _poll_request()
     │                                          │
     │                                          │  REQUEST_FILE ✓，LOCK_FILE ✗
     │                                          ▼
     │                              request = json.load(REQUEST_FILE)
     │                              os.remove(REQUEST_FILE)
     │                                          │
     │                                          ▼
     │                              handler.handle(request)
     │                                          │
     │                                          ▼
     │                              facade.get_shader_info(42, "pixel")
     │                                          │
     │                                          ▼
     │                              BlockInvoke → controller.GetPipelineState()
     │                              （RenderDoc 原生 API，replay 线程）
     │                                          │
     │                                          ▼
     │                              json.dump(response, open(RESPONSE_FILE, "w"))
     │                                          │
     ▼                                          │
client.py: os.path.exists(RESPONSE_FILE) ✓     │
     │                                          │
     ▼                                          │
time.sleep(0.01)  # 等文件写完               │
response = json.load(RESPONSE_FILE)            │
os.remove(RESPONSE_FILE)                       │
return response["result"]                      │
     │                                          │
     ▼
server.py: get_shader_info() 返回结果
     │
     ▼
FastMCP: 序列化成 MCP 协议报文，写回 stdout
     │
     ▼
Claude 收到数据
```

---

## 附录 A：客户端 / 服务端基础概念

### A.1 客户端与服务端

**服务端（Server）**：一直运行着，等待别人来找它，提供服务。  
**客户端（Client）**：需要时主动去找服务端，发请求，拿结果。

类比：餐厅（服务端）一直开着门，顾客（客户端）来了才点餐。

在本项目中，角色分配很有趣——有**三层**客户端/服务端关系：

| 关系 | 客户端 | 服务端 |
|------|--------|--------|
| 第 1 层 | Claude AI | MCP Server（`mcp_server/`）|
| 第 2 层 | MCP Server（`bridge/client.py`）| RenderDoc 扩展（`socket_server.py`）|

MCP Server 在第 1 层是服务端，在第 2 层是客户端——它是个"中间人"。

### A.2 IPC（进程间通信）

**IPC = Inter-Process Communication**，进程间通信。

同一台电脑上不同的程序（进程）默认是互相隔离的，无法直接访问对方的内存。IPC 是让它们"对话"的机制。常见方式：

| 方式 | 原理 | 本项目使用？ |
|------|------|------------|
| **Socket** | 通过网络协议栈通信，可跨机器 | 否（RenderDoc 内置 Python 没有 socket 模块） |
| **文件** | 双方读写同一个文件 | **是**（request.json / response.json）|
| **管道（Pipe）** | 操作系统提供的字节流通道 | 是（MCP 协议用 stdio，本质是管道） |
| **共享内存** | 双方映射同一块内存区域 | 否 |

### A.3 Socket 是什么

Socket 是网络通信的"插座"，让两个程序可以像打电话一样互发数据。即使在同一台机器上，也可以用 `127.0.0.1`（本机地址）通过 socket 通信。

这个项目原本打算用 socket（从 `MCPBridgeServer` 的命名、`host/port` 参数可以看出），但因为 RenderDoc 内置的 Python 解释器裁剪了 socket 模块，最终改成了文件 IPC。名字没有跟着改。

### A.4 stdio（标准输入输出）

每个命令行程序默认有三个"管道"：
- **stdin**（标准输入）：程序从这里读数据（对应键盘输入）
- **stdout**（标准输出）：程序把数据写到这里（对应终端显示）
- **stderr**（标准错误）：程序把错误信息写到这里

Claude Desktop 启动 MCP Server 进程后，把自己的输出接到 MCP Server 的 stdin，把 MCP Server 的输出接到自己的输入，形成双向通信管道——就像两根水管对接起来。MCP 协议的全部报文就在这对管道里流动，不需要网络，不需要端口。

---

## 附录 B：MCP 协议报文格式

MCP 协议基于 JSON-RPC 2.0，通过 stdio 传输。每条消息是一行 JSON，以换行符分隔。

### B.1 Claude 发给 MCP Server 的工具调用请求

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_shader_info",
    "arguments": {
      "event_id": 42,
      "stage": "pixel"
    }
  }
}
```

### B.2 MCP Server 返回给 Claude 的成功响应

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"source\": \"...\", \"constants\": {...}}"
      }
    ]
  }
}
```

### B.3 MCP Server 与 RenderDoc 扩展之间的文件 IPC 格式

这是项目**自定义**的简化 JSON-RPC（非标准）：

**request.json**：
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "get_shader_info",
  "params": {
    "event_id": 42,
    "stage": "pixel"
  }
}
```

**response.json（成功）**：
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "result": {
    "source": "...",
    "constants": {}
  }
}
```

**response.json（失败）**：
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "error": {
    "code": -32000,
    "message": "No capture loaded"
  }
}
```

注意：这两层 JSON 格式是**独立设计**的，第一层（MCP 协议）由 FastMCP 框架自动处理，开发者只需关心第二层（文件 IPC 格式）。
