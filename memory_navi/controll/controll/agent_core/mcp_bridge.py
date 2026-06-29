"""MCP ↔ 同步世界 的桥。

官方 `mcp` SDK 是 async，且 streamable-http 的 ClientSession 需在其生命周期内保持打开。
做法：起一条守护线程跑专用 asyncio loop，在 loop 上开一个常驻 ClientSession，
其余（同步的）Supervisor/skills/记忆作者通过 run_coroutine_threadsafe 阻塞式调用。
这样保持一条热连接（不每次握手），又给上层一个干净的同步接口。

对外：
  - RosTools.list_openai_tools(allowlist) -> OpenAI tools=[] schema
  - RosTools.call(name, args) -> ToolOutput(text, images, is_error, raw)
  - RosTools.connect(ip, port) -> 便捷调用 connect_to_robot
"""
import asyncio
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .image_utils import ImagePart


@dataclass
class ToolOutput:
    """归一化后的 MCP 工具结果。"""
    text: str = ""
    images: list = field(default_factory=list)  # list[ImagePart]
    is_error: bool = False
    raw: object = None


def _normalize(result) -> ToolOutput:
    """CallToolResult.content（TextContent/ImageContent...）-> ToolOutput。"""
    texts, images = [], []
    for block in (getattr(result, "content", None) or []):
        btype = getattr(block, "type", None)
        if btype == "text":
            texts.append(getattr(block, "text", ""))
        elif btype == "image":
            images.append(
                ImagePart(b64=block.data, mime=getattr(block, "mimeType", "image/jpeg"))
            )
    return ToolOutput(
        text="\n".join(t for t in texts if t),
        images=images,
        is_error=bool(getattr(result, "isError", False)),
        raw=result,
    )


class McpBridge:
    """常驻 MCP ClientSession（streamable-http）+ 后台 loop。"""

    def __init__(self, url: str, connect_timeout: float = 30.0):
        self._url = url
        self._session = None
        self._shutdown = None  # asyncio.Event，在 loop 线程内创建
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="mcp-loop")
        self._thread.start()

        ready: Future = Future()
        self._loop.call_soon_threadsafe(lambda: self._loop.create_task(self._serve(ready)))
        ready.result(timeout=connect_timeout)  # 连接失败会在此抛出

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _serve(self, ready: Future):
        try:
            async with streamablehttp_client(self._url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._shutdown = asyncio.Event()
                    ready.set_result(True)
                    await self._shutdown.wait()
        except Exception as e:  # noqa: BLE001
            if not ready.done():
                ready.set_exception(e)

    def _run(self, coro, timeout: float):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    # ---- 同步 API ----
    def list_tools(self, timeout: float = 30.0):
        return self._run(self._session.list_tools(), timeout).tools

    def call_tool(self, name: str, args: dict, timeout: float = 30.0):
        return self._run(self._session.call_tool(name, args or {}), timeout)

    def close(self):
        if self._shutdown is not None:
            self._loop.call_soon_threadsafe(self._shutdown.set)


class RosTools:
    """MCP 工具的同步门面：schema 转换 + 调用归一化。"""

    def __init__(self, bridge: McpBridge, call_timeout: float = 30.0):
        self._bridge = bridge
        self._call_timeout = call_timeout

    def list_openai_tools(self, allowlist=None) -> list:
        """把 MCP Tool.inputSchema 转成 OpenAI function-tool schema；可按名过滤。"""
        out = []
        for t in self._bridge.list_tools():
            if allowlist is not None and t.name not in allowlist:
                continue
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": (t.description or "")[:1024],
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                },
            })
        return out

    def call(self, name: str, args: dict) -> ToolOutput:
        res = self._bridge.call_tool(name, args or {}, self._call_timeout)
        return _normalize(res)

    def connect(self, ip: str, port: int) -> ToolOutput:
        """便捷：设置/测试 rosbridge 目标（等价 probe 的起手式）。"""
        return self.call("connect_to_robot", {"ip": ip, "port": int(port)})
