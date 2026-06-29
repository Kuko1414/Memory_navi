#!/usr/bin/env python3
"""垂直切片：端到端验证整套工作流的接口。

Proof 1 — Qwen3-VL-8B 通过 function-call + MCP 看到 ROS topic
          （自主调用 get_topics / get_topic_type，报告相机话题消息类型）。
Proof 2 — Claude（记忆作者）观察相机图像 → 产出 C 格式 JSON 词条 → 写文件系统记忆 → 读回。

前置：rosbridge:9090、MCP server(streamable-http):9000、Webots 仿真、vLLM 8B:8000（带 tool-calling flag）、
      ANTHROPIC_API_KEY。详见 plan。

直接运行：
  conda activate vllm
  python memory_navi/controll/controll/slice_demo.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))  # .../controll/controll
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from agent_core import config
from agent_core.cloud.memory_author import MemoryAuthor
from agent_core.cloud.providers import AnthropicProvider
from agent_core.llm_client import make_vllm_client, resolve_model
from agent_core.mcp_bridge import McpBridge, RosTools
from agent_core.memory.fs_memory import FsMemory
from agent_core.tool_loop import run_tool_loop

PROOF1_TOOLS = {"get_topics", "get_topic_type", "get_topic_details", "subscribe_once"}


def _hr(title: str):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def proof1_fc_mcp(ros: RosTools) -> bool:
    _hr("Proof 1 — Qwen 经 FC+MCP 看到 ROS topic")
    client = make_vllm_client(config.VLLM_8B_BASE_URL)
    model = resolve_model(client, config.VLLM_8B_MODEL_FALLBACK)
    print(f"[vLLM] model = {model}")

    tools = ros.list_openai_tools(allowlist=PROOF1_TOOLS)
    print(f"[MCP] 暴露给模型的工具：{[t['function']['name'] for t in tools]}")

    messages = [
        {"role": "system", "content": "你是机器人执行体，可通过工具（MCP）查询 ROS。需要时调用工具，最后用中文简洁回答。"},
        {"role": "user", "content": f"列出当前所有 ROS topic，并报告相机话题 {config.CAMERA_TOPIC} 的消息类型(message type)。"},
    ]
    res = run_tool_loop(
        client, model, messages, tools, dispatch=ros.call,
        max_iters=config.TOOL_LOOP_MAX_ITERS,
        temperature=config.TOOL_LOOP_TEMPERATURE,
        max_tokens=config.TOOL_LOOP_MAX_TOKENS,
        max_image_turns=config.TOOL_LOOP_MAX_IMAGE_TURNS,
    )

    print("\n[工具调用轨迹]")
    for step in res.tool_trace:
        flag = "ERR" if step["is_error"] else "ok"
        print(f"  - {step['name']}({json.dumps(step['args'], ensure_ascii=False)}) -> [{flag}] {step['text'][:120]}")
    print(f"\n[模型最终回答]\n{res.text}")

    called = {s["name"] for s in res.tool_trace}
    answer = (res.text or "").lower()
    ok = ("get_topics" in called) and ("sensor_msgs" in answer or "image" in answer)
    print(f"\n[Proof 1] {'PASS ✅' if ok else 'FAIL ❌'} "
          f"(调用了 get_topics={'get_topics' in called}, 回答含 Image/sensor_msgs={'sensor_msgs' in answer or 'image' in answer})")
    return ok


def proof2_memory_author(ros: RosTools) -> bool:
    _hr("Proof 2 — Claude 观察相机图像 → 写 C 格式记忆")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("[Proof 2] SKIP ⚠️  未设置 ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN")
        return False

    mem = FsMemory(config.MEMORY_ROOT, config.ENV_NAME)
    provider = AnthropicProvider()
    print(f"[Claude] model = {provider._model}")
    author = MemoryAuthor(ros, provider, mem)

    result = author.record(area="roomA", trigger="vertical_slice")
    if not result.ok:
        print(f"[Proof 2] FAIL ❌  {result.error}")
        return False

    print("\n[Claude 产出并通过 C schema 校验的记录]")
    print(json.dumps(result.record, ensure_ascii=False, indent=2))

    readback = mem.load_area("roomA")
    n_obj = len(readback.get("objects", [])) if readback else 0
    ok = readback is not None and readback.get("area") == "roomA"
    print(f"\n[读回 roomA/area.json] area={readback.get('area') if readback else None}, objects={n_obj}")
    print(f"\n[Proof 2] {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


def main() -> int:
    _hr("垂直切片：MCP 桥 / tool loop / 读盘+降采样 / Claude provider / C schema / 文件系统记忆")
    print(f"[MCP] 连接 {config.MCP_URL} ...")
    try:
        bridge = McpBridge(config.MCP_URL, connect_timeout=config.MCP_CONNECT_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        print(f"❌ 无法连接 MCP server（{config.MCP_URL}）：{e}")
        print("   请先启动：cd memory_navi/ros-mcp-server && "
              "uv run server.py --transport streamable-http --host 127.0.0.1 --port 9000")
        return 2

    ros = RosTools(bridge, call_timeout=config.MCP_CALL_TIMEOUT)
    conn = ros.connect(config.ROSBRIDGE_IP, config.ROSBRIDGE_PORT)
    print(f"[MCP] connect_to_robot -> {conn.text[:200]}")

    results = {}
    try:
        results["proof1"] = _safe(proof1_fc_mcp, ros)
        results["proof2"] = _safe(proof2_memory_author, ros)
    finally:
        bridge.close()

    _hr("结果")
    for k, v in results.items():
        print(f"  {k}: {'PASS ✅' if v else 'FAIL/SKIP ❌'}")
    return 0 if all(results.values()) else 1


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"❌ {fn.__name__} 抛异常：{e}")
        return False


if __name__ == "__main__":
    sys.exit(main())
