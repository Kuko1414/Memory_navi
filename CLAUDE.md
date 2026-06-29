# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目是什么

一个室内 embodied-VLM 智能体，采用刻意的**角色分工**（而非多模型联合决策）：
- **本地执行器** — vLLM Qwen3-VL-8B（FP16，端口 8000；4B 在 8001 作降级）。通过 MCP bridge 以
  function-calling 驱动 ROS，低延迟，负责导航与技能执行。
- **云端记忆作者** — Claude（经局域网 gateway）。由**结构性条件**触发（进入新区域，或 `inspect`
  技能），观察相机并写出结构化的 "C-format" JSON 语义记录。不在常规循环里；写一次本地复用，
  warmup 后显著减少云调用。
- **代码层** — 确定性的安全急停、传感器活性检测、几何/TF、Nav2。**永不**由模型把关。

空间记忆是"词典式"的：拓扑图（STG）用于远距离导航，外加每个区域的详细 JSON 记录用于近距离任务。
意图与设计动机以 `Report/proposal.md` 为准；进度/变更记录在 `Process.md` 和 `Memory.md`。
工作区的框架和topic内容在`Architecture.md`

## 关键环境约束

conda **base 是 Python 3.13，会破坏 ROS2 Humble（Python 3.10）**。在 source ROS2、或运行任何涉及
rosbridge / `ros2` / MCP server / safety node / TF 的命令之前，务必先 `conda deactivate`。
agent_core + vLLM 一侧运行在 `vllm` conda 环境（Python 3.11）。这两个世界是**有意分开**的。

## 进程拓扑（谁和谁通信）

```
[conda vllm py3.11]                         [py3.10 / system ROS2]
 agent_core (执行器编排)
  ├─ vLLM Qwen3-VL-8B :8000 (FC 执行器) ───┐
  ├─ vLLM Qwen3-VL-4B :8001 (降级)         │ HTTP
  └─ Claude gateway (记忆作者, HTTPS)       │
        │                                  ▼
        └─ MCP client ──HTTP:9000──► ros-mcp-server (FastMCP, ~31 个工具)
                                          │ ws:9090
                                          ▼
                                      rosbridge ◄──► ROS2 / Webots (命名空间 /agentN)

 safety_node.py  (独立 rclpy): /agentN/scan < 0.15m → 置零 /agentN/cmd_vel；状态写到
                                /agentN/safety/status + /dev/shm/agent_safety_<ns>.json
 sim_bringup TF (rclpy): map→base_link (GPS+IMU 真值) + static base_link→{lidar,camera}
```

非显而易见的关键点：原始相机帧（`/agentN/camera/image_color`，bgra8 约 1.23MB）**无法通过 rosbridge
JSON 传输**。需运行 `image_jpeg_relay.py` 生成 `/agentN/camera/image_color/compressed`（约 28–30KB）
——这才是 agent/MCP 实际消费的话题。

## 端口

| 8000 | vLLM 8B (model id `qwen3-vl-8b`) | 3080 | LibreChat UI | 27017 | MongoDB (LibreChat) |
| 9000 | MCP server (streamable-http) | 9090 | rosbridge websocket |

> UI 为 **LibreChat**（conda env `librechat`，源码跑在 `~/LibreChat`）；通过 custom endpoint 连 vLLM、
> 经 `mcpServers`(streamable-http) 连 ros-mcp-server，在「Agents」里让 Qwen 自己 function-call 操控 Webots。
> 旧的 Streamlit UI 已移除。4B(:8001) 暂不用（`start_all.sh` 内注释保留）。

## 启动整个栈

下面每条 ROS 命令都假设先执行 `conda deactivate && source /opt/ros/humble/setup.bash`。

```bash
# vLLM 模型 + chat UI（conda vllm）。会加上 --enable-auto-tool-choice --tool-call-parser hermes（Qwen3 FC 必需）
./start_all.sh            # all | 8b | 4b | chat | stop | status

# ROS2 / Webots 一侧（分开的终端，已 conda deactivate）
source /home/kuko/webots_ws/install/setup.bash
ros2 launch webots_ros2_robomaster robot_launch.py          # Webots 仿真 (agent0/agent1)
ros2 launch rosbridge_server rosbridge_websocket_launch.xml  # bridge on :9090
python3 memory_navi/image_jpeg_relay.py /agent0/camera/image_color /agent0/camera/image_color/compressed
python3 memory_navi/safety_node.py
ros2 launch memory_navi/sim_bringup/agent_tf.launch.py

# MCP server（memory_navi/ros-mcp-server，py3.10 via uv）
uv run server.py --transport streamable-http --host 127.0.0.1 --port 9000
uv run python probe_topics.py     # 连通性检查，无需 LLM
```

冒烟测试 / 垂直切片（需要 Webots + rosbridge + MCP + vLLM 8B 全部就绪，且设置了
`ANTHROPIC_AUTH_TOKEN`）：`python memory_navi/controll/controll/slice_demo.py` —— Proof 1 = Qwen
FC+MCP 读取 ROS 话题，Proof 2 = Claude 观察相机并写出 C-format 记忆 JSON。退出码 0 = 两者均通过。

## Claude gateway 鉴权（非标准）

记忆作者使用 `ANTHROPIC_BASE_URL`（局域网中转）+ **`ANTHROPIC_AUTH_TOKEN`**（bearer）鉴权——
**不是** `ANTHROPIC_API_KEY`。默认模型 `claude-sonnet-4-6`（在 `agent_core/config.py` 中可配）。

## 测试

```bash
# agent_core / controll（conda vllm）
pytest memory_navi/controll/test/            # ament copyright/flake8/pep257 风格检查
# MCP server（uv；集成测试需要 Docker + ROS）
cd memory_navi/ros-mcp-server && uv run pytest tests/ -m "not integration"
```

首次构建 ROS2 工作区：`cd /home/kuko/webots_ws && colcon build`，然后 source `install/setup.bash`。
MCP server 依赖：`cd memory_navi/ros-mcp-server && uv sync --python /usr/bin/python3.10`。

## 代码地图（关键部分）

- `memory_navi/controll/controll/agent_core/` — 精简执行器核心：
  `config.py`（所有 endpoint/路径/阈值）、`llm_client.py`（vLLM OpenAI client + `resolve_model`）、
  `mcp_bridge.py`（后台 asyncio MCP `ClientSession`，`RosTools.list_openai_tools/.call`）、
  `tool_loop.py`（非流式 FC 循环；工具产出的图片作为一轮 user 图像回灌）、
  `image_utils.py`（`ImagePart` 加载/降采样/转换）。
- `agent_core/cloud/` — 记忆作者：`schema.py`（C-format JSON Schema + `validate_memory_record`）、
  `providers.py`（`AnthropicProvider` 用 tool-use 强制 JSON；`LocalVLMProvider` 4B 降级）、
  `memory_author.py`（`MemoryAuthor.record()`：抓帧 → 标注 → 校验 → 写区域 JSON）。
- `agent_core/memory/fs_memory.py` — 文件系统 STG：`load_area`/`upsert_area`/`topology_path`（BFS）/
  原子写。记录位于 `memory_navi/memory/<env>/<area>/area.json`；图在 `topology.json`。
- `memory_navi/ros-mcp-server/` — FastMCP↔ROS 桥；工具在 `ros_mcp/tools/`，图片落盘在
  `ros_mcp/utils/websocket.py`。`ToolOutput{text, images, is_error}`。
- `memory_navi/safety_node.py`、`memory_navi/image_jpeg_relay.py`、`memory_navi/sim_bringup/` ——
  确定性代码层（独立于任何模型）。

## Supervisor 模型（已设计，Phase 4 —— 尚未实现）

决策引擎是**6 条确定性规则**，按优先级评估，循环中无 AI：
0 安全（`lidar<0.15m` 或 `sensors_ok==False` → EmergencyStop）· 1 云触发（`entered_new_area`
或 `skill=="inspect"` → DispatchCloud record）· 2 `retry>=3` → SafePause+Notify · 3 有未访问区域 →
explore · 4 用户目标在地图中 → navigate · 5 全部已访问 → TaskComplete。新的决策逻辑请保持这种
确定性风格——基于置信度 / 多模型联合的决策已被明确否决（见 `Report/proposal.md`）。

## 状态（依据 Process.md / Memory.md）

Phase 1–3 已完成：精简 agent core + 垂直切片、仿真硬件（lidar minRange 0.05、depth、TF）、
safety node。**Phase 4（supervisor + 技能 + `depth_projection.py` 几何管线）与 Phase 5
（A→B→A 实验 vs Qwen-only 基线）尚未开始。** 已知缺口：几何管线把 VLM 提供的 ROI → depth →
`abs_pose`（JSON 中的 `abs_pose`/`distance_m`/`delta` 字段由代码填充，目前为 null）；agent1 的 TF
需要命名空间隔离（agent0 已测）；云记录常只返回 summary（计划：升级到 `claude-opus-4-8` + 更严格的
system prompt）。
