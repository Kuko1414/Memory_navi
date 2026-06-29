# Architecture（接口与代码结构）

室内具身智能体：**大模型精确记忆 + 小模型高效执行**。本文件说明各接口的数据类型/用途，并列出工作区代码结构。
方法论以 [proposal.md](proposal.md) 为准；本轮工程进度见 [Process.md](Process.md)，改动记录见 [Memory.md](Memory.md)。

---

## 一、系统拓扑（进程与连线）

```
[py3.11 vLLM conda]                         [py3.10 uv venv]            [py3.10 ROS2]
 agent_core（执行者编排，无 rclpy）  --HTTP-->  ros-mcp-server :9000  --ws:9090--> rosbridge <-> ROS2 / Webots
   │                                          (streamable-http /mcp)                         │
   ├─ HTTP :8000  ──>  vLLM Qwen3-VL-8B（执行者，开 tool-calling）                            │
   ├─ HTTP :8001  ──>  vLLM Qwen3-VL-4B（快速/备用）                                          │
   └─ HTTPS ───────>  Claude 网关（记忆作者）                                                 │
                                                                                            │
 safety_node（独立 rclpy 进程）  <───────────── /agentN/scan,imu,gps,camera_info ────────────┤
   └─ 硬停: 零速 -> /agentN/cmd_vel ；状态 -> /dev/shm/agent_safety_<ns>.json                 │
 sim_bringup TF（rclpy）  map->base_link(GPS+IMU) + 静态 base_link->{lidar,camera} ──────────┘
```

角色分工：**Qwen3-VL-8B = 执行者**（function-call + MCP 看/控 ROS、低延迟）；**Claude = 记忆作者**（仅在触发点观察相机图、记录结构化语义）；**安全/几何 = 确定性代码**（独立、不经 LLM）。

---

## 二、对外接口（数据类型 + 用途）

### 2.1 vLLM（OpenAI 兼容）— 执行者 / 本地 VLM
- 端点：`http://localhost:8000/v1`（8B，编排）、`http://localhost:8001/v1`（4B，快速）。
- 调用：`chat.completions.create(model, messages, tools=[...], tool_choice="auto", stream=False)`。开了 `--enable-auto-tool-choice --tool-call-parser hermes`，返回 `message.tool_calls`。
- 视觉：消息内容块 `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}`。
- 用途：执行者读取 ROS 工具结果、低延迟推理；图像问答。

### 2.2 ros-mcp-server（MCP, streamable-http）— ROS 工具层
- 端点：`http://127.0.0.1:9000/mcp`；后端经 rosbridge `ws://127.0.0.1:9090` 通 ROS2。
- ~31 个工具，关键：`connect_to_robot(ip,port)`、`get_topics()`、`get_topic_type(topic)`、`get_topic_details`、`subscribe_once(topic,msg_type,expects_image,timeout)`、`subscribe_for_duration`、`publish_once(topic,msg_type,msg)`、`call_service`、`send_action_goal`、`view_saved_image`。
- 返回（经 agent_core 归一化为 `ToolOutput`）：`text:str`、`images:list[ImagePart(b64,mime)]`、`is_error:bool`。`subscribe_once(expects_image)` 抓帧时会存盘 `<server-cwd>/camera/received_image.jpeg` 并返回 base64。
- 用途：让 Qwen 通过 function-call 看/控 ROS；记忆作者抓相机帧。

### 2.3 Claude 网关（Anthropic Messages API）— 记忆作者
- 认证：环境变量 `ANTHROPIC_BASE_URL`（LAN 中继）+ `ANTHROPIC_AUTH_TOKEN`（bearer，非 `ANTHROPIC_API_KEY`）；`anthropic.Anthropic()` 自动读取。模型 `claude-sonnet-4-6`（可换 opus）。
- 调用：`messages.create(system=..., tools=[record_memory], tool_choice={type:tool,name:record_memory}, messages=[image_block + 文本])` → 取 `tool_use.input`（C 格式 JSON）。
- 用途：观察相机图 → 产出结构化语义记忆。

### 2.4 ROS2 话题（仿真侧，namespace `/agentN`）
| 话题 | 类型 | 用途 |
|---|---|---|
| `camera/image_color` | sensor_msgs/Image (bgra8 640×480) | 彩色相机（执行/记忆作者） |
| `camera/image_color/compressed` | sensor_msgs/CompressedImage (JPEG~28KB) | 由 `image_jpeg_relay.py` 转出，便于经 MCP 传输 |
| `camera/camera_info` | sensor_msgs/CameraInfo | 彩色内参（含 frame `camera_link`） |
| `camera/depth/image` | sensor_msgs/Image (深度) | 深度图（几何投影，下一步） |
| `camera/depth/camera_info` · `camera/depth/points` | CameraInfo · PointCloud2 | 深度内参 / 点云 |
| `scan` | sensor_msgs/LaserScan (360°, min 0.05, max 12, frame `lidar_link`) | 测距 → 安全硬停 |
| `imu` | sensor_msgs/Imu | 朝向（TF 真值） |
| `gps` | geometry_msgs/PointStamped | 位置（TF 真值） |
| `cmd_vel` | geometry_msgs/Twist | 运动指令（safety 触发时刷零速覆盖） |
| `cmd_wheels` | (轮速指令) | 底层轮控 |
| `safety/status` | std_msgs/String (JSON) | 安全节点状态广播 |
| `/tf` `/tf_static` | tf2_msgs/TFMessage | `map→base_link→{lidar_link,camera_link}` |

### 2.5 记忆记录（C 分层混合，`memory/<env>/<area>/area.json`）
- 字段：`area, type, summary, observed_at, view_pose, objects[], hazards[]`。
- `objects[i]`：`id,name,spatial(定性), roi(归一化 bbox), view{angle_deg,distance_m,distance_delta_m}, abs_pose{x,y,z}, abs_pose_delta_m, state, affordance[], confidence, verified_by[]`。
- 约定：`abs_pose/distance_m/delta` 由几何管线回填（VLM 不臆造，留 null）；`roi` 由 VLM 给、供代码在深度图取该区域。
- 拓扑：根 `topology.json` 边表 `{from,to,via,direction,width_m,confidence}` 供 BFS（"文件夹=词典、topology.json=连通图"）。

### 2.6 安全状态 `/dev/shm/agent_safety_<ns>.json`
`{tripped:bool, lidar_min_m:float|null, sensors_ok:bool, stale:[topic...], reason:"ok|near_obstacle|sensor_fault", ts}` —— 供未来 Supervisor 的 `safety_iface` 读。

---

## 三、代码结构（工作区 `/home/kuko/Kuko1414/`）

### 顶层
- `proposal.md` — 学术主张（source of truth）。`Report/` — 项目计划书 v3.0（html）。`SETUP.md` — 硬件/环境。
- `start_all.sh` — 一键起 vLLM 8B(:8000, model id `qwen3-vl-8b`) + LibreChat(:3080)，已加 tool-calling flag。
  （4B 暂不用，脚本内注释保留。）
- **UI = LibreChat**（conda env `librechat`，源码在 `~/LibreChat`）：custom endpoint 连 vLLM、
  `mcpServers`(streamable-http) 连 ros-mcp-server，在「Agents」里让 Qwen 自己 function-call 操控机器人。
  旧 Streamlit `vllm_chat_ui.py` 已移除。

### `memory_navi/` — 主系统
- **`ros-mcp-server/`** — FastMCP ↔ ROS2 桥（已部署）。`server.py`/`ros_mcp/main.py` 入口（streamable-http :9000）；`ros_mcp/tools/*`（topics/services/actions/nodes/parameters/images/connection + **新增 `perception.py::scan_summary` 全精度→标量**、**`agent_actions.py`**：`move`/`turn_left_deg`/`turn_right_deg`/`stop`/`look`/`get_pose`/`navigate_to`，move/turn 为**闭环读真值位姿**、safety-shm 兜停，新文件 additive 不动上游）；`ros_mcp/utils/websocket.py`（rosbridge + 图像存盘；**注意 `receive` 超时即 close 连接丢订阅**）；`probe_topics.py` 验证。
- **`controll/controll/agent_core/`** — 薄 agent 核心（执行者编排 + 记忆作者）：
  - `config.py` — 端点/模型/路径/阈值集中配置。
  - `llm_client.py` — vLLM OpenAI client 工厂（proxy-safe）+ `resolve_model`。
  - `image_utils.py` — `ImagePart`；读盘/降采样/转 OpenAI、Anthropic 图像块。
  - `mcp_bridge.py` — 后台 asyncio 线程常驻 MCP `ClientSession`；`RosTools.list_openai_tools()` / `.call()->ToolOutput`。
  - `tool_loop.py` — `run_tool_loop(...)`：非流式 function-call 循环，工具图像作 user image turn 回灌；`tool_choice` 已参数化（vLLM hermes 不支持 `"required"`）。
  - `executor.py` — `Executor`：复用上面件，allowlist 只暴露精选动作工具，`run(brief,...)` 留 supervisor 注入口；`call_timeout` 120s（闭环 move 慢）。
  - `harness.py` — Qwen 规划器(`plan_task` 结构化子目标)→ 代码执行器(`execute_step` move/turn 闭环+真值核验、report 强制 look)；感知反馈环 `explore_and_report`(看→决策→执行→核验)。
  - `cloud/schema.py` — C 格式 JSON Schema + `validate_memory_record`。
  - `cloud/providers.py` — `AnthropicProvider`(主, system+tool-use 强制 JSON) / `LocalVLMProvider`(4B 回退)。
  - `cloud/memory_author.py` — `MemoryAuthor.record()`：抓图→标注→校验→写区域 json。
  - `memory/fs_memory.py` — 文件系统 STG：`load_area/upsert_area/topology_path(BFS)`/原子写。
  - 🟡 `harness.py` 是 supervisor 雏形（规划+核验+探索环）；⬜ 仍未建：`supervisor.py`（6 规则）、`skills/`、`memory/memory_tools.py`、几何导航 `goto(x,y)`。
- `controll/controll/slice_demo.py` — **垂直切片入口**（Proof1: Qwen FC+MCP 读 topic；Proof2: Claude 观察图→写 C 记忆）。`control.py` 仍是桩。
- `controll/controll/executor_smoke.py` — 执行器冒烟（Qwen 经精选工具多步动作）。`controll/test/test_executor.py` — 执行器接线单测。
- `Report/harness_eval.md` + `Report/break_room_ground_truth.json` — harness 评估报告（幻觉 vs 不收敛）+ 评估答案 key。
- `memory/<env>/<area>/area.json` — STG 记忆产出（如 `memory/sim/roomA/area.json`）。
- `image_jpeg_relay.py` — 相机 raw bgra8(1.23MB) → JPEG CompressedImage 中继（rosbridge 传不动 raw 图）。
- `sim_bringup/webots_pose_tf.py` — `map→base_link` 真值 TF（GPS+IMU）。
- `sim_bringup/agent_tf.launch.py` — 静态 `base_link→{lidar_link,camera_link}` + 上面的 pose 节点。
- `safety_node.py` — 独立安全节点（存活性校验 + <0.15m 硬停 + 状态输出）。

### `/home/kuko/webots_ws/`（仿真，独立工作区）
- `src/webots_ros2_robomaster/worlds/break_room.wbt` — 当前世界，2 台车 agent0/agent1（各含 Camera+RangeFinder+Lidar"LDS-01"+GPS+IMU+Compass+Receiver）。
- `src/webots_ros2_robomaster/resource/robomaster.urdf` — webots_ros2_driver 设备映射（scan/camera/depth/gps/imu）。
- `launch/robot_launch.py` — 起 Webots + 2 个 WebotsController。

### 运行环境
- `vllm` conda 环境（py3.11）：跑 vLLM + agent_core + 记忆作者（`openai/anthropic/mcp/pillow`）。
- ROS2 Humble（系统 py3.10）：跑 rosbridge / Webots / safety_node / TF（注意：先剔除 miniconda 再 source ROS2，见 [[conda-ros2-python-conflict]]）。
