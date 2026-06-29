# memory_navi · ROS-MCP 接入层启动说明

本文件记录如何启动 `ros-mcp-server`，让 Qwen（或任何 MCP 宿主）能透过 MCP 看到/操作 ROS2 机器人。
本步只做"通路"，Qwen agent 本体另见后续。

## 架构与数据流

```
Webots 仿真 (webots_ros2_robomaster)
  /agent0/camera/image_color, /agent0/cmd_vel, /agent1/...   ← ROS2 (DDS)
        │
   rosbridge_server  ── ws://127.0.0.1:9090 ──┐         （ROS2 ↔ WebSocket 桥）
        │                                       │
   ros-mcp-server (FastMCP)  ←── MCP (stdio / http) ──  MCP 宿主
        │                                                 · 验证: probe_topics.py（无 LLM）
        │                                                 · 将来: memory_navi 的 Qwen agent
   本地 vLLM: Qwen3-VL-8B  @ http://127.0.0.1:8000/v1     （agent 调用，本步不涉及）
```

## 端口 / 路径速查

| 项目 | 值 |
|------|-----|
| rosbridge WebSocket | `127.0.0.1:9090` |
| vLLM (Qwen3-VL) OpenAI 接口 | `http://127.0.0.1:8000/v1` |
| ros-mcp-server 目录 | `memory_navi/ros-mcp-server/` |
| MCP server 入口 | `uv run server.py`（默认 stdio；另支持 `--transport http/streamable-http --host --port`） |
| 连接目标（默认） | `127.0.0.1:9090`，运行时可用 `connect_to_robot(ip,port)` 改 |
| 仿真工程 | `/home/kuko/webots_ws`（已修好的相机/命名空间） |

## 0. 一次性前置

> ⚠️ **conda 坑**：base 环境是 Python 3.13，会和 ROS2 Humble(3.10) 冲突。
> **凡是跑 ROS2 / rosbridge 的终端，先 `conda deactivate`**（直到提示符没有 `(base)`）。

```bash
# (a) 安装 rosbridge（需要 sudo 密码，自己执行一次）
sudo apt update && sudo apt install -y ros-humble-rosbridge-suite

# (b) ros-mcp-server 依赖（已用清华源装好；如需重装）
cd ~/Kuko1414/memory_navi/ros-mcp-server
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"   # 国内加速
export UV_PYTHON_PREFERENCE="only-system"
uv sync --python /usr/bin/python3.10
```

## 1. 启动顺序（三个终端）

**终端 A — Webots 仿真**
```bash
conda deactivate
source /opt/ros/humble/setup.bash
source ~/webots_ws/install/setup.bash
ros2 launch webots_ros2_robomaster robot_launch.py
```

**终端 B — rosbridge（ROS2↔WebSocket，:9090）**
```bash
conda deactivate
source /opt/ros/humble/setup.bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
# 确认监听: ss -tlnp | grep 9090
```

**终端 C — Open WebUI 启动**

```bash
conda activate open-webui
open-webui serve

## 2. 把 MCP server 当常驻服务（可选，给将来的 Qwen agent 用）

```bash
cd ~/Kuko1414/memory_navi/ros-mcp-server
# stdio（宿主自己拉起子进程，调试最简单）：
uv run server.py
# 或 HTTP 常驻服务（解耦、好封装）：
uv run server.py --transport streamable-http --host 127.0.0.1 --port 9000
```
> 一个 server 实例 = 一个 rosbridge 连接，但一个实例即可覆盖整张 ROS 图（用话题名区分 agent0/agent1）。
> 要同时连"仿真+真车"等多目标，再起一个实例（不同端口）。

## 3. 切换到真车

不用改代码：把 `connect_to_robot` 的目标改成真车即可。
- 探针：`--ip 192.168.101.138 --port 9090`
- agent / 宿主里：调用 `connect_to_robot(ip="192.168.101.138", port=9090)`
- 前提：真车上 rosbridge 已运行（参考 `humble_ws/future/launch_ros_mcp.md`）。

## 4. 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `port_check: open=false` / Connection refused | rosbridge 没起 | 终端B 起 rosbridge，确认 :9090 监听 |
| `No topics found` | ROS2 图里没节点 | 终端A 起仿真，`ros2 topic list` 确认 |
| `uv sync` 卡住不动 | PyPI 国内慢 | 用 `UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple` |
| 跑 ros2 报 `rclpy._rclpy_pybind11` 缺失 | conda Py3.13 抢了 python | 先 `conda deactivate` 再 source ROS |
| 相机话题没图 | 仿真没在跑/被暂停 | 见 webots_ws，时钟要在走 |

## 5. 关键文件

- `ros-mcp-server/server.py` → `ros_mcp/main.py`（FastMCP 入口，默认连 127.0.0.1:9090）
- `ros-mcp-server/ros_mcp/tools/`（topics / images / connection / ... 工具）
- `ros-mcp-server/probe_topics.py`（本目录新增的通路验证脚本）
