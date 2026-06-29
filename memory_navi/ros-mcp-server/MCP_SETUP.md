# 整栈 Bringup（缺口补全任务用）

> 给“缺口补全实验”(`completion_demo.py`) 用的一键启动清单。每个终端一行标题，照着开。
> **铁律（见 CLAUDE.md）**：凡涉及 ROS2 的终端，先 `conda deactivate`（conda base 是 py3.13，会
> 破坏 ROS2 Humble 的 py3.10）。只有 agent_core / vLLM 那一侧用 `conda activate vllm`(py3.11)。
>
> 端口：vLLM 8B `:8000` · MCP server `:9000` · rosbridge `:9090`。命名空间 `agent0`。

## 进程清单（共 8 个）

| # | 终端做什么 | conda | 命令 |
|---|---|---|---|
| 1 | Webots 仿真(agent0/agent1) | deactivate | conda deactivate && source /opt/ros/humble/setup.bash && source ~/webots_ws/install/setup.bash && ros2 launch webots_ros2_robomaster robot_launch.py |
| 2 | rosbridge (:9090) | deactivate | conda deactivate && source /opt/ros/humble/setup.bash && ros2 launch rosbridge_server rosbridge_websocket_launch.xml |
| 3 | 彩色图 JPEG 中继 | deactivate | conda deactivate && source /opt/ros/humble/setup.bash && python3 memory_navi/image_jpeg_relay.py /agent0/camera/image_color /agent0/camera/image_color/compressed |
| 4 | **深度摘要中继**（本任务新增） | deactivate | conda deactivate && source /opt/ros/humble/setup.bash && python3 memory_navi/depth_summary_relay.py /agent0/camera/depth/image /agent0/camera/depth/summary |
| 5 | 安全急停节点 | deactivate | conda deactivate && source /opt/ros/humble/setup.bash && python3 memory_navi/safety_node.py |
| 6 | TF (map→base_link + 静态) | deactivate | conda deactivate && source /opt/ros/humble/setup.bash && ros2 launch memory_navi/sim_bringup/agent_tf.launch.py |
| 7 | MCP server (:9000) | (用 uv 的 py3.10) | cd memory_navi/ros-mcp-server && uv run server.py --transport streamable-http --host 127.0.0.1 --port 9000 |
| 8 | vLLM 8B (:8000) | activate vllm | conda activate vllm && ./start_all.sh 8b --enable-auto-tool-choice --tool-call-parser hermes |

> #4 深度中继是这次任务**新增**的：原始深度图(32FC1 ≈1.23MB)和原始彩色图一样过不了 rosbridge，
> 故在 ROS 端原生订阅、压成「左→右每列中位距离」小数组发到 `/agent0/camera/depth/summary`，
> MCP 的 `depth_summary` 工具再去订这个小话题。**不开 #4，`depth_summary` 会超时**（巡检会退化成只用 scan）。

## 启动顺序 & 健康检查

1. 先 #1 Webots → #2 rosbridge（其余依赖它们）。
2. 再 #3 #4 #5 #6（中继/安全/TF）、#7 MCP、#8 vLLM。
3. 连通性自检（不需 LLM）：
   ```bash
   cd memory_navi/ros-mcp-server && uv run python probe_topics.py
   ```
   关注这几条话题是否都在：
   - `/agent0/camera/image_color/compressed`（#3 出）
   - `/agent0/camera/depth/summary`（#4 出，类型 `std_msgs/Float32MultiArray`，data 长度=列数）
   - `/agent0/scan` `/agent0/gps` `/agent0/imu`
4. 都就绪后跑实验：
   ```bash
   conda activate vllm
   python memory_navi/controll/controll/completion_demo.py
   ```
   退出码 0 = 到达/朝向/语义/距离全过且无幻觉。结果存 `Report/last_completion_run.json`，
   可离线重打分：`python Report/score_completion.py Report/last_completion_run.json`。

## 排错

- `depth_summary` 超时 → #4 没开，或深度话题名不对（确认 `/agent0/camera/depth/image` 在）。
- `geo_goto` 读不到位姿 / move 不停 → #6 TF 或 #1 GPS/IMU 没上；`ws_manager.receive` 超时须 >位姿周期(已设 0.5s)。
- Executor 连不上 → #7 MCP(:9000) 或 #8 vLLM(:8000) 没起。
- 转向/位移不准 → 已是闭环读真值；若仍偏，查 safety 是否频繁刹停（前方太近）。
