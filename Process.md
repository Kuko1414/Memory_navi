# Process（总任务与实现流程）

## 总任务

构建**室内具身智能体**：云端大模型（Claude）当"记忆作者"在触发点精确记录区域语义（词典式 STG 记忆），本地小模型（Qwen3-VL-8B）当"执行者"基于语义拓扑低延迟导航/执行 skill，确定性代码负责安全与几何。验证"记忆由大模型、执行由小模型、安全由代码"的角色分工优于多模型共同决策（详见 [proposal.md](proposal.md)）。接口与代码结构见 [Architecture.md](Architecture.md)。

实现按"先打通最小工作流，再逐层加厚"推进。`done` = 已实现并验证。

---

## 实现流程

### 阶段一 · 基础设施
1. 部署 vLLM 服务 Qwen3-VL-8B(:8000) + 4B(:8001)，OpenAI 兼容。**done**
2. 部署 ros-mcp-server（FastMCP，streamable-http :9000）经 rosbridge 通 ROS2，~31 工具，probe 验证。**done**
3. 开启 vLLM function-calling（`--enable-auto-tool-choice --tool-call-parser hermes`，Qwen3 用 Hermes 模板）。**done**

### 阶段二 · 薄 Agent 核心（执行者 ↔ MCP ↔ 记忆作者）
4. `mcp_bridge.py`：后台 asyncio 线程常驻 MCP `ClientSession` + 同步 `RosTools`（schema 转换/调用归一化）。**done**
5. `tool_loop.py`：非流式 function-call 循环，工具图像作 user image turn 回灌。**done**
6. `llm_client.py` / `image_utils.py` / `config.py`：vLLM client、图像读盘+降采样+格式转换、集中配置。**done**
7. 记忆作者 `cloud/`：`schema.py`（C 格式）+`providers.py`（Claude system+tool-use 强制 JSON / 4B 回退）+`memory_author.py`（抓图→标注→校验→写盘）。**done**
8. `memory/fs_memory.py`：文件系统式 STG（文件夹=词典、topology.json=连通图、BFS、原子写）。**done**

### 阶段三 · 垂直切片端到端验证
9. `slice_demo.py`：**Proof 1** Qwen 经 FC+MCP 自主 `get_topics→get_topic_type` 报出相机话题类型。**done**
10. **Proof 2** Claude 经网关观察相机图（JPEG）→C 格式 JSON→校验→写 `memory/sim/roomA/area.json`→读回。**done**
11. 相机 `image_jpeg_relay.py`：raw bgra8(1.23MB) 过不了 rosbridge → 转 JPEG(~28KB) CompressedImage。**done**
12. Claude 网关接入（`ANTHROPIC_BASE_URL`+`ANTHROPIC_AUTH_TOKEN` bearer，`claude-sonnet-4-6`）实测通。**done**

### 阶段四 · 仿真硬件（Lidar + 深度 + TF）
13. Lidar：修 URDF 映射（相对 `scan`+`alwaysOn`）+ 降 `minRange 0.2→0.05`（否则测不到 0.15m）→ `/agentN/scan`。**done**
14. 深度：每台车 Camera 旁加 `RangeFinder`（同位姿/分辨率）+ URDF 映射 → `/agentN/camera/depth/{image,camera_info,points}`。**done**
15. TF：暴露 GPS/IMU；`webots_pose_tf.py`(GPS+IMU 真值 `map→base_link`) + 静态 `base_link→{lidar_link,camera_link}`（`agent_tf.launch.py`）。**done**（仅 agent0 完整）

### 阶段五 · 安全底层
16. `safety_node.py`（独立 rclpy 进程）：50Hz 短窗**传感器存活性校验**（scan/imu/gps/color-info/depth-info）+ **<0.15m 硬停**（刷零速到 cmd_vel）+ 状态(`/agentN/safety/status` + `/dev/shm`)。**done**
17. 实测：启动全 stale→`sensor_fault`，上线→`ok`；阈值测试→`near_obstacle` 且 cmd_vel 被刷零速。**done**

### 阶段六 · 执行工具层 + harness 加固（无幻觉 navi demo）
18. 精选动作/感知 MCP 工具（`ros-mcp-server/ros_mcp/tools/`，新文件 additive 不动 vendored 上游）：
    `perception.py::scan_summary`（全精度 scan→最近障碍/扇区/clear_path 标量，止住上下文膨胀）、
    `agent_actions.py::{move,turn_left_deg,turn_right_deg,stop,look,get_pose,navigate_to}`。**done**
19. 可复用执行器 `agent_core/executor.py::Executor`（复用 tool_loop/mcp_bridge，allowlist 只暴露精选工具，
    留 `brief` 注入口给 supervisor）；`config.py` 加 `ACTION_ALLOWLIST`/`ACTION_SYSTEM_PROMPT`。**done**
20. **运动闭环**：开环按墙钟计时严重欠走（mecanum 麦轮各向异性摩擦+仿真<1×实时）→ `move/turn` 改闭环读
    真值位姿（gps+imu，到目标/ safety-shm tripped 即停）。实测 move0.5→0.506m、turn90→±90°、0.14m 刹停。**done**
21. **幻觉定位**：单段对话自跑多步任务第 6 步出现幻觉（叙述代替指令/没做到却说做到/看图说谎）；
    route-B（每子目标全新短上下文+代码核验）**零幻觉**。判定为 harness 问题，**不换模型**。见 [Report/harness_eval.md](Report/harness_eval.md)。**done**
22. harness `agent_core/harness.py`：Qwen 当规划器（结构化子目标 JSON）→ 代码当执行器（move/turn 代码
    闭环下发+真值核验；report 强制 look+核验）。完成标志达成：demo 任务 Qwen 自分解 7 步、ALL VERIFIED、
    看图报告与真实帧吻合、零幻觉。**done**
23. 记忆驱动探索实验（`harness.explore_and_report` + 真实坐标答案 key 客观打分）：预置缺口记忆让 Qwen 走一步
    验一步绕到红柜后方。**结果：VLM 自主目标导航不收敛**（原地打转，距目标 3.52m 没到）。核心结论：**幻觉(放任)
    与不收敛(管死)是同一问题两面，正确解 = 确定性代码做导航 + VLM 只做语义**。详见 [Report/harness_eval.md](Report/harness_eval.md)
    §7、[Report/break_room_ground_truth.json](Report/break_room_ground_truth.json)。**done（结论性失败，已记录）**

> 已知：vLLM(hermes) 不支持 `tool_choice="required"`；`ws_manager.receive` 超时会断连丢订阅（receive 超时须>位姿周期）。
> **下一步**：完善 prompt + 强化流程；给 VLM 补"上级"——代码用坐标几何导航到目标区，VLM 只 look+描述；
> 按需加 skill/function-call（如 `goto(x,y)`/`face(object)`）。效果仍差时再评估微调。

---

## 未来工作（未实现）

- **UI**（用户自行调研 LibreChat 等原生 MCP 的工具调用 UI；Streamlit 弃用）。**done**
- **记忆作者质量**：换 `claude-opus-4-8` + 更正式 system prompt，让 `objects` 数组可靠充实（当前网关/sonnet 偏只回 summary，schema 已把 objects 设为可选）。
- **几何投影 `geometry/depth_projection.py`**：VLM 给 ROI → 代码在深度图取鲁棒距离(中位数/最近簇) → 内参反投影 + TF → `abs_pose` + `delta`（远粗近精）。
- **Supervisor**：6 规则确定性状态机（安全/触发云端/重试/探索/导航/完成），`decide()` 可替换；rule0 永远代码。
- **5 个 Skill**：explore/navigate/inspect/capture_scene/emergency_stop（受限 prompt + 工具 allowlist 防幻觉）。
- **本地记忆工具**：`query_topology/update_json/mark_area/get_pose` 暴露给执行者。
- **安全加固**：多机器人 TF frame 命名空间化（agent1）；硬停改 `twist_mux` 优先级（而非刷零速）；车轮存活性（开 `joint_state_broadcaster` 或监 `cmd_wheels`）。
- **一键 bringup**：把 safety_node + TF 合进 robomaster launch。
- **实验**：A→B→A 多轮 vs 基线（Qwen-only 无记忆），指标：单轮延迟、云端调用率、成功率、碰撞数。

---

## 阶段七 · 缺口补全实验（分层 harness 实跑，2026-06-23）

任务：发给小车的记忆**故意删掉红柜后办公区**，让它绕到**红柜后方(x>红柜2.62)**回看办公桌、无幻觉补全。
分层落地：**代码导航**（`agent_core/navigator.py`：`geo_goto`/`geo_face_point`/`geo_route` 用坐标+闭环
move/turn 把车确定性开到办公区内）+ **Qwen 只做语义**（`harness.inspect_and_report`：只看图报物体+方位，
距离由代码按 bearing 从 depth 列接地）+ **回写缺口记忆** + **持答案客观打分**（`Report/score_completion.py`，
答案 `Report/break_room_ground_truth.json` / 人读版 `break_room_map.md`）。新增 depth 中继 `memory_navi/depth_summary_relay.py`
（原始深度图过不了 rosbridge，仿 jpeg relay 压成每列中位距离小话题）。入口 `controll/completion_demo.py`。

**结果：第4轮 PASS** —— 真正到红柜后方 (x=5.2)、朝向偏差 0.4°、Qwen 无 priming 仍正确报“办公桌/办公椅”、无幻觉。

### 途中遇到的问题与处理（务必记牢，后续细化）
1. **启动笔误**：jpeg 中继被填成深度中继的输出话题→话题类型冲突、`look` 拿不到图；**深度采样行带偏低**被近处
   地板主导(读数恒~0.85m)→把行带上移到地平线略上方(0.25~0.55)才读到真实物距。
2. **“红柜后”语义误解**：初版把观察点放在红柜【前方】(0.5,1.0) 越矮墙看，实为没绕到后方。**判据修正：x 必须 > 红柜 2.62**
   （scorer 加 `behind_cabinet` 硬门；旧的“前方越墙看”现在会判 FAIL）。
3. **隔断迷宫 + 无 Nav2**：`wall(1)@x=2.65`(y<1.005) 挡直行，须北上越其北端→沿 y=1.5 走廊东进→南下进办公区。
   无 Nav2 路径规划，**手标路点绕行**（`route` 在答案里），偏脆，换布局要重标。
4. **脱困**：mecanum 麦轮侧滑会**偶发触发 safety 急停**（蹭到侧向柜子）。`geo_goto` 改为：safety 停后查前方是否真被挡，
   若前方畅通=侧向瞬态→**重试**（上限3次），不再一停就放弃。实测两次瞬态停都自动恢复。
5. **对比坐标(打分)的坑**：① scorer 起初把“朝西看到的物体”距离匹配到**身后**最近的同类物（相机根本看不到）→ 加
   **前方视野(FOV±70°)过滤**后才对（办公桌真值 2.1m vs 报 2.13m，1% 误差）。② depth 距离是 bearing→列**中位**，
   测的是该方向最近主面、非具体物体（椅子在桌前会被桌面距离盖过）→ **距离改“参考”不作硬门**（用户：可粗糙）。
6. **不透题（关键纪律）**：移除喂给 Qwen 的 `area_hint`（“红柜后办公区/越过紫墙看”）与 inspector 里“重点是办公区
   (桌/显示器/椅)”的示例——这些是 priming，会诱发/虚高召回。**Qwen 只汇报当前画面所见**；导航坐标只在**代码侧**，
   且**绝不喂给 Qwen**（当前仍从答案硬编=占位，应由感知几何“看红柜→深度定位→算后方”推导，见下）。

### 待细化（下一步）
- **监督者目标由感知推导**替换硬编答案：看红柜→`depth_projection` 定位红柜世界坐标→算“其后方”观察点→`geo_route`。这才真正“不透题”。
- **路线脆**：上 Nav2(`navigate_to` 已接好，缺 Nav2 进程) 或给 `geo_goto` 加基于 lidar 的反应式绕障，替代手标路点。
- **逐物体距离**：bearing→列中位偏粗；可让 VLM 给归一化 ROI、代码取该框内最近簇深度。
- **取景低**：相机 0.21m 且离桌近→只看到桌腿/椅底，显示器上半部出画；非硬伤但可调站距/视角。

---

## 阶段八 · 教 Qwen 调空间工具自主导航（v4 系列，2026-06-25）

给 Qwen 新增 MCP 工具 `if_in_memory`（查物体有无坐标）、`nav_distance`（走到参照物左/右/前 n 米）、
`nav_object`（走到两参照物中间）、`record_area`（模板化登记新物体，Qwen 只填 id/name/confidence，
坐标代码自动补全）。渐进逼近回路：每步看图+记忆+位置变化 delta → Qwen 决策 → `nav_distance` 或定性方向
→ scan 门控执行 → 重观测。被挡回退 `geo_route`。

**进展**：Qwen 从 v3 的"只会 direction_hint"进步到**稳定调 `nav_distance`**（8/8 步全是 NavDistance）。
`if_in_memory`/`not_in_memory`/`RecordArea` 响应链正确工作。geo_route 回退始终可靠→PASS。

**卡点**：
1. Qwen JSON 输出偶尔截断（raw=`{`），导致 Phase A 无物体写入 memory→NavDistance 全 not_in_memory
2. NavDistance MCP 层导航（`_geo_goto_simple`）不收敛——没有 transient safety 重试、step 太保守，
   远不如 `agent_core/navigator.geo_goto` 鲁棒
3. Qwen 用硬编码 `dist_m=2.5` 而非引用 depth 读数（prompt 约束不够）
4. **观测视角缺陷**（用户指出）：当前到观察点后只看一帧就报告——如果车在正确位置但朝向错误
   （比如面对墙而非办公区），Qwen 看不到任何办公区物体、只能报告"墙壁"，不会主动转视角扫视。
   同样，转向后 Qwen 看不到新方向的画面就盲走。

**下一步（用户定）**：
- 观测前"环顾"：到位后不只看一帧，而是 turn left 30°→look→turn right 60°→look 做全景扫视再报告
- 转向前先看：执行 turn 之前先 look 确认转向方向合理
- NavDistance 计算留 MCP，导航执行移交 `navigator.geo_goto`
- 修 Qwen JSON 截断 + dist_m 引用 depth
