# Memory（更改记录 / Changelog）

> 最新在上。每条以日期开头。详情见 [Architecture.md](Architecture.md) / [Process.md](Process.md)。

---

## 6.22 Monday: 执行工具层 + 闭环运动 + harness 加固（幻觉 vs 不收敛 实验）

### 1. 精选动作/感知 MCP 工具（`ros-mcp-server/ros_mcp/tools/`，新文件 additive，不动 vendored 上游）
- `perception.py::scan_summary`：全精度 scan → 最近障碍{dist,bearing}/8 扇区/clear_path 标量（~50 token，止住上下文膨胀）。
- `agent_actions.py`：`move`/`turn_left_deg`/`turn_right_deg`/`stop`/`look`/`get_pose`/`navigate_to`，命名空间从 `ROS_NAMESPACE` 读。
- **运动改闭环**：开环按墙钟计时严重欠走（move 0.5m→0.07m、turn 90°→30°；根因 mecanum 麦轮各向异性摩擦
  `coulombFriction[0,2,0]` + 仿真<1×实时）。改成边动边读**真值位姿**（gps x,y + imu yaw，到目标即停；
  读 `/dev/shm/agent_safety_<ns>.json` tripped 即停）。实测 **move 0.5→0.506m、turn 90→±90°、0.14m 刹停**。
  注：`webots_ros2_robomaster` 轮径/轮距/电机限速本来就对，**没改**。

### 2. 可复用执行器 + harness（`agent_core/`，新建）
- `executor.py::Executor`：复用 `tool_loop`/`mcp_bridge`，allowlist 只暴露精选工具，留 `brief` 注入口给 supervisor；`call_timeout` 120s。
- `tool_loop.py`：参数化 `tool_choice`（实测 **vLLM(hermes) 不支持 `"required"`**，返回空 → 不依赖强制工具）。
- `harness.py`：Qwen 当**规划器**（结构化子目标 JSON）→ 代码当**执行器**（move/turn 代码闭环+真值核验，report 强制 look）；
  另含**感知反馈环** `explore_and_report`（看→决策→执行→核验循环）。
- `config.py`：加 `ACTION_ALLOWLIST` / `ACTION_SYSTEM_PROMPT`。

### 3. 核心实验结论（详见 `Report/harness_eval.md`）——幻觉与不收敛是同一枚硬币两面
- **不加约束**（单段对话自跑整任务）→ 第 6 步**幻觉**（叙述代替指令 / 没做到说做到 / 看图说谎）。
- **严格约束**（route-B：每子目标新上下文 + 代码核验）→ **零幻觉**；简单 demo（左走→右转→看图报告）Qwen 自分解 7 步、ALL VERIFIED、报告与真实帧吻合。
- **更难任务**（缺口记忆 + 绕到红柜后方）→ 真实坐标做答案 key 客观打分：**VLM 自主目标导航不收敛**（原地打转，距目标 3.52m 没到）。
- **结论**：放任则编、管死则瘫；正确解 = **确定性代码做导航 + VLM 只做语义**（验证 proposal 角色分工）。VLM 需"上级"下达空间指令。
- 评估资产：`Report/break_room_ground_truth.json`（红柜=`cabinet(1)`@(2.62,0.22)，后方办公区≈(3.18,-0.3)，真实=办公桌/椅/显示器/键盘/绿植）。

### 4. 途中小问题（一句话）
- `ws_manager.receive` 超时即 close 连接丢订阅 → receive 超时须 > 位姿周期（gps 10Hz），设 0.5s。
- `tool_choice="required"` vLLM 不支持；`turn_deg` 带符号被 Qwen 搞混 → 拆 `turn_left/right_deg`；agent0/agent1 同在原点附近易混淆"没动"。

---

## 6.20 Saturday: 薄 Agent 核心 + 垂直切片打通 + 仿真硬件(Lidar/深度/TF) + 安全底层节点

### 1. 薄 Agent 核心 `memory_navi/controll/controll/agent_core/`（新建）
- 决策：**自己搓薄核心**，不上 LangGraph/AutoGen 等重型框架（MCP 已是工具层；顶层确定性；护延迟/依赖；论文可解释）。
- 新增模块：`config / llm_client / image_utils / mcp_bridge(后台 asyncio 常驻 MCP 会话+同步 RosTools) / tool_loop(非流式 FC 循环) / cloud{schema,providers,memory_author} / memory/fs_memory(文件系统 STG)`。

### 2. 垂直切片 `slice_demo.py`（新建，两证明全绿 ✅）
- Proof 1：Qwen3-VL-8B 经 function-call + MCP **自主** `get_topics→get_topic_type`，报出 `/agent0/camera/image_color = sensor_msgs/Image`。
- Proof 2：Claude（记忆作者）观察相机图 → C 格式 JSON → 校验 → 写 `memory/sim/roomA/area.json` → 读回。
- 配套：`start_all.sh` 加 `--enable-auto-tool-choice --tool-call-parser hermes`（开 Qwen3 FC）。
- 路上修的真问题：① 相机 raw bgra8(1.23MB) 过不了 rosbridge → 新增 `image_jpeg_relay.py` 转 JPEG(~28KB)；② Claude 走网关 `ANTHROPIC_BASE_URL`+`ANTHROPIC_AUTH_TOKEN`(bearer, `claude-sonnet-4-6`)，非 API_KEY。
- 已知：网关/sonnet 偏只回 summary，`objects` 数组暂设为可选（待 opus + 强 system prompt）。

### 3. 仿真硬件（改 `webots_ws/src/webots_ros2_robomaster/`，两台车 agent0/agent1）
- `robomaster.urdf`：Lidar 映射修正（绝对`/scan`+lazy → 相对`scan`+`alwaysOn`）；**新增 RangeFinder 深度**映射 → `/agentN/camera/depth/{image,camera_info,points}`；新增 GPS 映射；IMU 改 `/agentN/imu` 常发。
- `break_room.wbt`：Lidar `minRange 0.2→0.05`（**关键**，否则 0.15m 测不到）；Camera 旁加 `RangeFinder`（同位姿/640×480）。
- TF（新建 `memory_navi/sim_bringup/`）：`webots_pose_tf.py`（GPS+IMU 真值 `map→base_link`）+ `agent_tf.launch.py`（静态 `base_link→{lidar_link,camera_link}`）。实测 `map→base_link` = 出生位姿，正确。
- 实测新话题全部上线：`/agent0/scan`(min 0.05, 360°)、`/agent0/camera/depth/*`、`/agent0/gps`(PointStamped)、`/agent0/imu`(Imu)、`/tf`。

### 4. 安全底层 `memory_navi/safety_node.py`（新建，已测）
- 独立 rclpy 进程，不依赖 vLLM/MCP/agent_core。
- **传感器存活性**：50Hz 短窗校验 scan/imu/gps/color-info/depth-info；任一超阈值→`sensor_fault`。
- **安全距离硬停**：`/agentN/scan` 最小距离 <0.15m → 刷零速到 `/agentN/cmd_vel`（绕过 LLM）。
- **状态**：`/agentN/safety/status` + `/dev/shm/agent_safety_<ns>.json`。
- 实测：启动全 stale→sensor_fault；上线→ok；阈值测试→near_obstacle 且 cmd_vel 被刷零速 ✅。

### 5. 文档（新建）
- `Architecture.md` / `Process.md` / `Memory.md`（本文件）。

### 复审待定项（已交付用户）
多机器人 TF frame 命名空间化（agent1）；硬停改 twist_mux 优先级；车轮存活性；safety_node+TF 并入一键 launch；记忆作者换 opus。

---

## 6.25 Thursday: 分层 harness 实跑验证 + 空间工具 v4 + 观测视角设计

### 1. 分层架构确立（文献 + 实测）
- 经 34 篇文献深研（`Report/architecture_literature_review.md`），确定三层分工：
  Qwen(边缘)标 ROI+可通行方向 · 代码(边缘)做 depth 回投+TF+导航+安全 · Claude(云端)仅卡住时低频文本援助+写记忆。
- 核心铁律：**不让任何神经网络直出米制坐标**（SpatialVLM 仅 37% 精度）。几何 = depth+pinhole+TF = 纯代码。

### 2. 新增模块（本次新建/改）
- **`agent_core/geometry/depth_projection.py`**：pinhole 反投影 + TF 链 → 物体世界坐标；
  `derive_observation_point` 从物体坐标推导航观察点。10 随机测试 <0.001m 误差，3px jitter ~3cm。
- **`depth_summary_relay.py`**（新增 ROS 节点）：原始 depth 32FC1 过不了 rosbridge →
  ROS 端原生压缩成左→右每列中位距离小数组 `/agent0/camera/depth/columns`，MCP `perception.py::depth_summary` 读取。
- **`ros_mcp/tools/spatial_nav.py`**：新增 `if_in_memory`(查物体坐标) / `nav_distance`(走到参照物左/右/前 n 米) /
  `nav_object`(走到两参照物之间) / `record_area`(模板化登记新物体，Qwen 只填 id/name/confidence，坐标代码自动补)。
- **`harness.py`** 升级 INSPECT_SYS（+bbox_center +passable）、新增 `inspect_with_memory`/`explore_with_memory`/`verify_passable`。
- **`completion_demo.py`** v4：渐进逼近回路（位置感知 delta + NavXxx 工具调用 + geo_route 回退）。
- **`Report/score_completion.py`** 新增 target_from_perception / target_deviation_m / claude_calls。
- **`Report/break_room_ground_truth.json`** + **`break_room_map.md`**：完整全局语义地图（答案，仅供评分）。
- **`ros-mcp-server/MCP_SETUP.md`**：8 进程一键启动清单（含新增的 depth 中继）。

### 3. 关键实验结果（缺口补全，break_room agent0）
- **v2（depth 回投去硬编）**：PASS ✅。导航目标从 Qwen ROI+depth 动态算、(5.18,0.27)，偏差 vs 答案 0.77m，
  Claude=0 次，无幻觉。隔断绕行路点仍手标。
- **v3（Qwen 记忆驱动探索）**：Qwen 6 步一致选 left、方向正确，但从未自判 arrived（太保守）。步数耗尽后回退 geo_route → PASS。
- **v4（空间工具 nav_distance）**：Qwen 稳定调用 nav_distance 但导航执行不收敛（MCP 层 `_geo_goto_simple` 不如 agent_core `navigator.geo_goto` 鲁棒）。
  geo_route 回退始终可靠。

### 4. 观测视角缺陷（用户指出，待修）
- 当前到位后只看**一帧**就报告——若车在正确位置但面壁，Qwen 只看到墙、不会主动转视角扫视。
- 转向前不先看图——Qwen 在转之前看画面、转之后不知道新方向有什么，盲走可能撞桌。
- **正解**：到位后"环顾扫视"（turn 30°→look→turn -60°→look→回到原位，综合多帧报告）；
  转向前先 look 确认方向；看不到≠没有→应 `reshoot` 换角度。

### 5. 途中学到的问题
- JPEG 中继被误配成 depth 输出话题 → 话题类型冲突；depth 采样行带偏低被地板主导→上移到地平线上方(0.25~0.55)。
- "红柜后"语义误解：初版观察点 (0.5,1.0) 在红柜前方越墙看，不算"后方"→修正为 x>2.62 硬门。
- 隔断迷宫无 Nav2→手标路点绕行（偏脆）；mecanum 侧滑偶发 safety 误停→`geo_goto` 加 transient safety 重试。
- 评分距离匹配到身后看不见的同类物→加前方视野 FOV±70° 过滤；depth 距离 bearing→列中位偏粗→改为"参考"不做硬门。
- **不透题原则**：移除嗂 Qwen 的 area_hint 与期望物清单（防 priming）。导航坐标只在代码侧，不进 Qwen prompt。
- safety_node（原生 rclpy）与 MCP move（经 rosbridge）都写 cmd_vel 互相覆盖→正解应改 twist_mux 优先级。

---

## 6.25 问题清单 & 解决方案（19Q，全部已定方案）

> 从 `current_problem.md` / `planTosolve.md` 合并。每条格式：问题 → 方案。

### 感知与观测
- **Q1 JSON 截断（raw=`{`）**：根因 max_tokens=700 偏小。修→1000；代码侧检测不闭合时 retry。
- **Q2 到位面壁**：加环顾扫视 `sweep_observe`（每 30° look，max 11 次），info≥3 条或扫满即停。
- **Q3 转向盲走**：转向→look→Qwen 确认"前方可通"→再 move。
- **Q4 dist_m 硬编 2.5**：NavDistance 改为 `NavDistance(Object, u, v, direction)` — Qwen 选像素，代码查深度（3×3 小区块取中位）。
- **Q5 depth 行带过滤过度**：行带扩到 0.15–0.55，保留部分地面信息（token 量不变）。地面分格检测暂不替 Qwen 做。

### 导航与执行
- **Q6 NavDistance 不收敛**：计算留 MCP 工具，导航执行移交 `navigator.geo_goto`（transient safety 重试已鲁棒）。
- **Q7 隔断无 Nav2**：当前先做 scan 反应式绕障；`navigate_to` 接口预留，条件成熟切换。保持接口一致。
- **Q8 cmd_vel 互相覆盖**：正解 twist_mux（safety 最高优先级）。当前软件 guard 兜底。
- **Q9 mecanum 瞬态 safety**：`geo_goto` 已有重试，MCP `_geo_goto_simple` 也加同逻辑。

### 记忆与语义
- **Q10 RecordArea abs_pose=null**：RecordArea 后 completion_demo 侧立即 depth_projection 回填坐标。
- **Q11 物体名不匹配**：`lookup_by_position(bearing, depth_m)` — 按位姿+深度反查 ID，余量 ~0.3m。
- **Q12 Claude 脱困未触发**：代码检测连续 N 次失败→主动触发。Claude 收 pose+scan+已知物体坐标+记忆地图，纯文本推理。

### 架构
- **Q13 behind 写死**：Qwen 出 `spatial_intent: "behind"|"left_of"|"right_of"|"front"`（纯词），代码翻译方向→几何。
- **Q14 face_target 来自答案**：去掉。到位后环顾扫视（Q2）自然找最佳视角。

### 评估
- **Q16 scorer 只评最终报告**：确认。途中效率指标后续加。
- **Q17 depth 列中位偏粗**：改为按 bbox_center 多采样点取中位。

### 工程（自修）
- **Q15** topic 冲突已修 · **Q18** relays 写 start_all.sh · **Q19** bash 默认去 conda · **Q8** twist_mux 记待办

---

## 7.1 初探稳定化：改用 Qwen 作者 + 客观召回打分（阶段九续）

- **打分地基**：`Report/score_explore.py`（召回=位置≤1m 且类别命中/该区可见真值；报命中/漏记/误标/幻觉）
  + `Report/explore_room_ground_truth.json`（break_room 全屋真值 + observable 分母标记）。修 **A3 实锤 bug**：
  aliases 合并未去重（曾两次 `red_cabinet`）→ 归一化去重、剔除等于主名。
- **作者对比（用户"谁准用谁"）**：同帧序列 **Qwen 54.5% > Claude 36.4%**（Claude 英文乱标、方差大；Qwen 原生中文、免费）。
  **GPT 端点 = Codex/ChatGPT-账户代理，只放行 codex 模型、视觉调用全 500 → 不可用**。
  → explore **改用本地 Qwen 作者**（inspect 出物体+ROI，代码 depth_roi 回填几何、去重整理），**移除 Claude 云调用**。
- **稳定化改动**：**C1** `INSPECT_SYS` 强制规范中文单数名 + 禁墙/地/天花板/自身/门 + 给 ROI（名字全清洁，16 条噪声清零）；
  **A4** `_dedup_objects` 加 `_name_compat`（只并同类，桌椅不再并坨）+ `DEDUP_M 0.7→0.6`；
  **A6** `MAX_VANTAGES 16→26` + `MIN_VANTAGE_SPACING_M=1.0`（整屋覆盖 + 止住原地重扫）；
  **A7** `_validate_doors`（count≥2 + 近周界，16→5 门）；**A2** size 任一维>2.5m 标 `size_unreliable` 置 null；
  **B1** `_merge_obj_pair` 几何字段取**更近帧**（非高置信更远帧）。删死代码 `EXPLORE_SYS/_explore_decide/_quadrants/_pixel_to_world`。
- **验收**：scope=整个 break_room、多区同一 `area.json`。**召回 55%→65%(13/20)**，达用户"至少 13-15"下限；名字全清洁、门已校验。
  单测 `test_explore_stable.py` 9 项全绿。**教训**：跨帧命名发散是召回杀手（C1 强制规范名一举解决）；
  名感知去重比纯位置去重关键（近距不同类家具不能并）；几何字段应按**距离**融合(近帧准)而非置信度。
  残留：SW 密集角 6 近似矮盒分不出；覆盖会往空旷远区乱跑浪费 vantage；跨轮需 Webots `Ctrl+Shift+R` 复位车。
  关联 [[layered-harness-code-nav-vlm-semantics]] [[no-answer-leak-to-qwen]] [[dual-model-shared-memory-merge]]。

---

## 6.30 共享记忆 + 几何回填 + 初探重写（阶段九）

### 统一记忆 & 双模型协作
- **MCP 记忆工具共用**：`memory_tools.py` 注册 `read_area_memory/upsert_object/get_room_boundary/list_areas`，
  Qwen 与 Claude 同一份本地 `area.json` 读写。物体规范字段：id/name/aliases/abs_pose/size/roi/confidence/verified_by。
- **一个写一个擦（已修）**：`upsert_object` 原用 `dict.update()` **整字段覆盖**，Claude 写 `verified_by` 擦掉 Qwen 的。
  改为**数组字段(verified_by/aliases/affordance)并集、标量覆盖**。`dual_collab_test.py` 验证：同物两模型 verified_by
  并存、按 name 合并不重复。**教训：多写者共享记录时，数组字段必须 union 语义，不能 last-writer-wins。**

### 几何回填（方法B size + 方法a boundary）
- **分工**：模型只标 ROI，**代码算几何**（用户："没必要让模型来算"）。`depth_roi` MCP 工具读一帧 depth 批量算 ROI 深度统计；
  `depth_projection.roi_to_size/roi_center_pixel/boundary_from_points`；`MemoryAuthor._backfill_geometry` 填 size+abs_pose。
- **厚度坑（已修）**：ROI 含连续地面/墙的深度斜坡时，min-max 厚度被拉到整跨度（沙发"厚 5.5m"）。
  改 **gap 切簇取含 median 那簇 + 簇内 p15/p85 分位数**修剪 → 0.85m。
- **视角硬约束（非 bug）**：机器人相机仅 0.21m，桌面物体（显示器）bbox 越过桌面看到**远墙**，depth 不是物体表面 →
  size 虚大、abs_pose 偏。数学修不了，**深度无效的对象 size/abs_pose 留 null 不瞎编**（实测 8/9 填、null 的都对）。
- **Claude 结构化**：网关用 tool_use 回空 objects → `AnthropicProvider.annotate` 改**纯 JSON 主路径**。
- **schema 修**：`view_pose` 键 `yaw_deg`→`yaw`（否则 record 校验失败丢整帧）。

### 初探模式（explore_probe）重写 — 代码 frontier + Qwen 顾问 + Claude 语义
- **P1 Qwen 选视角→原地打转（核心，已修）**：让 Qwen 选"去哪看"会卡死（8 步 5 步强制脱困、没出起点 2m 盒）。
  **正解=代码持有覆盖保证**：`PITCH=1.4m` 网格 frontier 从起点铺开、VFH 开过去、撞墙格标 blocked、覆盖完才停。
  **Qwen 降为顾问**（`_advisor`）：只报门/开口方向给代码**重排未覆盖格优先级**，删不掉格、停不了覆盖。
  **回答"Qwen 要彻底退出导航吗"——不，降为顾问（建议方向+标门），代码兜底覆盖。** 重跑覆盖 7.2×6.8m、0 卡死。
- **P2 hazard 字符串崩记录（已修）**：Claude 返回 `hazards:["..."]`（字符串）不符 schema 对象 → 丢整帧标注。
  `_finalize` 把字符串 hazard 包成 `{note:str}`。
- **P3 记忆噪声（已修）**：44 物体含 19 墙地影/反光 + 2 自观测(robot_body) + 显示器拆 5 条。
  `SYSTEM_RECORD_JSON` 收紧：只记离散家具/设备，不记建筑表面/视觉假象/机器人自身部件，规范单数命名。重跑 9 个干净物体。
- 入口：`explore_probe.py`(初探) / `autonomy_probe.py`(补全) / `dual_collab_test.py` / `geom_backfill_test.py`。
  关联 [[dual-model-shared-memory-merge]] [[harness-hallucination-vs-convergence]] [[layered-harness-code-nav-vlm-semantics]]。

### 初探模式三类修复（覆盖收敛 / 环视 / 写盘正确性，同日续）
- **覆盖 streak 根因**：`_pick_target` 的 Qwen-hint 优先级是**绝对二元**（命中提示锥碾压所有非提示格、无视距离）→
  一直追"门在前方"往北 streak、东半区不探、卡死。**修=距离带为主键、hint 仅同距带 tiebreak（近优先）**；
  外加 **bbox 给 frontier 播种**（墙点扫到哪、远角就入 frontier）、终止改 covered/上限/卡死（删 BUDGET=8）、
  撞墙 `_blocked_cone`(共线更远格一并 blocked)+净位移卡死检测+逃向最远格。**教训：让"顾问"用绝对优先级，等于又把决策权交回模型→退化成模型驱动 streak；顾问必须是不能跨越硬约束的弱信号。**
- **几何校验去重（用户关键思路）**："据小车位姿+深度算出物体世界位置，重复(同位)就过滤/合并"。`_dedup_objects`：
  可信物体(界内+地面高度带)按**世界位置**去重(跨命名同位=同物→合一、别名入 aliases)；不可信(高处/越界/无 abs_pose)按名归并+标 `size_unreliable`+坐标留 null；跨桶去重。**99 原始观测→18 干净物体（首跑 71 噪声）**。
  **教训：单帧 depth 反投对高处/远物不可信(打到远墙)，位置去重前必须先按"高度带+边界"过滤不可信观测，否则同物散成多条。**
- **空缺感知**：记录时把已记物体名 `known_objects` 喂记忆作者→只补缺口、不每帧重复登记（源头降噪，配合事后几何去重）。
- **单一并集写路径**：`FsMemory.upsert_object`（数组并集+同名近=同实例/远=多实例），explore 落盘改逐物体 upsert（不再整条覆盖）；
  门 `_cluster_doors` 世界点聚类去重 → 每门写 `topology.json` 边(to=占位未探区)。`MemoryAuthor.record` 加 `write=`/`known_objects=`。
  关联 [[dual-model-shared-memory-merge]] [[observation-turn-look-design]] [[no-answer-leak-to-qwen]]。
