# Memory（更改记录 / Changelog）

> 最新在上。每条以日期开头。详情见 [Architecture.md](Architecture.md) / [Process.md](Process.md)。

---

## 7.6 Sunday: 混合标注全栈实跑 7 轮 + 均匀网格覆盖 → 召回天花板=相机高度（详见 Process.md 阶段十四）

### 本会话落地（代码+单测绿、实测跑通、待提交）
- **YOLO 权重损坏修复**：`mobileclip2_b.ts` 被 ultralytics 无代理自动下载成 202MB 截断（首轮 120s"超时"实为下载失败），
  用仓库根 242MB 有效副本覆盖 → 冷加载 3.6s、0 失败。`.gitignore` 补 `mobileclip2_b.ts`。
- **禁记类硬过滤**（`_FORBIDDEN_NAMES`/`_process_rep`）：代码丢 地板/墙/天花板/门/机器人自身 → 记忆 0 残留。
- **Qwen 全图也看一遍**：混合每帧 = YOLO 框命名 ∪ Qwen 全图 inspect（补 YOLO 漏检）。
- **观测站距 0.5→0.7m**：关键是 `pf.OBS_MIN_CLEAR_M`（观测点质量门=车真正停车距离；先只改标注侧没用，补这个才生效）。
- **常驻 YOLO 服务** `yolo_service.py`：模型只加载一次，不再每帧 subprocess+重载，0 失败。
- **单帧共现反合并**（`COOCCUR_EPS_M`）：同帧同名 abs_pose 差>0.2m=两实例禁并（治密集同名家具间距≈坐标噪声）。
- **评分：显示器东×4→region「东区」**（用户认可，GT observable 20→17）：0.13m 相机分不出 0.5m 间距的个数,整片算1召回点。
- **均匀网格覆盖** `_grid_coverage`（**取代 frontier 密度 + Claude 调度官**）：bbox 内 1.8m 铺格,最近未覆盖格→路由→柔性环视→
  标覆盖;网格满则穿插推 frontier 扩张 bbox。治"往一个方向扎、极角落漏"。修两 bug(开局误判全覆盖退出/标漂移站位致同格死循环)。
- **navigator._pose 加固**：保证含 yaw_deg(重试+补默认),治瞬时坏读 KeyError 崩整轮。

### 召回轨迹(GT17,混合)与最终结论
`40→30→45→47→47→47`。**v6c 均匀网格把车开到南墙前 1.2m(厨台/水槽/矮柜),但照样漏——已不是覆盖是识别**:低矮物体
0.13m 相机 1.2m 处低于台面认不出。**召回天花板彻底=相机高度(0.13m)**。覆盖(均匀网格正解)+感知软件(过滤/去重/merge)到头且干净。
**唯一正交杠杆=抬相机 0.13→~0.8m**(一直往后放)。

### "为什么无 YOLO 能到 65%"(用户问,git 取证)
**65% 是真的**(同 GT observable=20,13/20)。但那是提交 `2fe4994` 的**纯 Qwen 精简版(702 行)**——git 证实**无** FAR_LABEL_M
距离门/Claude 调度官/整理官,这仨是**下次提交 `23ee29d` 才加**。落差主因:①**距离门 FAR_LABEL_M=2.5** 大房间丢>2.5m 合法物(65%版没这门,最大嫌疑);
②Claude 调度官方差(已用均匀网格取代);③跨轮车漂移(7轮没 Webots 复位)+单轮方差。**教训:23ee29d 那批"精度/防幻觉"改动用召回换精度、可能过度**。

### filters-off 验证(v7) + 自适应站距失败(v8) → 回退定案 v7
- **v7（关掉 ROI 大小/远距离门 + LOS 穿墙反证，`FILTER_ROI_LOS=0`）**：**召回 47%→59%(10/17)、幻觉不升(19→17)**。
  **证实退化=过度预过滤**——v6c 里 LOS 反证丢 49 + ROI 门丢 24 **大量是贴墙/稍远真家具被误杀**；关掉后回来、YOLO ROI+深度+dedup 足够压幻觉。
  **默认改为关**（`FILTER_ROI_LOS` 默认 0）；噪声交整理层。**印证"关键在整理不在记录"。**
- **东office 没进去（用户目测）**：v7 车到东区但**站进办公区内(2.3<x<3.6)观测点=0**，从开阔东侧 1.5m 外斜看密集堆→桌↔显示器↔椅交叉误标。根因 0.7m 站距门+密集家具站不进去。
- **v8 自适应站距(密集→0.4)失败**：密集近站触发 0 次、还是没进、召回反跌 53%。**机制错**：绑在"重选用尽"上，但车总在开阔侧先找到 0.7 点第0次就返回、轮不到密集兜底；**选点器优化 clearance 最大而非离目标最近**。**已回退 v7**。

### 当前定案配置 + 下一轮（用户定，本轮不做）
**定案=v7**：混合标注全套 + 均匀网格覆盖 + **filters off 默认** + 无自适应站距。最优 v7 58.8%(10/17,逼近历史65%)。
**下一轮解决**：① **整理层（主线）**——把高召回原料(~17幻觉+10误标)用上下文组织成词典式：A 功能子区结构化 + B 上下文纠错(办公区孤立"柜子"多半是桌,救东office) + C 关系(on/in) + D 墙后/自由空间幻觉在整理层剔除(替代预过滤,不误杀)。
② **东office 正确修法**：目标格有物体时一开始就用短站距(standoff_min=0.4),靠 APF 引力拉车进去(非"退无可退才放宽")。③可选:FAR_LABEL_M 大房间放宽重审、相机高度 0.13→0.8m。
关联 [[hybrid-perception-yolo-roi-qwen-name]] [[layered-harness-code-nav-vlm-semantics]] [[explore-coverage-variance-obsgate]]。

---

## 7.6 续: 混合标注 Phase A+B 落地 —— 弃权门验证实验（详见 Process.md 阶段十三）

### 本会话已落地（代码+单测绿+装配 smoke 通过、未提交；**端到端实跑需全栈，本会话未跑**）
- **Phase A 可复用件（纯函数优先）**：
  - `yoloe.py`：`raw_detections_to_boxes`（YOLO 低 conf `raw_detections`→只带 idx/roi/center/conf/label 的框，**不套 canonical_name**、
    label-agnostic）+ `assemble_hybrid_objects`（只留 `keep and 完整 and 有名`，保 YOLO ROI+Qwen name+`verified_by=["hybrid"]`，**不产坐标**）。
  - `harness.py`：`NAME_BOXES_SYS`（弃权门 prompt：先判完整/部分/勉强，**只"完整"才命名 keep=true**；框标错就弃权别配合编名）+
    `name_boxes`（无状态一次性 vision completion，**方案一=整图画编号框一次批量判**）+ 纯 `_parse_box_judgments`（容错+漏报默认弃权）。
  - `draw_dual_boxes.py`：`draw_numbered_boxes`（复用 `_roi_px`/`_load_font`，画蓝框+大号编号，返回 PIL）。
- **Phase B 弃权门验证实验（`HYBRID_REVIEW_DIR`，镜像 `DUAL_REVIEW_DIR`）**：`explore_probe.py` 加 `_hybrid_yolo_boxes`（消费
  `raw_detections`）、`_sweep_vantage` hybrid 分支（混合作驱动、同帧 Qwen-only inspect 作基线旁路）、`_hybrid_review_dump`
  （逐帧存 raw+**编号框图=Qwen 实际所见**+每框 YOLO/Qwen 判定+两路留存）、`_finalize_hybrid_review`/`_write_hybrid_index`
  （`area_hybrid` vs `area_baseline` 各 `score_explore` 打分 + `index.md`）。产物 `Report/hybrid_review/`。
- **单测全绿**：`test_{hybrid_assemble,name_boxes_parse,numbered_drawer}.py`；pycodestyle 新增代码零违规；`import explore_probe`
  全链路 smoke（YOLO 框→Qwen 判定→装配出 `办公椅`@YOLO ROI）通过。

### 范围与判据（用户定）
只做到验证实验就停：跑一轮存图存判定供人工/Claude 审。**弃权门通过判据**：对 YOLO 的墙/半截物/空墙框 Qwen `keep=false` 可靠弃权
+ 混合召回 ≥ Qwen-only 基线（≥0.70）。通过再开工 `PERCEPTION_BACKEND=hybrid`（Phase C）。**风险**：每帧 YOLO 重载慢（先接受）；
完整度门 vs 召回张力（须同时报召回）；`_annotation_ok` FAR 门需回填后补远距丢弃。关联 [[hybrid-perception-yolo-roi-qwen-name]]。

---

## 7.5 续: 深度过滤修复 + 双标注对比 → 定案混合标注架构（详见 Process.md 阶段十二）

### 本会话已落地（代码+单测绿、flake8 干净、未提交）
- **深度反证治幻觉**：`explore_probe` 加 **occ 空旷反证**(`_occupancy_phantoms`，abs_pose 落已扫空旷自由格+邻域无 occupied→丢，
  贴墙真家具豁免) + **LOS 穿墙反证**(`_backfill_geometry_local` 串 occ，观测→坐标视线穿 occupied→丢)。单测 `test_occupancy_phantom.py`(7项)。
  **修正**：上轮"车冲出房间 x=-5.6"是 Claude 幻觉（仅一房间），非真 bug，未修。
- **双标注对比工具**：`DUAL_REVIEW_DIR` 同轨迹**同帧**同跑 Qwen+YOLO，各建记忆图分别 `score_explore` 打分；`harness.inspect_and_report(look=)`
  注入共享帧；`draw_dual_boxes.py`(Noto CJK 字体画中文框，绿=留存/橙=过滤)出三图并列 index。产物 `Report/dual_review/`。
- **YOLO 审阅实验** `YOLOE_REVIEW_DIR`：存 YOLO 画框图+坐标+打分 → `Report/yolo_gate_experiment/`。

### 双标注实测（同轨迹同帧，用户判读）
- **Qwen**：ROI **漂移严重**(坐标错)、命名准、**少幻觉**，但平 0.9 过度自信、桌上一排报成 3 显示器。
- **YOLO**：ROI **锁死不漂移**(坐标准)、**爱幻觉**(空墙高置信)、命名 OOD 乱标、**过度标注**(半截柜子→桌/沙发)、静默漏检。
- 召回单轮 Qwen 6/20 vs YOLO 10/20（**噪声大、别过读**；稳定结论是**定性互补**）。

### 定案：混合标注架构（用户定，取代 §11.6 硬门）
**YOLO 出 ROI(定位)→Qwen 读 ROI 出 ID/名(命名)→代码 depth 映射+坐标重合过滤→入记忆。** 关键 **Qwen 弃权门**：不死板标每个 ROI，
看不清/不确定/形态非几乎完整呈现的**一律不标**→筛掉 YOLO 误标的墙/柜/工作台（不再要求 YOLO 无幻觉）。Claude 补充：弃权门是命门须先验证
(Qwen 平0.9 过度自信是威胁)、YOLO conf 要放低(漏检=召回天花板)、相机高度仍是物理天花板。关联 [[hybrid-perception-yolo-roi-qwen-name]]
[[dual-annotator-comparison]] [[yolo-perception-camera-height]]。

---

## 7.5 Saturday: YOLO 感知支线验证 + 混合架构决策（详见 Process.md 阶段十一）

### YOLO 部署（Codex+本会话，代码后端，未提交）
- 独立 `yolo` conda env（CPU torch 2.12 + ultralytics 8.4.87 + CLIP fork；GPU 被 vLLM 8B 占满仅剩 5.6G 故走 CPU，~30ms/图）。
- `agent_core/perception/yoloe.py`（纯转换函数：0..1000 ROI / 英类→中文名表 / 低 conf 过滤，可脱 YOLO 单测）；
  探针 `yolo_offline_probe.py`/`yolo_live_probe.py`；`explore_probe` 加 `PERCEPTION_BACKEND=qwen|yoloe` 开关。测全绿。
- **`mobileclip2_b.ts`(YOLOE 开集依赖 242MB)走 clash 代理 127.0.0.1:7897 ~10MB/s 25s 下完**（直连 GitHub 20-30KB/s）。权重在仓库根、已 gitignore。

### 关键结论（当前相机高度 0.13m）
- **YOLO 不是 Qwen 可调工具**：MCP 无 yolo 工具、prompt 未提；且 **explore 无 Qwen FC**（确定性编排 + Qwen 当标注器）。yoloe 是代码后端替换。
- **闭集不够**：普通 yolo26n(COCO)缺柜子/办公桌/门只出绿植；**必须开集(YOLOE)或微调**。
- **YOLO 单独用没救回**：live 召回 **5/20=25% FAIL**（vs Qwen 65%）。17 记录=命中5/误标6/幻觉5。离线家具 conf 0.04~0.11(静默漏检)、绿植 0.71。
- **根因精化（不是"距离"，是"视角+域"）**：相机 z=0.13m 极端低仰角(看桌底/侧棱)= 姿态 OOD + Webots 合成纹理 = 外观 OOD →
  区域 embedding 落训练流形外 → 开集 softmax **近随机 argmax**(误标 conf 0.28~0.44) → **A 标成 B**。坐标错 ≠ YOLO 算法，是**幻觉框反投到墙面**
  + **覆盖 bug 车冲出房间**(x=-5.6 虚空)。**YOLO 在合成域也高置信度幻觉(空墙绿植 conf 0.68)→ 推翻"YOLO 不幻觉"。**
- **失败模式互补**：Qwen 会命名不会定位(ROI 漂)+爱幻觉；YOLO 会定位(框锁像素)不会命名(OOD 误标)+静默漏检。**→ 引出混合架构。**

### 用户定的新方向（交下一会话，详见 Process.md §11.6）
1. **深度过滤优化**（位置错=深度/几何过滤不完善；先修 §11.5 两 bug：占用兜底冲出房间 + 深度放行幻觉位置）。
2. **YOLO 分割 + Qwen 解读语义一起注入记忆**：代码实现但**要教会 Qwen 参与语义解读**（≠ 纯代码后端）。
3. **YOLO 当 Qwen 的门**：只有 YOLO 标出的才让 Qwen 识别、不标不识别（治静默漏检+幻觉）。**必须先跑前置实验证明**：①YOLO 标的 Qwen 都能对；
   ②YOLO 无幻觉标注。**方法：把一次实验 YOLO 看到的所有图 + 标注位置存本地供人+Claude 审阅。** 备注：0.13m 下大概率不成立 → 实质是相机高度实验。
4. **相机高度仍是首要物理根因**；YOLO 置信度是干净量尺（跳到 0.5+ = 确认根因是相机）。
- 关联 [[explore-coverage-variance-obsgate]] [[precision-over-recall-explore]] [[no-answer-leak-to-qwen]]。

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
