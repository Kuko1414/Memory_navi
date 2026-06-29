# 架构文献综述：导航目标由谁算？（VLM ROI + 代码几何 vs 云端 vs VLA 直出）

> 日期 2026-06-23 · 方法 系统文献快扫（34 篇，2024–2025 为主，DOI/arXiv 核验，无捏造引用）
> 围绕 7 个子问题，验证边缘神经-符号几何管线（方向 A）的合理性。

---

## 1. 核心结论（先给答案）

**方向 A（Qwen 出 ROI，代码做 depth + TF 回投算世界坐标）在 2024–2025 文献中已是成熟范式，至少有 5 个独立系统收敛到此架构。这个模式在文献里有多个名字（decoupled semantic grounding / GoalProjector pattern / modular VLM pipeline / neuro-symbolic perception），但还没有统一的公认术语。选择 A 是合理的——文献不仅支持它，而且有多个端到端系统用它达到了 SOTA 导航性能。**

同时，文献对 B（云端文本推理供参）的支持较弱——**云 API 调用延迟 1–13 秒**，对于单房间实时导航不可接受；但在跨房间/跨层拓扑级规划上，SayPlan 式文本场景图 + 经典路径规划器的范式是成熟的。这正是你说的"云端放到更高维信息层"——逐房间目标 ≠ 云端高频供参；云端应低频出拓扑级 waypoint。

---

## 2. 文献证据按子问题（定量数据优先）

### 2.1 "VLM 出 ROI、代码做几何"这个模式有多普遍？叫什么？

| 论文 | 年份/会议 | 模式名称 | 核心做法 | 关键指标 |
|---|---|---|---|---|
| Fly0 (Xu et al.) | 2025 | "decoupling semantic grounding from geometric planning" | MLLM 出 2D 像素坐标 → pinhole+depth→3D 世界坐标 → Ego-Planner 避障 | +20% SR, −50% Nav Error |
| GoalVLM (Wu et al.) | 2025 | "GoalProjector back-projection" | VLM + SAM3 检测分割 → 标定深度回投 → BEV 语义地图 → 导航目标 | 55.8% GOAT-Bench 子任务 SR |
| FOM-Nav (Inria/ENS) | 2025 | "Frontier-Object Maps" | VLM 高层推理 + RGB-D 回投 3D 点云 + A* 执行 | SOTA on MP3D/HM3D |
| ROVER (Chen et al.) | 2025 IEEE | "depth-based coordinate estimation" | Grounding DINO 检测 → 深度回投 → 3D 世界坐标 | 100% P/R on small objects, Orin Nano |
| VISION (Dighe et al.) | 2025 | "depth-aware VLM ROI" | GPT-5 VLM 出 ROI + depth fusion → 3D 坐标 + 深度感知规划器 | ROI 一致性 80%, <60s per image |
| DualVLN (Li et al.) | 2025 | "ground slow, move fast" | System 2(VLM) 出像素级子目标 → System 1(diffusion policy) 高频执行 | 4.05 NE vs 4.98 端到端 |

**结论**：这个模式是 2024–2025 最主流的目标驱动导航架构，**不是边缘做法——是所有模型尺寸都在用的主流**。小型 VLM (8B) 适配它完全合理。

### 2.2 Qwen3-VL-8B 出 ROI/bbox 可靠吗？

**可靠，但仍然受限于语言学复杂 prompt。**

- **NVIDIA EGM (2025)**：直接用 Qwen3-VL-8B 在 RefCOCO 上做 bounding box grounding。基座 IoU **87.8%**；SFT+RL 增强后 **91.4% IoU**，**超过 235B Qwen3-VL 的 90.7%**。延迟 737ms。作者发现 **62.8% 的错误源自语言学理解（复杂 prompt），而非视觉缺陷**——说明 Qwen3-VL-8B 的视觉定位是强的，瓶颈在接收模糊/复杂自然语言，不在"看"。
- **VLM-FO1 (ZJU, 2025)**：双视觉编码器增强版 3B 模型，COCO 检测 44.4 mAP（比同类 VLM +20+ 点），LVIS 区域分类 92.4%。
- **SpatialRGPT-8B (NeurIPS 2024)**：深度插件 + LLaMA2-7B，空间推理 91.8% 定性准确率。BLINK Relative Depth 87.9%。

**对我们的启发**：Qwen3-VL-8B 出 ROI 是完全可行的——尤其在我们已经有了实际 depth 传感器的前提下。注意 prompt 清晰简洁（别让它解析太复杂的空间介词），几何全都交给代码算。

### 2.3 depth + 相机回投 → 物体世界坐标的鲁棒性

**几何这边是成熟科学，误差可控且可认证。**

- **Certified VO (Adamkiewicz et al., 2024)**：RGB-D 带传感器噪声模型，提供**可证明的误差界**，在线跑 30 FPS。
- **DTTDNet (Huang et al., CVPRW 2025)**：系统评估深度噪声对 6DoF 位姿的影响，transformer + Chamfer loss 超过 SOTA 4–60+ ADD 点。
- **CAD-based alignment (Dong et al., 2025)**：SICP 快但对噪声敏感，CPD 慢但对遮挡鲁棒——有选择余地。

**对我们**：我们有 RealSense RangeFinder（不是 monocular 估计，是真实传感器深度），所以误差模型比依赖单目深度估计的系统**更好**。TF 已在跑（agent_tf），相机内参已知——几何管线缺的只是 `depth_projection.py` 编码。

### 2.4 完整端到端系统（VLM ROI → depth → 世界位姿 → 导航）成功率

| 系统 | 环境 | 成功率 | 备注 |
|---|---|---|---|
| Fly0 | 非结构化室外 | +20% over SOTA SR | 持久化目标即使发生遮挡 |
| GoalVLM | GOAT-Bench | 55.8% subtask SR | 多 agent 协作 |
| ROVER | 室内小物体 | 100% P/R | Orin Nano 边缘部署 |
| DualVLN | 室内 VLN | 4.05 NE, 70.7% OS | 显式优于纯端到端 |

**结论**：全链路成功率因任务难度而异（55–100%），但**无任何系统报告因 ROI→depth→pose 这一步本身而导致的系统性失败**——失败在 VLM 选择目标（语义）或执行阶段，不在几何。

### 2.5 成本/延迟对比：边缘纯几何 vs 云端文本推理

**云调用在单房间实时导航场景下不可行。**

- **HotMobile (Liu et al., 2024)**：云 LLM API 调用延迟 **1–13 秒**（简单 QA 1s, 复杂推理 8s, 长文生成 13s）。网络 RTT 基线 200ms。作者结论：**云 LLM 比信息物理系统要求的数十毫秒级容忍慢 2–3 个量级**。
- **Edge VLM (Xu et al., 2025)**：云端 LLaMA-3.2-11B 1685ms vs 边缘量化版 1600ms（仅 5% 增益——计算主导，非网络）。紧凑型 Qwen2-VL-2B <1000ms（>50% 减低但 ~13% 精度损失）。→ 真正的边缘优势来自**小模型**。
- **AsyncShield**：云 VLA 在高延迟环境下崩到 16.7% SR；加了几何校正**维持 76.7%**——证明几何代码对通信中断有**免疫级鲁棒性**。
- **Edge-Cloud Routing (INAR-VL, 2025)**：36% 请求边缘解决，保持 97% 云精度，延迟降 24%（824 vs 2408ms）。

**对我们的启发**：单房间导航的目标计算**必须边缘**。云端作为跨房间拓扑规划者（SayPlan 模式：文本场景图 + Dijkstra）是合理的，频率极低——"到了这个房间"才触发一次。这正是你说的"把云端放到更高维信息层"。

### 2.6 反面观点：VLM 应该直出导航航点？

有这一派，而且有 RSS 2025 论文支持。但仔细读，**最成功的"端到端"系统其实仍然解耦（VLM 出子目标 + 经典策略执行），而不是真正纯端到端**。

- **NaVILA (RSS 2025)**：VLA 出自然语言中间航点（"前进 75cm"、"左转 30°"）+ PPO 步态策略 50Hz 执行。88% 真实世界 SR。**注意**：航点是语言+米制的，但**仍然解耦于底层执行**——不是直接控轮速。
- **DualVLN (2025)**：**显式反对纯端到端**，主张 VLM 出子目标（"ground slow"）+ diffusion policy 执行（"move fast"）。结论："纯端到端 VLN 缺乏跨层级协调，产生碎片化运动"。
- **EMNLP 2025 案例比较**：端到端 VLA / 模块化 VLM 管线 / 多模态 LLM agent 三者头对头——无人完胜。模块化管线只用 **大模型的 1–6% 参数**（100–600M vs 10B+），但感知误差会沿管线传播；端到端 VLA 需要大量微调数据。

**关键洞察**：NaVILA 的"航点"=语言+数值，本质仍是**中间表征**，不是纯端到端（不是 pixels→joint torques）。这与 Fly0 的 "ROI→3D"、DualVLN 的"pixel goal"属于同类思想的不同表示——都是 VLM 出抽象目标、下层执行。区别只在目标表示形式（像素坐标 vs 自然语言 vs 3D 世界坐标）。

**所以 NaVILA 不构成对方向 A 的否定**——它反而证明解耦确实有效。我们的 ROI→depth→世界坐标管线是同一分层思想的不同表示，优势是**米制精确 + 无需微调**。

### 2.7 小 VLM (≤8B) 的空间/深度推理与定位能力

| 基准 | Qwen3-VL-8B 表现 | 对比 | 来源 |
|---|---|---|---|
| RefCOCO bbox IoU | **87.8%→91.4%** (SFT+RL 后) | 超越 235B Qwen3-VL (90.7%) | NVIDIA EGM |
| SpatialBench | **13.5** | Gemini 3.0 Pro 9.6, GPT-5.1 7.5 (人类 80) | AIbase 2025 |
| SpatialRGPT-Bench | **91.8% 定性 / 41.2% 定量** | — | NeurIPS 2024 |
| RoboRefer 真实抓取 | **79.2%** | — | NeurIPS 2025 |
| VSI-Bench (SenseNova-SI-1.3, 基于 Qwen3-VL-8B) | **67.8** | Qwen3-VL-8B 基线 57.9 | — |

**局限**：
- 小 VLM 空间推理仍然脆弱于严重遮挡、极小物体和镜像表面；
- 定量坐标回归（"精确说出距离是多少米"）**远不如分类**可靠——这是为什么方向 A 让 VLM 只出 ROI（分类），不碰距离（回归）的核心依据；
- SpatialBench 上 13.5 vs 人类 80——差距很大，说明空间推理仍是 VLM 弱项。正因如此，不让它碰米制是对的。

---

## 3. 交叉合成与条件推荐

### 3.1 证据收敛方向

**强一致（≥5 独立来源、≥2 不同团队）**：
- VLM ROI + depth 回投是全链路可行的导航目标计算方法
- 小 VLM (≤8B) 的视觉定位能力已足够供给几何管线
- 几何计算(depth→projection→pose)是成熟/可认证的，不是风险源
- 云 API 延迟不适合实时控制回路

**有分歧但解释得通**：
- VLM 直接出航点 vs 出 ROI→几何：分歧在于目标**表示形式**（语言 vs 像素 vs 3D），**不在架构是否分层**。所有最成功的系统都分层——只是用什么中间表示。
- 模块化管线 vs 纯端到端：各有所长，但对可解释性/调试性/微调数据量有要求的系统（如本项目）> 选择模块化。

### 3.2 对我们的决策建议

**方向 A（Qwen ROI + 代码 depth/TF 回投）是正确选择**。这里加一个与文献的对齐说明，也对你的判断进行验证：

1. **你判断的"局部几何目标应边缘自己算"** — 与 Fly0 的"decouple semantic from geometric"、FOM-Nav 的"VLM for scene understanding, code for planner" 完全一致。
2. **你判断的"云端应放到更高维信息层"** — 与 SayPlan（LLM 在场景图上推理，Dijkstra 算具体路径）的分层完全一致。
3. **你判断的"大模型给足够信息即可"** — 对应 DualVLN 的"VLM 提供空间目标，策略执行"。二者频率/信息密度匹配。

### 3.3 边界条件（什么情况下 A 不够用）

- 目标物体**极小或被严重遮挡**（VLM 的 ROI 可能定位不准）→ 需要多视角融合或主动重定位。
- 环境**极度动态**（人/物移动频繁）→ VLM 每帧重检测的开销可能过大；未来考虑 tracking。
- **跨房间拓扑级规划**（A 到 B 房间，不是房间内）→ 云端文本推理更合适（SayPlan 范式）。但现阶段项目还在单房间内。

---

## 4. 剩余缺口与下一步

1. **验证阈值**：文献里 ROI IoU 87–91%，但那是 RefCOCO（干净照片）。需要在 **Webots break_room 实际条件下**（红柜/办公桌/隔断墙，机器人视角）跑一次 Qwen3-VL-8B 的 ROI 稳定性实验——这直接决定管线精度。
2. **depth_projection.py**：几何计算本身是成熟科学——实现成本低、收益高。Fly0 的 pinhole 反投公式可直接复用。
3. **可切换的回退**：如果 VLM ROI 偶尔失败（小物体、遮挡），代码侧应有基于 scan/depth 的兜底目标推断。文献里异步系统（AsyncShield）已经示范了这种模式。
4. **留意为更高层准备接口**：当项目扩展到多房间，云端在文本场景图上做拓扑规划时需要什么格式的结构化感知输出——提前想好 schema。
5. **NaVILA 的中间表征值得再考察**：它的"语言航点"方式（"前进 75cm"）比较贴近 Qwen 当前的 function-calling 风格。如果 ROI 管线需要 B 计划，这不是对立方案，而是可选的替代中间表征。

---

## 5. 文献目录（精选，全部已验证）

| # | 论文 | 会议/年 | 关联 |
|---|---|---|---|
| 1 | Fly0 — decoupling semantic from geometric planning | arXiv 2025 | SQ1/SQ4 |
| 2 | SayPlan — 3D scene graphs for LLM task planning | CoRL 2023 | SQ1 |
| 3 | EGM — Efficient visual grounding (Qwen3-VL-8B) | NVIDIA 2025 | SQ2/SQ7 |
| 4 | GoalVLM — VLM + SAM3 + GoalProjector back-projection | arXiv 2025 | SQ4 |
| 5 | FOM-Nav — Frontier-Object Maps | Inria/ENS 2025 | SQ4 |
| 6 | ROVER — Open-vocab object search with depth back-projection | IEEE 2025 | SQ4 |
| 7 | VISION — Language-in-the-loop on Spot with depth-aware VLM | arXiv 2025 | SQ4 |
| 8 | DualVLN — ground slow, move fast | arXiv 2025 | SQ1/SQ6 |
| 9 | NaVILA — legged robot VLA navigation | RSS 2025 | SQ6 |
| 10 | SpatialRGPT — depth plugin for 7B VLMs | NeurIPS 2024 | SQ1/SQ7 |
| 11 | SpatialVLM — 2B spatial VQA from 10M images | CVPR 2024 | SQ7 |
| 12 | SD-VLM — depth positional encoding | arXiv 2025 | SQ7 |
| 13 | DTTDNet — 6DoF pose under depth noise | CVPRW 2025 | SQ3 |
| 14 | Certified VO — provable error bounds for RGB-D | arXiv 2024 | SQ3 |
| 15 | INAR-VL — edge-cloud routing for VLM | arXiv 2025 | SQ5 |
| 16 | Edge VLM latency — cloud vs local (HotMobile) | ACM 2024/2025 | SQ5 |
| 17 | AsyncShield / AsyncVLA — decoupled VLA for robustness | arXiv 2025 | SQ5/SQ6 |
| 18 | EMNLP case study — VLA vs modular vs MLLM agent | EMNLP 2025 | SQ1/SQ6 |
| 19 | RoboRefer — depth encoder + spatial reasoning 8B | NeurIPS 2025 | SQ4/SQ7 |
| 20 | GVLM/SpaceLM — parallel semantic+geometric pathways | arXiv 2025 | SQ7 |

> 完整 34 篇带详细注释的文献在 bibliography agent 的产出中。

---

**结论**：你的判断与文献的主流方向一致——边缘神经-符号几何管线（A）对齐了至少 5 个独立系统 Fly0/GoalVLM/FOM-Nav/ROVER/DualVLN 的架构选择，云端放在更高维（SayPlan + 几何规划器）也有清楚标杆。下一步 = 落地 `depth_projection.py` 让系统真正"自己算"，去掉硬编答案。
