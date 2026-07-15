#!/usr/bin/env python3
"""初期房间探索 v3：从零覆盖 + Qwen 本地语义标注 + Claude 象限调度官（叠加不替换）。

角色分工（本模块严格遵守，见 Report/proposal.md）：
- Qwen（本地、免费）——逐帧看图标物体+ROI（感知），代码用 depth_roi 回填几何。
- Claude（云端、结构性低频触发）——【象限调度官/Role-1】：只读符号地图(bbox+四象限覆盖统计+物体坐标
  + 代码筛好的候选格)，【从候选里选一个 id】指出下一步该补哪个欠覆盖象限；绝不看像素、不产坐标。
- 代码——几何/去重/导航(geo_goto)/安全/覆盖保证。frontier 兜底无条件跑，守住 ≥65% 召回下限。

三段式主流程：STAGE A recon（起点环视→四角→中心 360° 自举 bbox）→ STAGE B 调度官轮询（Claude 选象限、
scan 复核假墙、VFH 去补、物体格停 ≥0.5m）→ STAGE C frontier 兜底（覆盖保证）。三段共用 _absorb_sweep
把观测折进同一状态袋，统一喂不改动的去重+写盘路径。

运行：conda run -n vllm python explore_probe.py（需 ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL gateway）。
"""
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPORT_DIR = os.path.join(REPO, "Report")
for p in (HERE, REPORT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent_core import config, harness
from agent_core import navigator as nav
from agent_core.cloud.providers import make_director
from agent_core.executor import Executor
from agent_core.geometry import depth_projection as dp
from agent_core.geometry import occupancy as oc
from agent_core.geometry import potential_field as pf
from agent_core.memory.fs_memory import FsMemory
from agent_core.perception import roi_candidates as rc
from agent_core.perception import yoloe

AREA = "explore_room"          # 独立 area，不碰 autonomy_probe 的 break_room 记录
RUN_OUT = os.path.join(REPORT_DIR, "explore_run.json")
LAST_RUN_METRICS = {}          # run_pipeline 结束时填；供 ExploreMode 取指标（不改内部流水线）
START_XY = (-0.5, -1.0)        # 休息室起点（初始条件，非答案）
START_TOL_M = 1.5

# 感知后端：默认 YOLOE（微调 low_view_furniture_v1 全权语义标注；Qwen 不参与命名）。
# 角色分工：YOLO 出物体名+ROI（感知），代码用 depth_roi 回填几何；profile 逐类阈值 + confirmed/candidate 隔离。
# 仍可 PERCEPTION_BACKEND=qwen / hybrid 回退旧路径（微调前的 Qwen 标注/混合标注）。
PERCEPTION_BACKEND = os.environ.get("PERCEPTION_BACKEND", "yoloe").strip().lower()
YOLOE_PYTHON = os.environ.get("YOLOE_PYTHON", "/home/kuko/miniconda3/envs/yolo/bin/python")
# 部署 profile（逐类发布阈值 + candidate-only 类 monitor/kitchen_counter/sink）；yolo_offline_probe
# 读它做 confirmed(过阈值→可导航) / candidate(低于阈值/仅候选类→交复核) 隔离。见 deploy/README.md。
YOLOE_PROFILE = os.environ.get(
    "YOLOE_PROFILE",
    os.path.join(REPO, "memory_navi/training/low_view_furniture/deploy/low_view_furniture_v1.json"),
).strip()
# 微调权重（*.pt 不进 git，留本地；路径须与 profile 内 weights 一致）。
YOLOE_MODEL = os.environ.get(
    "YOLOE_MODEL",
    os.path.join(REPO, "models/low_view_furniture/low_view_furniture_v1/best.pt"),
)
YOLOE_DEVICE = os.environ.get("YOLOE_DEVICE", "cpu")
# conf 下限取 profile 的 candidate_conf(0.05)：低分先落 candidate，profile 逐类阈值再决定升 confirmed。
YOLOE_CONF = float(os.environ.get("YOLOE_CONF", "0.05"))
YOLOE_TIMEOUT_S = float(os.environ.get("YOLOE_TIMEOUT_S", "120"))
YOLOE_TMP_DIR = os.environ.get("YOLOE_TMP_DIR", "/tmp/yoloe_explore")
# 审阅实验：非空则把每朝向 YOLO 画框标注图 + 检测(含反投世界坐标)存该目录，供人+Claude 审阅
# 「①YOLO 标的 Qwen 能否识别 ②YOLO 是否幻觉标注」（见 Process.md §11.6-3）。默认空=不影响现有流程。
YOLOE_REVIEW_DIR = os.environ.get("YOLOE_REVIEW_DIR", "").strip()
# 双标注对比实验：非空则同一轨迹同一帧【同时】跑 Qwen 与 YOLO 两个标注器，逐帧存原图 + 各自标注清单，
# 各建一张记忆图并用 score_explore.py 分别打分，全部落该目录，供审阅「各标什么/ID是什么/各错在哪」。
DUAL_REVIEW_DIR = os.environ.get("DUAL_REVIEW_DIR", "").strip()
# 混合标注·弃权门验证实验（Phase B）：非空则同一帧【YOLO 出低 conf ROI → Qwen 逐框命名+完整度判定】作驱动，
# 【同帧再跑 Qwen-only inspect】作基线对照；逐帧存原图+编号框图+每框判定，两路各建记忆图分别打分。
# 供人+Claude 审「弃权门能否可靠弃 YOLO 的墙/半截物/空墙框」。默认空=不影响现有流程。
HYBRID_REVIEW_DIR = os.environ.get("HYBRID_REVIEW_DIR", "").strip()
HYBRID_YOLO_CONF = float(os.environ.get("HYBRID_YOLO_CONF", "0.05"))  # 低 conf 多提议保召回，靠弃权门兜假阳
# 开关（阶段十四定案=默认关）：关掉 ROI 大小/远距离门 + LOS 穿墙反证。实测(v7)关掉后召回 47%→59%、幻觉不升
# （YOLO ROI 强跟踪 + 深度值过滤 ROI-drift/占用反证 + 语义链接 dedup/consolidate 仍在压幻觉）。这俩预过滤会误杀
# 贴墙/稍远真家具→默认关；噪声(幻觉/误标)交给下一步【整理层】用上下文清理。设 FILTER_ROI_LOS=1 可恢复旧硬过滤。
FILTER_ROI_LOS = os.environ.get("FILTER_ROI_LOS", "0").strip().lower() not in ("0", "false", "no", "off")
_REVIEW_SEQ = [0]             # 全局递增序号（脚本内唯一命名，不用时间/随机）
_VANTAGE_SEQ = [0]           # 全局递增 vantage 序号（供审阅定位是第几个观测点）
_YOLO_RECORDS = []          # 双标注模式下 YOLO 旁路的逐 vantage 记录（另建 YOLO 记忆图打分）
_YOLO_CANDIDATE_RECORDS = []  # 低阈值/不达发布门类别；单独落盘，绝不参与导航或 confirmed 记忆
_BASELINE_RECORDS = []      # 混合验证模式下 Qwen-only 基线旁路记录（另建基线记忆图对照打分）

# —— 覆盖收敛参数（代码持有覆盖保证；vantage=云端标注成本上限，nav_steps=平移硬上限，两者解耦）——
MAX_VANTAGES = 26            # 全景视角上限（回退到 v4 稳定版：34 摊薄预算反伤召回、DFS 又 0 触发；26 配 Fix1/2
#                             步效修复=稳定 ~50-70%(方差)。DFS 下探/改动2 保留但空转，留待日后"织入常规选点"再启用）
MAX_NAV_STEPS = 200          # 平移步(geo_step_open)硬上限（180→200：小幅buffer；真正省步靠"不往家具里设点+早停弹跳"而非加预算）
MIN_VANTAGE_SPACING_M = 1.0  # 新 vantage 距上一个 vantage 至少这么远才环视（防 _drive_to 卡住时原地重扫浪费）
STUCK_LIMIT = 2              # 连续 STUCK_LIMIT 轮净位移<阈值 → 触发脱困/放弃


def _hr(t):
    print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


def _pose(ex, retries=3):
    last = {}
    for _ in range(retries):
        p = harness._pose(ex)
        if isinstance(p, dict) and "x" in p:
            return p
        last = p
        time.sleep(0.3)
    return {"x": 0.0, "y": 0.0, "yaw_deg": 0.0, "_pose_error": str(last)}


def _qwen_obj_abs_pose(o, pose):
    """Qwen inspect 物体(bbox_center[0-1000] + 代码接地 distance_m) → 世界 abs_pose。

    与 Claude 物体走同一几何反投(back_project)，使两套地图能用同一打分器公平对比。
    缺 bbox_center/距离/位姿则返回 None（该物体无位置、按不可信处理）。
    """
    bc = o.get("bbox_center")
    dist = o.get("distance_m")
    if not (isinstance(bc, (list, tuple)) and len(bc) == 2):
        return None
    if not (isinstance(dist, (int, float)) and dist > 0):
        return None
    if not (isinstance(pose, dict) and pose.get("x") is not None and pose.get("yaw_deg") is not None):
        return None
    K = dp.DEFAULT_K
    try:
        upx, vpx = dp.qwen_norm_to_pixel(float(bc[0]), float(bc[1]), K["width"], K["height"])
        return dp.back_project(upx, vpx, float(dist), K, None,
                               robot_pose={"x": float(pose["x"]), "y": float(pose["y"]),
                                           "yaw_deg": float(pose["yaw_deg"])})["abs_pose"]
    except Exception:  # noqa: BLE001
        return None


# 禁记类（建筑表面 / 门作为物体 / 机器人自身）——代码层硬过滤，不靠模型自觉（实测 Qwen 会把
# 近距离红柜看成"门"、把桌面/地板报成"地板"、偶把自身报成"机器人"）。门另经 advisor→doors_raw 记入
# 拓扑边，故从 objects 里丢；地板/墙/天花板/踢脚线/自身一律不入物体库。
_FORBIDDEN_NAMES = {
    "地板", "地面", "floor",
    "墙", "墙面", "墙壁", "wall",
    "天花板", "ceiling", "踢脚线", "梁", "隔断", "隔断墙", "partition",
    "门", "门口", "通道", "door", "doorway", "gate",
    "机器人", "机器人本体", "底盘", "轮子", "robot", "wheel", "self", "chassis",
}


def _is_forbidden_name(name):
    """建筑表面/门/机器人自身等禁记类 → True（代码硬过滤，见 _process_rep）。"""
    return _norm_name(name) in _FORBIDDEN_NAMES


def _annotation_ok(o):
    """Task 3 标注过滤谓词：远距离过报 / 视野过小(团在一起) → 丢弃。返回 (ok, reason)。

    只用 Qwen 输出 + 代码接地距离(distance_m)，在几何回填【前】过滤：省一次脏 ROI 的深度读，
    也避免远/团物体污染去重。见 Report/qwen_labeling_issue/README.md 对策①。
    """
    if not FILTER_ROI_LOS:                   # 实验：关掉 ROI 大小/远距离门（对照召回/幻觉）
        return True, ""
    dist = o.get("distance_m")
    if isinstance(dist, (int, float)) and dist > FAR_LABEL_M:
        return False, f"dist {dist:.2f}m>{FAR_LABEL_M}m(远距离过报)"
    roi = o.get("roi")
    if isinstance(roi, dict):
        _, _, w, h = dp._roi_to_frac(roi)
        if w * h < MIN_ROI_AREA_FRAC:
            return False, f"roi面积 {w * h:.4f}<{MIN_ROI_AREA_FRAC}(视野过小)"
        if min(w, h) < MIN_ROI_DIM_FRAC:
            return False, f"roi最短边 {min(w, h):.3f}<{MIN_ROI_DIM_FRAC}(视野过小)"
    return True, ""


def _backfill_geometry_local(ex, objects, view_pose, occ=None):
    """方法B 几何回填：对带 roi 的 Qwen 物体，读一帧 depth 批量算 size + abs_pose（原地写）。

    与 MemoryAuthor._backfill_geometry 同一管线（depth_roi 近带聚类 + roi_to_size + roi 中心反投），
    使 Qwen 作者地图与 Claude 走同样的几何接地（abs_pose 比 bearing 列距离更准）。失败静默跳过。
    """
    indexed = [(i, o) for i, o in enumerate(objects)
               if isinstance(o, dict) and isinstance(o.get("roi"), dict)]
    if not indexed:
        return
    for _, o in indexed:
        o.setdefault("geometry_status", rc.GEOMETRY_NO_DEPTH)
    rois = []
    for _, o in indexed:
        probe = yoloe.mask_polygon_to_depth_probe_roi(
            o.get("mask_polygon"),
            dp.DEFAULT_K["width"],
            dp.DEFAULT_K["height"],
        )
        o["_mask_depth_probe_roi"] = probe
        rois.append(probe or o["roi"])
    try:
        out = ex.ros.call("depth_roi", {"rois_json": json.dumps(rois, ensure_ascii=False)})
        data = json.loads((out.text or "").strip())
    except Exception:  # noqa: BLE001
        return
    if not data.get("ok"):
        return
    stats_list = data.get("stats") or []
    pose = {"x": float(view_pose["x"]), "y": float(view_pose["y"]),
            "yaw_deg": float(view_pose.get("yaw_deg", view_pose.get("yaw", 0.0)))}
    for (_, o), stats in zip(indexed, stats_list):
        if not stats:
            o["geometry_status"] = rc.GEOMETRY_NO_DEPTH
            continue
        o["depth_stats"] = stats
        med = stats.get("median_m")
        # Task 1 ROI 漂移护栏：ROI 中心深度 vs bearing band 距离严重不一致 = ROI 大概率飘到背景地板/远墙
        #   (经典漂移，见 README 错误①) → 该帧坐标不可信 → 标记丢弃(_roi_drift)，探索走近会重标。
        band = o.get("distance_m")
        if (isinstance(med, (int, float)) and med > 0
                and isinstance(band, (int, float)) and band > 0
                and abs(med - band) > ROI_DEPTH_MISMATCH_M):
            o["_roi_drift"] = True
            o["geometry_status"] = rc.GEOMETRY_DEPTH_MISMATCH
            o["size_unreliable"] = True
            o["abs_pose"] = None
            continue
        probe = o.get("_mask_depth_probe_roi")
        sz = dp.roi_to_size(o["roi"], stats)
        # A2 size 清洗：桌面/远物 bbox 越过物体看到远墙 → 尺寸线性虚大。任一维 > 常理家具上限
        #   = depth 打在远面，size 不可信 → 标记并置 null（不瞎编尺寸；abs_pose 保留供去重/召回）。
        dims = [sz.get("width_m"), sz.get("height_m"), sz.get("depth_m")]
        if any(v is not None and v > SANE_MAX_M for v in dims):
            o["size_unreliable"] = True
            o["size"] = {"width_m": None, "height_m": None, "depth_m": None}
        else:
            o["size"] = sz
        if med:
            try:
                uc, vc = dp.roi_center_pixel(probe or o["roi"])
                # Task 1：反投用近带表面深度(near_min_m)而非整框中位，减少 ROI 混入远背景致坐标外推。
                nmin = stats.get("near_min_m")
                if probe:
                    # The probe is already inside the segmentation mask. Median is more
                    # robust than near_min here because a single foreground depth pixel
                    # must not pull the object onto a neighbouring surface.
                    surf_d = med
                    o["geometry_source"] = "mask_polygon_depth_median"
                    o["depth_probe_roi"] = probe
                else:
                    surf_d = nmin if (isinstance(nmin, (int, float)) and nmin > 0) else med
                    o["geometry_source"] = "bbox_depth_near_min"
                if "distance_m" not in o and isinstance(surf_d, (int, float)) and surf_d > 0:
                    o["distance_m"] = round(float(surf_d), 3)
                    o["distance_src"] = "depth_roi"
                o["abs_pose"] = dp.back_project(uc, vc, surf_d, robot_pose=pose)["abs_pose"]
                # Task 1(1b) LOS 穿墙反证：观测位姿→abs_pose 线段若中途穿过 occupied 格 = 相机隔墙
                #   看不到该处 → 反投坐标不可信（幻觉框钉到墙后）→ 标记丢弃（沿用 _roi_drift 风格）。
                ap = o["abs_pose"]
                if (FILTER_ROI_LOS and occ is not None and isinstance(ap, dict)
                        and ap.get("x") is not None
                        and not oc.line_free(occ, pose["x"], pose["y"],
                                             float(ap["x"]), float(ap.get("y") or 0.0))):
                    o["_los_blocked"] = True    # 实验关闭时不做 LOS 穿墙反证
                    o["geometry_status"] = rc.GEOMETRY_LOS_BLOCKED
                    o["size_unreliable"] = True
                    o["abs_pose"] = None
                else:
                    o["geometry_status"] = rc.GEOMETRY_OK
            except Exception:  # noqa: BLE001
                o["geometry_status"] = rc.GEOMETRY_OUT_OF_BOUNDS


def _scan_world_points(ex, pose):
    """各扇区最近障碍 → 世界点（粗边界/墙点）。"""
    try:
        s = json.loads(ex.ros.call("scan_summary", {"sectors": 12}).text)
    except (ValueError, TypeError):
        return []
    pts = []
    for label, d in (s.get("sectors") or {}).items():
        if not isinstance(d, (int, float)) or d <= 0:
            continue
        ang = nav._SECTOR8_ANG.get(label)
        if ang is None and isinstance(label, str) and label.startswith("sec_"):
            try:
                ang = int(label[4:])
            except ValueError:
                ang = None
        if ang is None:
            continue
        wa = math.radians(pose["yaw_deg"] + ang)
        pts.append([round(pose["x"] + d * math.cos(wa), 2), round(pose["y"] + d * math.sin(wa), 2)])
    return pts


# ===== 代码网格 frontier 覆盖（持有覆盖保证）+ Qwen 顾问（只重排+标门）=====
PITCH = 1.4              # 格距（米）；覆盖以此为步进向四周铺开，撞墙的格标 blocked
HINT_CONE_DEG = 30.0    # 顾问提示锥：只在此锥内才算"命中提示方向"
HINT_BAND_M = 0.75 * PITCH  # 距离量化带：hint 只在同一带内重排，永不跨距离碾压（近优先）
SWEEP_HEADINGS = (0.0, 90.0, 180.0, 270.0)  # 每个 vantage 原地环视的世界朝向
NEAR_LABEL_M = 0.7      # 标注铁律：正前 scan 最近障碍必须 > 此值才让 Qwen 标注（0.5→0.7：太近只剩一块颜色→
#                         把红柜看成"门"等误标；拉远到 0.7m 视角更全，见阶段十三实测点1）
NEAR_LABEL_CLOSE_M = 0.55  # 动态观测下限：贴墙家具退不开时,若 front∈[此值,0.7) 且 Qwen 判"值得贴近看"→放宽标注
#                         (南墙厨台/水槽/矮柜贴墙,退不到0.7m否则永远漏；0.55 比 0.45 稍远,防太近误标)
BBOX_CLAMP_M = 6.0      # frontier 播种 bbox 钳制半径（坏墙点不至于把 frontier 炸开）
BBOX_MAX_CELLS = 120    # 播种格数上限保护
DOOR_CLUSTER_M = 1.2    # 门世界点聚类阈值（去重）
DOOR_NOMINAL_M = 1.5    # 门世界点估计的名义距离（开口处 scan 常无回波，用名义距离投点供聚类）
DEDUP_M = 0.6           # 几何校验：同类物体 abs_pose 距离 ≤ 此值=同一物体多视角重复 → 合并；
#                         更远(如南墙一排矮柜 0.65m+ 间距)=不同实例各自成条（配合 _name_compat 名感知）
COOCCUR_EPS_M = 0.2     # 单帧共现反合并的"同物阈"：同一帧两个同名框但 abs_pose 差 > 此值=感知已分辨出的
#                         两个实例(如东排显示器 0.54m) → 禁止被 DEDUP_M 并回去；≤此值=同物双源(YOLO+Qwen)→仍并。
#                         治"名字对但被同名邻居吃掉"(密集同名家具间距≈坐标噪声, 纯位置去重分不开; 用单帧共现当铁证)
RELIABLE_Z = (-0.2, 1.6)  # 可信高度带（地面家具）；超出=高处/墙挂物，单帧 depth 反投不可信 → 按名归并并标记
SANE_MAX_M = 2.5        # A2 尺寸清洗：任一维 > 此值=depth 打到远墙的虚大尺寸 → 标 size_unreliable 并置 null
BOUNDARY_MARGIN = 0.6   # abs_pose 超出房间边界此余量=depth 打到远墙的错误投影 → 视作不可信

# ===== 标注过滤（Task 3：远距离过报 + 视野过小团在一起 → 不入库；见 Report/qwen_labeling_issue）=====
FAR_LABEL_M = 2.5       # 距离带门：distance_m > 此值的标注一律丢弃。实测 >2.5m 对率骤降(>3.5m 仅~7%对)；
#                         远处一团模糊会被 Qwen 报成"桌+显示器+柜"一坨全接地到同一远距离。探索会走近再干净抓。
MIN_ROI_AREA_FRAC = 0.006  # ROI 归一化面积(w·h, 0~1)下限：小于此=物体在画面里太小/太远 → 视角不全易团 → 丢弃
MIN_ROI_DIM_FRAC = 0.05    # ROI 最短边(min(w,h), 0~1)下限：细长/极小框同样过滤（团在一起的远物典型）
# Task 1 ROI 漂移护栏：ROI 中心深度(median_m) 与 bearing band 距离相差 > 此值 = ROI 大概率飘到背景地板/
#   远墙(经典漂移，见 README 错误①) → 该帧坐标不可信 → 丢弃（探索走近会重标）。
ROI_DEPTH_MISMATCH_M = 0.8
CONFUSION_SNAP_MAX_DEPTH_ERR_M = 0.45
CONFUSION_SNAP_MIN_VALID = 30

# ===== Claude 象限调度官（Role-1 覆盖规划；叠加在代码 frontier 兜底之上，绝不替换）=====
# 角色分工不变：Claude 只读符号地图【选】去哪(从代码筛好的候选里选 id，不产坐标)，代码保证走得成/不漏。
RECON_INSET_M = 0.8            # recon：bbox 四角向内缩这么多作为 4 个 recon 目标点
RECON_MAX_ITERS = 10          # recon/中心 geo_goto 每次 max_iters
DIRECTOR_MAX_ROUNDS = 8       # Claude plan_coverage 调用轮数硬上限（成本/延迟）
DIRECTOR_NAV_BUDGET = int(0.6 * MAX_NAV_STEPS)  # 调度官最多吃这么多平移步 → 兜底始终留 ≥40%
DIRECTOR_MAX_ITERS = 12       # 调度官目标 geo_goto 每次 max_iters
DIRECTOR_EMPTY_LIMIT = 2      # 连续这么多轮空计划(done 之外的无效/幻觉) → 提前交给兜底
DIRECTOR_FAIL_LIMIT = 3       # 连续这么多个目标都到不了(即使绕行) → 该向不可达，提前交给兜底(保住 backstop 预算)
DIRECTOR_MAX_LEGS = 14        # 调度官 geo_goto_around 反应式绕行腿数上限（绕红柜进东侧办公区要够腿；早停靠"净进展"检测而非砍腿数）
QUAD_COVER_TARGET = 0.6       # 象限 coverage < 此值 且 仍有可达空洞 = under_covered
CAND_PER_QUAD = 3             # 每象限给 Claude 的候选格上限（payload 紧凑）
REOPEN_FREE_NEIGHBORS = 3     # blocked 格 8 邻里 ≥ 此数是 visited → 疑似假墙，作 reopen 候选
REOPEN_FRONT_CONE_DEG = 30.0  # 假墙 scan 复核：前向锥半角
REOPEN_CLEAR_MARGIN = 0.4     # 复核判开：前向最近障碍 ≥ 到目标格距离 + 此余量（匹配 front_block_m）
OBJECT_STANDOFF_M = 0.7       # 停车铁律：目标格落着物体则只贴近到 ≥ 此距离（0.5→0.7：太近只见一块颜色误标）
CONSOLIDATE_CLUSTER_M = 0.3   # Role-2 整理：位置聚簇阈值。实测0.3最优：只并近乎重合的同物重复(52→38)、
#                               召回不掉(60%)；再大(0.6)会把密集区里挨着的不同物体(显示器vs桌)误并、掉召回

ADVISOR_SYS = (
    "你是室内机器人的【探索顾问】：不开车、不做导航决策，只看图给方向提示。\n"
    "代码已用网格保证房间全覆盖；你只需指出画面里【值得优先去看】的方向，"
    "尤其是【门 / 通往其他房间的开口 / 还没探索的暗口或走廊】。\n"
    "只输出一个 JSON（不要解释）：{\"hints\":[{\"dir\":\"left|center|right\",\"kind\":\"door|opening|gap\",\"reason\":\"\"}]}\n"
    "画面被墙堵死、没有明显门/开口就返回 {\"hints\":[]}。不要建议朝墙的方向。"
)


def _cell(x, y, pitch=PITCH):
    return (int(round(x / pitch)), int(round(y / pitch)))


def _cell_center(c, pitch=PITCH):
    return (c[0] * pitch, c[1] * pitch)


def _neighbors(c):
    return [(c[0] + dx, c[1] + dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))]


# ===== occupancy 覆盖（Q1 修复）：稠密射线建栅格 → frontier 候选 → A* 路由穿缝到远侧 =====
OCC_RES_M = 0.4          # occupancy 栅格分辨率（比 PITCH 细，够射线标 free + 缝检测）
CAND_TOTAL = 12          # 给 Claude 的候选总数上限
LOWCOV_RADIUS_M = 1.2    # 低覆盖判定：已知自由格距任一 visited > 此值 = 观测过没走近


def _update_occ(ex, occ, pose):
    """读一帧 scan_rays 更新 occupancy（射线标 free/occupied + 车格 visited）。失败静默跳过。

    破 Q1 偏置的数据源：稠密射线让被隔断遮挡区一旦从开口扫到就作 free 进栅格、其邻 unknown 成 frontier。
    """
    if occ is None:
        return
    try:
        data = json.loads(ex.ros.call("scan_rays", {"max_beams": 180}).text)
    except Exception:  # noqa: BLE001
        return
    beams = data.get("beams")
    if not beams:
        return
    oc.update_from_beams(occ, pose, beams, range_max_m=data.get("range_max_m") or 12.0)
    oc.mark_visited(occ, pose)


def _occ_candidates(occ, visited_cells, blocked_cells, pose, *, cap=CAND_TOTAL):
    """occupancy frontier + 低覆盖格 → 贴 PITCH 网格的候选（破"只从已扫 bbox 派生"偏置）。

    只保留 A* 从车格【可达】的候选（到不了的先不给 Claude，等扫到更多自由空间再连通）；近优先、去重、限量。
    返回 [{id, x, y, cell(PITCH), occ_cell, reopen_blocked:False}]（同 _candidate_cells 契约，Claude 只选 id）。
    """
    start = occ.cell_of(pose["x"], pose["y"])
    raw = oc.frontier_cells(occ) | oc.low_coverage_cells(occ, radius_m=LOWCOV_RADIUS_M)
    best = {}                                    # PITCH 格 -> (dist, occ 格)
    for c in raw:
        wx, wy = occ.center(c)
        pc = _cell(wx, wy)
        if pc in visited_cells or pc in blocked_cells:
            continue
        d = math.hypot(wx - pose["x"], wy - pose["y"])
        if pc not in best or d < best[pc][0]:
            best[pc] = (d, c)
    cands = []
    for pc, (_d, occ_cell) in sorted(best.items(), key=lambda kv: kv[1][0]):
        if len(cands) >= cap:
            break
        if oc.astar(occ, start, occ_cell) is None:   # 不可达 → 暂不作候选
            continue
        cx, cy = _cell_center(pc)
        cands.append({"id": f"c{len(cands)}", "x": round(cx, 1), "y": round(cy, 1),
                      "cell": pc, "occ_cell": occ_cell, "reopen_blocked": False})
    return cands


def _quad_visit_counts(visited, bbox):
    """每象限已访问位姿数（覆盖均衡用）。bbox 空/无 visited → 全 0。"""
    counts = {"NE": 0, "NW": 0, "SE": 0, "SW": 0}
    if not bbox or not visited:
        return counts
    xmid = (bbox["xmin"] + bbox["xmax"]) / 2.0
    ymid = (bbox["ymin"] + bbox["ymax"]) / 2.0
    for v in visited:
        counts[_quad_of(v[0], v[1], xmid, ymid)] += 1
    return counts


def _least_covered_quad(cands, bbox, visited):
    """把候选按象限分组，返回(访问最少且有候选的象限, 该象限候选列表)。破 frontier 近邻贪心的方向漂移
    （治"预算被一个方向吃光、别的区整片漏"的覆盖方差）。无 bbox/候选 → (None, cands)。"""
    if not cands or not bbox:
        return None, cands
    xmid = (bbox["xmin"] + bbox["xmax"]) / 2.0
    ymid = (bbox["ymin"] + bbox["ymax"]) / 2.0
    vc = _quad_visit_counts(visited, bbox)
    by_quad = {}
    for c in cands:
        by_quad.setdefault(_quad_of(c["x"], c["y"], xmid, ymid), []).append(c)
    target_q = min(by_quad.keys(), key=lambda q: (vc.get(q, 0), q))   # 访问最少优先, 名字 tiebreak
    return target_q, by_quad[target_q]


def _balance_filter(cands, bbox, visited):
    """覆盖均衡硬门：把候选限制在【访问最少的象限】(有候选者)——供 STAGE B 给 Claude 前收窄，
    让它只能在欠覆盖象限里选 id（代码持有覆盖均衡, Claude 只在允许范围内选）。"""
    _q, sub = _least_covered_quad(cands, bbox, visited)
    return sub or cands


def _balanced_pick(cands, bbox, visited):
    """覆盖均衡选点：访问最少象限里离 pose 最近的候选(cands 已近优先排序)。供兜底替代"取最近 frontier"。"""
    _q, sub = _least_covered_quad(cands, bbox, visited)
    return sub[0] if sub else (cands[0] if cands else None)


def _origin_balance_bbox(origin=START_XY):
    """Return a synthetic bbox whose midpoint stays fixed at the exploration origin."""
    ox, oy = origin
    return {"xmin": ox - 1.0, "xmax": ox + 1.0, "ymin": oy - 1.0, "ymax": oy + 1.0}


def _balanced_grid_target(targets, pose, vantage_xys, origin=START_XY):
    """Choose a nearby target in the least-observed origin-relative quadrant.

    The discovered bbox can grow strongly toward one room wing.  Using its moving
    midpoint for balancing then keeps rewarding that same wing.  The robot start is a
    stable, answer-free reference, while ``vantage_xys`` counts actual observations and
    is not distorted by long failed navigation traces.
    """
    if not targets:
        return None
    cands = [{"x": x, "y": y} for x, y in targets]
    _q, subset = _least_covered_quad(
        cands,
        _origin_balance_bbox(origin),
        vantage_xys,
    )
    pool = subset or cands
    chosen = min(
        pool,
        key=lambda target: math.hypot(
            target["x"] - pose["x"],
            target["y"] - pose["y"],
        ),
    )
    return chosen["x"], chosen["y"]


def _exclude_observed_candidates(cands, vantage_xys, radius=None):
    """Drop frontier candidates already observed from a safe nearby vantage.

    A viewpoint planner may deliberately stop away from the requested frontier cell.
    ``visited_cells`` then does not contain that target even though the sweep covered it,
    so occupancy-only filtering repeatedly selects the same frontier.  Grid coverage
    records requested targets in ``vantage_xys``; honor that evidence here.
    """
    radius = COVER_RADIUS_M if radius is None else radius
    return [
        candidate
        for candidate in cands
        if not any(
            math.hypot(candidate["x"] - vx, candidate["y"] - vy) <= radius
            for vx, vy in vantage_xys
        )
    ]


def _snap_to_free(occ, start, target_xy, max_r_cells=6):
    """目标格落在占据/未知(常是沙发/桌腿内部) → snap 到最近的【已知自由且 A* 可达】格中心。

    近圈优先(0.4m 一环, ≤max_r_cells 环)。找到返回 (x,y)+True；目标本就自由或无可达自由格 → 返回原点+False。
    这是"据雷达/位姿算点合不合理"的几何层：不把导航目标设进家具里，从源头免掉靠近-弹开反复排斥。
    """
    gx, gy = occ.cell_of(target_xy[0], target_xy[1])
    if occ.state((gx, gy)) in oc._KNOWN_FREE:
        return target_xy, False                       # 目标本就在自由格，无需 snap
    for radius in range(1, max_r_cells + 1):
        ring = [(gx + dx, gy + dy)
                for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
                if max(abs(dx), abs(dy)) == radius and occ.state((gx + dx, gy + dy)) in oc._KNOWN_FREE]
        ring.sort(key=lambda c: (c[0] - gx) ** 2 + (c[1] - gy) ** 2)   # 近目标优先
        for c in ring:
            if oc.astar(occ, start, c):
                return occ.center(c), True
    return target_xy, False


def _route_to(ex, occ, target_xy, *, tol_m=0.5, max_legs=14):
    """occupancy A* → 简化 waypoints → geo_route_around 逐段绕行到 target_xy（门桥接执行）。

    Step 0.2 证：反应式直冲穿不了长墙；A* 在已知自由格上自动生成"绕墙穿缝"走廊路点，再逐段绕行执行。
    目标落家具内部(占据/未知) → 先 snap 到最近的自由可达格，避免蛮力冲进家具靠近-弹开反复排斥。
    A* 无路(远候选未连通) → 退化 geo_goto_around 直接尽力（下一轮扫到更多自由空间会连通）。返回导航结果 dict。
    """
    p = _pose(ex)
    start = occ.cell_of(p["x"], p["y"])
    tgt, snapped = _snap_to_free(occ, start, target_xy)   # 不往家具里设点
    goal = occ.cell_of(tgt[0], tgt[1])
    path = oc.astar(occ, start, goal)
    has_path = bool(path and len(path) >= 2)
    if has_path:
        wps = oc.path_waypoints(occ, path)
        if wps:
            wps[-1] = (round(tgt[0], 2), round(tgt[1], 2))   # 末点用(snap 后的)精确目标
            r = nav.geo_route_around(ex, wps, tol_m=tol_m, max_legs=max_legs,
                                     max_step_m=1.0, front_block_m=0.6, max_stuck=5)
            r["route_wps"] = wps
            r["astar_path"] = True                # 诊断：A* 在已知自由格上找到走廊
            r["snapped"] = snapped
            return r
    r = nav.geo_goto_around(ex, tgt[0], tgt[1], tol_m=tol_m,
                            max_legs=max_legs, max_step_m=1.0, front_block_m=0.6)
    r["astar_path"] = has_path                    # 诊断：A* 无路(远候选未连通) → 退化直冲
    r["snapped"] = snapped
    return r


def _advisor(ex, rep):
    """Qwen 顾问：看图报门/开口/未探索方向（不决定去哪，只给方向提示）。失败返回 []。"""
    content = [{"type": "text", "text": "画面里有没有门/通道/未探索的暗口？给方向(left/center/right)。只输出 JSON。"}]
    img = rep.get("image")
    if img is not None:
        content.append(harness.to_openai_image_url(img))
    try:
        resp = ex.client.chat.completions.create(
            model=ex.model,
            messages=[{"role": "system", "content": ADVISOR_SYS}, {"role": "user", "content": content}],
            temperature=0.2, max_tokens=200, stream=False)
        obj = harness._extract_obj(resp.choices[0].message.content or "")
        return obj.get("hints", []) if isinstance(obj, dict) else []
    except Exception:  # noqa: BLE001
        return []


APPROACH_SYS = ("你判断机器人正前方【近处】是否有贴墙或密集家具(柜子/办公桌/水槽/厨台/沙发/绿植/显示器等)"
                "值得靠近看清并记录。只有确实有家具主体在正前近处才 true；空地/墙面/门/已远离 → false。只输出 JSON。")


def _approach_worth(ex, img):
    """Qwen 语义判断：正前近处是否有值得贴近看的贴墙/密集家具（南墙厨台/水槽/矮柜退不开时用）。

    用户思路：'能不能通过/值不值得'这类语义判断交给 Qwen，代码据其结论调用底层(放宽近距标注)。
    返回 True=值得贴近标注；否/异常/无图 → False（保守跳过，不冒近距误标风险）。
    """
    if img is None:
        return False
    content = [{"type": "text",
                "text": '正前方近处有没有贴墙或密集家具(柜子/桌子/水槽/厨台/沙发等)值得靠近看清? '
                        '只输出 {"worth": true/false, "obj": "名称或空"}。'},
               harness.to_openai_image_url(img)]
    try:
        resp = ex.client.chat.completions.create(
            model=ex.model,
            messages=[{"role": "system", "content": APPROACH_SYS},
                      {"role": "user", "content": content}],
            temperature=0.0, max_tokens=60, stream=False)
        obj = harness._extract_obj(resp.choices[0].message.content or "")
        return bool(isinstance(obj, dict) and obj.get("worth"))
    except Exception:  # noqa: BLE001
        return False


def _pick_target(frontier, pose, hint_bearings):
    """在未覆盖格里选下一个目标。**近优先为主键**，顾问提示只在同一距离带内重排（mild tiebreak）。

    排序键 = (距离带, 命中提示?0:1, 距离, 格)：距离带为主 → 远的提示格永远碾压不了近的非提示格，
    消除"门在前方就一直往北 streak"。顾问删不掉格、停不了覆盖。末尾带格保证确定性（消 set 抖动）。
    """
    best, best_key = None, None
    for c in frontier:
        cx, cy = _cell_center(c)
        d = math.hypot(cx - pose["x"], cy - pose["y"])
        b = math.degrees(math.atan2(cy - pose["y"], cx - pose["x"]))
        amin = min((abs(((b - hb + 180) % 360) - 180) for hb in hint_bearings), default=180.0)
        aligned = amin <= HINT_CONE_DEG
        band = round(d / HINT_BAND_M)
        key = (band, 0 if aligned else 1, round(d, 2), c)
        if best_key is None or key < best_key:
            best_key, best = key, c
    return best


def _drive_to(ex, target_xy, max_steps=4, tol=0.6):
    """VFH 朝目标格心走，最多 max_steps 步。返回 (reached, cur_pose, n_steps)。

    n_steps = 实际下发的 geo_step_open 平移步数（供调用方计入 MAX_NAV_STEPS 硬上限）。
    连续不动(moved<0.08=撞墙)即提前返回 reached=False。
    """
    cur = _pose(ex)
    steps = 0
    for _ in range(max_steps):
        if math.hypot(target_xy[0] - cur["x"], target_xy[1] - cur["y"]) <= tol:
            return True, cur, steps
        st = nav.geo_step_open(ex, nav.bearing_deg(cur["x"], cur["y"], target_xy[0], target_xy[1]))
        steps += 1
        cur = _pose(ex)
        if (st.get("moved_m") or 0) < 0.08:
            return False, cur, steps
    reached = math.hypot(target_xy[0] - cur["x"], target_xy[1] - cur["y"]) <= tol
    return reached, cur, steps


def _seed_frontier_from_bbox(bbox, visited_cells, blocked_cells, pitch=PITCH):
    """用扫到的房间粗边界 bbox 给 frontier 播种（最高杠杆修复：让远角/东半区一开始就入 frontier）。

    枚举 bbox 内未访问/未 blocked 的格。钳制 bbox + 格数上限，坏墙点不至于把 frontier 炸开。
    """
    if not bbox:
        return set()
    xmin = max(bbox["xmin"], -BBOX_CLAMP_M)
    xmax = min(bbox["xmax"], BBOX_CLAMP_M)
    ymin = max(bbox["ymin"], -BBOX_CLAMP_M)
    ymax = min(bbox["ymax"], BBOX_CLAMP_M)
    if (max(0.0, xmax - xmin) * max(0.0, ymax - ymin)) / (pitch * pitch) > BBOX_MAX_CELLS:
        return set()
    cells = set()
    for cx in range(int(math.floor(xmin / pitch)), int(math.ceil(xmax / pitch)) + 1):
        for cy in range(int(math.floor(ymin / pitch)), int(math.ceil(ymax / pitch)) + 1):
            c = (cx, cy)
            if c not in visited_cells and c not in blocked_cells:
                cells.add(c)
    return cells


def _blocked_cone(frontier, pose, target_xy, cone_deg=25.0):
    """撞墙后：与失败目标【共线且更远】的 frontier 格也标 blocked（墙后的格同样到不了）。

    防止 _pick_target 立刻又选一个共线更远的等距格、再撞同一堵墙（修复 step6→7 同位姿卡死）。
    """
    tb = math.degrees(math.atan2(target_xy[1] - pose["y"], target_xy[0] - pose["x"]))
    td = math.hypot(target_xy[0] - pose["x"], target_xy[1] - pose["y"])
    out = set()
    for c in frontier:
        cx, cy = _cell_center(c)
        d = math.hypot(cx - pose["x"], cy - pose["y"])
        b = math.degrees(math.atan2(cy - pose["y"], cx - pose["x"]))
        if d >= td - 1e-6 and abs(((b - tb + 180) % 360) - 180) <= cone_deg:
            out.add(c)
    return out


def _front_clear(ex):
    """读正前方 scan 最近障碍距离（米）；读不到返回 None。"""
    try:
        s = json.loads(ex.ros.call("scan_summary", {}).text)
        return s.get("front_min_m")
    except Exception:  # noqa: BLE001
        return None


def _ensure_view_distance(ex, min_clear=NEAR_LABEL_M, max_back=3):
    """标注铁律：正前 scan 最近障碍必须 > min_clear 才标注（太近视角不全→误标）。
    太近就后退拉开（有向安全已允许后退脱离死区）。返回最终正前距离(米)或 None。"""
    front = _front_clear(ex)
    tries = 0
    while front is not None and front < min_clear and tries < max_back:
        r = ex.ros.call("move", {"distance_m": -0.35})   # 后退拉开视距
        # 后方也被挡(safety 否决后退)→退不动，停止尝试
        if '"traveled_m":0.0' in (r.text or "") and '"status":"safety_stop"' in (r.text or ""):
            break
        front = _front_clear(ex)
        tries += 1
    return front


def _yoloe_inspect_image(img):
    """Run the isolated YOLOE environment on one ImagePart and return a report-like dict.

    This intentionally shells out to the `yolo` conda env so explore can keep running in
    the vLLM/agent_core environment without importing Ultralytics.
    """
    if img is None:
        return {"objects": [], "candidate_objects": [], "image": img, "raw": "no image"}
    frame_dir = os.path.join(YOLOE_TMP_DIR, str(os.getpid()))
    out_dir = os.path.join(frame_dir, "offline")
    os.makedirs(frame_dir, exist_ok=True)
    frame_path = os.path.join(frame_dir, "frame.jpg")
    with open(frame_path, "wb") as f:
        f.write(base64.b64decode(img.b64))

    cmd = [
        YOLOE_PYTHON,
        os.path.join(HERE, "yolo_offline_probe.py"),
        "--images",
        frame_path,
        "--out",
        out_dir,
        "--model",
        YOLOE_MODEL,
        "--device",
        YOLOE_DEVICE,
        "--conf",
        str(YOLOE_CONF),
    ]
    if YOLOE_PROFILE:                       # 逐类阈值 + confirmed/candidate 隔离（默认开）
        cmd += ["--profile", YOLOE_PROFILE]
    try:
        res = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=YOLOE_TIMEOUT_S
        )
        with open(os.path.join(out_dir, "detections.json"), encoding="utf-8") as f:
            data = json.load(f)
        images = data.get("images") or []
        obj0 = images[0] if images else {}
        objects = obj0.get("objects") or []
        candidate_objects = obj0.get("candidate_objects") or []
        ann = obj0.get("annotated")            # 画框标注图（相对 out_dir），供审阅实验存档
        ann_path = os.path.join(out_dir, ann) if ann else None
        return {
            "objects": objects,
            "candidate_objects": candidate_objects,
            "image": img,
            "raw": res.stdout,
            "annotated": ann_path,
        }
    except Exception as e:  # noqa: BLE001
        print(f"    [YOLOE失败] {type(e).__name__}: {str(e)[:160]} → 本朝向不记物体")
        return {
            "objects": [],
            "candidate_objects": [],
            "image": img,
            "raw": str(e),
            "annotated": None,
        }


# ===== 常驻 YOLO 检测服务（模型只加载一次，消除每帧 subprocess+重载）=====
_YOLO_PROC = [None]      # 惰性单例：Popen 句柄或 None（启动失败/不可用）


def _yolo_service():
    """惰性启动常驻 YOLO 服务（yolo env，模型只加载一次）。返回 Popen 或 None。"""
    p = _YOLO_PROC[0]
    if p is not None:
        return p if p.poll() is None else None
    os.makedirs(YOLOE_TMP_DIR, exist_ok=True)
    err_path = os.path.join(YOLOE_TMP_DIR, "service.err")
    try:
        errf = open(err_path, "w")   # noqa: SIM115 (进程存活期间需一直打开)
        p = subprocess.Popen(
            [YOLOE_PYTHON, os.path.join(HERE, "yolo_service.py"),
             "--model", YOLOE_MODEL, "--device", YOLOE_DEVICE, "--conf", str(HYBRID_YOLO_CONF)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf,
            text=True, bufsize=1, cwd=HERE)
        ready = (p.stdout.readline() or "").strip()   # 阻塞等模型加载完（首行 READY）
        if ready != "READY":
            print(f"    [YOLO服务] 启动异常(首行={ready!r})，见 {err_path} → 回退每帧 subprocess")
            _YOLO_PROC[0] = None
            return None
        print(f"    [YOLO服务] 就绪（模型已加载，后续每帧免重载）pid={p.pid}")
        _YOLO_PROC[0] = p
        return p
    except Exception as e:  # noqa: BLE001
        print(f"    [YOLO服务启动失败] {e} → 回退每帧 subprocess")
        _YOLO_PROC[0] = None
        return None


def _yolo_service_stop():
    """收尾：优雅关停常驻 YOLO 服务。"""
    p = _YOLO_PROC[0]
    if p is None:
        return
    try:
        if p.poll() is None:
            p.stdin.write("__QUIT__\n")
            p.stdin.flush()
            p.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    _YOLO_PROC[0] = None


def _yolo_detect_raw(img):
    """跑 YOLO（低 conf）出 raw_detections。优先走常驻服务；不可用则回退单帧 subprocess。
    返回 (raw_detections, width, height)；失败 → ([], 0, 0)。"""
    if img is None:
        return [], 0, 0
    frame_dir = os.path.join(YOLOE_TMP_DIR, str(os.getpid()))
    os.makedirs(frame_dir, exist_ok=True)
    frame_path = os.path.join(frame_dir, "frame.jpg")
    with open(frame_path, "wb") as f:
        f.write(base64.b64decode(img.b64))
    # 首选：常驻服务（一次加载）
    proc = _yolo_service()
    if proc is not None:
        try:
            proc.stdin.write(frame_path + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            data = json.loads(line) if line else {}
            if data.get("ok"):
                return (data.get("raw_detections") or [],
                        int(data.get("width") or 0), int(data.get("height") or 0))
            print(f"    [YOLO服务返回错误] {str(data.get('error'))[:120]}")
        except Exception as e:  # noqa: BLE001
            print(f"    [YOLO服务调用失败] {type(e).__name__}: {str(e)[:120]} → 本帧回退 subprocess")
    # 回退：单帧 subprocess（每帧重载，慢但稳）
    out_dir = os.path.join(frame_dir, "hybrid")
    cmd = [YOLOE_PYTHON, os.path.join(HERE, "yolo_offline_probe.py"),
           "--images", frame_path, "--out", out_dir,
           "--model", YOLOE_MODEL, "--device", YOLOE_DEVICE, "--conf", str(HYBRID_YOLO_CONF)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=YOLOE_TIMEOUT_S)
        with open(os.path.join(out_dir, "detections.json"), encoding="utf-8") as f:
            data = json.load(f)
        obj0 = (data.get("images") or [{}])[0]
        return (obj0.get("raw_detections") or [],
                int(obj0.get("width") or 0), int(obj0.get("height") or 0))
    except Exception as e:  # noqa: BLE001
        print(f"    [混合YOLO失败] {type(e).__name__}: {str(e)[:160]} → 本朝向无框")
        return [], 0, 0


def _hybrid_yolo_boxes(img):
    """混合后端定位源：YOLO（低 conf）→ raw_detections → 只带 ROI/编号的框（命名交给 Qwen 弃权门）。
    返回 (boxes, width, height)；失败/无图 → ([], 0, 0)。"""
    raw, w, h = _yolo_detect_raw(img)
    boxes = yoloe.raw_detections_to_boxes(raw, w, h, conf_thres=HYBRID_YOLO_CONF)
    return boxes, w, h


def _backfill_box_geometry(ex, boxes, view_pose, occ=None):
    """给 YOLO 候选框补 depth/TF 几何，供离线 GPT 仲裁公平打分。

    这里不产语义名、不进主记忆，只把每个候选框的 abs_pose/size/distance_m 挂回 box。
    后续 GPT 或 Qwen 决定 keep/name；坐标仍来自同一套代码几何管线。
    """
    if not boxes:
        return
    tmp = []
    for b in boxes:
        tmp.append({
            "name": "候选框",
            "confidence": b.get("confidence"),
            "roi": b.get("roi"),
            "bbox_center": b.get("bbox_center"),
        })
    _backfill_geometry_local(ex, tmp, view_pose, occ=occ)
    for b, o in zip(boxes, tmp):
        if o.get("_roi_drift"):
            b["roi_drift"] = True
        if o.get("_los_blocked"):
            b["los_blocked"] = True
        for k in ("abs_pose", "size", "distance_m", "bbox_center", "geometry_status",
                  "depth_stats", "roi_quality"):
            if o.get(k) is not None:
                b[k] = o[k]


def _object_from_yolo_box(box, name, *, semantic_source="qwen_full"):
    """Build a memory object whose semantic name comes from Qwen and ROI from YOLO."""
    obj = {
        "name": name,
        "confidence": box.get("confidence"),
        "roi": box.get("roi"),
        "bbox_center": box.get("bbox_center"),
        "distance_m": box.get("distance_m"),
        "abs_pose": box.get("abs_pose"),
        "size": box.get("size"),
        "roi_source": "yoloe",
        "semantic_source": semantic_source,
        "geometry_status": box.get("geometry_status", rc.GEOMETRY_OK),
        "verified_by": ["qwen", "yolo_roi"],
    }
    if box.get("detector_label"):
        obj["detector_label"] = box["detector_label"]
    if box.get("depth_stats"):
        obj["depth_stats"] = box["depth_stats"]
    return {k: v for k, v in obj.items() if v is not None}


def _snap_qwen_roi(ex, obj):
    """Return a snapped ROI candidate for one Qwen full-scene object, or None."""
    probes = rc.generate_depth_snap_rois(obj)
    if not probes:
        return None
    try:
        out = ex.ros.call("depth_roi", {"rois_json": json.dumps(probes, ensure_ascii=False)})
        data = json.loads((out.text or "").strip())
    except Exception:  # noqa: BLE001
        return None
    if not data.get("ok"):
        return None
    picked = rc.choose_depth_snap_roi(
        obj, probes, data.get("stats") or [],
        depth_tol_m=ROI_DEPTH_MISMATCH_M,
        sane_max_m=SANE_MAX_M,
    )
    if picked.get("status") != rc.GEOMETRY_OK or not picked.get("roi"):
        obj["geometry_status"] = picked.get("status") or rc.GEOMETRY_NO_DEPTH
        obj["roi_quality"] = picked.get("roi_quality")
        return None
    return {
        "roi": picked["roi"],
        "roi_source": "qwen_depth_snap",
        "semantic_source": "qwen_full",
        "geometry_status": rc.GEOMETRY_OK,
        "depth_stats": picked.get("depth_stats"),
        "roi_quality": picked.get("roi_quality"),
    }


def _resolve_qwen_candidate_rois(ex, objects, candidate_boxes=None):
    """Replace Qwen raw ROIs with validated candidate ROIs where possible.

    Qwen full-scene objects that cannot be matched/snapped are preserved as
    semantic sightings for reports, but they do not get a trusted ROI.
    """
    entries, all_probes = [], []
    for o in objects or []:
        if not isinstance(o, dict):
            continue
        if o.get("detector_source") in ("hybrid", "yoloe") or o.get("verified_by"):
            oo = dict(o)
            oo.setdefault("roi_source", "yoloe")
            oo.setdefault("semantic_source", "qwen_box")
            entries.append(("object", oo))
            continue
        name = o.get("name")
        match = rc.match_yolo_candidate(o, candidate_boxes or [])
        if match is not None and name:
            entries.append(("object", _object_from_yolo_box(match, name, semantic_source="qwen_full")))
            continue
        probes = rc.generate_depth_snap_rois(o)
        if probes:
            start = len(all_probes)
            all_probes.extend(probes)
            entries.append(("snap", o, start, len(probes)))
        else:
            entries.append(("sighting", o, rc.GEOMETRY_NO_DEPTH))

    stats_all = []
    if all_probes:
        try:
            out = ex.ros.call("depth_roi", {"rois_json": json.dumps(all_probes, ensure_ascii=False)})
            data = json.loads((out.text or "").strip())
            if data.get("ok"):
                stats_all = data.get("stats") or []
        except Exception:  # noqa: BLE001
            stats_all = []

    out, sightings = [], []
    for entry in entries:
        kind = entry[0]
        if kind == "object":
            out.append(entry[1])
            continue
        if kind == "snap":
            _kind, o, start, count = entry
            probes = all_probes[start:start + count]
            stats = stats_all[start:start + count] if stats_all else []
            picked = rc.choose_depth_snap_roi(
                o, probes, stats,
                depth_tol_m=ROI_DEPTH_MISMATCH_M,
                sane_max_m=SANE_MAX_M,
            )
            if picked.get("status") == rc.GEOMETRY_OK and picked.get("roi"):
                oo = dict(o)
                oo.update({
                    "roi": picked["roi"],
                    "roi_source": "qwen_depth_snap",
                    "semantic_source": "qwen_full",
                    "geometry_status": rc.GEOMETRY_OK,
                    "depth_stats": picked.get("depth_stats"),
                    "roi_quality": picked.get("roi_quality"),
                })
                out.append(oo)
                continue
            status = picked.get("status") or rc.GEOMETRY_NO_DEPTH
            o = dict(o)
            o["geometry_status"] = status
            o["roi_quality"] = picked.get("roi_quality")
        else:
            _kind, o, status = entry
            o = dict(o)
            o["geometry_status"] = status
        sight = {
            "name": o.get("name"),
            "spatial": o.get("bearing") or o.get("spatial"),
            "roi": o.get("roi"),
            "bbox_center": o.get("bbox_center"),
            "distance_m": o.get("distance_m"),
            "geometry_status": o.get("geometry_status", rc.GEOMETRY_NO_DEPTH),
            "semantic_source": "qwen_full",
            "semantic_sighting": True,
        }
        sightings.append({k: v for k, v in sight.items() if v is not None})
    return out, sightings


def _south_wall_label_context(pose, heading):
    """Return kitchen/south-wall class hints for low-angle south-wall frames."""
    try:
        y = float(pose.get("y", 0.0))
        h = float(heading) % 360.0
    except (TypeError, ValueError):
        return None, ""
    facing_south = 225.0 <= h <= 315.0
    near_south_band = y <= -0.8
    if not (facing_south or near_south_band):
        return None, ""
    return ["厨台", "水槽", "柜子", "办公桌"], "当前帧靠近/朝向南墙厨房带"


def _high_confusion_snap_ok(o):
    """Extra single-frame gate for monitor/desk/chair when only Qwen snap supports ROI."""
    if not rc.is_high_confusion_name(o.get("name")):
        return True, ""
    if o.get("roi_source") != "qwen_depth_snap":
        return True, ""
    q = o.get("roi_quality") or {}
    depth_err = q.get("depth_error_m")
    n_valid = q.get("n_valid")
    if isinstance(depth_err, (int, float)) and depth_err > CONFUSION_SNAP_MAX_DEPTH_ERR_M:
        return False, f"高混淆类snap depth_err {depth_err:.2f}>{CONFUSION_SNAP_MAX_DEPTH_ERR_M}"
    if isinstance(n_valid, int) and n_valid < CONFUSION_SNAP_MIN_VALID:
        return False, f"高混淆类snap有效深度 {n_valid}<{CONFUSION_SNAP_MIN_VALID}"
    return True, ""


def _review_dump(rep, kept, observer_pose, heading, vantage_idx):
    """审阅实验存档：把本朝向的 YOLO 画框标注图 + 原始检测 + 反投世界坐标(kept)存 YOLOE_REVIEW_DIR。

    产出供人+Claude 审阅 ①YOLO 标的真家具在不在 ②YOLO 有没有在空墙幻觉。YOLOE_REVIEW_DIR 空则跳过。
    """
    if not YOLOE_REVIEW_DIR:
        return
    img_dir = os.path.join(YOLOE_REVIEW_DIR, "imgs")
    raw_dir = os.path.join(YOLOE_REVIEW_DIR, "imgs_raw")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    seq = _REVIEW_SEQ[0]
    _REVIEW_SEQ[0] += 1
    img_rel = None
    ann = rep.get("annotated")
    if ann and os.path.isfile(ann):
        img_rel = os.path.join("imgs", f"{seq:04d}.jpg")
        try:
            shutil.copyfile(ann, os.path.join(YOLOE_REVIEW_DIR, img_rel))
        except Exception:  # noqa: BLE001
            img_rel = None
    # 未标记原图（YOLO 实际看到的帧，无框）——供对比审阅
    raw_rel = None
    imgpart = rep.get("image")
    b64 = getattr(imgpart, "b64", None)
    if b64:
        raw_rel = os.path.join("imgs_raw", f"{seq:04d}.jpg")
        try:
            with open(os.path.join(YOLOE_REVIEW_DIR, raw_rel), "wb") as rf:
                rf.write(base64.b64decode(b64))
        except Exception:  # noqa: BLE001
            raw_rel = None
    row = {
        "seq": seq, "vantage_idx": vantage_idx, "heading": round(float(heading), 1),
        "observer_pose": {"x": round(observer_pose["x"], 3), "y": round(observer_pose["y"], 3),
                          "yaw_deg": round(observer_pose.get("yaw_deg", 0.0), 1)},
        "annotated_image": img_rel,
        "raw_image": raw_rel,
        # YOLO 原始检测（画框所示，未过 _annotation_ok）——审阅召回/幻觉的一手证据
        "yolo_raw": [{"name": o.get("name"), "confidence": o.get("confidence"),
                      "roi": o.get("roi"), "bbox_center": o.get("bbox_center")}
                     for o in (rep.get("objects") or []) if isinstance(o, dict)],
        # 过滤+几何接地后留存物体（带反投世界坐标 abs_pose）——看坐标接地对不对
        "kept": [{"name": o.get("name"), "confidence": o.get("confidence"),
                  "roi": o.get("roi"), "abs_pose": o.get("abs_pose")} for o in kept],
    }
    with open(os.path.join(YOLOE_REVIEW_DIR, "manifest.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _finalize_review(area_path):
    """审阅实验收尾：拷贝 run 工件 + area.json、跑打分器写 score.txt、生成人读 index.md。

    产出 YOLOE_REVIEW_DIR 一站式证据集：逐朝向标注图 + 反投坐标清单 + 召回打分。YOLOE_REVIEW_DIR 空则跳过。
    """
    if not YOLOE_REVIEW_DIR:
        return
    try:
        for src in (RUN_OUT, area_path):
            if src and os.path.isfile(src):
                shutil.copyfile(src, os.path.join(YOLOE_REVIEW_DIR, os.path.basename(src)))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 审阅工件拷贝失败: {e}")
    # 打分器（纯 stdlib，用当前解释器跑）→ score.txt
    scorer = os.path.join(HERE, "eval", "score_explore.py")
    try:
        res = subprocess.run([sys.executable, scorer], capture_output=True, text=True, timeout=60)
        with open(os.path.join(YOLOE_REVIEW_DIR, "score.txt"), "w", encoding="utf-8") as f:
            f.write(res.stdout + ("\n[stderr]\n" + res.stderr if res.stderr else ""))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 打分器运行失败: {e}")
    # 人读 index.md：逐朝向 YOLO 原始检测 vs 留存(带坐标)，供审阅召回/幻觉
    rows = []
    mpath = os.path.join(YOLOE_REVIEW_DIR, "manifest.jsonl")
    if os.path.isfile(mpath):
        with open(mpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    n_raw = sum(len(r.get("yolo_raw") or []) for r in rows)
    n_kept = sum(len(r.get("kept") or []) for r in rows)
    lines = [f"# YOLO 标注审阅实验（backend={PERCEPTION_BACKEND}, model={YOLOE_MODEL}）", "",
             f"- 朝向帧数：{len(rows)}  YOLO 原始检测：{n_raw}  过滤+接地留存：{n_kept}",
             "- 审阅要点：①YOLO 框住的真家具命名对不对（WHAT）②有没有在空墙/虚空画框（幻觉）"
             "③留存 abs_pose 接地对不对（WHERE）",
             "- 打分见 `score.txt`；记忆见 `area.json`；轨迹见 `explore_run.json`。", ""]
    for r in rows:
        p = r.get("observer_pose") or {}
        lines.append(f"## 帧{r['seq']:04d} · v{r.get('vantage_idx')} · heading={r.get('heading')}° "
                     f"· pose=({p.get('x')},{p.get('y')},{p.get('yaw_deg')}°)")
        if r.get("raw_image"):
            lines.append(f"原图 ![]({r['raw_image']})")
        if r.get("annotated_image"):
            lines.append(f"标注 ![]({r['annotated_image']})")
        raw = r.get("yolo_raw") or []
        lines.append(f"- YOLO 原始检测 {len(raw)}：" + (", ".join(
            f"{o.get('name')}({o.get('confidence')})" for o in raw) or "无"))
        kept = r.get("kept") or []
        for o in kept:
            ap = o.get("abs_pose") or {}
            lines.append(f"  - 留存 **{o.get('name')}** conf={o.get('confidence')} "
                         f"abs_pose=({ap.get('x')},{ap.get('y')})")
        lines.append("")
    with open(os.path.join(YOLOE_REVIEW_DIR, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[审阅实验] {YOLOE_REVIEW_DIR}  帧={len(rows)} YOLO检测={n_raw} 留存={n_kept}  "
          f"→ index.md / score.txt / manifest.jsonl")


def _obj_brief(o):
    """物体精简摘要（供审阅清单）：名/置信度/roi。"""
    out = {
        "name": o.get("name"),
        "confidence": o.get("confidence"),
        "roi": o.get("roi"),
        "memory_status": o.get("memory_status"),
        "candidate_reason": o.get("candidate_reason"),
    }
    return {key: value for key, value in out.items() if value is not None}


def _obj_kept_brief(o):
    """留存物体摘要：名/置信度/反投世界坐标。"""
    ap = o.get("abs_pose") or {}
    out = {
        "name": o.get("name"),
        "confidence": o.get("confidence"),
        "abs_pose": {"x": ap.get("x"), "y": ap.get("y")} if ap.get("x") is not None else None,
        "roi": o.get("roi"),
        "roi_source": o.get("roi_source"),
        "semantic_source": o.get("semantic_source"),
        "geometry_status": o.get("geometry_status"),
        "memory_status": o.get("memory_status"),
        "candidate_reason": o.get("candidate_reason"),
    }
    for k in ("roi_quality", "detector_label"):
        if o.get(k) is not None:
            out[k] = o.get(k)
    return {k: v for k, v in out.items() if v is not None}


def _dual_review_dump(
    look,
    rep_q,
    kept_q,
    rep_y,
    kept_y,
    kept_y_candidates,
    observer_pose,
    heading,
    vantage_idx,
):
    """双标注逐帧存档：原图 + Qwen/YOLO 各自【上报物体】与【过滤接地后留存】清单。DUAL_REVIEW_DIR 空则跳过。"""
    if not DUAL_REVIEW_DIR:
        return
    raw_dir = os.path.join(DUAL_REVIEW_DIR, "imgs_raw")
    os.makedirs(raw_dir, exist_ok=True)
    seq = _REVIEW_SEQ[0]
    _REVIEW_SEQ[0] += 1
    raw_rel = None
    imgpart = look.images[0] if (look and look.images) else None
    b64 = getattr(imgpart, "b64", None)
    if b64:
        raw_rel = os.path.join("imgs_raw", f"{seq:04d}.jpg")
        try:
            with open(os.path.join(DUAL_REVIEW_DIR, raw_rel), "wb") as rf:
                rf.write(base64.b64decode(b64))
        except Exception:  # noqa: BLE001
            raw_rel = None
    row = {
        "seq": seq, "vantage_idx": vantage_idx, "heading": round(float(heading), 1),
        "observer_pose": {"x": round(observer_pose["x"], 3), "y": round(observer_pose["y"], 3),
                          "yaw_deg": round(observer_pose.get("yaw_deg", 0.0), 1)},
        "raw_image": raw_rel,
        # 各标注器【上报的全部物体】（未过滤，看它到底把画面里什么叫成了什么）
        "qwen_reported": [_obj_brief(o) for o in (rep_q.get("objects") or [])
                          if isinstance(o, dict)],
        "yolo_reported": [
            _obj_brief(o)
            for o in (
                list(rep_y.get("objects") or [])
                + list(rep_y.get("candidate_objects") or [])
            )
            if isinstance(o, dict)
        ],
        # 过滤+几何接地后【留存】（进各自记忆图、参与打分）
        "qwen_kept": [_obj_kept_brief(o) for o in kept_q],
        "yolo_kept": [_obj_kept_brief(o) for o in kept_y],
        "yolo_candidates": [_obj_kept_brief(o) for o in kept_y_candidates],
    }
    with open(os.path.join(DUAL_REVIEW_DIR, "manifest.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean_candidate_records(candidate_records, bbox, visited, occ, confirmed):
    """Geometry-clean candidates and remove low-score duplicates of confirmed objects."""
    candidates = _dedup_objects(candidate_records, bbox)
    candidates, _ = _free_space_phantoms(candidates, visited)
    candidates, _ = _occupancy_phantoms(candidates, occ)
    out = []
    for candidate in candidates:
        ap = candidate.get("abs_pose")
        duplicate = any(
            isinstance(ap, dict)
            and ap.get("x") is not None
            and isinstance(obj.get("abs_pose"), dict)
            and obj["abs_pose"].get("x") is not None
            and _name_compat(candidate, obj)
            and _abs_dist(ap, obj["abs_pose"]) <= DEDUP_M
            for obj in confirmed
        )
        if duplicate:
            continue
        candidate["memory_status"] = "candidate"
        out.append(candidate)
    return out


def _finalize_dual(
    qwen_area_path,
    yolo_records,
    yolo_candidate_records,
    bbox,
    visited,
    occ,
    director,
):
    """双标注收尾：Qwen 主记忆已写盘；这里由 YOLO 旁路记录另建一张 YOLO 记忆图，两张各自打分并写 index.md。"""
    if not DUAL_REVIEW_DIR:
        return
    scorer = os.path.join(HERE, "eval", "score_explore.py")

    def _score(area_path, tag):
        try:
            res = subprocess.run([sys.executable, scorer, area_path], capture_output=True,
                                 text=True, timeout=60)
            fp = os.path.join(DUAL_REVIEW_DIR, f"score_{tag}.txt")
            with open(fp, "w", encoding="utf-8") as f:
                f.write(res.stdout + ("\n[stderr]\n" + res.stderr if res.stderr else ""))
            return res.stdout
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ 打分器({tag})失败: {e}")
            return ""

    # Qwen 记忆图（主记忆，正常流程已写）→ 拷贝一份 + 打分
    try:
        if qwen_area_path and os.path.isfile(qwen_area_path):
            shutil.copyfile(qwen_area_path, os.path.join(DUAL_REVIEW_DIR, "area_qwen.json"))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 拷贝 qwen area 失败: {e}")
    out_q = _score(qwen_area_path, "qwen")
    # YOLO 记忆图：同一套后处理（去重 + 反证 + 整理）从 YOLO 旁路记录另建，写独立 json 打分
    yolo_clean = _dedup_objects(yolo_records, bbox)
    yolo_clean, _p1 = _free_space_phantoms(yolo_clean, visited)
    yolo_clean, _p2 = _occupancy_phantoms(yolo_clean, occ)
    try:
        yolo_clean = _consolidate_memory(yolo_clean, director)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ YOLO 记忆整理跳过: {e}")
    for obj in yolo_clean:
        obj["memory_status"] = "confirmed"
    yolo_area = os.path.join(DUAL_REVIEW_DIR, "area_yolo.json")
    with open(yolo_area, "w", encoding="utf-8") as f:
        json.dump({"area": AREA, "type": "explore", "objects": yolo_clean},
                  f, ensure_ascii=False, indent=2)
    out_y = _score(yolo_area, "yolo")

    candidate_clean = _clean_candidate_records(
        yolo_candidate_records,
        bbox,
        visited,
        occ,
        yolo_clean,
    )
    candidate_area = os.path.join(DUAL_REVIEW_DIR, "area_yolo_with_candidates.json")
    with open(candidate_area, "w", encoding="utf-8") as f:
        json.dump(
            {
                "area": AREA,
                "type": "explore",
                "objects": yolo_clean + candidate_clean,
                "confirmed_count": len(yolo_clean),
                "candidate_count": len(candidate_clean),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    out_y_candidates = _score(candidate_area, "yolo_with_candidates")

    def _recall(txt):
        m = re.search(r"召回率 = (\d+)/(\d+)", txt or "")
        return m.group(0) if m else "n/a"
    # 画框（Qwen/YOLO 各一套）+ 逐帧三图并列 index.md（best-effort，缺 PIL 不致命）
    _mp = os.path.join(DUAL_REVIEW_DIR, "manifest.jsonl")
    n_rows = sum(1 for _ in open(_mp, encoding="utf-8")) if os.path.isfile(_mp) else 0
    try:
        import draw_dual_boxes
        draw_dual_boxes.draw_dual(DUAL_REVIEW_DIR)
        draw_dual_boxes.write_index(DUAL_REVIEW_DIR)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 画框/索引生成跳过: {e}")
    print(f"[双标注对比] {DUAL_REVIEW_DIR}  帧={n_rows}  "
          f"Qwen召回={_recall(out_q)}  YOLO confirmed召回={_recall(out_y)}  "
          f"YOLO 含候选召回={_recall(out_y_candidates)}")


def _hybrid_review_dump(look, boxes, judg, kept_h, rep_base, kept_base,
                        observer_pose, heading, vantage_idx):
    """混合验证逐帧存档：原图 + 编号框图（Qwen 实际所见）+ 每框 YOLO conf/detector_label 与 Qwen 判定，
    以及混合留存(带 abs_pose) 与 Qwen-only 基线上报/留存。HYBRID_REVIEW_DIR 空则跳过。"""
    if not HYBRID_REVIEW_DIR:
        return
    for d in ("imgs_raw", "imgs_yolo", "imgs_qwen", "imgs_numbered", "imgs_final"):
        os.makedirs(os.path.join(HYBRID_REVIEW_DIR, d), exist_ok=True)
    seq = _REVIEW_SEQ[0]
    _REVIEW_SEQ[0] += 1
    imgpart = look.images[0] if (look and look.images) else None
    b64 = getattr(imgpart, "b64", None)
    raw_rel = yolo_rel = qwen_rel = num_rel = final_rel = None
    if b64:
        raw_rel = os.path.join("imgs_raw", f"{seq:04d}.jpg")
        try:
            with open(os.path.join(HYBRID_REVIEW_DIR, raw_rel), "wb") as rf:
                rf.write(base64.b64decode(b64))
        except Exception:  # noqa: BLE001
            raw_rel = None
        # 三套结果图：①YOLO 标注(橙,所有提议框 label=英类conf) ②Qwen 注入(绿,仅留存框 label=中文名)
        # ③编号图(蓝,Qwen 判框时实际所见——只给编号不给 YOLO 类名，防 priming)
        try:
            import io
            import draw_dual_boxes
            from PIL import Image as PILImage
            src = base64.b64decode(b64)
            yolo_items = [{"roi": b.get("roi"),
                           "label": f"{b.get('idx')}:{b.get('detector_label')} "
                                    f"{b.get('confidence')}"} for b in (boxes or [])]
            qwen_items = [{"roi": b.get("roi"),
                           "label": ((judg or {}).get(b.get("idx")) or {}).get("name")}
                          for b in (boxes or [])
                          if ((judg or {}).get(b.get("idx")) or {}).get("keep")]
            final_items = [{"roi": o.get("roi"),
                            "label": f"{o.get('name')}:{o.get('roi_source') or '?'}"}
                           for o in (kept_h or [])]
            for rel, items, color, drawer in (
                (os.path.join("imgs_yolo", f"{seq:04d}.jpg"), yolo_items,
                 draw_dual_boxes.YOLO_COLOR, "labeled"),
                (os.path.join("imgs_qwen", f"{seq:04d}.jpg"), qwen_items,
                 draw_dual_boxes.QWEN_COLOR, "labeled"),
                (os.path.join("imgs_final", f"{seq:04d}.jpg"), final_items,
                 draw_dual_boxes.QWEN_COLOR, "labeled"),
                (os.path.join("imgs_numbered", f"{seq:04d}.jpg"), boxes, None, "numbered"),
            ):
                base = PILImage.open(io.BytesIO(src))
                if drawer == "numbered":
                    ann = draw_dual_boxes.draw_numbered_boxes(base, items)
                else:
                    ann = draw_dual_boxes.draw_labeled_boxes(base, items, color=color)
                ann.save(os.path.join(HYBRID_REVIEW_DIR, rel), quality=90)
            yolo_rel = os.path.join("imgs_yolo", f"{seq:04d}.jpg")
            qwen_rel = os.path.join("imgs_qwen", f"{seq:04d}.jpg")
            final_rel = os.path.join("imgs_final", f"{seq:04d}.jpg")
            num_rel = os.path.join("imgs_numbered", f"{seq:04d}.jpg")
        except Exception as _e:  # noqa: BLE001
            print(f"    [混合存图失败] {type(_e).__name__}: {str(_e)[:120]}")
    row = {
        "seq": seq, "vantage_idx": vantage_idx, "heading": round(float(heading), 1),
        "observer_pose": {"x": round(observer_pose["x"], 3), "y": round(observer_pose["y"], 3),
                          "yaw_deg": round(observer_pose.get("yaw_deg", 0.0), 1)},
        "raw_image": raw_rel, "yolo_image": yolo_rel, "qwen_image": qwen_rel,
        "numbered_image": num_rel, "final_image": final_rel,
        # YOLO 低 conf 出的每个框（编号 + conf + 英类 + roi）
        "yolo_boxes": [{"idx": b.get("idx"), "confidence": b.get("confidence"),
                        "detector_label": b.get("detector_label"), "roi": b.get("roi"),
                        "bbox_center": b.get("bbox_center"), "distance_m": b.get("distance_m"),
                        "abs_pose": b.get("abs_pose"), "size": b.get("size"),
                        "geometry_status": b.get("geometry_status"),
                        "depth_stats": b.get("depth_stats"),
                        "roi_quality": b.get("roi_quality"),
                        "roi_drift": b.get("roi_drift", False),
                        "los_blocked": b.get("los_blocked", False)}
                       for b in (boxes or [])],
        # Qwen 逐框判定（命名 / 完整度 / 是否留存=弃权门结果）
        "qwen_judgments": [{"idx": i, "name": j.get("name"),
                            "completeness": j.get("completeness"), "keep": j.get("keep")}
                           for i, j in sorted((judg or {}).items())],
        # 混合过滤+接地后留存（带反投世界坐标）
        "hybrid_kept": [_obj_kept_brief(o) for o in kept_h],
        "hybrid_roi_source_counts": rc.roi_source_counts(kept_h),
        # Qwen-only 基线：同帧上报 + 留存（对照混合召回）
        "baseline_reported": [_obj_brief(o) for o in (rep_base.get("objects") or [])
                              if isinstance(o, dict)],
        "baseline_semantic_sightings": rep_base.get("_semantic_sightings") or [],
        "baseline_kept": [_obj_kept_brief(o) for o in kept_base],
    }
    with open(os.path.join(HYBRID_REVIEW_DIR, "manifest.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _finalize_hybrid_review(hybrid_area_path, baseline_records, bbox, visited, occ, director):
    """混合验证收尾：混合主记忆已写盘（=area_hybrid）；由 Qwen-only 旁路记录另建基线记忆图，
    两张各自打分并生成人读 index.md（逐帧 原图|编号框图 + 每框 YOLO/Qwen 判定 + 两路留存）。"""
    if not HYBRID_REVIEW_DIR:
        return
    scorer = os.path.join(HERE, "eval", "score_explore.py")

    def _score(area_path, tag):
        try:
            res = subprocess.run([sys.executable, scorer, area_path], capture_output=True,
                                 text=True, timeout=60)
            with open(os.path.join(HYBRID_REVIEW_DIR, f"score_{tag}.txt"), "w",
                      encoding="utf-8") as f:
                f.write(res.stdout + ("\n[stderr]\n" + res.stderr if res.stderr else ""))
            return res.stdout
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ 打分器({tag})失败: {e}")
            return ""

    # 混合记忆图（主记忆，正常流程已写）→ 拷贝一份为 area_hybrid.json + 打分
    try:
        if hybrid_area_path and os.path.isfile(hybrid_area_path):
            shutil.copyfile(hybrid_area_path,
                            os.path.join(HYBRID_REVIEW_DIR, "area_hybrid.json"))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 拷贝 hybrid area 失败: {e}")
    out_h = _score(hybrid_area_path, "hybrid")
    # Qwen-only 基线记忆图：同一套后处理从基线旁路记录另建，写独立 json 打分
    base_clean = _dedup_objects(baseline_records, bbox)
    base_clean, _p1 = _free_space_phantoms(base_clean, visited)
    base_clean, _p2 = _occupancy_phantoms(base_clean, occ)
    try:
        base_clean = _consolidate_memory(base_clean, director)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 基线记忆整理跳过: {e}")
    base_area = os.path.join(HYBRID_REVIEW_DIR, "area_baseline.json")
    with open(base_area, "w", encoding="utf-8") as f:
        json.dump({"area": AREA, "type": "explore", "objects": base_clean},
                  f, ensure_ascii=False, indent=2)
    out_b = _score(base_area, "baseline")

    def _recall(txt):
        m = re.search(r"召回率 = (\d+)/(\d+)", txt or "")
        return m.group(0) if m else "n/a"
    _write_hybrid_index(_recall(out_h), _recall(out_b))
    print(f"[混合弃权门验证] {HYBRID_REVIEW_DIR}  "
          f"混合召回={_recall(out_h)}  Qwen基线召回={_recall(out_b)}  → index.md 供审阅")


def _write_hybrid_index(hybrid_recall, baseline_recall):
    """人读 index.md：逐帧 原图|编号框图 并列 + 每框 YOLO(conf,label)→Qwen(name/完整度/keep) + 两路留存。"""
    mpath = os.path.join(HYBRID_REVIEW_DIR, "manifest.jsonl")
    if not os.path.isfile(mpath):
        return
    rows = [json.loads(x) for x in open(mpath, encoding="utf-8") if x.strip()]
    n_box = sum(len(r.get("yolo_boxes") or []) for r in rows)
    n_keep = sum(1 for r in rows for j in (r.get("qwen_judgments") or []) if j.get("keep"))
    geom_bad = sum(
        1 for r in rows for b in (r.get("yolo_boxes") or [])
        if b.get("geometry_status") not in (None, rc.GEOMETRY_OK)
    )
    sightings = sum(len(r.get("baseline_semantic_sightings") or []) for r in rows)
    source_counts = {}
    for r in rows:
        for o in (r.get("hybrid_kept") or []):
            src = o.get("roi_source") or "unknown"
            source_counts[src] = source_counts.get(src, 0) + 1
    source_txt = ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items())) or "n/a"
    lines = [
        "# 混合标注·弃权门验证实验（YOLO 出 ROI + Qwen 命名/弃权 vs Qwen-only 基线）", "",
        f"- 帧数：{len(rows)}  YOLO 框总数：{n_box}  Qwen 留存(未弃权)：{n_keep}"
        f"  弃权率≈{1 - n_keep / n_box:.0%}" if n_box else f"- 帧数：{len(rows)}",
        f"- **混合记忆召回：{hybrid_recall}**（`score_hybrid.txt`）",
        f"- **Qwen-only 基线召回：{baseline_recall}**（`score_baseline.txt`）",
        f"- ROI 来源分布：{source_txt}",
        f"- YOLO 候选几何失败：{geom_bad}；Qwen 语义暂存未入库：{sightings}",
        "- **弃权门通过判据**：对 YOLO 的墙/半截物/空墙框，Qwen `keep=false` 可靠弃权（弃权精度高）；"
        "且混合召回 ≥ 基线（≥0.70 项目门）。",
        "- 每帧五图：原图 / YOLO 标注(橙,全部提议) / Qwen 注入(绿,仅留存命名) / 最终采用ROI / 编号图。", ""]
    for r in rows:
        p = r.get("observer_pose") or {}
        seq = r.get("seq")
        lines.append(f"## 帧{seq:04d} · v{r.get('vantage_idx')} · heading={r.get('heading')}° "
                     f"· pose=({p.get('x')},{p.get('y')})")
        quad = []
        for key, tag in (("raw_image", "原图"), ("yolo_image", "YOLO"),
                         ("qwen_image", "Qwen"), ("final_image", "最终"),
                         ("numbered_image", "编号")):
            if r.get(key):
                quad.append(f"![{tag}]({r[key]})")
        lines.append(" ".join(quad))
        judg = {j.get("idx"): j for j in (r.get("qwen_judgments") or [])}
        for b in (r.get("yolo_boxes") or []):
            j = judg.get(b.get("idx")) or {}
            mark = "✅留存" if j.get("keep") else "🚫弃权"
            lines.append(
                f"- 框{b.get('idx')} YOLO({b.get('detector_label')},{b.get('confidence')}) "
                f"→ Qwen: {j.get('name') or '—'} [{j.get('completeness')}] {mark}")
        def _hk(o):
            ap = o.get("abs_pose") or {}
            return (f"{o.get('name')}@({ap.get('x')},{ap.get('y')})"
                    f"[roi={o.get('roi_source') or '?'} geo={o.get('geometry_status') or '?'}]")
        hk = "、".join(_hk(o) for o in (r.get("hybrid_kept") or []))
        bk = "、".join(o.get("name") for o in (r.get("baseline_reported") or []) if o.get("name"))
        lines.append(f"- **混合留存**：{hk or '无'}")
        lines.append(f"- Qwen-only 基线上报：{bk or '无'}")
        ss = "、".join(o.get("name") for o in (r.get("baseline_semantic_sightings") or [])
                       if o.get("name"))
        if ss:
            lines.append(f"- Qwen-only 语义暂存未入库：{ss}")
        lines.append("")
    with open(os.path.join(HYBRID_REVIEW_DIR, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _process_rep(ex, rep, cur, occ=None, candidate_boxes=None):
    """一次标注(rep.objects) → 过滤(_annotation_ok) + 几何回填(depth_roi→size/abs_pose) +
    护栏(ROI漂移/LOS穿墙) → 返回留存物体列表。Qwen 与 YOLO 两路共用同一后处理（公平对比）。"""
    heading_objs = []
    for o in (rep.get("objects") or []):
        if not (isinstance(o, dict) and o.get("name")):
            continue
        if _is_forbidden_name(o.get("name")):    # 建筑面/门/自身 → 硬过滤，不入物体库
            print(f"    [丢弃标注] {o.get('name')}: 禁记类(建筑面/门/机器人自身)")
            continue
        co = {"name": o["name"], "confidence": o.get("confidence"),
              "spatial": o.get("bearing"), "roi": o.get("roi"),
              "bbox_center": o.get("bbox_center"), "distance_m": o.get("distance_m")}
        for k in ("detector_source", "verified_by", "roi_source", "semantic_source",
                  "geometry_status", "abs_pose", "size", "detector_label", "depth_stats",
                  "roi_quality", "detector_score", "mask_polygon", "model_version",
                  "memory_status", "candidate_reason", "publish_threshold"):
            if o.get(k) is not None:
                co[k] = o[k]
        ok, why = _annotation_ok(co)
        if not ok:
            print(f"    [丢弃标注] {co['name']}: {why}")
            continue
        heading_objs.append(co)
    heading_objs, sightings = _resolve_qwen_candidate_rois(ex, heading_objs, candidate_boxes)
    rep["_semantic_sightings"] = sightings
    _backfill_geometry_local(ex, heading_objs, cur, occ=occ)
    kept = []
    for o in heading_objs:
        o.pop("_mask_depth_probe_roi", None)
        o.pop("mask_polygon", None)
        if o.pop("_roi_drift", False):    # Task 1 护栏：ROI 深度不一致(疑似漂移) → 丢弃该帧
            print(f"    [丢弃标注] {o.get('name')}: ROI深度与band距离不一致(疑似漂移)")
            continue
        if o.pop("_los_blocked", False):  # Task 1(1b)：观测→坐标视线穿墙(疑似钉墙后) → 丢弃该帧
            print(f"    [丢弃标注] {o.get('name')}: 观测视线穿墙(反投坐标在墙后,疑似幻觉)")
            continue
        if o.get("geometry_status") not in (None, rc.GEOMETRY_OK):
            print(f"    [丢弃标注] {o.get('name')}: geometry_status={o.get('geometry_status')}")
            continue
        if o.get("abs_pose") is None:
            print(f"    [语义暂存] {o.get('name')}: 无验证ROI，不写入坐标记忆")
            continue
        ok_conf, why_conf = _high_confusion_snap_ok(o)
        if not ok_conf:
            print(f"    [丢弃标注] {o.get('name')}: {why_conf}")
            continue
        o.setdefault("geometry_status", rc.GEOMETRY_OK)
        o.setdefault("roi_source", "unknown")
        o.setdefault("semantic_source", "unknown")
        kept.append(o)
    return kept


def _sweep_vantage(ex, known_names, headings=SWEEP_HEADINGS, occ=None):
    """到格后【原地环视】：逐朝向标注器 inspect（出物体+ROI）+ 顾问标门 + 收墙点；
    代码用 depth_roi 给每个物体回填 size+abs_pose（贴物体的框，比 bearing 列距离更准）。

    双标注模式(DUAL_REVIEW_DIR)：同一帧同时给 Qwen 与 YOLO 标注，两路各自后处理并存档对比，
    Qwen 作驱动（顾问/门/写主记忆），YOLO 旁路收集供对比打分。
    混合验证模式(HYBRID_REVIEW_DIR)：混合(YOLO ROI+Qwen 命名/弃权)作驱动写主记忆，Qwen-only inspect 旁路作基线。
    返回 {objects, hint_bearings, doors_raw, wall_pts[, objects_yolo | objects_baseline]}。
    """
    objs_all, objs_all_y, hint_bearings, doors_raw, wall_pts = [], [], [], [], []
    objs_all_candidates, objs_all_y_candidates = [], []
    objs_all_base = []
    off_map = {"left": 45.0, "center": 0.0, "right": -45.0}
    _VANTAGE_SEQ[0] += 1
    vantage_idx = _VANTAGE_SEQ[0]
    for h in headings:
        p = _pose(ex)
        nav.geo_face_point(ex, p["x"] + math.cos(math.radians(h)), p["y"] + math.sin(math.radians(h)))
        front = _ensure_view_distance(ex)     # 太近先后退，保证正前 scan>0.5 再标注
        cur = _pose(ex)
        wall_pts.extend(_scan_world_points(ex, cur))
        if front is not None and front < NEAR_LABEL_M:
            # 动态观测距离：贴墙家具退不开(front∈[CLOSE,0.7))→问 Qwen 是否值得贴近；值得则放宽标注，否则跳过
            close_ok = False
            if front >= NEAR_LABEL_CLOSE_M:
                _look = ex.ros.call("look", {})
                if _approach_worth(ex, _look.images[0] if _look.images else None):
                    close_ok = True
                    print(f"    [贴近观测] 正前{front:.2f}m∈[{NEAR_LABEL_CLOSE_M},{NEAR_LABEL_M}) "
                          f"Qwen判贴墙家具值得贴近 → 放宽标注(治南墙漏记)")
            if not close_ok:
                print(f"    [跳过标注] 正前仅 {front:.2f}m<{NEAR_LABEL_M}m 且退不开 → 本朝向不标注(防近距误标)")
                continue
        look = None
        if DUAL_REVIEW_DIR or HYBRID_REVIEW_DIR or PERCEPTION_BACKEND == "yoloe":
            look = ex.ros.call("look", {})
            img = look.images[0] if look.images else None
        hints = sorted(set(known_names)) or None
        rep_y = rep_base = None
        hyb_boxes, hyb_judg = [], {}
        if DUAL_REVIEW_DIR:                    # Qwen 与 YOLO 看同一帧
            rep = harness.inspect_and_report(ex, look=look, name_hints=hints)
            rep_y = _yoloe_inspect_image(img)
        elif HYBRID_REVIEW_DIR:               # 混合(驱动) vs Qwen-only(基线) 看同一帧
            hyb_boxes, _w, _h = _hybrid_yolo_boxes(img)
            _backfill_box_geometry(ex, hyb_boxes, cur, occ=occ)
            allowed_names, ctx_hint = _south_wall_label_context(cur, h)
            hyb_judg = harness.name_boxes(
                ex, look, hyb_boxes, name_hints=hints,
                allowed_names=allowed_names, context_hint=ctx_hint,
            )
            rep_base = harness.inspect_and_report(ex, look=look, name_hints=hints)
            # 【让 Qwen 全图也看一遍】YOLO 框命名(定位准) ∪ Qwen 全图 inspect(补 YOLO 漏检=召回天花板)
            #   → 合并交下游按位置/名去重(_dedup_objects)。基线仍单取 rep_base 对照。
            merged = (yoloe.assemble_hybrid_objects(hyb_boxes, hyb_judg)
                      + list(rep_base.get("objects") or []))
            rep = {"objects": merged, "image": img, "raw": ""}
        elif PERCEPTION_BACKEND == "yoloe":
            rep = _yoloe_inspect_image(img)
        else:
            rep = harness.inspect_and_report(ex, name_hints=hints)
        for hh in _advisor(ex, rep):           # 导航顾问用驱动标注器(dual=Qwen)
            if not isinstance(hh, dict):
                continue
            off = off_map.get((hh.get("dir") or "").lower())
            if off is None:
                continue
            wb = cur["yaw_deg"] + off
            hint_bearings.append(wb)
            if hh.get("kind") == "door":
                doors_raw.append({"from": [round(cur["x"], 2), round(cur["y"], 2)],
                                  "bearing": wb, "dir": hh.get("dir"),
                                  "reason": str(hh.get("reason", ""))[:50]})
        kept = _process_rep(ex, rep, cur, occ, candidate_boxes=hyb_boxes)
        frame_id = f"v{vantage_idx}h{int(h)}"         # 单帧共现反合并的帧身份（vantage+朝向）
        for _o in kept:
            _o["_frame"] = frame_id
        if DUAL_REVIEW_DIR:
            kept_y = _process_rep(ex, rep_y, cur, occ)
            candidate_rep = {"objects": rep_y.get("candidate_objects") or []}
            kept_y_candidates = _process_rep(ex, candidate_rep, cur, occ)
            for candidate in kept_y_candidates:
                candidate["memory_status"] = "candidate"
            _dual_review_dump(
                look,
                rep,
                kept,
                rep_y,
                kept_y,
                kept_y_candidates,
                cur,
                h,
                vantage_idx,
            )
            for _o in kept_y:
                _o["_frame"] = frame_id
            for _o in kept_y_candidates:
                _o["_frame"] = frame_id
            objs_all_y.extend(kept_y)
            objs_all_y_candidates.extend(kept_y_candidates)
        elif HYBRID_REVIEW_DIR:
            kept_base = _process_rep(ex, rep_base, cur, occ, candidate_boxes=None)
            _hybrid_review_dump(look, hyb_boxes, hyb_judg, kept, rep_base, kept_base,
                                cur, h, vantage_idx)
            for _o in kept_base:
                _o["_frame"] = frame_id
            objs_all_base.extend(kept_base)
        elif PERCEPTION_BACKEND == "yoloe":
            candidate_rep = {"objects": rep.get("candidate_objects") or []}
            kept_candidates = _process_rep(ex, candidate_rep, cur, occ)
            for candidate in kept_candidates:
                candidate["memory_status"] = "candidate"
                candidate["_frame"] = frame_id
            objs_all_candidates.extend(kept_candidates)
            _review_dump(rep, kept, cur, h, vantage_idx)   # 单 YOLO 审阅存档
        objs_all.extend(kept)
    ret = {"objects": objs_all, "hint_bearings": hint_bearings,
           "doors_raw": doors_raw, "wall_pts": wall_pts}
    if DUAL_REVIEW_DIR:
        ret["objects_yolo"] = objs_all_y
        ret["objects_yolo_candidates"] = objs_all_y_candidates
    if HYBRID_REVIEW_DIR:
        ret["objects_baseline"] = objs_all_base
    if PERCEPTION_BACKEND == "yoloe":
        ret["objects_candidates"] = objs_all_candidates
    return ret


# ===== Task 2：APF 势场观测点（Claude 每次选点，保留反应式后退兜底）=====
VIEWPOINT_ARRIVE_TOL_M = 0.3   # 已在选中的观测点附近此距离内 → 不折腾，直接环视
OBS_REPICK_TRIES = 2           # 到位后观测点不合格时，回 APF 重选的最大次数


def _read_sectors(ex, n=36):
    """读 scan_summary(细分扇区) → {label: dist_m}；失败返回 {}。"""
    try:
        scan = json.loads(ex.ros.call("scan_summary", {"sectors": n}).text) or {}
        return scan.get("sectors", {}) or {}
    except Exception:  # noqa: BLE001
        return {}


def _pick_viewpoint(ex, director, target_xy=None):
    """环视前：建 APF 势场 → Claude 从候选选安全站位 → 导过去 → 【到位用真实 scan 复核质量门】。

    质量门 `pf.scan_obs_quality`：最近障 ≥0.5m 且四周均匀(CV≤OBS_CV_MAX)才算合格观测点；不合格则把该点
    排除、从当前位姿回 APF 重选(至多 OBS_REPICK_TRIES 次)——从源头不停在贴墙/角落/桌底这类坏点。
    角色分工不变：代码算势场/筛候选/判质量/导航，Claude 只【选 id】不产坐标。无候选/异常/全不合格 →
    返回 None（调用方就地尽力环视一次，用户选定兜底，不丢覆盖）。返回合格观测点 (x,y) 或 None。
    """
    fixed_target = target_xy
    bad_xys = []                                    # 已判不合格的观测点，重选时按邻近剔除
    for attempt in range(OBS_REPICK_TRIES + 1):
        p = _pose(ex)
        if "x" not in p:
            return None
        tgt = fixed_target if fixed_target is not None else (p["x"], p["y"])
        payload = pf.build_viewpoint_payload(_read_sectors(ex), p, tgt)
        cands = [c for c in (payload.get("_candidates_full") or [])
                 if all(math.hypot(c["x"] - bx, c["y"] - by) >= VIEWPOINT_ARRIVE_TOL_M
                        for bx, by in bad_xys)]
        if not cands:
            break
        plan_in = {"ascii_field": payload["ascii_field"], "target_xy": payload["target_xy"],
                   "pose": payload["pose"],
                   "candidates": [{"id": c["id"], "x": c["x"], "y": c["y"],
                                   "clearance_m": c["clearance_m"], "potential": c["potential"]}
                                  for c in cands]}
        plan = director.pick_viewpoint(plan_in)
        tid = plan.get("target_id") if isinstance(plan, dict) else None
        chosen = next((c for c in cands if c["id"] == tid), None)
        if chosen is None:
            break
        if math.hypot(chosen["x"] - p["x"], chosen["y"] - p["y"]) >= VIEWPOINT_ARRIVE_TOL_M:
            nav.geo_goto_around(ex, chosen["x"], chosen["y"],
                                tol_m=VIEWPOINT_ARRIVE_TOL_M, max_legs=10)
        q = pf.scan_obs_quality(_read_sectors(ex))     # 到位用真实 scan 复核质量门
        if q["ok"]:
            print(f"    [APF观测点] Claude 选 {chosen['id']}@({chosen['x']},{chosen['y']}) "
                  f"clear={chosen['clearance_m']}m min={q['min_clear_m']} cv={q['cv']} "
                  f"rationale={str(plan.get('rationale', ''))[:46]}")
            return (chosen["x"], chosen["y"])
        cur = _pose(ex)
        bad_xys.append((cur.get("x", chosen["x"]), cur.get("y", chosen["y"])))
        tail = (f"→ 回APF重选({attempt + 1}/{OBS_REPICK_TRIES})"
                if attempt < OBS_REPICK_TRIES else f"→ 重选{OBS_REPICK_TRIES}次用尽")
        print(f"    [观测点不宜] min={q['min_clear_m']} cv={q['cv']} "
              f"(阈 {pf.OBS_MIN_CLEAR_M}/{pf.OBS_CV_MAX}) {tail}")
    print("    [观测点] 无合格观测点 → 就地尽力观测")
    return None


def _goto_and_sweep(ex, director, known_names, target_xy=None, occ=None):
    """Task 2 包装：APF+Claude 选观测点并导过去 → 更新 occupancy → _sweep_vantage 环视。

    返回 (sweep, final_pose)。final_pose 为选点移动后的真值位姿，供调用方更新 last_vantage_xy / visited。
    occ 非空则在观测点用 scan_rays 更新栅格（frontier 覆盖的数据源）。
    """
    _pick_viewpoint(ex, director, target_xy)
    _update_occ(ex, occ, _pose(ex))
    sweep = _sweep_vantage(ex, known_names, occ=occ)
    return sweep, _pose(ex)


# ===== 几何校验去重（用户："据位姿+深度算物体位置，重复就过滤"）=====
def _abs_dist(a, b):
    """两 abs_pose 的 xy 平面距离；任一缺坐标返回 1e9（视作不同物体）。"""
    if not (isinstance(a, dict) and isinstance(b, dict)) or a.get("x") is None or b.get("x") is None:
        return 1e9
    return math.hypot(a["x"] - b["x"], (a.get("y") or 0.0) - (b.get("y") or 0.0))


def _norm_name(n):
    """归一化物体名用于匹配：小写、空白/下划线统一（reception desk≡reception_desk）。"""
    return re.sub(r"[\s_]+", " ", (n or "").strip().lower())


def _name_set(o):
    """物体的归一化名集合：主名 + 别名。"""
    s = {_norm_name(o.get("name"))}
    s.update(_norm_name(a) for a in (o.get("aliases") or []))
    return {x for x in s if x}


def _name_compat(a, b):
    """两物体是否【同类】（主名/别名有交集）。用于位置去重时只合并同类，
    避免把挨得近的【不同类】家具（桌+显示器+椅）按位置并成一坨（实测 office 召回杀手）。"""
    return bool(_name_set(a) & _name_set(b))


def _dedup_aliases(aliases, main_name):
    """别名保序去重（按归一化名），并剔除等于主名的项（修复重复 red_cabinet / 别名==主名）。"""
    out, seen = [], {_norm_name(main_name)}
    for a in aliases or []:
        na = _norm_name(a)
        if not na or na in seen:
            continue
        seen.add(na)
        out.append(a)
    return out


def _in_bbox(ap, b, m=BOUNDARY_MARGIN):
    return (b["xmin"] - m) <= ap["x"] <= (b["xmax"] + m) and (b["ymin"] - m) <= ap["y"] <= (b["ymax"] + m)


def _merge_obj_pair(keep, inc):
    """重复观测合并进 keep：高置信度的取作主条（旧名进 aliases），数组字段并集。"""
    kc = keep.get("confidence", 0) or 0
    ic = inc.get("confidence", 0) or 0
    aliases = list(keep.get("aliases") or [])
    if ic > kc:
        old = keep.get("name")
        # 语义字段按置信度取高；几何字段(abs_pose/size/roi)不在此列——由下面 B1 按【距离】决定
        for f in ("name", "confidence", "spatial", "state"):
            if inc.get(f) is not None:
                keep[f] = inc[f]
        if old:
            aliases.append(old)               # 被降级的旧主名进别名
    elif inc.get("name"):
        aliases.append(inc["name"])           # 低置信度观测的名进别名
    aliases.extend(inc.get("aliases") or [])  # 合并 inc 自带别名
    deduped = _dedup_aliases(aliases, keep.get("name"))
    if deduped:
        keep["aliases"] = deduped
    else:
        keep.pop("aliases", None)
    for k in ("verified_by", "affordance"):
        merged = list(keep.get(k) or [])
        for v in (inc.get(k) or []):
            if v not in merged:
                merged.append(v)
        if merged:
            keep[k] = merged
    # B1 多视角融合：几何字段(abs_pose/size)取【更近那帧】——距离更小=深度更准、越少打到远墙。
    kd, idd = keep.get("distance_m"), inc.get("distance_m")
    if inc.get("abs_pose") and idd is not None and (kd is None or idd < kd):
        keep["abs_pose"] = inc["abs_pose"]
        keep["distance_m"] = idd
        for f in ("size", "roi", "spatial", "roi_source", "semantic_source",
                  "geometry_status", "roi_quality", "detector_label"):
            if inc.get(f) is not None:
                keep[f] = inc[f]
        keep.pop("size_unreliable", None)        # 采用更近帧的尺寸可信度
        if inc.get("size_unreliable"):
            keep["size_unreliable"] = True
    elif keep.get("abs_pose") is None and inc.get("abs_pose"):
        keep["abs_pose"] = inc["abs_pose"]       # keep 无位置则先补上（inc 有就用）
        if inc.get("distance_m") is not None:
            keep["distance_m"] = inc["distance_m"]
        for f in ("size", "roi", "spatial", "roi_source", "semantic_source",
                  "geometry_status", "roi_quality", "detector_label"):
            if inc.get(f) is not None:
                keep[f] = inc[f]


# 物体多视角投票门（休眠件，未接入写盘）：可信物体须 ≥ 这么多个【不同 vantage】独立报到才入库。
# 实测在【单次覆盖】探索下过狠——真家具多只被走到一次，K=2 把召回打到 5%(见 Report/README §4.5)。
# 故写盘改用零召回代价的 _free_space_phantoms；本门保留供【将来多次重访的密集覆盖模式】按需启用。
MIN_OBJ_VIEWS = 2


def _dedup_objects(
    vantage_records,
    boundary,
    min_views=1,
    dropped_out=None,
    *,
    dedup_m=DEDUP_M,
):
    """几何校验去重 + 多视角一致性投票：可信物体(在界内+地面高度带)按【世界位置】去重(name-agnostic)；
    不可信物体(无 abs_pose/越界/高处)按【归一化名】归并并标 size_unreliable。

    位置重复=同一物体的多视角重复观测 → 合并而非新增（直接解决 71 条噪声里的"同物多记"）。
    min_views>1 时启用投票：合并后仅被【单一 vantage】支持的可信物体判为孤帧幻觉丢弃（被丢者
    追加进 dropped_out 供日志）。投票纯几何(比对反投世界坐标)由代码执行——Qwen 只出单帧证词、
    代码当陪审团（对齐门校验 DOOR_MIN_COUNT）。默认 min_views=1（不投票，供中途覆盖统计取全量）。
    """
    reliable, unreliable = [], {}
    for vi, rec in enumerate(vantage_records):     # vi=vantage 序号，即该观测的"独立视角"身份
        for o in rec.get("objects", []):
            if not isinstance(o, dict) or not o.get("name"):
                continue
            ap = o.get("abs_pose")
            ok_pos = isinstance(ap, dict) and ap.get("x") is not None
            zok = ok_pos and (ap.get("z") is None or RELIABLE_Z[0] <= ap["z"] <= RELIABLE_Z[1])
            inb = ok_pos and (not boundary or _in_bbox(ap, boundary))
            if ok_pos and zok and inb:
                candidates = []
                for r in reliable:
                    # 同类 + 位置近 = 同一物体的多视角重复 → 合并；不同类即使挨着也各自成条
                    distance = _abs_dist(ap, r["abs_pose"])
                    if distance <= dedup_m and _name_compat(o, r):
                        # 单帧共现反合并：同一帧已在此簇报过同名、且位置差 > 同物阈 = 感知已分辨出的
                        #   另一个实例(密集同名家具) → 不并进 r，跳过继续找/新建条（保住实例数）。
                        if (o.get("_frame") in r["_frames"]
                                and distance > COOCCUR_EPS_M):
                            continue
                        candidates.append((distance, r))
                # Wider YOLO association radii require nearest-neighbour matching;
                # first-match greedily attaches a second physical instance to the
                # first cluster and leaves the correct cluster as a false singleton.
                hit = min(candidates, key=lambda item: item[0])[1] if candidates else None
                if hit:
                    _merge_obj_pair(hit, o)
                    hit["_views"].add(vi)        # 记下又一个独立视角佐证了此位置
                    hit["_frames"].add(o.get("_frame"))
                else:
                    no = dict(o)
                    no["_views"] = {vi}
                    no["_frames"] = {o.get("_frame")}
                    reliable.append(no)
            else:
                key = _norm_name(o["name"])
                oo = dict(o)
                oo["size_unreliable"] = True
                if not (ok_pos and inb):
                    oo["abs_pose"] = None        # 越界/无效投影不瞎编坐标
                if key in unreliable:
                    _merge_obj_pair(unreliable[key], oo)
                else:
                    unreliable[key] = oo
    # 多视角投票：仅单一 vantage 支持的可信物体=孤帧幻觉 → 丢弃（precision-over-recall）
    voted = []
    for o in reliable:
        n_views = len(o.pop("_views", None) or ())    # 顺手剥除内部记账字段(非 schema、勿写盘)
        o.pop("_frames", None)
        if n_views >= min_views:
            voted.append(o)
        elif dropped_out is not None:
            dropped_out.append(o)
    reliable = voted
    # 跨桶去重：同一物体若已有【可信位置】，丢弃它的不可信高视角重复条（按名/别名匹配），避免双计。
    rel_names = set()
    for o in reliable:
        rel_names.add(_norm_name(o.get("name")))
        for a in (o.get("aliases") or []):
            rel_names.add(_norm_name(a))
    unr_final = [o for o in unreliable.values() if _norm_name(o.get("name")) not in rel_names]
    result = reliable + unr_final
    for o in result:                     # 剥除内部记账字段(非 schema、勿写盘)
        o.pop("_frame", None)
        o.pop("_frames", None)
    return result


FREE_SPACE_R_M = 0.3    # 自由空间反证半径：物体反投坐标落在任一已走过位姿此半径内=车曾占据该处
# → 车穿不过真家具 → 判近距离幻觉丢弃。取车体半径量级(robomaster 半宽~0.16、含误差留余)——
# 真物体绝不会落在车实际到达的点上，故【零召回代价】，只删压在自身轨迹上的近距离幻觉。


def _free_space_phantoms(objects, visited_poses, radius=FREE_SPACE_R_M):
    """自由空间反证：把 abs_pose 落在任一已走过位姿 radius 内的物体判为幻觉（车穿不过真家具）。

    独立于多视角冗余的正交信号——用车【自身轨迹认证的自由空间】反证，故不伤召回（真物体不在此）。
    返回 (kept, dropped)。无坐标或无轨迹时一律保留（无从反证，不误杀）。
    """
    if not visited_poses:
        return list(objects), []
    kept, dropped = [], []
    for o in objects:
        ap = o.get("abs_pose")
        if not (isinstance(ap, dict) and ap.get("x") is not None):
            kept.append(o)                       # 无坐标无从反证 → 保留
            continue
        ox, oy = float(ap["x"]), float(ap.get("y") or 0.0)
        in_free = any(math.hypot(ox - vx, oy - vy) <= radius for vx, vy in visited_poses)
        (dropped if in_free else kept).append(o)
    return kept, dropped


OCC_FREE_CLEAR_M = 0.6   # occ 自由空间反证：abs_pose 落在已扫自由格、且此半径内无 occupied = 空旷地板幻觉
# → 车雷达已把该处标 free(看到地板)、附近又没有墙/障碍 → 该处不该有家具 → 判幻觉丢。
# 「附近无 occupied」是精度护栏：贴墙真家具(墙线记 occupied)因此豁免，不误杀（配合 precision-over-recall）。


def _occ_free_phantom(occ, x, y, clear_m=OCC_FREE_CLEAR_M):
    """abs_pose (x,y) 落在 occ 确信自由格(free/visited) 且邻域 clear_m 内无 occupied → 空旷地板幻觉。

    比 _free_space_phantoms 的「细轨迹」反证覆盖面大得多（整片已扫自由空间）。occ 空/坐标在
    未知或占据格 → 返回 False（无从反证或可能真有家具，不误杀）。纯查表、可脱 ROS 单测。
    """
    if occ is None or not occ.cells:
        return False
    c = occ.cell_of(x, y)
    if occ.state(c) not in oc._KNOWN_FREE:       # 未知/占据处不判（可能真有家具/墙）
        return False
    rc = max(1, int(math.ceil(clear_m / occ.res)))
    for dx in range(-rc, rc + 1):
        for dy in range(-rc, rc + 1):
            if occ.state((c[0] + dx, c[1] + dy)) == oc.OCCUPIED:
                return False                     # 邻域有墙/障碍 → 可能贴墙真家具 → 豁免
    return True


def _occupancy_phantoms(objects, occ, clear_m=OCC_FREE_CLEAR_M):
    """occ 自由空间反证：abs_pose 落在已扫空旷自由格的物体判为幻觉。返回 (kept, dropped)。

    与 _free_space_phantoms 正交（后者只查车轨迹 0.3m 内）；无坐标/occ 空 → 保留（不误杀）。
    """
    if occ is None or not occ.cells:
        return list(objects), []
    kept, dropped = [], []
    for o in objects:
        ap = o.get("abs_pose")
        if not (isinstance(ap, dict) and ap.get("x") is not None):
            kept.append(o)
            continue
        if _occ_free_phantom(occ, float(ap["x"]), float(ap.get("y") or 0.0), clear_m):
            dropped.append(o)
        else:
            kept.append(o)
    return kept, dropped


# ===== Role-2：代码按位置聚簇 → Claude 只给每簇规范名 → 代码按簇合并（几何归代码、语义归 Claude）=====
def _position_clusters(records, cluster_m=CONSOLIDATE_CLUSTER_M):
    """把去重后记录按世界位置贪心聚簇（name-agnostic，代码持有几何）。
    返回 [(member_indices, (cx,cy) or None), ...]；无坐标的各自单簇。"""
    clusters, singles = [], []
    for i, o in enumerate(records):
        ap = o.get("abs_pose")
        if not (isinstance(ap, dict) and ap.get("x") is not None):
            singles.append(i)
            continue
        x, y = float(ap["x"]), float(ap.get("y") or 0.0)
        hit = next((c for c in clusters
                    if math.hypot(x - c["sx"] / c["n"], y - c["sy"] / c["n"]) <= cluster_m), None)
        if hit:
            hit["ids"].append(i)
            hit["sx"] += x
            hit["sy"] += y
            hit["n"] += 1
        else:
            clusters.append({"ids": [i], "sx": x, "sy": y, "n": 1})
    out = [(c["ids"], (round(c["sx"] / c["n"], 2), round(c["sy"] / c["n"], 2))) for c in clusters]
    out += [([i], None) for i in singles]
    return out


def _merge_members(members, canonical):
    """把一簇成员合并成一个物体（几何取更近帧、别名并集），强制规范名。"""
    base = dict(members[0])
    for m in members[1:]:
        _merge_obj_pair(base, m)
    aliases = list(base.get("aliases") or []) + [m.get("name") for m in members]
    base["name"] = canonical
    ded = _dedup_aliases(aliases, canonical)
    if ded:
        base["aliases"] = ded
    else:
        base.pop("aliases", None)
    base["verified_by"] = list(dict.fromkeys((base.get("verified_by") or []) + ["claude_consolidate"]))
    return base


def _majority_name(records, ids):
    """簇内多数（归一化）名对应的原始名——Claude 没判定时的保守兜底。"""
    from collections import Counter
    top = Counter(_norm_name(records[i].get("name")) for i in ids).most_common(1)[0][0]
    return next((records[i].get("name") for i in ids if _norm_name(records[i].get("name")) == top),
                records[ids[0]].get("name"))


def _apply_consolidation(records, clusters, plan):
    """按 Claude 的【每簇一个规范名】整理：每簇成员合并成一个物体（坐标由代码取更近帧、别名并集）；
    drop 整簇剔除；Claude 未判定的簇 → 按多数名合并保留（不丢）。plan 无效 → 返回原 records。
    每簇恒 1 个物体（同簇=同一物理物体的多视角，绝不因名字矛盾拆成多个 → 消除同位重名）。"""
    if not isinstance(plan, dict) or not (plan.get("clusters") or plan.get("drop")):
        return records
    drop = {i for i in (plan.get("drop") or []) if isinstance(i, int)}
    names = {c["id"]: c["name"] for c in (plan.get("clusters") or [])
             if isinstance(c, dict) and isinstance(c.get("id"), int) and c.get("name")}
    out = []
    for cid, (ids, _cen) in enumerate(clusters):
        if cid in drop:
            continue
        name = names.get(cid) or _majority_name(records, ids)
        out.append(_merge_members([records[i] for i in ids], name))
    return out


def _cluster_size(records, ids):
    """簇代表尺寸 [宽,高](米)：取距离最近且尺寸可信的成员；全不可信 → None（尺寸未知，不作删依据）。"""
    best = None
    for i in ids:
        o = records[i]
        sz = o.get("size") or {}
        if o.get("size_unreliable") or sz.get("width_m") is None:
            continue
        d = o.get("distance_m")
        key = d if isinstance(d, (int, float)) else 1e9
        if best is None or key < best[0]:
            best = (key, sz)
    if best is None:
        return None
    return [round(best[1].get("width_m") or 0.0, 2), round(best[1].get("height_m") or 0.0, 2)]


def _consolidate_memory(records, director, cluster_m=CONSOLIDATE_CLUSTER_M):
    """探索写盘前一次记忆整理：代码按位置聚簇 → Claude 给每簇规范名 + 判尺寸离谱则删 → 代码合并。
    Claude 失败/空计划 → 原样返回（安全降级）。"""
    clusters = _position_clusters(records, cluster_m)
    payload = [{"id": cid,
                "x": cen[0] if cen else None, "y": cen[1] if cen else None,
                "names": [records[i].get("name") for i in ids],
                "size": _cluster_size(records, ids)}     # [宽,高]米 或 null；供 Claude 判尺寸合理性
               for cid, (ids, cen) in enumerate(clusters)]
    plan = director.consolidate_memory(payload)
    out = _apply_consolidation(records, clusters, plan)
    if out is records:
        print("[整理] Claude 未返回有效计划 → 跳过整理（原样落库）")
    else:
        print(f"[整理] 记忆整理官：{len(records)} 记录 / {len(clusters)} 位置簇 → {len(out)} 物体")
    return out


def _prepare_confirmed_for_storage(records, director, backend):
    """Apply only the consolidation that is safe for the active perception backend.

    YOLO confirmed detections already passed a class-specific precision threshold and
    geometry deduplication.  A text-only director cannot inspect the source pixels, so
    allowing it to rename or drop those detections removes real adjacent instances in
    dense furniture rows.  Qwen observations still use the legacy semantic consolidation.
    """
    if backend == "yoloe":
        print(
            f"[整理] YOLO confirmed {len(records)} 条已通过阈值和几何去重 "
            "→ 跳过无视觉语义删改"
        )
        return list(records)
    return _consolidate_memory(records, director)


def _upsert_confirmed_object(memory, area, obj, backend):
    """Store one confirmed record without re-merging separated YOLO instances."""
    if backend == "yoloe":
        # _dedup_objects already associated repeated YOLO observations.  Keep the
        # smaller storage tolerance so adjacent cabinets/chairs resolved in one frame
        # are not collapsed by FsMemory's general-purpose 0.8 m default.
        return memory.upsert_object(
            area,
            obj,
            instance_tol_m=CONSOLIDATE_CLUSTER_M,
        )
    return memory.upsert_object(area, obj)


def _cluster_doors(doors_raw, cluster_m=DOOR_CLUSTER_M, nominal_m=DOOR_NOMINAL_M):
    """门空间去重：把每条门(观察位姿+世界 bearing)沿名义距离投成世界点，贪心聚类合并重复门。

    返回 [{pose:{x,y}, dir, reason, count}]：同一扇门多视角记录被并成一条，带估计世界点。
    """
    clusters = []   # 每个 {sx, sy, n, dir, reason}
    for d in doors_raw:
        fx, fy = d["from"][0], d["from"][1]
        wb = math.radians(d.get("bearing", 0.0))
        px, py = fx + nominal_m * math.cos(wb), fy + nominal_m * math.sin(wb)
        hit = None
        for c in clusters:
            if math.hypot(px - c["sx"] / c["n"], py - c["sy"] / c["n"]) <= cluster_m:
                hit = c
                break
        if hit is None:
            clusters.append({"sx": px, "sy": py, "n": 1,
                             "dir": d.get("dir"), "reason": d.get("reason", "")})
        else:
            hit["sx"] += px
            hit["sy"] += py
            hit["n"] += 1
    out = []
    for c in clusters:
        out.append({"pose": {"x": round(c["sx"] / c["n"], 2), "y": round(c["sy"] / c["n"], 2)},
                    "dir": c["dir"], "reason": c["reason"], "count": c["n"]})
    return out


DOOR_MIN_COUNT = 2        # 门校验：少于这么多视角都报到的=噪声，丢弃（Qwen 顾问 door-happy）
DOOR_BOUNDARY_MARGIN = 1.3  # 门校验：真出口在房间周界附近；离边界 bbox 超过此距离的"门"判噪声


def _validate_doors(doors, bbox):
    """门校验（用户："门也需要校验，哪来11个门"）：①≥DOOR_MIN_COUNT 个视角都报到；
    ②门世界点在房间周界附近（离 bbox 边 ≤ margin；房间正中间的"门"不合理）。两条都过才留。"""
    if not bbox:
        return [d for d in doors if d.get("count", 0) >= DOOR_MIN_COUNT]
    out = []
    for d in doors:
        if d.get("count", 0) < DOOR_MIN_COUNT:
            continue
        x, y = d["pose"]["x"], d["pose"]["y"]
        edge = min(abs(x - bbox["xmin"]), abs(x - bbox["xmax"]),
                   abs(y - bbox["ymin"]), abs(y - bbox["ymax"]))
        if edge <= DOOR_BOUNDARY_MARGIN:
            out.append(d)
    return out


# ===== Claude 象限调度官：纯函数 helpers（可单测，不碰 ROS/网络）=====
def _absorb_sweep(sweep, *, wall_points, vantage_records, doors_raw, known_names):
    """三阶段(recon/director/兜底)【统一】折叠一次环视结果 → 保证 _dedup_objects 对所有来源一视同仁。
    只折叠环视产物(墙点/物体/门/名字)；visited/steps_log 由各调用点按需另记。返回 hint_bearings。"""
    wall_points.extend(sweep["wall_pts"])
    vantage_records.append({"objects": sweep["objects"]})
    if "objects_yolo" in sweep:                # 双标注：YOLO 旁路记录另存（不进 Qwen 主记忆）
        _YOLO_RECORDS.append({"objects": sweep["objects_yolo"]})
    if "objects_yolo_candidates" in sweep:
        _YOLO_CANDIDATE_RECORDS.append({"objects": sweep["objects_yolo_candidates"]})
    if "objects_candidates" in sweep:          # 纯 YOLO：候选也仅旁路，不进 confirmed 主记忆
        _YOLO_CANDIDATE_RECORDS.append({"objects": sweep["objects_candidates"]})
    if "objects_baseline" in sweep:            # 混合验证：Qwen-only 基线旁路另存（不进混合主记忆）
        _BASELINE_RECORDS.append({"objects": sweep["objects_baseline"]})
    doors_raw.extend(sweep["doors_raw"])
    known_names.extend(o.get("name") for o in sweep["objects"] if o.get("name"))
    return sweep["hint_bearings"]


def _recon_corners(bbox, inset=RECON_INSET_M):
    """bbox 四角向内缩 inset → 4 个 recon 目标点（自举房间范围）。bbox 退化/太小 → []。"""
    if not bbox:
        return []
    x0, x1, y0, y1 = bbox["xmin"], bbox["xmax"], bbox["ymin"], bbox["ymax"]
    if (x1 - x0) < 2 * inset or (y1 - y0) < 2 * inset:
        return []
    return [(x0 + inset, y0 + inset), (x1 - inset, y0 + inset),
            (x1 - inset, y1 - inset), (x0 + inset, y1 - inset)]


def _order_nearest(points, pose):
    """贪心就近串起点序（减少 recon 里程）。"""
    remaining = list(points)
    cx, cy = pose.get("x", 0.0), pose.get("y", 0.0)
    ordered = []
    while remaining:
        nxt = min(remaining, key=lambda p: math.hypot(p[0] - cx, p[1] - cy))
        ordered.append(nxt)
        remaining.remove(nxt)
        cx, cy = nxt
    return ordered


def _ang180(a):
    return ((a + 180.0) % 360.0) - 180.0


def _quad_of(x, y, xmid, ymid):
    """按 bbox 中点判象限：NE/NW/SE/SW。"""
    return ("N" if y >= ymid else "S") + ("E" if x >= xmid else "W")


def _quad_centroid(q, bbox, xmid, ymid):
    """象限几何中心点（候选格按到此点距离排序，优先补象限深处）。"""
    x = (xmid + bbox["xmax"]) / 2.0 if q[1] == "E" else (bbox["xmin"] + xmid) / 2.0
    y = (ymid + bbox["ymax"]) / 2.0 if q[0] == "N" else (bbox["ymin"] + ymid) / 2.0
    return (x, y)


def _bbox_cells(bbox, pitch=PITCH):
    """bbox 内所有格（与 _seed_frontier_from_bbox 同一 clamp/上限）；不过滤 visited/blocked。"""
    if not bbox:
        return set()
    xmin = max(bbox["xmin"], -BBOX_CLAMP_M)
    xmax = min(bbox["xmax"], BBOX_CLAMP_M)
    ymin = max(bbox["ymin"], -BBOX_CLAMP_M)
    ymax = min(bbox["ymax"], BBOX_CLAMP_M)
    if (max(0.0, xmax - xmin) * max(0.0, ymax - ymin)) / (pitch * pitch) > BBOX_MAX_CELLS:
        return set()
    cells = set()
    for cx in range(int(math.floor(xmin / pitch)), int(math.ceil(xmax / pitch)) + 1):
        for cy in range(int(math.floor(ymin / pitch)), int(math.ceil(ymax / pitch)) + 1):
            cells.add((cx, cy))
    return cells


def _objects_xy(vantage_records, bbox):
    """去重后【有可信 abs_pose】的物体 (name,x,y) 列表 —— 喂 Claude payload / 象限物体计数。"""
    out = []
    for o in _dedup_objects(vantage_records, bbox):
        ap = o.get("abs_pose")
        if isinstance(ap, dict) and ap.get("x") is not None:
            out.append((o.get("name"), round(float(ap["x"]), 1), round(float(ap.get("y") or 0.0), 1)))
    return out


def _quadrant_stats(bbox, visited_cells, blocked_cells, obj_xy):
    """每象限覆盖统计：observed=visited∪blocked，coverage=observed/total。
    under_covered = coverage<QUAD_COVER_TARGET 且【仍有非 blocked 未访问的可达空洞】
    （全 blocked/全访问的象限不算欠覆盖，防调度官死盯不可达区）。"""
    quads = ("NE", "NW", "SE", "SW")
    base = {q: {"visited": 0, "blocked": 0, "total": 0, "n_objects": 0, "_open": False} for q in quads}
    if not bbox:
        return {q: {"visited": 0, "blocked": 0, "total": 0, "n_objects": 0,
                    "observed": 0, "coverage": 0.0, "under_covered": False} for q in quads}
    xmid = (bbox["xmin"] + bbox["xmax"]) / 2.0
    ymid = (bbox["ymin"] + bbox["ymax"]) / 2.0
    for c in _bbox_cells(bbox):
        cx, cy = _cell_center(c)
        q = _quad_of(cx, cy, xmid, ymid)
        base[q]["total"] += 1
        if c in visited_cells:
            base[q]["visited"] += 1
        elif c in blocked_cells:
            base[q]["blocked"] += 1
        else:
            base[q]["_open"] = True
    for (_n, ox, oy) in obj_xy:
        base[_quad_of(ox, oy, xmid, ymid)]["n_objects"] += 1
    out = {}
    for q in quads:
        v = base[q]
        observed = v["visited"] + v["blocked"]
        cov = observed / v["total"] if v["total"] else 1.0
        out[q] = {"visited": v["visited"], "blocked": v["blocked"], "total": v["total"],
                  "n_objects": v["n_objects"], "observed": observed, "coverage": round(cov, 2),
                  "under_covered": bool(cov < QUAD_COVER_TARGET and v["_open"])}
    return out


def _candidate_cells(bbox, visited_cells, blocked_cells, per_quad=CAND_PER_QUAD):
    """给 Claude 的【代码筛好的候选格】(反幻觉边界：它只能选 id、不产坐标)：
    ①每象限最靠质心的未覆盖格 top-N；②reopen 候选=被自由空间包围(≥REOPEN_FREE_NEIGHBORS 个 visited 邻居)的 blocked 格。
    返回 [{id, quad, x, y, cell, reopen_blocked}]。"""
    if not bbox:
        return []
    xmid = (bbox["xmin"] + bbox["xmax"]) / 2.0
    ymid = (bbox["ymin"] + bbox["ymax"]) / 2.0
    by_quad = {q: [] for q in ("NE", "NW", "SE", "SW")}
    for c in _bbox_cells(bbox):
        cx, cy = _cell_center(c)
        by_quad[_quad_of(cx, cy, xmid, ymid)].append(c)
    cands = []
    for q, qcells in by_quad.items():
        cen = _quad_centroid(q, bbox, xmid, ymid)
        openc = [c for c in qcells if c not in visited_cells and c not in blocked_cells]
        openc.sort(key=lambda c: math.hypot(_cell_center(c)[0] - cen[0], _cell_center(c)[1] - cen[1]))
        for c in openc[:per_quad]:
            cx, cy = _cell_center(c)
            cands.append({"quad": q, "x": round(cx, 1), "y": round(cy, 1),
                          "cell": c, "reopen_blocked": False})
    for c in sorted(blocked_cells):
        if sum(1 for n in _neighbors(c) if n in visited_cells) >= REOPEN_FREE_NEIGHBORS:
            cx, cy = _cell_center(c)
            cands.append({"quad": _quad_of(cx, cy, xmid, ymid), "x": round(cx, 1),
                          "y": round(cy, 1), "cell": c, "reopen_blocked": True})
    for i, cc in enumerate(cands):
        cc["id"] = f"c{i}"
    return cands


def _build_plan_payload(bbox, quad_stats, obj_xy, candidates, last_rejected, round_i, rounds_left,
                        occ_summary=None):
    """把符号地图压成 Claude payload（候选剥掉内部 cell 字段，只留 id/x/y[/quad]）。

    occ_summary 非空则附 occupancy 覆盖计数（free/occupied/visited/frontier），供 Claude 判覆盖进度。
    候选来自 occupancy frontier（可能无 quad 键），故 quad/reopen 用 .get 容错。
    """
    b = {k: round(v, 1) for k, v in bbox.items()} if bbox else {}
    objs = [{"name": n, "x": x, "y": y} for (n, x, y) in obj_xy]
    cands = []
    for c in candidates:
        cc = {"id": c["id"], "x": c["x"], "y": c["y"]}
        if c.get("quad"):
            cc["quad"] = c["quad"]
        if c.get("reopen_blocked"):
            cc["reopen_blocked"] = True
        cands.append(cc)
    out = {"bbox": b, "quadrants": quad_stats, "objects": objs, "candidates": cands,
           "last_rejected_reopen": last_rejected, "round": round_i, "rounds_left": rounds_left}
    if occ_summary is not None:
        out["coverage"] = occ_summary
    return out


def _validate_plan_choice(plan, candidates):
    """把 Claude 的 target_id 映回候选；缺失/幻觉/格式错 → None（安全跳过本轮，不致命）。"""
    if not isinstance(plan, dict):
        return None
    tid = plan.get("target_id")
    if not tid:
        return None
    for c in candidates:
        if c["id"] == tid:
            return c
    return None


def _stop_rule(target_cell, obj_xy, standoff=OBJECT_STANDOFF_M):
    """停车铁律：目标格落着已记录物体 → 抬 front_block_m/tol 只贴近到 ≥standoff（太近看不清）。
    否则默认。返回 (front_block_m, tol_m)。"""
    cx, cy = _cell_center(target_cell)
    near = any(math.hypot(ox - cx, oy - cy) <= PITCH / 2.0 for (_n, ox, oy) in obj_xy)
    if near:
        return (standoff, standoff + 0.2)
    return (0.4, 0.5)


def _is_forward_open(sectors, dist_to_cell, cone_deg=REOPEN_FRONT_CONE_DEG, margin=REOPEN_CLEAR_MARGIN):
    """假墙 scan 复核谓词（纯函数）：前向锥(|角|≤cone)内最近障碍 ≥ 到目标格距离+margin → 判『开』。
    sectors: {角度(度,机体系,0=前): 最近障碍米}。无有效前向读数 → False（保守当墙，绝不误开进实墙）。"""
    fronts = [d for a, d in (sectors or {}).items()
              if isinstance(d, (int, float)) and d > 0 and abs(_ang180(a)) <= cone_deg]
    if not fronts:
        return False
    return min(fronts) >= dist_to_cell + margin


def _min_vantage_spacing_ok(cur, last_vantage_xy):
    """新 vantage 距上一个 ≥ MIN_VANTAGE_SPACING_M 才值得重扫（防原地重复浪费）。"""
    if last_vantage_xy is None:
        return True
    return math.hypot(cur["x"] - last_vantage_xy[0], cur["y"] - last_vantage_xy[1]) >= MIN_VANTAGE_SPACING_M


# ===== Claude 象限调度官：需 ROS 的 helper（scan 复核）=====
def _reverify_open(ex, cell):
    """假墙重开前的 scan 复核（护栏）：面向目标格 → 读 scan 扇区 → 前向锥是否真有缺口。
    真墙 → False（调用方回填 blocked 并反馈 Claude）；假墙 → True（可 discard 后驱动）。"""
    cx, cy = _cell_center(cell)
    p = _pose(ex)
    dist_to_cell = math.hypot(cx - p["x"], cy - p["y"])
    try:
        nav.geo_face_point(ex, cx, cy)
        sectors = nav._scan_sectors(ex, sectors=12)
    except Exception:  # noqa: BLE001
        return False
    return _is_forward_open(sectors, dist_to_cell)


COVER_PITCH_M = 1.8     # 均匀覆盖网格格距（bbox 内每格保证一个观测点）。取 1.8：使南墙边缘格离起点 vantage
#                         > COVER_RADIUS → 不被起点顺带"覆盖"，逼车真的走到南带观测（治南墙漏）。
COVER_RADIUS_M = 1.0    # 网格格中心此半径内有过 vantage = 该格已覆盖（略 > 半格距，避免相邻格缝隙）


def _backtrack_open_push(ex, occ, visited, visited_cells, *, bbox=None, max_nodes=10, dive_steps=6):
    """DFS 回溯破停滞（用户思路）：覆盖判"无可达 frontier"时不收尾，先回溯近期访问节点(DFS 栈)，
    找一个仍有【朝未访问区的开阔方向】的旧节点，VFH 朝那深入(可进未知、边走边扫)重连 frontier——
    沿连通开阔空间(如隔断后西南角)走进去。某开阔岔路已走过→换分支；节点无未探分支→回退更早节点。
    有 bbox 时**偏置朝【访问最少象限】**的开阔方向(奔欠覆盖区去，而非随便探)。

    返回 True=扫出新自由格(值得 continue 继续覆盖)；False=所有回溯节点都无未探开阔分支(真收尾)。
    上界：回溯 ≤max_nodes 节点、每节点深入 ≤dive_steps 步，调用方再以 pushes 兜底防 wander。
    """
    before = sum(1 for s in occ.cells.values() if s in oc._KNOWN_FREE)
    qdir = None                                      # 欠覆盖象限中心方向(偏置用)
    if bbox:
        counts = _quad_visit_counts(visited, bbox)
        lq = min(counts, key=counts.get)             # 访问最少象限 = 覆盖最欠
        xmid = (bbox["xmin"] + bbox["xmax"]) / 2.0
        ymid = (bbox["ymin"] + bbox["ymax"]) / 2.0
        qx = (bbox["xmin"] + xmid) / 2 if "W" in lq else (xmid + bbox["xmax"]) / 2
        qy = (bbox["ymin"] + ymid) / 2 if "S" in lq else (ymid + bbox["ymax"]) / 2
    seen = []
    for past in reversed(visited[-max_nodes:]):
        node = (round(past[0], 1), round(past[1], 1))
        if node in seen:
            continue
        seen.append(node)
        _route_to(ex, occ, (past[0], past[1]), tol_m=0.5, max_legs=DIRECTOR_MAX_LEGS)   # 导回 DFS 节点
        p = _pose(ex)
        if bbox:
            qdir = math.degrees(math.atan2(qy - p["y"], qx - p["x"]))
        try:
            sec = json.loads(ex.ros.call("scan_summary", {"sectors": 12}).text).get("sectors", {})
        except Exception:  # noqa: BLE001
            continue
        best = None                                  # 挑"够开阔 且 前方格未访问"的方向(未走过的岔路)
        for label, dist in sec.items():
            if not isinstance(dist, (int, float)) or dist < 1.0:
                continue
            ang = pf._label_angle(label)
            if ang is None:
                continue
            world = p.get("yaw_deg", 0.0) + ang
            tx = p["x"] + 1.5 * math.cos(math.radians(world))
            ty = p["y"] + 1.5 * math.sin(math.radians(world))
            if _cell(tx, ty) in visited_cells:       # 该开阔方向已走过 → DFS 换分支
                continue
            # 评分=净空 + 朝欠覆盖象限的方向偏置(±50° 内加权，奔西南角这类欠覆盖区去)
            score = dist
            if qdir is not None and abs(((world - qdir + 180) % 360) - 180) < 50:
                score += 2.5
            if best is None or score > best[0]:
                best = (score, world, dist)
        if best is None:
            continue                                 # 该节点无未探开阔分支 → 回退更早节点
        print(f"[DFS回溯] 退回节点({p['x']:.1f},{p['y']:.1f}) 朝开阔未访问方向 {best[1]:.0f}°(净空{best[2]:.1f}m) 深入")
        for _ in range(dive_steps):
            r = nav.geo_step_open(ex, best[1], step_m=1.0, clearance_m=0.4)
            _update_occ(ex, occ, _pose(ex))
            if r.get("status") in ("safety_stop", "no_room", "error") or (r.get("moved_m") or 0) < 0.1:
                break
        after = sum(1 for s in occ.cells.values() if s in oc._KNOWN_FREE)
        if after > before + 2:
            print(f"[DFS回溯] 扫出新自由格 {after - before} → 重连 frontier 继续覆盖")
            return True
    return False


def _uncovered_grid_targets(bbox, occ, vantage_xys, blocked_grid):
    """已知 bbox 内按 COVER_PITCH 均匀铺格中心 → 过滤：占用格 / 已被 vantage 覆盖(COVER_RADIUS 内) /
    已标 blocked。返回未覆盖格中心 [(x,y)]（均匀覆盖硬保证的目标池；bbox 随环视扩张,新边格自动纳入）。"""
    if not bbox:
        return []
    out = []
    nx = int((bbox["xmax"] - bbox["xmin"]) / COVER_PITCH_M) + 1
    ny = int((bbox["ymax"] - bbox["ymin"]) / COVER_PITCH_M) + 1
    for ix in range(max(1, nx)):
        x = bbox["xmin"] + COVER_PITCH_M * (ix + 0.5)
        if x > bbox["xmax"]:
            continue
        for iy in range(max(1, ny)):
            y = bbox["ymin"] + COVER_PITCH_M * (iy + 0.5)
            if y > bbox["ymax"]:
                continue
            if (round(x, 1), round(y, 1)) in blocked_grid:
                continue
            if occ.state(occ.cell_of(x, y)) == oc.OCCUPIED:
                continue
            if any(math.hypot(x - vx, y - vy) <= COVER_RADIUS_M for vx, vy in vantage_xys):
                continue
            out.append((round(x, 2), round(y, 2)))
    return out


def _grid_coverage(ex, *, director, occ, visited, wall_points, vantage_records, doors_raw,
                   steps_log, visited_cells, blocked_cells, known_names,
                   nav_steps, vantages, last_vantage_xy, last_pose, vantage_xys):
    """均匀网格覆盖（代码持有覆盖保证，取代 frontier 密度/Claude 选点）：反复取【最近的未覆盖格】→ 路由过去
    (A*/反应式) → 柔性环视 → 记 vantage_xys 标该格已覆盖。保证 bbox 内每个可达格都物理走到一个观测点(含南带,
    治极角落漏/方向漂移)；bbox 随环视扩张 → 新边格自动纳入(兼顾发现)。无可达未覆盖格/预算尽 → 收。"""
    stop_reason = "nav_cap"
    blocked_grid = set()
    stuck = 0
    pushes = 0                       # DFS 回溯破停滞次数上界(防 wander)
    last_xy = (last_pose["x"], last_pose["y"])
    while nav_steps < MAX_NAV_STEPS and vantages < MAX_VANTAGES:
        pose = _pose(ex)
        _update_occ(ex, occ, pose)
        bbox = dp.boundary_from_points(wall_points + visited)
        targets = _uncovered_grid_targets(bbox, occ, vantage_xys, blocked_grid)
        if targets:
            tx, ty = _balanced_grid_target(targets, pose, vantage_xys)
        else:
            # 已知 bbox 内网格已覆盖 → 推最近 frontier 扩张 bbox（发现更多房间 → 下轮新边格再纳入均匀覆盖）。
            # 只有"无未覆盖格 且 无可达 frontier"才算真完成——否则会像 v6 那样一开局就误判全覆盖退出。
            fcands = _occ_candidates(occ, visited_cells, blocked_cells, pose, cap=CAND_TOTAL)
            fcands = _exclude_observed_candidates(fcands, vantage_xys)
            if not fcands:
                # 可达区已探完 → DFS 回溯：找旧节点上"未走过的开阔岔路"朝欠覆盖象限下探(进未知、破死锁)
                if pushes < 8 and _backtrack_open_push(ex, occ, visited, visited_cells, bbox=bbox):
                    pushes += 1
                    continue
                stop_reason = "covered"
                print("[网格覆盖] 网格全覆盖 且 无可达 frontier → 覆盖完成")
                break
            ftgt = _balanced_pick(
                fcands,
                _origin_balance_bbox(),
                vantage_xys,
            )
            tx, ty = ftgt["x"], ftgt["y"]
            print(f"[网格覆盖] 已知区网格已满 → 推 frontier({tx:.1f},{ty:.1f}) 扩张 bbox")
        r = _route_to(ex, occ, (tx, ty), tol_m=0.6, max_legs=DIRECTOR_MAX_LEGS)
        nav_steps += r.get("n_steps") or len(r.get("steps") or [])
        cur = r.get("pose") or _pose(ex)
        last_pose = cur
        visited.append([round(cur["x"], 2), round(cur["y"], 2)])
        visited_cells.add(_cell(cur["x"], cur["y"]))
        near = math.hypot(cur["x"] - tx, cur["y"] - ty) <= COVER_RADIUS_M
        if near and _min_vantage_spacing_ok(cur, last_vantage_xy) and vantages < MAX_VANTAGES:
            sw, cur = _goto_and_sweep(ex, director, known_names, target_xy=(tx, ty), occ=occ)
            last_pose = cur
            _absorb_sweep(sw, wall_points=wall_points, vantage_records=vantage_records,
                          doors_raw=doors_raw, known_names=known_names)
            vantages += 1
            last_vantage_xy = (cur["x"], cur["y"])
            # 标【目标格】已覆盖(不只标漂移后的站位)——否则站位退离目标格 >COVER_RADIUS 时该格永不被标覆盖
            #   → 反复重选同格(v6b bug)。同时标观测站位,顺带覆盖周边格。
            vantage_xys.append((tx, ty))
            vantage_xys.append((cur["x"], cur["y"]))
            print(f"[网格覆盖{vantages}] 格({tx:.1f},{ty:.1f})→观测({cur['x']:.2f},{cur['y']:.2f}) "
                  f"Qwen记{len(sw['objects'])}物体 nav={nav_steps}")
            steps_log.append({"step": vantages - 1, "pose": cur,
                              "qwen_objects": [o.get("name") for o in sw["objects"]]})
        elif near:
            vantage_xys.append((tx, ty))               # 已被邻近 vantage 覆盖(间距太近) → 标该格覆盖不重扫
            vantage_xys.append((cur["x"], cur["y"]))
        else:
            blocked_grid.add((round(tx, 1), round(ty, 1)))   # 到不了 → 标 blocked,不再枉试
            d0 = math.hypot(cur["x"] - tx, cur["y"] - ty)
            print(f"[网格覆盖·诊断] 格({tx:.1f},{ty:.1f}) 到不了 status={r.get('status')} "
                  f"astar={'有路' if r.get('astar_path') else '无路'} 剩余{d0:.2f}m "
                  f"legs={len(r.get('steps') or [])} → 标 blocked nav={nav_steps}")
        moved = math.hypot(cur["x"] - last_xy[0], cur["y"] - last_xy[1])
        last_xy = (cur["x"], cur["y"])
        stuck = stuck + 1 if moved < 0.15 else 0
        if stuck >= STUCK_LIMIT:
            blocked_grid.add((round(tx, 1), round(ty, 1)))   # 卡死 → 放弃当前格换下一个
            stuck = 0
    return nav_steps, vantages, stop_reason, last_pose


def _frontier_backstop(ex, *, director, occ, visited, wall_points, vantage_records, doors_raw,
                       steps_log, visited_cells, blocked_cells, known_names,
                       nav_steps, vantages, last_vantage_xy, last_pose):
    """occupancy frontier 覆盖【兜底】(recon+director 之后无条件跑，守住覆盖下限)。
    每轮：当前观测点环视(更新 occ)→取最近【可达】frontier 候选→A* 路由过去→标 visited/blocked→卡死逃逸。
    frontier 耗尽=覆盖完成(unknown 边界扫光)。返回 (nav_steps, vantages, stop_reason, last_pose)。"""
    stop_reason = "nav_cap"
    stuck = 0
    pushes = 0                       # DFS 回溯破停滞次数上界(防 wander)
    cur = last_pose
    last_iter_xy = (last_pose["x"], last_pose["y"])
    while True:
        if vantages >= MAX_VANTAGES:
            stop_reason = "vantage_cap"
            break
        if nav_steps >= MAX_NAV_STEPS:
            stop_reason = "nav_cap"
            break

        pose = _pose(ex)
        last_pose = pose
        visited.append([round(pose["x"], 2), round(pose["y"], 2)])
        visited_cells.add(_cell(pose["x"], pose["y"]))

        if not _min_vantage_spacing_ok(pose, last_vantage_xy):
            print(f"  [跳过环视] 距上一 vantage <{MIN_VANTAGE_SPACING_M}m，不重扫")
            _update_occ(ex, occ, pose)         # 不环视也刷一帧栅格(frontier 需要最新自由空间)
        else:
            sweep, pose = _goto_and_sweep(ex, director, known_names, occ=occ)
            last_pose = pose
            _absorb_sweep(sweep, wall_points=wall_points, vantage_records=vantage_records,
                          doors_raw=doors_raw, known_names=known_names)
            vantages += 1
            last_vantage_xy = (pose["x"], pose["y"])
            cobjs = [o.get("name") for o in sweep["objects"]]
            print(f"[视角{vantages}] pose=({pose['x']:.2f},{pose['y']:.2f}) 环视 Qwen记{len(cobjs)}物体 "
                  f"门提示={len(sweep['doors_raw'])}")
            steps_log.append({"step": vantages - 1, "pose": pose, "qwen_objects": cobjs,
                              "n_doors": len(sweep["doors_raw"])})

        cands = _occ_candidates(occ, visited_cells, blocked_cells, pose, cap=CAND_TOTAL)
        if not cands:
            # 可达区已探完 → DFS 回溯：找旧节点上"未走过的开阔岔路"朝欠覆盖象限下探(进未知、破死锁)
            if pushes < 8 and _backtrack_open_push(
                    ex, occ, visited, visited_cells,
                    bbox=dp.boundary_from_points(wall_points + visited)):
                pushes += 1
                continue
            stop_reason = "covered"
            print("[覆盖] occupancy 无可达 frontier → unknown 边界扫光，覆盖完成")
            break
        # 覆盖均衡：取【访问最少象限】的最近 frontier，而非全局最近（破近邻贪心的方向漂移）
        bs_bbox = dp.boundary_from_points(wall_points + visited)
        tgt = _balanced_pick(cands, bs_bbox, visited)
        tx, ty = tgt["x"], tgt["y"]
        r = _route_to(ex, occ, (tx, ty), tol_m=0.5, max_legs=DIRECTOR_MAX_LEGS)
        nav_steps += r.get("n_steps") or len(r.get("steps") or [])
        cur = r.get("pose") or _pose(ex)
        if r.get("arrived"):
            visited_cells.add(tgt["cell"])
            visited_cells.add(_cell(cur["x"], cur["y"]))
            print(f"  → 格{tgt['cell']}({tx:.1f},{ty:.1f}) 到位 ({cur['x']:.2f},{cur['y']:.2f}) nav={nav_steps}")
        else:
            blocked_cells.add(tgt["cell"])
            d0 = math.hypot(cur["x"] - tx, cur["y"] - ty)
            print(f"  → 格{tgt['cell']}({tx:.1f},{ty:.1f}) 不可达·诊断 status={r.get('status')} "
                  f"astar={'有路' if r.get('astar_path') else '无路'} 剩余{d0:.2f}m "
                  f"legs={len(r.get('steps') or [])} → 标 blocked nav={nav_steps}")

        moved = math.hypot(cur["x"] - last_iter_xy[0], cur["y"] - last_iter_xy[1])
        last_iter_xy = (cur["x"], cur["y"])
        stuck = stuck + 1 if moved < 0.15 else 0
        if stuck >= STUCK_LIMIT:
            far = _occ_candidates(occ, visited_cells, blocked_cells, cur, cap=CAND_TOTAL)
            if far:
                fc = max(far, key=lambda c: math.hypot(c["x"] - cur["x"], c["y"] - cur["y"]))
                r2 = _route_to(ex, occ, (fc["x"], fc["y"]), tol_m=0.6, max_legs=6)
                nav_steps += r2.get("n_steps") or len(r2.get("steps") or [])
                cur = r2.get("pose") or _pose(ex)
                esc = math.hypot(cur["x"] - last_iter_xy[0], cur["y"] - last_iter_xy[1])
                last_iter_xy = (cur["x"], cur["y"])
                print(f"  [脱困] 逃向最远 frontier({fc['x']:.1f},{fc['y']:.1f}) 位移{esc:.2f}m nav={nav_steps}")
                if esc < 0.15:
                    stop_reason = "stuck"
                    break
            else:
                stop_reason = "stuck"
                break
            stuck = 0
    return nav_steps, vantages, stop_reason, last_pose


def run_pipeline(*, ex=None, mem=None, area=None, report_dir=None):
    """探索流水线（原 main 体）。可被 ExploreMode/supervisor 注入 ex/mem/area 复用。

    area 经 `global AREA` 生效——脚本单进程单跑，内部大量引用模块级 AREA，此为最小侵入 seam。
    report_dir 仅为接口一致性保留（工件路径 RUN_OUT 仍走模块常量）。返回 0 成功。
    """
    global AREA
    if area:
        AREA = area
    _hr("explore_probe v2：从零覆盖 + Qwen 本地语义标注（代码做几何/去重/整理）")
    ex = ex or Executor()
    mem = mem or FsMemory(config.MEMORY_ROOT, config.ENV_NAME)
    # 作者 = 本地 Qwen（inspect 出物体+ROI），代码用 depth_roi 回填几何、去重整理；不调云端。
    # （Phase1 实测 Qwen 召回 > Claude，且免费、原生中文 → explore 用 Qwen 作者。）
    print(f"[标注] 本地 Qwen ({ex.model})  |  云端作者=不用（explore 从零，全本地）")

    # 初探=从零：清掉上轮该 area 的旧记忆，避免跨轮 merge 污染
    _ap = os.path.join(config.MEMORY_ROOT, config.ENV_NAME, AREA, "area.json")
    if os.path.exists(_ap):
        os.remove(_ap)

    # 起点守卫：软复位到休息室起点（VFH 能从墙后绕回来）+ 朝东
    sp = _pose(ex)
    d_start = math.hypot(sp.get("x", 99) - START_XY[0], sp.get("y", 99) - START_XY[1])
    print(f"[起始位姿] {sp}  距起点 {d_start:.2f}m")
    if d_start > START_TOL_M:
        # 经北侧开阔区(1.5,1.5)中转再回起点——直接朝起点常被 wall(1) 挡住绕不过来。
        # 用 geo_goto_around（反应式绕行，会先退一点解钉）而非直冲，能从贴墙/近隔断的搁浅位自救。
        print(f"[软复位] 绕行经 (1.5,1.5) 中转开回 {START_XY}…")
        for wp in ((1.5, 1.5), START_XY):
            nav.geo_goto_around(ex, wp[0], wp[1], tol_m=0.5, max_legs=20)
        cp = _pose(ex)
        if math.hypot(cp["x"] - START_XY[0], cp["y"] - START_XY[1]) > START_TOL_M:
            print(f"⚠️ 软复位未回到起点(仍在 {cp.get('x'):.1f},{cp.get('y'):.1f})；"
                  "如需干净起点请在 Webots 里重置仿真(Ctrl+Shift+R)后重跑。")
    nav.geo_face_point(ex, START_XY[0] + 1.0, START_XY[1])   # 朝东起手

    visited, wall_points, vantage_records, steps_log, doors_raw = [], [], [], [], []
    vantage_xys = []                # 已观测(vantage)位姿 [(x,y)]——均匀网格覆盖判"该格是否已覆盖"用
    rec_type = "lounge"
    stop_reason = "nav_cap"

    # 共享状态袋：recon / director / 兜底 三阶段同一份 → 统一喂不改动的去重+写盘路径
    start = _pose(ex)
    last_pose = start
    visited_cells, blocked_cells = {_cell(start["x"], start["y"])}, set()
    nav_steps, vantages = 0, 0
    last_vantage_xy = None          # 上一个实际环视点（间距门槛用）
    known_names = []                # 已记录物体名（作命名锚点喂 Qwen，减少跨帧命名发散）
    director = make_director()  # Claude 可用则用；DISABLE_CLAUDE=1/无额度时离线规则兜底
    occ = oc.OccGrid(res_m=OCC_RES_M)   # occupancy 覆盖栅格（Q1：稠密射线 → frontier 候选 → A* 路由）
    _YOLO_RECORDS[:] = []           # 双标注 YOLO 旁路记录（每轮重置）
    _YOLO_CANDIDATE_RECORDS[:] = [] # candidate-only/低阈值旁路（永不进 confirmed 导航记忆）
    _BASELINE_RECORDS[:] = []       # 混合验证 Qwen-only 基线旁路记录（每轮重置）

    def _recon_goto_sweep(txp, typ, tag):
        """recon：VFH 开到点 → 记 visited → 满足间距则环视吸收。更新共享计数。"""
        nonlocal nav_steps, vantages, last_vantage_xy, last_pose
        r = nav.geo_goto_around(ex, txp, typ, tol_m=0.5, max_legs=RECON_MAX_ITERS,
                                max_step_m=1.0, front_block_m=0.4)
        nav_steps += len(r.get("steps") or [])
        cur = r.get("pose") or _pose(ex)
        last_pose = cur
        visited.append([round(cur["x"], 2), round(cur["y"], 2)])
        visited_cells.add(_cell(cur["x"], cur["y"]))
        if _min_vantage_spacing_ok(cur, last_vantage_xy) and vantages < MAX_VANTAGES:
            sw, cur = _goto_and_sweep(ex, director, known_names, occ=occ)   # APF 选观测点后 cur 已更新
            last_pose = cur
            _absorb_sweep(sw, wall_points=wall_points, vantage_records=vantage_records,
                          doors_raw=doors_raw, known_names=known_names)
            vantages += 1
            last_vantage_xy = (cur["x"], cur["y"])
            vantage_xys.append((cur["x"], cur["y"]))
            print(f"[RECON-{tag}{vantages}] pose=({cur['x']:.2f},{cur['y']:.2f}) Qwen记{len(sw['objects'])}物体")

    try:
        # ===== STAGE A · RECON：自举 bbox（起点环视 → 四内缩角 → 中心 360°）=====
        _hr("STAGE A · RECON（起点环视 → 四角 → 中心 360°，自举房间范围）")
        s0, p0 = _goto_and_sweep(ex, director, known_names, occ=occ)   # 起点也先 APF 选观测点
        last_pose = p0
        _absorb_sweep(s0, wall_points=wall_points, vantage_records=vantage_records,
                      doors_raw=doors_raw, known_names=known_names)
        visited.append([round(p0["x"], 2), round(p0["y"], 2)])
        vantages += 1
        last_vantage_xy = (p0["x"], p0["y"])
        vantage_xys.append((p0["x"], p0["y"]))
        steps_log.append({"step": 0, "phase": "recon_start", "pose": p0,
                          "qwen_objects": [o.get("name") for o in s0["objects"]]})

        prov_bbox = dp.boundary_from_points(wall_points + visited)
        corners = _order_nearest(_recon_corners(prov_bbox), start)
        print(f"[RECON] 临时 bbox={prov_bbox}  内缩角点={[(round(x, 1), round(y, 1)) for x, y in corners]}")
        for (cxp, cyp) in corners:
            if nav_steps >= DIRECTOR_NAV_BUDGET or vantages >= MAX_VANTAGES:
                break
            _recon_goto_sweep(cxp, cyp, "角")
        # 中心 360°（用四角精炼后的 bbox）
        rb = dp.boundary_from_points(wall_points + visited)
        if rb and nav_steps < DIRECTOR_NAV_BUDGET and vantages < MAX_VANTAGES:
            _recon_goto_sweep((rb["xmin"] + rb["xmax"]) / 2.0, (rb["ymin"] + rb["ymax"]) / 2.0, "中心")

        # ===== STAGE B · 均匀网格覆盖（代码持有覆盖保证，取代 frontier 密度/Claude 频域选点）=====
        _hr("STAGE B · 均匀网格覆盖（bbox 内每格保证一个观测点；含南带，治极角落漏/方向漂移）")
        nav_steps, vantages, stop_reason, last_pose = _grid_coverage(
            ex, director=director, occ=occ, visited=visited, wall_points=wall_points,
            vantage_records=vantage_records, doors_raw=doors_raw, steps_log=steps_log,
            visited_cells=visited_cells, blocked_cells=blocked_cells, known_names=known_names,
            nav_steps=nav_steps, vantages=vantages, last_vantage_xy=last_vantage_xy,
            last_pose=last_pose, vantage_xys=vantage_xys)

        # ===== STAGE C · FRONTIER 兜底（网格覆盖后的薄兜底：扫光残留 frontier，守住覆盖下限）=====
        _hr("STAGE C · frontier 兜底（无条件补全覆盖，用 nav 剩余额度）")
        nav_steps, vantages, stop_reason, last_pose = _frontier_backstop(
            ex, director=director, occ=occ, visited=visited, wall_points=wall_points,
            vantage_records=vantage_records,
            doors_raw=doors_raw, steps_log=steps_log, visited_cells=visited_cells,
            blocked_cells=blocked_cells, known_names=known_names,
            nav_steps=nav_steps, vantages=vantages, last_vantage_xy=last_vantage_xy, last_pose=last_pose)
        ex.ros.call("stop", {})
    finally:
        ex.close()
        _yolo_service_stop()      # 关停常驻 YOLO 服务（若启用）

    # —— 几何校验去重（用户："据位姿+深度算位置，重复就过滤"）：把所有视角的原始观测先收敛 ——
    bbox = dp.boundary_from_points(wall_points + visited)
    doors = _validate_doors(_cluster_doors(doors_raw), bbox)
    raw_count = sum(len(r.get("objects", []) or []) for r in vantage_records)
    clean_objs = _dedup_objects(vantage_records, bbox)
    # 自由空间反证（治近距离幻觉）：反投坐标落在车已走过位姿的车体半径内=车穿过该处→物理不可能→丢。
    # 零召回代价（真物体不会落在车走过的点上），只删压在自身轨迹上的近距离幻觉。
    clean_objs, phantoms = _free_space_phantoms(clean_objs, visited)
    # occ 自由空间反证（正交扩展）：abs_pose 落在已扫空旷自由格(附近无墙) = 空旷地板幻觉 → 丢。
    clean_objs, occ_phantoms = _occupancy_phantoms(clean_objs, occ)
    print(f"[去重] 原始观测 {raw_count} → 几何校验后 {len(clean_objs)} 物体"
          f"（轨迹反证丢 {len(phantoms)} + occ空旷反证丢 {len(occ_phantoms)}）")
    for o in phantoms:
        ap = o.get("abs_pose") or {}
        print(f"    [丢弃幻觉] {o.get('name')} @({ap.get('x')},{ap.get('y')}) "
              f"落在车已走过处(物理不可能)")
    for o in occ_phantoms:
        ap = o.get("abs_pose") or {}
        print(f"    [丢弃幻觉] {o.get('name')} @({ap.get('x')},{ap.get('y')}) "
              f"落在已扫空旷自由格(该处无家具)")
    # —— Role-2：Qwen 记录可交给 Claude 整理；YOLO confirmed 保留校准后的视觉标签 ——
    clean_objs = _prepare_confirmed_for_storage(
        clean_objs,
        director,
        PERCEPTION_BACKEND,
    )
    # —— 落盘（单一并集写路径）：逐物体 upsert_object（数组并集、同名远位=多实例）——
    n_obj = 0
    for o in clean_objs:
        try:
            _upsert_confirmed_object(mem, AREA, o, PERCEPTION_BACKEND)
            n_obj += 1
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ upsert_object 失败({o.get('name')}): {e}")
    # area 级标量字段：load 现有(已含上面写入的 objects)→补 type/summary/boundary/view_pose/doors→整写
    rec = mem.load_area(AREA) or {"area": AREA, "objects": []}
    rec["area"] = AREA
    rec["type"] = rec_type
    rec.setdefault("hazards", [])
    rec["boundary"] = bbox
    rec["doors"] = doors
    rec["view_pose"] = {"x": last_pose["x"], "y": last_pose["y"], "yaw": last_pose["yaw_deg"]}
    rec["summary"] = f"初期探索覆盖 {len(visited)} 视角；记忆 {len(rec.get('objects', []))} 物体、{len(doors)} 门。"
    try:
        path = mem.upsert_area(AREA, rec)
        print(f"[写回区域记录] {path}")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 区域写回失败: {e}")
    # —— 门 → 拓扑边（聚类去重后每门一条边；to 用占位未探区，多房间导航实现后再接真区）——
    for k, d in enumerate(doors):
        try:
            mem.add_edge(AREA, f"unexplored_{AREA}_{k}", via=d["pose"],
                         direction=d.get("dir"), confidence=round(min(1.0, d.get("count", 1) / 3.0), 2))
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ add_edge 失败: {e}")

    final_objs = [o.get("name") for o in rec.get("objects", [])]
    run = {
        "area": AREA, "n_vantages": len(visited), "stop_reason": stop_reason,
        "visited": visited, "coarse_bbox": bbox, "wall_points": wall_points[:400],
        "merged_objects": final_objs,
        "doors": doors, "nav_steps": nav_steps, "steps": steps_log,
    }
    with open(RUN_OUT, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
    LAST_RUN_METRICS.clear()
    LAST_RUN_METRICS.update({
        "area": AREA, "n_vantages": len(visited), "n_objects": len(final_objs),
        "n_doors": len(doors), "stop_reason": stop_reason, "nav_steps": nav_steps,
    })

    _hr("结果")
    print(f"覆盖视角={len(visited)}  停因={stop_reason}  nav_steps={nav_steps}  作者=Qwen(本地)  粗边界bbox={bbox}")
    print(f"门(聚类去重 {len(doors)})={doors}")
    print(f"记忆物体({len(final_objs)})={final_objs}")
    print(f"[run 工件] {RUN_OUT}  [记忆] {os.path.join(config.MEMORY_ROOT, config.ENV_NAME, AREA, 'area.json')}")
    if PERCEPTION_BACKEND == "yoloe" and _YOLO_CANDIDATE_RECORDS:
        candidates = _clean_candidate_records(
            list(_YOLO_CANDIDATE_RECORDS),
            bbox,
            visited,
            occ,
            clean_objs,
        )
        candidate_path = os.path.join(REPORT_DIR, "yolo_candidates.json")
        with open(candidate_path, "w", encoding="utf-8") as f:
            json.dump(
                {"area": AREA, "objects": candidates, "memory_status": "candidate"},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[候选旁路] {len(candidates)} 条 → {candidate_path}（不参与导航）")
    _qwen_area = os.path.join(config.MEMORY_ROOT, config.ENV_NAME, AREA, "area.json")
    _finalize_review(_qwen_area)
    _finalize_dual(
        _qwen_area,
        list(_YOLO_RECORDS),
        list(_YOLO_CANDIDATE_RECORDS),
        bbox,
        visited,
        occ,
        director,
    )
    # 混合验证：主记忆(_qwen_area)此模式下即混合记忆 → 作 area_hybrid；基线由旁路另建对照
    _finalize_hybrid_review(_qwen_area, list(_BASELINE_RECORDS), bbox, visited, occ, director)
    return 0


def main():
    """CLI 入口薄包装：等价原行为（构造自建 ex/mem，area=模块级 AREA）。"""
    return run_pipeline()


if __name__ == "__main__":
    sys.exit(main())
