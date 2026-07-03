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
import json
import math
import os
import re
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
from agent_core.cloud.providers import AnthropicProvider
from agent_core.executor import Executor
from agent_core.geometry import depth_projection as dp
from agent_core.memory.fs_memory import FsMemory

AREA = "explore_room"          # 独立 area，不碰 autonomy_probe 的 break_room 记录
RUN_OUT = os.path.join(REPORT_DIR, "explore_run.json")
START_XY = (-0.5, -1.0)        # 休息室起点（初始条件，非答案）
START_TOL_M = 1.5

# —— 覆盖收敛参数（代码持有覆盖保证；vantage=云端标注成本上限，nav_steps=平移硬上限，两者解耦）——
MAX_VANTAGES = 18             # 全景视角上限（从26降，缩短单轮：recon~6 + 调度官~6 + 兜底~6 足够铺满单间）
MAX_NAV_STEPS = 130          # 平移步(geo_step_open)硬上限，与 vantage 解耦的安全兜底（随屋大调高）
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


def _backfill_geometry_local(ex, objects, view_pose):
    """方法B 几何回填：对带 roi 的 Qwen 物体，读一帧 depth 批量算 size + abs_pose（原地写）。

    与 MemoryAuthor._backfill_geometry 同一管线（depth_roi 近带聚类 + roi_to_size + roi 中心反投），
    使 Qwen 作者地图与 Claude 走同样的几何接地（abs_pose 比 bearing 列距离更准）。失败静默跳过。
    """
    indexed = [(i, o) for i, o in enumerate(objects)
               if isinstance(o, dict) and isinstance(o.get("roi"), dict)]
    if not indexed:
        return
    rois = [o["roi"] for _, o in indexed]
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
            continue
        sz = dp.roi_to_size(o["roi"], stats)
        # A2 size 清洗：桌面/远物 bbox 越过物体看到远墙 → 尺寸线性虚大。任一维 > 常理家具上限
        #   = depth 打在远面，size 不可信 → 标记并置 null（不瞎编尺寸；abs_pose 保留供去重/召回）。
        dims = [sz.get("width_m"), sz.get("height_m"), sz.get("depth_m")]
        if any(v is not None and v > SANE_MAX_M for v in dims):
            o["size_unreliable"] = True
            o["size"] = {"width_m": None, "height_m": None, "depth_m": None}
        else:
            o["size"] = sz
        if stats.get("median_m"):
            try:
                uc, vc = dp.roi_center_pixel(o["roi"])
                o["abs_pose"] = dp.back_project(uc, vc, stats["median_m"], robot_pose=pose)["abs_pose"]
            except Exception:  # noqa: BLE001
                pass


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
NEAR_LABEL_M = 0.5      # 标注铁律：正前 scan 最近障碍必须 > 此值才让 Qwen 标注（太近视角不全→误标→撑爆去重）
BBOX_CLAMP_M = 6.0      # frontier 播种 bbox 钳制半径（坏墙点不至于把 frontier 炸开）
BBOX_MAX_CELLS = 120    # 播种格数上限保护
DOOR_CLUSTER_M = 1.2    # 门世界点聚类阈值（去重）
DOOR_NOMINAL_M = 1.5    # 门世界点估计的名义距离（开口处 scan 常无回波，用名义距离投点供聚类）
DEDUP_M = 0.6           # 几何校验：同类物体 abs_pose 距离 ≤ 此值=同一物体多视角重复 → 合并；
#                         更远(如南墙一排矮柜 0.65m+ 间距)=不同实例各自成条（配合 _name_compat 名感知）
RELIABLE_Z = (-0.2, 1.6)  # 可信高度带（地面家具）；超出=高处/墙挂物，单帧 depth 反投不可信 → 按名归并并标记
SANE_MAX_M = 2.5        # A2 尺寸清洗：任一维 > 此值=depth 打到远墙的虚大尺寸 → 标 size_unreliable 并置 null
BOUNDARY_MARGIN = 0.6   # abs_pose 超出房间边界此余量=depth 打到远墙的错误投影 → 视作不可信

# ===== Claude 象限调度官（Role-1 覆盖规划；叠加在代码 frontier 兜底之上，绝不替换）=====
# 角色分工不变：Claude 只读符号地图【选】去哪(从代码筛好的候选里选 id，不产坐标)，代码保证走得成/不漏。
RECON_INSET_M = 0.8            # recon：bbox 四角向内缩这么多作为 4 个 recon 目标点
RECON_MAX_ITERS = 10          # recon/中心 geo_goto 每次 max_iters
DIRECTOR_MAX_ROUNDS = 8       # Claude plan_coverage 调用轮数硬上限（成本/延迟）
DIRECTOR_NAV_BUDGET = int(0.6 * MAX_NAV_STEPS)  # 调度官最多吃这么多平移步 → 兜底始终留 ≥40%
DIRECTOR_MAX_ITERS = 12       # 调度官目标 geo_goto 每次 max_iters
DIRECTOR_EMPTY_LIMIT = 2      # 连续这么多轮空计划(done 之外的无效/幻觉) → 提前交给兜底
DIRECTOR_FAIL_LIMIT = 3       # 连续这么多个目标都到不了(即使绕行) → 该向不可达，提前交给兜底(保住 backstop 预算)
DIRECTOR_MAX_LEGS = 14        # 调度官 geo_goto_around 反应式绕行腿数上限（绕红柜进东侧办公区要够腿）
QUAD_COVER_TARGET = 0.6       # 象限 coverage < 此值 且 仍有可达空洞 = under_covered
CAND_PER_QUAD = 3             # 每象限给 Claude 的候选格上限（payload 紧凑）
REOPEN_FREE_NEIGHBORS = 3     # blocked 格 8 邻里 ≥ 此数是 visited → 疑似假墙，作 reopen 候选
REOPEN_FRONT_CONE_DEG = 30.0  # 假墙 scan 复核：前向锥半角
REOPEN_CLEAR_MARGIN = 0.4     # 复核判开：前向最近障碍 ≥ 到目标格距离 + 此余量（匹配 front_block_m）
OBJECT_STANDOFF_M = 0.5       # 停车铁律：目标格落着物体则只贴近到 ≥ 此距离（太近/太远都看不清）
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


def _sweep_vantage(ex, known_names, headings=SWEEP_HEADINGS):
    """到格后【原地环视】：逐朝向 Qwen inspect（出物体+ROI，本地、免费）+ 顾问标门 + 收墙点；
    代码用 depth_roi 给每个物体回填 size+abs_pose（贴物体的框，比 bearing 列距离更准）。

    name_hints=known_names（本区域已记物体名）→ 让 Qwen 沿用同名，减少跨帧命名发散（提升类别召回）。
    标注前强制正前 scan>NEAR_LABEL_M（太近视角不全易误标、撑爆去重）——退不开的朝向只收墙点不标注。
    返回 {objects(本 vantage 全部带几何的 Qwen 物体), hint_bearings(世界度), doors_raw, wall_pts}。
    """
    objs_all, hint_bearings, doors_raw, wall_pts = [], [], [], []
    off_map = {"left": 45.0, "center": 0.0, "right": -45.0}
    for h in headings:
        p = _pose(ex)
        nav.geo_face_point(ex, p["x"] + math.cos(math.radians(h)), p["y"] + math.sin(math.radians(h)))
        front = _ensure_view_distance(ex)     # 太近先后退，保证正前 scan>0.5 再标注
        cur = _pose(ex)
        wall_pts.extend(_scan_world_points(ex, cur))
        if front is not None and front < NEAR_LABEL_M:
            print(f"    [跳过标注] 正前仅 {front:.2f}m<{NEAR_LABEL_M}m 且退不开 → 本朝向不标注(防近距误标)")
            continue
        rep = harness.inspect_and_report(ex, name_hints=(sorted(set(known_names)) or None))
        for hh in _advisor(ex, rep):
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
        # 收集本朝向 Qwen 物体（带 roi）→ 几何回填 size+abs_pose（无 roi 的退化用 bbox_center+距离反投）
        heading_objs = []
        for o in (rep.get("objects") or []):
            if not (isinstance(o, dict) and o.get("name")):
                continue
            heading_objs.append({"name": o["name"], "confidence": o.get("confidence"),
                                 "spatial": o.get("bearing"), "roi": o.get("roi"),
                                 "bbox_center": o.get("bbox_center"), "distance_m": o.get("distance_m")})
        _backfill_geometry_local(ex, heading_objs, cur)
        for o in heading_objs:
            if o.get("abs_pose") is None:
                o["abs_pose"] = _qwen_obj_abs_pose(o, cur)
        objs_all.extend(heading_objs)
    return {"objects": objs_all, "hint_bearings": hint_bearings,
            "doors_raw": doors_raw, "wall_pts": wall_pts}


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
        for f in ("size", "roi", "spatial"):
            if inc.get(f) is not None:
                keep[f] = inc[f]
        keep.pop("size_unreliable", None)        # 采用更近帧的尺寸可信度
        if inc.get("size_unreliable"):
            keep["size_unreliable"] = True
    elif keep.get("abs_pose") is None and inc.get("abs_pose"):
        keep["abs_pose"] = inc["abs_pose"]       # keep 无位置则先补上（inc 有就用）
        if inc.get("distance_m") is not None:
            keep["distance_m"] = inc["distance_m"]


def _dedup_objects(vantage_records, boundary):
    """几何校验去重：可信物体(在界内+地面高度带)按【世界位置】去重(name-agnostic)；
    不可信物体(无 abs_pose/越界/高处)按【归一化名】归并并标 size_unreliable。

    位置重复=同一物体的多视角重复观测 → 合并而非新增（直接解决 71 条噪声里的"同物多记"）。
    """
    reliable, unreliable = [], {}
    for rec in vantage_records:
        for o in rec.get("objects", []):
            if not isinstance(o, dict) or not o.get("name"):
                continue
            ap = o.get("abs_pose")
            ok_pos = isinstance(ap, dict) and ap.get("x") is not None
            zok = ok_pos and (ap.get("z") is None or RELIABLE_Z[0] <= ap["z"] <= RELIABLE_Z[1])
            inb = ok_pos and (not boundary or _in_bbox(ap, boundary))
            if ok_pos and zok and inb:
                hit = None
                for r in reliable:
                    # 同类 + 位置近 = 同一物体的多视角重复 → 合并；不同类即使挨着也各自成条
                    if _abs_dist(ap, r["abs_pose"]) <= DEDUP_M and _name_compat(o, r):
                        hit = r
                        break
                if hit:
                    _merge_obj_pair(hit, o)
                else:
                    reliable.append(dict(o))
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
    # 跨桶去重：同一物体若已有【可信位置】，丢弃它的不可信高视角重复条（按名/别名匹配），避免双计。
    rel_names = set()
    for o in reliable:
        rel_names.add(_norm_name(o.get("name")))
        for a in (o.get("aliases") or []):
            rel_names.add(_norm_name(a))
    unr_final = [o for o in unreliable.values() if _norm_name(o.get("name")) not in rel_names]
    return reliable + unr_final


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


def _build_plan_payload(bbox, quad_stats, obj_xy, candidates, last_rejected, round_i, rounds_left):
    """把符号地图压成 Claude payload（候选剥掉内部 cell 字段，只留 id/quad/x/y/reopen_blocked）。"""
    b = {k: round(v, 1) for k, v in bbox.items()} if bbox else {}
    objs = [{"name": n, "x": x, "y": y} for (n, x, y) in obj_xy]
    cands = [{"id": c["id"], "quad": c["quad"], "x": c["x"], "y": c["y"],
              "reopen_blocked": c["reopen_blocked"]} for c in candidates]
    return {"bbox": b, "quadrants": quad_stats, "objects": objs, "candidates": cands,
            "last_rejected_reopen": last_rejected, "round": round_i, "rounds_left": rounds_left}


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


def _frontier_backstop(ex, *, visited, wall_points, vantage_records, doors_raw, steps_log,
                       visited_cells, blocked_cells, known_names,
                       nav_steps, vantages, last_vantage_xy, last_pose):
    """代码网格 frontier 覆盖回路 —— 覆盖保证【兜底】(recon+director 之后无条件跑，守住 ≥65%)。
    从当前 bbox 重新播种 frontier(自动排除已 visited/blocked)，跑到覆盖完/上限/卡死。
    返回 (nav_steps, vantages, stop_reason, last_pose)。逻辑同旧主回路，仅参数化 + 共用 _absorb_sweep。"""
    stop_reason = "nav_cap"
    stuck = 0
    frontier = set()
    cur = last_pose
    last_iter_xy = (last_pose["x"], last_pose["y"])
    while True:
        if wall_points:
            frontier |= _seed_frontier_from_bbox(
                dp.boundary_from_points(wall_points + visited), visited_cells, blocked_cells)
        frontier -= visited_cells
        frontier -= blocked_cells
        if not frontier:
            stop_reason = "covered"
            print("[覆盖] 所有可达格已覆盖 → 完成")
            break
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
            hint_bearings = []
            print(f"  [跳过环视] 距上一 vantage <{MIN_VANTAGE_SPACING_M}m，不重扫")
        else:
            sweep = _sweep_vantage(ex, known_names)
            hint_bearings = _absorb_sweep(sweep, wall_points=wall_points, vantage_records=vantage_records,
                                          doors_raw=doors_raw, known_names=known_names)
            vantages += 1
            last_vantage_xy = (pose["x"], pose["y"])
            cobjs = [o.get("name") for o in sweep["objects"]]
            print(f"[视角{vantages}] pose=({pose['x']:.2f},{pose['y']:.2f}) 环视 Qwen记{len(cobjs)}物体 "
                  f"门提示={len(sweep['doors_raw'])}")
            steps_log.append({"step": vantages - 1, "pose": pose, "qwen_objects": cobjs,
                              "n_doors": len(sweep["doors_raw"])})

        frontier |= _seed_frontier_from_bbox(
            dp.boundary_from_points(wall_points + visited), visited_cells, blocked_cells)
        frontier -= visited_cells
        frontier -= blocked_cells
        if not frontier:
            stop_reason = "covered"
            break
        target_cell = _pick_target(frontier, pose, hint_bearings)
        tx, ty = _cell_center(target_cell)
        reached, cur, ns = _drive_to(ex, (tx, ty))
        nav_steps += ns
        if reached:
            visited_cells.add(target_cell)
            visited_cells.add(_cell(cur["x"], cur["y"]))
            frontier |= set(_neighbors(target_cell))
            print(f"  → 格{target_cell}({tx:.1f},{ty:.1f}) 到位 ({cur['x']:.2f},{cur['y']:.2f}) nav={nav_steps}")
        else:
            blocked_cells.add(target_cell)
            blocked_cells |= _blocked_cone(frontier, pose, (tx, ty))
            print(f"  → 格{target_cell}({tx:.1f},{ty:.1f}) 撞墙 → 标 blocked(含共线锥) nav={nav_steps}")

        moved = math.hypot(cur["x"] - last_iter_xy[0], cur["y"] - last_iter_xy[1])
        last_iter_xy = (cur["x"], cur["y"])
        stuck = stuck + 1 if moved < 0.15 else 0
        if stuck >= STUCK_LIMIT:
            frontier -= visited_cells
            frontier -= blocked_cells
            if frontier:
                far = max(frontier, key=lambda c: math.hypot(
                    _cell_center(c)[0] - cur["x"], _cell_center(c)[1] - cur["y"]))
                fx, fy = _cell_center(far)
                _, cur, ns2 = _drive_to(ex, (fx, fy), max_steps=3)
                nav_steps += ns2
                esc = math.hypot(cur["x"] - last_iter_xy[0], cur["y"] - last_iter_xy[1])
                last_iter_xy = (cur["x"], cur["y"])
                print(f"  [脱困] 逃向最远格({fx:.1f},{fy:.1f}) 位移{esc:.2f}m nav={nav_steps}")
                if esc < 0.15:
                    stop_reason = "stuck"
                    break
            stuck = 0
    return nav_steps, vantages, stop_reason, last_pose


def main():
    _hr("explore_probe v2：从零覆盖 + Qwen 本地语义标注（代码做几何/去重/整理）")
    ex = Executor()
    mem = FsMemory(config.MEMORY_ROOT, config.ENV_NAME)
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
    rec_type = "lounge"
    stop_reason = "nav_cap"

    # 共享状态袋：recon / director / 兜底 三阶段同一份 → 统一喂不改动的去重+写盘路径
    start = _pose(ex)
    last_pose = start
    visited_cells, blocked_cells = {_cell(start["x"], start["y"])}, set()
    nav_steps, vantages = 0, 0
    last_vantage_xy = None          # 上一个实际环视点（间距门槛用）
    known_names = []                # 已记录物体名（作命名锚点喂 Qwen，减少跨帧命名发散）
    last_rejected_reopen = []       # scan 复核确认过的真墙 [x,y]（反馈 Claude，别再选）
    director = AnthropicProvider()  # 象限调度官 + 记忆整理官（同一实例）

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
            sw = _sweep_vantage(ex, known_names)
            _absorb_sweep(sw, wall_points=wall_points, vantage_records=vantage_records,
                          doors_raw=doors_raw, known_names=known_names)
            vantages += 1
            last_vantage_xy = (cur["x"], cur["y"])
            print(f"[RECON-{tag}{vantages}] pose=({cur['x']:.2f},{cur['y']:.2f}) Qwen记{len(sw['objects'])}物体")

    try:
        # ===== STAGE A · RECON：自举 bbox（起点环视 → 四内缩角 → 中心 360°）=====
        _hr("STAGE A · RECON（起点环视 → 四角 → 中心 360°，自举房间范围）")
        s0 = _sweep_vantage(ex, known_names)
        _absorb_sweep(s0, wall_points=wall_points, vantage_records=vantage_records,
                      doors_raw=doors_raw, known_names=known_names)
        visited.append([round(start["x"], 2), round(start["y"], 2)])
        vantages += 1
        last_vantage_xy = (start["x"], start["y"])
        steps_log.append({"step": 0, "phase": "recon_start", "pose": start,
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

        # ===== STAGE B · CLAUDE 象限调度官（选欠覆盖象限 → 派 Qwen 去补）=====
        _hr("STAGE B · 象限调度官（Claude 只从代码候选里选 id；scan 复核 + 绕行 VFH 执行）")
        empty_rounds = 0
        fail_streak = 0
        for round_i in range(DIRECTOR_MAX_ROUNDS):
            if nav_steps >= DIRECTOR_NAV_BUDGET or vantages >= MAX_VANTAGES:
                print("[调度官] 预算用尽 → 交给 frontier 兜底")
                break
            bbox = dp.boundary_from_points(wall_points + visited)
            obj_xy = _objects_xy(vantage_records, bbox)
            quad_stats = _quadrant_stats(bbox, visited_cells, blocked_cells, obj_xy)
            candidates = _candidate_cells(bbox, visited_cells, blocked_cells)
            if not candidates:
                print("[调度官] 无候选格（已铺满）→ 交给兜底")
                break
            payload = _build_plan_payload(bbox, quad_stats, obj_xy, candidates,
                                          last_rejected_reopen, round_i, DIRECTOR_MAX_ROUNDS - round_i)
            plan = director.plan_coverage(payload)
            if plan.get("done") is True:
                print(f"[调度官] 判定四象限已足够覆盖(done) rationale={plan.get('rationale', '')}")
                break
            tgt = _validate_plan_choice(plan, candidates)
            if tgt is None:
                empty_rounds += 1
                print(f"[调度官] 空/幻觉计划(第{empty_rounds}次) plan={plan}")
                if empty_rounds >= DIRECTOR_EMPTY_LIMIT:
                    print("[调度官] 连续空计划 → 提前交给兜底")
                    break
                continue
            empty_rounds = 0
            cell = tgt["cell"]
            print(f"[调度官R{round_i}] 选 {tgt['id']}@{tgt['quad']}({tgt['x']},{tgt['y']}) "
                  f"reopen={tgt['reopen_blocked']} rationale={plan.get('rationale', '')}")
            # 假墙重开护栏：scan 复核确认真缺口才去；真墙→驳回并反馈
            if tgt["reopen_blocked"]:
                if not _reverify_open(ex, cell):
                    blocked_cells.add(cell)
                    last_rejected_reopen.append([tgt["x"], tgt["y"]])
                    print("  [假墙复核] scan 判定真墙 → 驳回，回填 blocked 并反馈 Claude")
                    continue
                blocked_cells.discard(cell)
                print("  [假墙复核] scan 判定有缺口 → 准许重开")
            tx, ty = _cell_center(cell)
            fblock, tol = _stop_rule(cell, obj_xy)   # 目标格有物体则停 ≥0.5m
            # geo_goto_around：撞墙不放弃，反应式绕行（绕红柜经北缺口进东侧办公区），比 geo_goto 直冲强
            r = nav.geo_goto_around(ex, tx, ty, tol_m=tol, max_legs=DIRECTOR_MAX_LEGS,
                                    max_step_m=1.0, front_block_m=fblock)
            nav_steps += len(r.get("steps") or [])
            cur = r.get("pose") or _pose(ex)
            last_pose = cur
            visited.append([round(cur["x"], 2), round(cur["y"], 2)])
            visited_cells.add(_cell(cur["x"], cur["y"]))
            if r.get("arrived"):
                visited_cells.add(cell)
                fail_streak = 0
            else:
                blocked_cells.add(cell)   # 绕行仍到不了 → 标 blocked（兜底不再枉试）
                fail_streak += 1
                print(f"  → 未到位({r.get('status')}) → 标 blocked（连续失败{fail_streak}）")
                if fail_streak >= DIRECTOR_FAIL_LIMIT:
                    print("[调度官] 连续多目标绕行仍不可达 → 提前交给兜底（保住 backstop 预算）")
                    break
            if r.get("arrived") and _min_vantage_spacing_ok(cur, last_vantage_xy) and vantages < MAX_VANTAGES:
                sw = _sweep_vantage(ex, known_names)
                _absorb_sweep(sw, wall_points=wall_points, vantage_records=vantage_records,
                              doors_raw=doors_raw, known_names=known_names)
                vantages += 1
                last_vantage_xy = (cur["x"], cur["y"])
                print(f"  → 到位环视{vantages} pose=({cur['x']:.2f},{cur['y']:.2f}) Qwen记{len(sw['objects'])}物体 nav={nav_steps}")

        # ===== STAGE C · FRONTIER 兜底（覆盖保证；无条件跑，守住 ≥65%）=====
        _hr("STAGE C · frontier 兜底（无条件补全覆盖，用 nav 剩余额度）")
        nav_steps, vantages, stop_reason, last_pose = _frontier_backstop(
            ex, visited=visited, wall_points=wall_points, vantage_records=vantage_records,
            doors_raw=doors_raw, steps_log=steps_log, visited_cells=visited_cells,
            blocked_cells=blocked_cells, known_names=known_names,
            nav_steps=nav_steps, vantages=vantages, last_vantage_xy=last_vantage_xy, last_pose=last_pose)
        ex.ros.call("stop", {})
    finally:
        ex.close()

    # —— 几何校验去重（用户："据位姿+深度算位置，重复就过滤"）：把所有视角的原始观测先收敛 ——
    bbox = dp.boundary_from_points(wall_points + visited)
    doors = _validate_doors(_cluster_doors(doors_raw), bbox)
    raw_count = sum(len(r.get("objects", []) or []) for r in vantage_records)
    clean_objs = _dedup_objects(vantage_records, bbox)
    print(f"[去重] 原始观测 {raw_count} → 几何校验后 {len(clean_objs)} 物体")
    # —— Role-2：Claude 记忆整理官（同物不同名并组、不同类分开、幻觉剔除；坐标仍由代码聚合）——
    clean_objs = _consolidate_memory(clean_objs, director)
    # —— 落盘（单一并集写路径）：逐物体 upsert_object（数组并集、同名远位=多实例）——
    n_obj = 0
    for o in clean_objs:
        try:
            mem.upsert_object(AREA, o)
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

    _hr("结果")
    print(f"覆盖视角={len(visited)}  停因={stop_reason}  nav_steps={nav_steps}  作者=Qwen(本地)  粗边界bbox={bbox}")
    print(f"门(聚类去重 {len(doors)})={doors}")
    print(f"记忆物体({len(final_objs)})={final_objs}")
    print(f"[run 工件] {RUN_OUT}  [记忆] {os.path.join(config.MEMORY_ROOT, config.ENV_NAME, AREA, 'area.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
