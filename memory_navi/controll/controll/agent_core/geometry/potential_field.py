"""人工势场(APF)局部观测点选择（几何核心，纯函数，可脱离 ROS/LLM 单测）。

用途（Task 2）：把雷达 scan 的障碍点铺成机器人局部势场，供代码筛出"安全站位观测点"候选，
再由 Claude 从候选里选一个（角色分工不变：Claude 只选 id、绝不产坐标）。既避免车贴墙 / 钻桌底
这类坏观测点，也把"撞了才退 0.35m"的反应式弹跳前移成"预先选好离障碍适中的站位"。

坐标约定与 navigator / depth_projection 一致：世界系 +X 东、+Y 北/左，yaw=0 朝 +X，+yaw CCW。

势场（数值越低越好；给人读 / 给 Claude 参考，候选真正硬筛用的是 clearance）：
  U(cell) = U_rep + U_att
  斥力 U_rep = ½·η·(1/d − 1/d0)²   (d<d0，d=该格到最近障碍距离；d≤0 视作 inf 不可站)
  引力 U_att = ½·ξ·|cell − target|²  (朝目标，鼓励靠近但不逼近站位带下限)
候选观测点：clearance 落在 [standoff_min, standoff_max] 的自由格，按 U 升序稀疏取 top-K。
"""
import math

# scan_summary 8 扇区人类标签 → 机体系中心角(度，0=前，+左)；细分扇区用 "sec_<角度>"。
_SECTOR_LABEL_ANG = {"front": 0, "front_left": 45, "left": 90, "rear_left": 135,
                     "rear": 180, "rear_right": -135, "right": -90, "front_right": -45}


OBS_MIN_CLEAR_M = 0.7   # 观测点质量门：任一扇区最近障 < 此值 = 贴障/钻桌底/贴墙 → 不宜停下观测。
#                         0.5→0.7：0.13m 低相机贴近物体只见一块颜色→误标(红柜→门)；拉远才见全貌。
#                         无 ≥0.7 站位可达时 _pick_viewpoint 回退原地观测→不丢覆盖，只是"能退则退"。
OBS_CV_MAX = 0.6        # 观测点均匀性门：归一化方差(变异系数 CV=std/mean) ≤ 此值才算"四周距离均匀"
#                         CV 不受房间大小影响；只剔"一侧很近其余很远"的偏斜/近钉位("退后看墙"CV≈0.35 仍过)


def scan_obs_quality(sectors, *, min_clear_m=OBS_MIN_CLEAR_M, cv_max=OBS_CV_MAX):
    """代码算某位姿是否【适合停下观测】：最近障够远(不贴障) + 四周距离够均匀(不偏斜近钉)。

    sectors: {label: dist_m}（scan_summary 的扇区距离，值 None/非正跳过）。
    返回 {min_clear_m, mean_m, cv, uniform, ok}；ok = (min ≥ min_clear_m) and (cv ≤ cv_max)。
    无有效距离 → ok=False（无从判断，不误判为好点）。纯几何，可脱 ROS 单测。
    """
    ds = [float(d) for d in (sectors or {}).values()
          if isinstance(d, (int, float)) and d > 0]
    if not ds:
        return {"min_clear_m": None, "mean_m": None, "cv": None, "uniform": False, "ok": False}
    dmin = min(ds)
    dmean = sum(ds) / len(ds)
    var = sum((d - dmean) ** 2 for d in ds) / len(ds)
    cv = math.sqrt(var) / dmean if dmean > 0 else math.inf
    uniform = cv <= cv_max
    ok = (dmin >= min_clear_m) and uniform
    return {"min_clear_m": round(dmin, 2), "mean_m": round(dmean, 2),
            "cv": round(cv, 2) if cv != math.inf else None, "uniform": uniform, "ok": ok}


def apf_heading(pose, gx, gy, beams, *, k_att=1.0, eta=0.5, d0_m=1.0, rep_cap=1.5):
    """APF 合力 → 世界航向(度)：执行模式【点到点】导航的方向核（纯数学，可脱 ROS 单测）。

    执行模式是点到点问题（起点+航点+目标坐标皆已知），不该用探索/补全的岔口-VFH（"直路被堵就找最宽开口"
    会把车导离目标）。APF：引力朝目标 + 斥力离障碍，合力方向逐小步开过去，天然穿缝（缝两侧斥力对消、
    引力拉过去）。

    斥力用【线性】falloff `eta·(1/d − 1/d0)`（不用 1/d²，否则贴障时斥力爆炸把车吓退），且对整簇障碍的
    合斥力【封顶 rep_cap】——斥力只负责【侧向转开】，别盖过朝目标的前进引力(k_att)，才不会一见墙就掉头。

    Args:
        pose: {x, y, yaw_deg}（世界系；yaw=0 朝 +x，+yaw CCW）。
        gx, gy: 目标世界坐标（引力指向它，单位向量 × k_att）。
        beams: [[bearing_deg(体系,0=前/+左), dist_m], ...]（scan_rays；dist=-1/None/<=0 或 ≥d0_m 跳过）。
        eta: 斥力增益。d0_m: 斥力影响半径。rep_cap: 合斥力幅值上限（相对 k_att）。

    Returns:
        合力世界航向(度)。无近障→朝目标；近障→航向侧偏离障；对称缝→侧向对消、朝目标穿中。
    """
    px, py = float(pose["x"]), float(pose["y"])
    yaw = float(pose.get("yaw_deg", 0.0) or 0.0)
    dgx, dgy = gx - px, gy - py
    dg = math.hypot(dgx, dgy) or 1e-9
    ax, ay = k_att * dgx / dg, k_att * dgy / dg               # 引力：朝目标单位向量
    rx, ry = 0.0, 0.0
    for b in beams or []:
        try:
            deg, d = float(b[0]), float(b[1])
        except (TypeError, ValueError, IndexError):
            continue
        if d <= 0 or d >= d0_m:                                # -1(无返回)/远障 → 无斥力
            continue
        wa = math.radians(yaw + deg)                           # 障碍世界方位
        w = eta * (1.0 / d - 1.0 / d0_m)                       # 线性 falloff（近大远小、不爆炸）
        rx -= w * math.cos(wa)                                 # 沿"离开障碍"方向（反障碍方位）
        ry -= w * math.sin(wa)
    rmag = math.hypot(rx, ry)
    if rmag > rep_cap:                                         # 封顶：斥力只侧向转开、不盖过引力
        rx *= rep_cap / rmag
        ry *= rep_cap / rmag
    return math.degrees(math.atan2(ay + ry, ax + rx))


def _label_angle(label):
    """扇区标签 → 机体系角度(度)；无法解析返回 None。"""
    if label in _SECTOR_LABEL_ANG:
        return _SECTOR_LABEL_ANG[label]
    if isinstance(label, str) and label.startswith("sec_"):
        try:
            return int(label[4:])
        except ValueError:
            return None
    return None


def obstacle_points_from_scan(sectors, pose):
    """scan_summary(sectors=N) 的每扇区最近距离 → 世界障碍点 [(x,y),...]。

    sectors: {label: dist_m}（label 为 8 扇区人类标签或 "sec_<中心角度>"）。
    pose: {x, y, yaw_deg}。None / 非正距离跳过。
    """
    pts = []
    yaw = float(pose.get("yaw_deg", 0.0) or 0.0)
    px, py = float(pose["x"]), float(pose["y"])
    for label, d in (sectors or {}).items():
        if not isinstance(d, (int, float)) or d <= 0:
            continue
        ang = _label_angle(label)
        if ang is None:
            continue
        wa = math.radians(yaw + ang)
        pts.append((round(px + d * math.cos(wa), 3), round(py + d * math.sin(wa), 3)))
    return pts


def build_field(obstacle_pts, pose, target_xy=None, *,
                half_size_m=1.6, res_m=0.25, d0_m=0.8, eta=1.0, xi=0.3):
    """以机器人为中心铺一张局部势场网格（robot 在正中心格）。

    返回 {res, half, nx, ny, xs, ys, clear, U, pose, target_xy, d0_m}；
    clear[jy][ix] = 该格到最近障碍的距离(米，None=无障碍点)，U 同形状为势能。
    """
    ox, oy = float(pose["x"]), float(pose["y"])
    n = max(1, int(round(half_size_m / res_m)))
    xs = [round(ox + i * res_m, 3) for i in range(-n, n + 1)]
    ys = [round(oy + j * res_m, 3) for j in range(-n, n + 1)]
    nx, ny = len(xs), len(ys)
    tgt = tuple(target_xy) if target_xy else None
    clear = [[None] * nx for _ in range(ny)]
    U = [[0.0] * nx for _ in range(ny)]
    for jy, wy in enumerate(ys):
        for ix, wx in enumerate(xs):
            cmin = math.inf
            for ax, ay in obstacle_pts:
                d = math.hypot(wx - ax, wy - ay)
                if d < cmin:
                    cmin = d
            if cmin is math.inf:
                clear[jy][ix] = None
                urep = 0.0
            else:
                clear[jy][ix] = round(cmin, 3)
                if cmin <= 1e-3:
                    urep = math.inf
                elif cmin < d0_m:
                    urep = 0.5 * eta * (1.0 / cmin - 1.0 / d0_m) ** 2
                else:
                    urep = 0.0
            uatt = 0.0
            if tgt is not None:
                uatt = 0.5 * xi * ((wx - tgt[0]) ** 2 + (wy - tgt[1]) ** 2)
            U[jy][ix] = urep if urep is math.inf else round(urep + uatt, 4)
    return {"res": res_m, "half": half_size_m, "nx": nx, "ny": ny, "xs": xs, "ys": ys,
            "clear": clear, "U": U,
            "pose": {"x": ox, "y": oy, "yaw_deg": float(pose.get("yaw_deg", 0.0) or 0.0)},
            "target_xy": tgt, "d0_m": d0_m}


def _clear_at(field, x, y):
    """就近查某世界点的 clearance(米)；越界或该格无障碍数据返回 inf(视作开阔)。"""
    cell = _nearest_cell(field, (x, y))
    if cell is None:
        return math.inf
    c = field["clear"][cell[0]][cell[1]]
    return math.inf if c is None else c


def _line_of_sight(field, x0, y0, x1, y1, block_m=0.2):
    """机器人(x0,y0)→候选(x1,y1)直线是否不穿障碍：沿途采样点 clearance 均 ≥ block_m 才算通。

    防止选到障碍【背后】的低势能格(几何看着近目标、实际车过不去)。纯网格查表，无 ROS。
    """
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(dist / (field["res"] * 0.5)))
    for i in range(1, steps):          # 跳过起点自身(车脚下)
        t = i / steps
        if _clear_at(field, x0 + t * (x1 - x0), y0 + t * (y1 - y0)) < block_m:
            return False
    return True


def candidate_viewpoints(field, *, standoff_min=OBS_MIN_CLEAR_M, standoff_max=1.5, k=6,
                         min_sep_m=0.4, require_los=True):
    """从势场里筛安全站位观测点：clearance 落在 [min,max]、【车能直达(LOS)】的自由格，
    按势能升序稀疏取 top-K。

    稀疏化(min_sep_m)避免候选挤成一堆；require_los 剔除障碍背后不可达的格。
    返回 [{id, x, y, clearance_m, potential, _ix, _iy}]。
    """
    rx, ry = field["pose"]["x"], field["pose"]["y"]
    cands = []
    for jy in range(field["ny"]):
        for ix in range(field["nx"]):
            c = field["clear"][jy][ix]
            if c is None or not (standoff_min <= c <= standoff_max):
                continue
            u = field["U"][jy][ix]
            if u is math.inf:
                continue
            wx, wy = field["xs"][ix], field["ys"][jy]
            if require_los and not _line_of_sight(field, rx, ry, wx, wy):
                continue
            cands.append({"x": wx, "y": wy,
                          "clearance_m": round(c, 2), "potential": round(u, 3),
                          "_ix": ix, "_iy": jy})
    cands.sort(key=lambda c: (c["potential"], c["clearance_m"]))
    picked = []
    for c in cands:
        if all(math.hypot(c["x"] - p["x"], c["y"] - p["y"]) >= min_sep_m for p in picked):
            picked.append(c)
        if len(picked) >= k:
            break
    for i, c in enumerate(picked):
        c["id"] = f"v{i}"
    return picked


def _nearest_cell(field, xy):
    """世界点 → 最近格 (iy, ix)；越界返回 None。"""
    if not xy:
        return None
    x, y = xy
    if not (field["xs"][0] <= x <= field["xs"][-1] and field["ys"][0] <= y <= field["ys"][-1]):
        return None
    ix = min(range(field["nx"]), key=lambda i: abs(field["xs"][i] - x))
    iy = min(range(field["ny"]), key=lambda j: abs(field["ys"][j] - y))
    return (iy, ix)


def render_ascii(field, candidates=None):
    """势场 ASCII 图（北在上）：R=车 T=目标 数字=候选点 id 末位 #=贴障碍 +=近障碍带 .=开阔/无障碍。"""
    candidates = candidates or []
    id_at = {(c["_iy"], c["_ix"]): c["id"] for c in candidates if "_ix" in c and "_iy" in c}
    rc = _nearest_cell(field, (field["pose"]["x"], field["pose"]["y"]))
    tc = _nearest_cell(field, field.get("target_xy"))
    rows = []
    for jy in range(field["ny"] - 1, -1, -1):          # 北(大 y)在上
        cells = []
        for ix in range(field["nx"]):
            if rc == (jy, ix):
                cells.append("R")
            elif tc == (jy, ix):
                cells.append("T")
            elif (jy, ix) in id_at:
                cells.append(id_at[(jy, ix)][-1])
            else:
                c = field["clear"][jy][ix]
                if c is None:
                    cells.append(".")
                elif c < 0.25:
                    cells.append("#")
                elif c < field["d0_m"]:
                    cells.append("+")
                else:
                    cells.append(".")
        rows.append(" ".join(cells))
    legend = ("图例：R=车 T=目标 数字=候选观测点 #=贴障碍(危) +=近障碍带 .=开阔；"
              f"每格 {field['res']}m，北在上/东在右。")
    return legend + "\n" + "\n".join(rows)


def build_viewpoint_payload(sectors, pose, target_xy=None, **kw):
    """一步到位：scan 扇区 + 位姿(+目标) → Claude 选点 payload。

    返回 {ascii_field, candidates:[{id,x,y,clearance_m}], target_xy, pose}；
    candidates 已剥掉内部 _ix/_iy（Claude 只需 id + 可读坐标）。无候选则 candidates 为空。
    """
    field_kw = {k: kw[k] for k in ("half_size_m", "res_m", "d0_m", "eta", "xi") if k in kw}
    cand_kw = {k: kw[k] for k in ("standoff_min", "standoff_max", "k", "min_sep_m") if k in kw}
    pts = obstacle_points_from_scan(sectors, pose)
    field = build_field(pts, pose, target_xy, **field_kw)
    cands = candidate_viewpoints(field, **cand_kw)
    ascii_field = render_ascii(field, cands)
    pub = [{"id": c["id"], "x": c["x"], "y": c["y"], "clearance_m": c["clearance_m"],
            "potential": c["potential"]} for c in cands]
    return {"ascii_field": ascii_field, "candidates": pub,
            "target_xy": list(target_xy) if target_xy else None,
            "pose": {"x": field["pose"]["x"], "y": field["pose"]["y"],
                     "yaw_deg": field["pose"]["yaw_deg"]},
            "_candidates_full": cands}
