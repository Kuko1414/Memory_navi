"""几何导航 skill（监督者层）：用坐标 + 已闭环验证的原语把车确定性地开到目标位姿。

为什么在这里（见 Report/harness_eval.md §7）：实测 VLM 自主目标导航不收敛（原地打转），
正解是"确定性代码做导航（去哪/朝哪），VLM 只做语义（看到了什么）"。本模块就是导航那一层：
- geo_face_point：转到正对某世界坐标点（用 get_pose 真值航向闭环 + 代码核验残差）。
- geo_goto：循环 face→前进，把车开到目标点附近（scan 门控、safety 兜底、真值核验）。

底层 move / turn_left_deg / turn_right_deg / get_pose / scan_summary 均是 ros-mcp-server
已闭环到真值的工具（见 agent_actions.py / perception.py）；本模块只做编排，不引入新的控制逻辑。
约定与 harness.execute_step 一致：动作经 ex.ros.call 下发，位姿用 get_pose 真值核验。
"""
import json
import math


def _norm_deg(a: float) -> float:
    """归一化角度到 [-180, 180)。"""
    return ((a + 180.0) % 360.0) - 180.0


def _pose(ex) -> dict:
    """get_pose 真值位姿；保证含 x/y/yaw_deg（残缺=瞬时读失败 → 重试；仍缺则补默认，
    绝不让一次坏读用 KeyError 崩掉整轮探索——宁可这帧转不准，下一次读会恢复）。"""
    import time
    p = {}
    for _ in range(3):
        try:
            p = json.loads(ex.ros.call("get_pose", {}).text)
        except (ValueError, TypeError):
            p = {}
        if isinstance(p, dict) and "x" in p and "yaw_deg" in p:
            return p
        time.sleep(0.4)
    if not isinstance(p, dict):
        p = {}
    p.setdefault("x", 0.0)
    p.setdefault("y", 0.0)
    p.setdefault("yaw_deg", 0.0)
    return p


def _scan(ex) -> dict:
    try:
        return json.loads(ex.ros.call("scan_summary", {}).text)
    except (ValueError, TypeError):
        return {}


def bearing_deg(x0: float, y0: float, tx: float, ty: float) -> float:
    """从 (x0,y0) 指向 (tx,ty) 的世界航向（度；+x=0，CCW 为正）。"""
    return math.degrees(math.atan2(ty - y0, tx - x0))


def geo_face_point(ex, tx: float, ty: float, tol_deg: float = 8.0) -> dict:
    """转到正对世界坐标点 (tx,ty)。代码闭环 + 核验残差。

    返回 {desired_yaw, yaw_before, yaw_after, err_deg, verified, pose}。
    """
    p = _pose(ex)
    desired = bearing_deg(p["x"], p["y"], tx, ty)
    diff = _norm_deg(desired - p["yaw_deg"])
    if abs(diff) > 1.0:                       # 1° 内不值得转
        tool = "turn_left_deg" if diff > 0 else "turn_right_deg"
        ex.ros.call(tool, {"degrees": round(abs(diff), 1)})
    p2 = _pose(ex)
    err = _norm_deg(desired - p2["yaw_deg"])
    return {
        "desired_yaw": round(desired, 1),
        "yaw_before": p["yaw_deg"],
        "yaw_after": p2["yaw_deg"],
        "err_deg": round(err, 1),
        "verified": abs(err) <= tol_deg,
        "pose": p2,
    }


def geo_goto(
    ex,
    tx: float,
    ty: float,
    *,
    tol_m: float = 0.25,
    max_iters: int = 12,
    max_step_m: float = 1.0,
    front_block_m: float = 0.4,
    max_safety_retries: int = 3,
) -> dict:
    """循环 face→前进，把车开到 (tx,ty) 附近。导航全在确定性代码。

    每轮：读真值位姿→判到达→正对目标→scan 门控(前方<front_block 才算真挡)→闭环前进(封顶 max_step_m)。
    safety 刹停若是侧向障碍的瞬态(前方仍畅通)则重试（mecanum 侧滑会偶发误触发）；真正前挡/步数用尽才退出。

    返回 {arrived, dist_m, pose, steps, status}；status ∈ arrived|blocked|safety|max_iters。
    """
    steps = []
    status = "max_iters"
    safety_retries = 0
    for i in range(max_iters):
        p = _pose(ex)
        dist = math.hypot(tx - p["x"], ty - p["y"])
        if dist <= tol_m:
            status = "arrived"
            break
        face = geo_face_point(ex, tx, ty)
        scan = _scan(ex)
        front = scan.get("front_min_m")
        if front is not None and front < front_block_m:   # 正对目标后前方仍被近障挡住
            steps.append(f"{i}:blocked front={front} face_err={face['err_deg']}")
            status = "blocked"
            break
        step = min(dist, max_step_m)
        mv = json.loads(ex.ros.call("move", {"distance_m": round(step, 3)}).text)
        steps.append(
            f"{i}:face_err={face['err_deg']} "
            f"move={mv.get('traveled_m')}/{round(step, 2)} {mv.get('status')}"
        )
        if mv.get("status") == "safety_stop":
            front2 = _scan(ex).get("front_min_m")
            if front2 is not None and front2 >= front_block_m and safety_retries < max_safety_retries:
                safety_retries += 1                      # 前方畅通=侧向瞬态触发，重试
                steps.append(f"{i}:transient_safety front={front2} retry{safety_retries}")
                continue
            status = "safety"
            break
    p = _pose(ex)
    dist = math.hypot(tx - p["x"], ty - p["y"])
    arrived = dist <= tol_m
    return {
        "arrived": arrived,
        "dist_m": round(dist, 3),
        "pose": p,
        "steps": steps,
        "status": "arrived" if arrived else status,
    }


def _open_side(ex) -> tuple:
    """按 scan 扇区开阔度选沿墙手性（相对当前朝向：left=北侧/右手=南侧）。返回 (side, lc, rc)。"""
    sc = _scan(ex).get("sectors", {})
    lc = max(sc.get("front_left") or 0.0, sc.get("left") or 0.0)
    rc = max(sc.get("front_right") or 0.0, sc.get("right") or 0.0)
    return ("left" if lc >= rc else "right", round(lc, 2), round(rc, 2))


def geo_goto_around(
    ex,
    tx: float,
    ty: float,
    *,
    tol_m: float = 0.4,
    max_legs: int = 24,
    max_step_m: float = 1.0,
    front_block_m: float = 0.6,
    clear_margin_m: float = 0.25,
    wall_step_m: float = 0.7,
    slide_deg: float = 90.0,
    backoff_m: float = 0.35,
    max_stuck: int = 4,
    confine_legs: int = 5,
    confine_r: float = 0.6,
) -> dict:
    """去点 + 撞墙【反应式分段绕行】（贴墙滑行 bug-algorithm，含解钉退避）。

    与 geo_goto 的区别：geo_goto 撞墙即放弃；本函数撞墙后不放弃——**先退一点解钉**(防贴墙过近
    被 safety 锁死)，再转向开阔侧≈slide_deg 贴墙挪 wall_step，回顶重新正对目标；目标方向一通就续直冲。
    沿墙手性按 scan 开阔度选定后**保持不翻面**(翻面会原地振荡)；以**位置冻结**(非距目标变化)判卡死。

    振荡早停：连续 confine_legs 腿一直在动却困在 confine_r 小圈里（目标落沙发/桌腿间→靠近-弹开-滑-靠近
    反复排斥）→ 立即放弃，status="no_progress"（正当绕墙 detour 净位移会持续增大，不会误判）。

    返回 {arrived, dist_m, pose, steps, status}；status ∈ arrived|stuck|no_progress|max_legs。
    """
    steps = []
    side = None              # 'left'/'right' 沿墙手性，撞墙时按 scan 选定后保持
    prev_xy = None
    stuck = 0
    status = "max_legs"
    pos_hist = []            # 近几腿真值位姿：查"一直在动却困在小圈"(弹跳)，区别于正当绕墙(净位移增大)
    for leg in range(max_legs):
        p = _pose(ex)
        dist = math.hypot(tx - p["x"], ty - p["y"])
        if dist <= tol_m:
            status = "arrived"
            break
        # 振荡早停：连续 confine_legs 腿仍困在 confine_r 内(沙发/桌腿间反复排斥) → 放弃(不再枉弹)
        pos_hist.append((p["x"], p["y"]))
        if len(pos_hist) > confine_legs:
            back = pos_hist[-(confine_legs + 1)]
            if math.hypot(p["x"] - back[0], p["y"] - back[1]) < confine_r:
                status = "no_progress"
                break
        # 卡死检测：位置几乎不动(被钉在墙上) → 放弃（detour 中位置一直在变，不会误判）
        if prev_xy is not None:
            moved_xy = math.hypot(p["x"] - prev_xy[0], p["y"] - prev_xy[1])
            stuck = 0 if moved_xy > 0.1 else stuck + 1
            if stuck >= max_stuck:
                status = "stuck"
                break
        prev_xy = (p["x"], p["y"])

        face = geo_face_point(ex, tx, ty)
        front = _scan(ex).get("front_min_m")
        clear = front is None or front >= front_block_m

        if clear:
            # 直冲按前向实测余量封顶（同 VFH geo_step_open）：别一步冲到贴墙触发 safety 退避。
            cap = min(dist, max_step_m)
            if front is not None:
                cap = min(cap, max(0.15, front - clear_margin_m))
            mv = json.loads(ex.ros.call("move", {"distance_m": round(cap, 2)}).text)
            trav = mv.get("traveled_m", 0) or 0
            if mv.get("status") != "safety_stop" and trav >= 0.08:
                steps.append(f"{leg}:direct move={round(trav,2)} dist={round(dist, 2)} pose=({round(p['x'],2)},{round(p['y'],2)})")
                side = None                      # 直冲成功，松开沿墙手性
                continue
            steps.append(f"{leg}:pinned(move {mv.get('status')}/{round(trav,2)})")
        else:
            steps.append(f"{leg}:blocked front={front} pose=({round(p['x'],2)},{round(p['y'],2)})")

        # —— 绕墙：先退一点解钉，再转向开阔侧前进一段（下一轮顶部会重新正对目标）——
        if side is None:
            side, lc, rc = _open_side(ex)
            steps.append(f"{leg}:commit side={side} (L{lc} R{rc})")
        ex.ros.call("move", {"distance_m": -round(backoff_m, 2)})   # 退后解钉
        tool = "turn_left_deg" if side == "left" else "turn_right_deg"
        ex.ros.call(tool, {"degrees": slide_deg})
        # 转向后重感知再封顶滑行步长：别盲滑 wall_step 撞到新墙。
        side_front = _scan(ex).get("front_min_m")
        step = wall_step_m
        if side_front is not None:
            step = min(wall_step_m, max(0.15, side_front - clear_margin_m))
        mvw = json.loads(ex.ros.call("move", {"distance_m": round(step, 2)}).text)
        steps.append(f"{leg}:slide {side} b{backoff_m}+t{slide_deg}+f{round(step, 2)} "
                     f"moved={mvw.get('traveled_m')} {mvw.get('status')}")

    p = _pose(ex)
    dist = math.hypot(tx - p["x"], ty - p["y"])
    arrived = dist <= tol_m
    return {
        "arrived": arrived,
        "dist_m": round(dist, 3),
        "pose": p,
        "steps": steps,
        "status": "arrived" if arrived else status,
    }


_SECTOR8_ANG = {"front": 0, "front_left": 45, "left": 90, "rear_left": 135,
                "rear": 180, "rear_right": -135, "right": -90, "front_right": -45}


def _scan_sectors(ex, sectors: int = 12) -> dict:
    """读 scan_summary(细分扇区) → {中心角(度, 机体系 0=前/+左): 该扇区最近障碍距离 m}。"""
    try:
        s = json.loads(ex.ros.call("scan_summary", {"sectors": sectors}).text)
    except (ValueError, TypeError):
        return {}
    out = {}
    for label, d in (s.get("sectors") or {}).items():
        ang = _SECTOR8_ANG.get(label)
        if ang is None and isinstance(label, str) and label.startswith("sec_"):
            try:
                ang = int(label[4:])
            except ValueError:
                ang = None
        if ang is not None:
            out[ang] = d
    return out


def scan_openings(ex, *, sectors: int = 12, pass_min_m: float = 0.7,
                  fwd_halfwidth_deg: float = 100.0) -> list:
    """单帧 scan → 前向【离散可通行开口】列表（把可通行角聚成簇，供上层判 岔口/走廊/死胡同）。

    复用 _scan_sectors 拿 {角度(体系,0=前/+左): 最近障碍距离|None}；只取前向半圆
    (|归一化角| <= fwd_halfwidth_deg)，按角度排序；连续 dist>=pass_min_m 的扇区聚成一个开口，
    被 <pass_min_m 或无返回(None) 的扇区隔断即分簇。
    返回按 center_deg 排序的 [{center_deg, width_deg, min_clear_m, angles:[...]}]（角度均已归一化）。
    - len==0：死胡同（前向无可通行开口）；==1：单走廊；>=2：岔口（丁字/十字）。
    """
    sc = _scan_sectors(ex, sectors)
    if not sc:
        return []
    bin_deg = 360.0 / max(1, sectors)
    fwd = sorted(
        (_norm_deg(a), d) for a, d in sc.items()
        if abs(_norm_deg(a)) <= fwd_halfwidth_deg
    )
    groups, cur = [], []
    for a, d in fwd:
        if d is not None and d >= pass_min_m:
            cur.append((a, d))
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    out = []
    for grp in groups:
        angs = [a for a, _ in grp]
        ds = [d for _, d in grp]
        out.append({
            "center_deg": round(sum(angs) / len(angs), 1),
            "width_deg": round(max(angs) - min(angs) + bin_deg, 1),
            "min_clear_m": round(min(ds), 2),
            "angles": angs,
        })
    out.sort(key=lambda o: o["center_deg"])
    return out


def opening_for_bearing(openings, des_robot_deg, tol_deg=15.0):
    """目标方位(体系)是否落在某开口的角度覆盖内(两端各放宽半个 bin)；是则返回该开口，否则 None。"""
    for o in openings:
        if min(o["angles"]) - tol_deg <= des_robot_deg <= max(o["angles"]) + tol_deg:
            return o
    return None


def nearest_opening(openings, des_robot_deg):
    """离目标方位最近的开口（几何兜底）；openings 为空返回 None。"""
    if not openings:
        return None
    return min(openings, key=lambda o: abs(_norm_deg(o["center_deg"] - des_robot_deg)))


def geo_step_open(
    ex,
    desired_bearing_deg: float,
    *,
    step_m: float = 1.2,
    clearance_m: float = 0.4,
    pass_min_m: float = 0.7,
    sectors: int = 12,
) -> dict:
    """朝 desired_bearing 走【一安全开阔段】（VFH 式：选离期望方向最近的够开阔扇区，留 clearance 不贴墙）。

    替代 geo_goto 的直线冲撞：每段只走一小步、且封顶到 (扇区最近障碍 - clearance)，故**永不贴墙被 safety 钉死**；
    期望方向被挡时自动选最接近的开阔扇区（如 east 被隔断墙挡→走 NE），由调用方循环逐段重感知，自然绕墙。

    返回 {moved_m, status, pose, chosen_bearing, clear_m, passable, desired_robot}。
    status ∈ moved | safety_stop | no_room | error。
    """
    p = _pose(ex)
    if "x" not in p:
        return {"moved_m": 0.0, "status": "error", "pose": p, "chosen_bearing": None,
                "clear_m": None, "passable": False, "desired_robot": None}
    scan = _scan_sectors(ex, sectors)
    if not scan:
        return {"moved_m": 0.0, "status": "error", "pose": p, "chosen_bearing": None,
                "clear_m": None, "passable": False, "desired_robot": None}
    des_robot = _norm_deg(desired_bearing_deg - p["yaw_deg"])
    # 只考虑前向半圆(±100°)的扇区，避免选到身后倒退
    fwd = {a: d for a, d in scan.items() if abs(_norm_deg(a)) <= 100 and d is not None}
    pool = fwd or {a: d for a, d in scan.items() if d is not None}
    cands = [(a, d) for a, d in pool.items() if d >= pass_min_m]
    if cands:
        chosen, clear = min(cands, key=lambda kv: abs(_norm_deg(kv[0] - des_robot)))
        passable = True
    else:                                   # 全堵：选最开阔的（逃逸/不至于卡死）
        chosen, clear = max(pool.items(), key=lambda kv: kv[1])
        passable = False
    world_head = p["yaw_deg"] + chosen
    diff = _norm_deg(world_head - p["yaw_deg"])
    if abs(diff) > 1.0:
        tool = "turn_left_deg" if diff > 0 else "turn_right_deg"
        ex.ros.call(tool, {"degrees": round(abs(diff), 1)})
    go = max(0.0, min(step_m, (clear or 0.0) - clearance_m))
    moved, status = 0.0, "no_room"
    if go >= 0.15:
        mv = json.loads(ex.ros.call("move", {"distance_m": round(go, 2)}).text)
        moved = mv.get("traveled_m", 0) or 0.0
        status = "safety_stop" if mv.get("status") == "safety_stop" else "moved"
    p2 = _pose(ex)
    return {"moved_m": round(moved, 2), "status": status, "pose": p2,
            "chosen_bearing": round(chosen, 0), "clear_m": round(clear or 0, 2),
            "passable": passable, "desired_robot": round(des_robot, 0)}


def _fwd_clear(beams, des_robot_deg, halfwidth_deg=25.0):
    """朝 des_robot 方向(体系)±halfwidth 内的【最近障碍距离】(米)；无有效束返回 None。

    供 apf_goto 按"前向最近障 − clearance"封顶步长，别一步冲进 safety 刹停带。
    beams: scan_rays 的 [[bearing_deg(体系), dist_m], ...]（-1/None/非正跳过）。
    """
    best = None
    for b in beams or []:
        try:
            deg, d = float(b[0]), float(b[1])
        except (TypeError, ValueError, IndexError):
            continue
        if d <= 0:
            continue
        if abs(_norm_deg(deg - des_robot_deg)) <= halfwidth_deg and (best is None or d < best):
            best = d
    return best


def apf_goto(ex, gx, gy, *, tol_m=0.4, step_m=0.6, max_steps=30,
             d0_m=1.0, clearance_m=0.35, k_att=1.0, eta=0.5, beams=90, max_stuck=4):
    """APF【点到点】导航：引力朝目标 + 斥力离障碍 → 合力航向，逐小步开过去（执行模式用）。

    每步：读活体 scan_rays → potential_field.apf_heading 求合力世界航向 → 面向它 → 前向最近障封顶步长后 move。
    天然穿缝、不把车导离目标（区别于岔口-VFH）。连续位置冻结(<0.1m ×max_stuck) → geo_goto_around 反应式
    脱困（逃 APF 局部极小）。safety_stop 只是本步停、下轮重规划。不碰 geo_step_open/geo_goto_around 本体。

    返回 {arrived, dist_m, pose, steps, status}；status ∈ arrived|arrived_around|stuck|max_steps。
    """
    from agent_core.geometry import potential_field as pf
    prev_xy, stuck, logs = None, 0, []
    for i in range(max_steps):
        p = _pose(ex)
        dist = math.hypot(gx - p["x"], gy - p["y"])
        if dist <= tol_m:
            return {"arrived": True, "dist_m": round(dist, 3), "pose": p, "steps": logs,
                    "status": "arrived"}
        if prev_xy is not None:                               # 卡死检测：位置冻结 → 反应式脱困
            if math.hypot(p["x"] - prev_xy[0], p["y"] - prev_xy[1]) > 0.1:
                stuck = 0
            else:
                stuck += 1
                if stuck >= max_stuck:
                    ga = geo_goto_around(ex, gx, gy, tol_m=tol_m, max_legs=8)
                    pf2 = _pose(ex)
                    dd = math.hypot(gx - pf2["x"], gy - pf2["y"])
                    st = "arrived_around" if ga.get("status") == "arrived" else "stuck"
                    return {"arrived": dd <= tol_m, "dist_m": round(dd, 3), "pose": pf2,
                            "steps": logs, "status": st}
        prev_xy = (p["x"], p["y"])
        try:
            sr = json.loads(ex.ros.call("scan_rays", {"max_beams": beams}).text)
            blist = sr.get("beams") or []
        except (ValueError, TypeError):
            blist = []
        heading = pf.apf_heading(p, gx, gy, blist, k_att=k_att, eta=eta, d0_m=d0_m)
        des_robot = _norm_deg(heading - p["yaw_deg"])
        fclear = _fwd_clear(blist, des_robot)                 # 沿【将走方向】(当前体系)最近障 → 封顶步长
        if abs(des_robot) > 1.0:
            tool = "turn_left_deg" if des_robot > 0 else "turn_right_deg"
            ex.ros.call(tool, {"degrees": round(abs(des_robot), 1)})
        cap = step_m if fclear is None else max(0.15, fclear - clearance_m)
        go = max(0.0, min(step_m, dist, cap))
        moved, status = 0.0, "no_room"
        if go >= 0.15:
            mv = json.loads(ex.ros.call("move", {"distance_m": round(go, 2)}).text)
            moved = mv.get("traveled_m", 0) or 0.0
            status = "safety_stop" if mv.get("status") == "safety_stop" else "moved"
        logs.append(f"{i}:h={round(heading)}deg d={round(dist, 2)} go={round(go, 2)} "
                    f"moved={round(moved, 2)} {status}")
    p = _pose(ex)
    dist = math.hypot(gx - p["x"], gy - p["y"])
    return {"arrived": dist <= tol_m, "dist_m": round(dist, 3), "pose": p, "steps": logs,
            "status": "max_steps"}


def geo_route(ex, waypoints, **kw) -> dict:
    """按顺序 geo_goto 经过一串路点（绕过隔断墙进入目标区）。任一段被挡/safety 即中止并上报。

    waypoints: [(x,y), ...]（不含起点）。返回 {arrived, legs, pose, status}。
    """
    legs = []
    status = "arrived"
    for i, (wx, wy) in enumerate(waypoints):
        leg = geo_goto(ex, wx, wy, **kw)
        legs.append({"wp": [wx, wy], **leg})
        if not leg["arrived"]:
            status = f"stuck@wp{i}:{leg['status']}"
            break
    last = legs[-1] if legs else {"pose": _pose(ex)}
    return {"arrived": status == "arrived", "legs": legs, "pose": last["pose"], "status": status}


def geo_route_around(ex, waypoints, **kw) -> dict:
    """按顺序【geo_goto_around（能绕行）】经过一串路点——门导向 waypoint 桥接的执行原语。

    与 geo_route 的区别：每段用反应式绕行的 geo_goto_around（撞墙不放弃、贴墙滑行），而非直线 geo_goto。
    实证（Step 0.2）：反应式直冲穿不了长墙(墙前横跳)；但"北上走廊→穿开口横轴→南下"的多路点 + 本函数
    逐段绕行，能把车从西侧起点带过 wall(1) 北缝进东侧办公区（[(1.8,1.5),(3.2,1.4),(3.5,-0.6)] 三腿全到）。
    供 explore 覆盖回路：当目标候选在检测到的墙/开口另一侧时，代码据几何构造走廊路点再调本函数。

    waypoints: [(x,y), ...]（不含起点）。任一段未到位即中止并上报。返回 {arrived, legs, pose, status}。
    """
    legs = []
    status = "arrived"
    total_steps = 0
    for i, (wx, wy) in enumerate(waypoints):
        leg = geo_goto_around(ex, wx, wy, **kw)
        ns = len(leg.get("steps") or [])
        total_steps += ns
        legs.append({"wp": [wx, wy], "n_steps": ns,
                     **{k: leg[k] for k in ("arrived", "dist_m", "status", "pose")}})
        if not leg["arrived"]:
            status = f"stuck@wp{i}:{leg['status']}"
            break
    last = legs[-1] if legs else {"pose": _pose(ex)}
    return {"arrived": status == "arrived", "legs": legs, "pose": last["pose"],
            "status": status, "n_steps": total_steps}
