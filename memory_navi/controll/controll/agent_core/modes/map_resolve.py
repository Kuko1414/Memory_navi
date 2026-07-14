"""执行模式的【地图解析】纯函数（几何归代码、无 ROS/LLM/mcp 依赖 → 可 base-pytest 直测）。

从 execution.py 搬出，避免 `import harness`→mcp 让这些纯逻辑无法离线测（仿 navigator 迁移开口辅助）。
职责：把 Claude 的语义 via/manner + 已知目标坐标，解析成机器人可导航的世界落点；并提供
"这条 leg 是不是就是目标物本身"、"这条 leg 是否把车带离目标" 的判据，供执行循环丢弃/护栏。
"""
import math

from agent_core.geometry import depth_projection as dp
from agent_core.memory.fs_memory import _abs_dist, _norm_name, match_object


# ---- 子区 / 地标 ----
def _sa_center(sa):
    r = sa.get("range") or {}
    if r.get("xmin") is None:
        return None
    return {"x": round((r["xmin"] + r["xmax"]) / 2, 2), "y": round((r["ymin"] + r["ymax"]) / 2, 2)}


def _landmarks(rec, cap=22):
    """地图代表地标(name+坐标，0.7m 粗去重)，供 Claude 规划 & 代码解析。"""
    out, seen = [], []
    for o in rec.get("objects", []):
        ap = o.get("abs_pose") or {}
        if ap.get("x") is None:
            continue
        if any(math.hypot(ap["x"] - s[0], ap["y"] - s[1]) < 0.7 and s[2] == o.get("name")
               for s in seen):
            continue
        seen.append((ap["x"], ap["y"], o.get("name")))
        out.append({"name": o.get("name"), "x": round(ap["x"], 2), "y": round(ap["y"], 2)})
        if len(out) >= cap:
            break
    return out


def _via_candidates(rec, via):
    """把 Claude 的 via（子区 label / 门 label|id / 物体名）解析成候选落点。返回 [{name,x,y,src}]。

    解析顺序：子区 label → 门 label/id(门到门路由) → 物体名/别名。门为单点落点(src='door')，
    执行循环会以门 pose 作为"穿开口"子目标。
    """
    nv = _norm_name(via)
    for sa in rec.get("sub_areas", []):
        if _norm_name(sa.get("label")) == nv:
            c = _sa_center(sa)
            if c:
                return [{"name": sa["label"], "x": c["x"], "y": c["y"], "src": "sub_area"}]
    for d in rec.get("doors", []):
        pose = d.get("pose") or {}
        if pose.get("x") is None:
            continue
        labels = [d.get("label"), d.get("id")]
        if any(_norm_name(n) == nv for n in labels if n):
            return [{"name": d.get("label") or d.get("id") or "门",
                     "x": round(pose["x"], 2), "y": round(pose["y"], 2), "src": "door"}]
    out = []
    for o in rec.get("objects", []):
        ap = o.get("abs_pose") or {}
        if ap.get("x") is None:
            continue
        names = [o.get("name")] + (o.get("aliases") or [])
        if any(_norm_name(n) == nv or nv in _norm_name(n) or _norm_name(n) in nv
               for n in names if n):
            out.append({"name": o.get("name"), "x": round(ap["x"], 2), "y": round(ap["y"], 2),
                        "src": "object", "id": o.get("id")})
    return out


def _manner_point(base, manner, from_pose, standoff=0.9, lateral=1.0):
    """把 (base 地标, manner 方位) 解析成机器人落脚点（几何归代码）。"""
    bx, by = base["x"], base["y"]
    ang = math.atan2(by - from_pose["y"], bx - from_pose["x"])
    if manner == "behind":
        d = dp.derive_observation_point({"x": bx, "y": by, "z": 0.5}, behind=True,
                                        standoff_m=standoff + 0.5)
        return (round(d["x"], 2), round(d["y"], 2))
    if manner in ("left", "right"):
        perp = ang + (math.pi / 2 if manner == "left" else -math.pi / 2)
        px = bx - standoff * math.cos(ang) + lateral * math.cos(perp)
        py = by - standoff * math.sin(ang) + lateral * math.sin(perp)
        return (round(px, 2), round(py, 2))
    return (round(bx - standoff * math.cos(ang), 2), round(by - standoff * math.sin(ang), 2))


def _bearing_from(pose, x, y):
    a = math.degrees(math.atan2(y - pose["y"], x - pose["x"])) - pose["yaw_deg"]
    return (a + 180) % 360 - 180


# ---- leg 判据（根治 leg5：目标物本身的 leg 交末段；误解析 leg 护栏）----
def leg_hits_target(cands, tgt, tol=0.8):
    """这条 leg 解析出的候选里，是否有实例就是当前目标物本身（同名且 abs_pose≤tol）。

    是 → 执行循环丢弃该 leg（末段本就用已知 tgt 坐标逼近），杜绝 Qwen 在多实例里盲选目标(leg5 根因)。
    """
    tname = _norm_name(tgt.get("name"))
    tpose = {"x": tgt.get("x"), "y": tgt.get("y")}
    for c in cands or []:
        if _norm_name(c.get("name")) != tname:
            continue
        d = _abs_dist(tpose, {"x": c.get("x"), "y": c.get("y")})
        if d is not None and d <= tol:
            return True
    return False


def door_leg_redundant(pose, door_xy, tgt, edge_m=0.3):
    """机器人与目标是否【已在门同侧】→ 穿门无意义，执行循环应跳过该门 leg（治无谓绕路）。

    竖隔断假设（理想图两门均竖墙 x=±2.65）：门把空间沿 x 分两侧，`sign(robot.x−门.x)==sign(target.x−门.x)`
    且两者都离门 >edge_m ⇒ 同侧。上次失败正是起点已在目标侧(东)却仍去西边的门绕路。
    （非竖隔断/多轴分隔的一般情形留待后续；当前实验足够。）
    """
    dx = float((door_xy or [0, 0])[0])
    rx, tx = float(pose.get("x", 0.0)), float(tgt.get("x", 0.0))
    if abs(rx - dx) <= edge_m or abs(tx - dx) <= edge_m:
        return False
    return (rx - dx) * (tx - dx) > 0


def leg_leads_away(sx, sy, pose, tgt, margin=1.5):
    """这条 leg 的落点是否把车带离已知目标（落点离 tgt 比当前 pose 还远 margin 以上）＝误解析。

    是 → 执行循环跳过该 leg（中间 leg 兜底护栏；正当中转不会离目标更远这么多）。
    """
    d_leg = math.hypot(sx - tgt.get("x", sx), sy - tgt.get("y", sy))
    d_now = math.hypot(pose.get("x", sx) - tgt.get("x", sx), pose.get("y", sy) - tgt.get("y", sy))
    return d_leg > d_now + margin


# ---- 目标归一化（补记忆 id 供末段身份确认门比对）----
def _resolve_targets(rec, params):
    """归一化任务目标为 [{name,x,y,manner,desc,id}]。

    优先 params['targets']（执行实验：坐标已知的理想图目标），每个目标用 match_object 按坐标+名反查
    其记忆 id（gt_25/gt_27），供末段"这确实是 gt_25"确认；否则回退单目标子区语义解析。
    """
    objs = rec.get("objects", [])
    tgts = params.get("targets")
    if tgts:
        out = []
        for t in tgts:
            x, y = float(t["x"]), float(t["y"])
            name = t.get("name", "目标")
            m = match_object(objs, {"x": x, "y": y}, name)
            out.append({"name": name, "x": x, "y": y,
                        "manner": (t.get("manner") or "near").lower(),
                        "desc": t.get("desc", f"到『{name}』附近"),
                        "id": (m or {}).get("id")})
        return out
    # 回退：单目标子区解析
    subarea = params.get("target_subarea")
    name = params.get("target_name")
    manner = (params.get("target_manner") or "near").lower()
    if not name:
        return []
    sa = next((s for s in rec.get("sub_areas", [])
               if _norm_name(s.get("label")) == _norm_name(subarea or "")), None)
    member_ids = set(sa.get("member_ids", [])) if sa else set()
    cands = [o for o in objs
             if _norm_name(o.get("name")) == _norm_name(name)
             and (o.get("id") in member_ids if member_ids else True)
             and (o.get("abs_pose") or {}).get("x") is not None]
    if not cands:
        return []
    tgt = max(cands, key=lambda o: o["abs_pose"]["x"])
    ap = tgt["abs_pose"]
    return [{"name": name, "x": ap["x"], "y": ap["y"], "manner": manner,
             "desc": params.get("target_desc") or f"到『{name}』的{manner}",
             "id": tgt.get("id")}]
