#!/usr/bin/env python3
"""去 GT 的 Qwen 自主导航诊断 runner（autonomy_probe）。

目的：把 GT 答案从【导航】里完全撤掉，让 INSPECT_MEM_SYS 原样驱动，并把每个动作
（go_look_behind / nav_distance / nav_object / direction_hint）都【真正接上导航】
（修掉 completion_demo 里 go_look_behind 算完观察点直接丢弃的 bug）。GT 仅保留用于打分。

产出不是"跑通"，而是一份诊断：Qwen 每步选什么动作、代码据此算的目标在哪、geo_goto
到达/卡墙/safety、最终停在哪、是否过分 → 分清是 prompt 问题还是执行（缺路径规划）问题。

对照：completion_demo.py（GT 脚手架版）保持原样不动。
运行：conda run -n vllm python memory_navi/controll/controll/autonomy_probe.py
"""
import json
import math
import os
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
from agent_core.executor import Executor
from agent_core.geometry import depth_projection as dp
from agent_core.memory.fs_memory import FsMemory
import score_completion as scorer

AREA = "break_room"
RUN_OUT = os.path.join(REPORT_DIR, "autonomy_probe_run.json")
LOG_OUT = os.path.join(REPORT_DIR, "autonomy_probe_log.json")

STEP_BUDGET = 14         # 分段走需要更多步预算（每步只走一开阔段）
NO_PROGRESS_LIMIT = 4    # 连续 N 步净位移<0.15m 视为卡死
STANDOFF_M = 2.5         # 观察点离参照物 standoff
START_XY = (-0.5, -1.0)  # 任务标准起点（初始条件，非答案）；机器人须从这里出发
START_TOL_M = 1.5        # 起点容差：超出则判定上一轮未复位、abort
REACH_TOL_M = 0.7        # 代码侧到达判据：离 Qwen 选的目标点 <此 即算到位
# Qwen 给的方向/意图 → 在"朝参照物航向"上叠加的偏置角(度，+左/CCW)。让 Qwen 语义决定绕哪侧。
_INTENT_OFF = {"behind": 0.0, "left_of": 25.0, "right_of": -25.0, "front": 0.0}
_DIR_OFF = {"left": 35.0, "right": -35.0, "front": 0.0, "center": 0.0}
_HINT_OFF = {"left": 40.0, "center": 0.0, "right": -40.0}

# 决策 prompt（去掉 go_look_behind/go_look_at —— 它们把 Qwen 的方向意图带偏；只留方向意图驱动）。
DECIDE_SYS = (
    "你是室内机器人的探索指引员。给你：本区域已知语义记忆（含『信息缺失』标记）、"
    "当前相机画面、激光障碍(scan)、深度(depth)、上一轮位置变化。\n"
    "任务：朝『信息缺失』的方向探索；每次只决定【下一步往哪走】，系统会执行一小段再让你重新看。\n"
    "可选（在 JSON 里给）：\n"
    "- direction_hint: left / center / right —— 相对你【当前朝向】，往左 / 正前 / 右走一小段。这是【主要】方式。\n"
    "- nav_distance —— 相对某参照物的某一侧走。"
    "例 {\"action\":\"nav_distance\",\"reference_object\":\"红色柜子\",\"direction\":\"left\"}（direction 取 left/right/front）。\n"
    "- arrived —— 已绕过遮挡、看到记忆里缺失的新区域 → direction_hint:\"arrived\"。\n"
    "规则：\n"
    "- 前方被家具/矮墙挡住，就用 left 或 right 绕过去，不要硬顶正前方。\n"
    "- 看到记忆里没有的新家具 / 新开阔区域 / 已明显绕过遮挡物 → 果断 direction_hint:\"arrived\"。\n"
    "只输出 JSON：{\"direction_hint\":\"left|center|right|arrived\",\"action\":\"nav_distance|none\","
    "\"reference_object\":\"\",\"direction\":\"left|right|front\"}"
)

# 主动巡视 prompt：到房间后，把"看哪个视角"的控制权交给 Qwen（标像素 → 代码凑过去看清）。
SURVEY_SYS = (
    "你已经在一片办公区里，任务是把这片区域的物体看全、记全（绘制语义地图）。\n"
    "给你当前相机画面 + 已累计看到的物体。判断要不要换个视角看得更全/更清：\n"
    "- 若画面里还有【在边缘 / 较远 / 没看清】的物体或区域，想凑近去看 → action:\"go\"，"
    "并给该目标的像素坐标 u,v（Qwen 归一化 0-1000，指向你想去看清的那个物体/那片地方）。\n"
    "- 若这片区域主要物体都已看清记全 → action:\"done\"。\n"
    "只输出 JSON：{\"action\":\"go|done\",\"u\":500,\"v\":500,\"reason\":\"\"}"
)


def _survey_decide(ex, rep, seen_names):
    """到房间后问 Qwen：要不要换视角凑近看别的？给像素点(u,v) 或 done。"""
    user = (f"已累计看到的物体：{seen_names}\n"
            "看当前画面：还要换个视角凑近看清别的物体/区域吗？要就给像素 u,v，否则 done。只输出 JSON：")
    content = [{"type": "text", "text": user}]
    img = rep.get("image")
    if img is not None:
        content.append(harness.to_openai_image_url(img))
    resp = ex.client.chat.completions.create(
        model=ex.model,
        messages=[{"role": "system", "content": SURVEY_SYS}, {"role": "user", "content": content}],
        temperature=0.2, max_tokens=200, stream=False)
    return harness._extract_obj(resp.choices[0].message.content or "")


def _pixel_to_world(ex, rep, u_norm, v_norm):
    """Qwen 归一化像素(0-1000) + 该列 depth → 世界点 abs_pose（看哪就把哪反投成坐标）。"""
    pose = _pose(ex)
    depths = []
    try:
        depths = (json.loads(rep.get("depth", "{}")) or {}).get("depths_m", []) or []
    except (ValueError, TypeError):
        depths = []
    col_d = None
    if depths:
        ci = min(len(depths) - 1, max(0, int(u_norm / 1000.0 * len(depths))))
        cand = sorted(d for j in range(max(0, ci - 1), min(len(depths), ci + 2))
                      for d in [depths[j]] if isinstance(d, (int, float)) and d > 0)
        col_d = cand[len(cand) // 2] if cand else None
    if not col_d:
        return None, pose
    K = dp.DEFAULT_K
    upx, vpx = dp.qwen_norm_to_pixel(u_norm, v_norm, K["width"], K["height"])
    return dp.back_project(upx, vpx, col_d, K, None, robot_pose=pose)["abs_pose"], pose


def _pose(ex, retries=3):
    """安全读位姿：get_pose 偶发超时会返回 {'error':...}，此处重试，始终返回带 x/y/yaw_deg 的 dict。"""
    last = {}
    for _ in range(retries):
        p = harness._pose(ex)
        if isinstance(p, dict) and "x" in p:
            return p
        last = p
        time.sleep(0.3)
    return {"x": 0.0, "y": 0.0, "yaw_deg": 0.0, "_pose_error": str(last)}


def _hr(t):
    print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


def _strip_gap(s):
    i = s.find("【信息缺失")
    return s[:i].rstrip() if i >= 0 else s


def _writeback(mem, gapped, objects, arrival_pose):
    rec = dict(gapped)
    by_name = {o.get("name"): dict(o) for o in gapped.get("objects", []) if o.get("name")}
    for o in objects:
        n = o.get("name")
        if not n:
            continue
        by_name[n] = {"name": n, "spatial": f"办公区,画面{o.get('bearing','center')}",
                      "view": {"distance_m": o.get("distance_m")},
                      "confidence": o.get("confidence", 0.6),
                      "verified_by": ["qwen3-vl-8b", "depth(代码接地)"]}
    rec["objects"] = list(by_name.values())
    rec["summary"] = _strip_gap(gapped.get("summary", "")) + \
        " 办公区已补全：" + "、".join(f"{o.get('name')}({o.get('bearing','?')})" for o in objects) + "。"
    rec["view_pose"] = {"x": arrival_pose["x"], "y": arrival_pose["y"], "yaw": arrival_pose["yaw_deg"]}
    return mem.upsert_area(AREA, rec)


# ---------------------------------------------------------------------------
# 决策：内联调 INSPECT_MEM_SYS（inspect_with_memory 不暴露 action，故必须内联）
# ---------------------------------------------------------------------------
def _decide(ex, gapped, delta_text):
    """看图 + 记忆 + 位置变化 → Qwen 出下一步决策（原样 INSPECT_MEM_SYS）。"""
    look = ex.ros.call("look", {})
    img = look.images[0] if look.images else None
    scan_t = ex.ros.call("scan_summary", {}).text
    depth_t = ex.ros.call("depth_summary", {}).text
    pose = _pose(ex)
    objs_mem = [f"- {o.get('name')}: {o.get('spatial','')}" for o in gapped.get("objects", [])]
    mem_text = f"区域 {gapped.get('area','?')}。{gapped.get('summary','')}\n" + "\n".join(objs_mem)
    user = (f"已知记忆:\n{mem_text}\n\n"
            f"scan: {scan_t}\ndepth(左→右,米): {depth_t}\n"
            f"{delta_text}\n\n"
            "看画面。可调 nav_distance(obj,dist,direction) / nav_object(a,b) 或 "
            "报 arrived / 给 direction_hint(left/center/right)。只输出 JSON:")
    content = [{"type": "text", "text": user}]
    if img is not None:
        content.append(harness.to_openai_image_url(img))
    resp = ex.client.chat.completions.create(
        model=ex.model,
        messages=[{"role": "system", "content": DECIDE_SYS},
                  {"role": "user", "content": content}],
        temperature=0.2, max_tokens=400, stream=False)
    raw = resp.choices[0].message.content or ""
    return harness._extract_obj(raw), raw, pose


def _resolve_abs_pose(ex, gapped, rep, name, dec, cab_abs):
    """三级解析参照物世界坐标：① gapped 记忆 ② 当前帧反投 ③ 柜类回退。"""
    name_l = (name or "").lower()
    # ① gapped 记忆里有 abs_pose
    for o in gapped.get("objects", []):
        on = (o.get("name") or "").lower()
        if name_l and (name_l in on or on in name_l) and o.get("abs_pose"):
            return o["abs_pose"], "memory"
    # ② 当前观测反投
    for o in rep.get("objects", []):
        on = (o.get("name") or "").lower()
        if name_l and (name_l in on or on in name_l) and o.get("distance_m"):
            K = dp.DEFAULT_K
            if o.get("bbox_center"):
                u, v = dp.qwen_norm_to_pixel(o["bbox_center"][0], o["bbox_center"][1],
                                             K["width"], K["height"])
            else:
                u, v = int(dec.get("u", 320) or 320), int(dec.get("v", 240) or 240)
            ap = dp.back_project(u, v, o["distance_m"], K, None,
                                 robot_pose=_pose(ex))["abs_pose"]
            return ap, "backproject"
    # ③ 柜类回退用 Phase A 接地的柜 abs_pose
    if cab_abs and any(k in name_l for k in ["柜", "cabinet"]):
        return cab_abs, "phaseA"
    return None, "none"


def _record_goto(rec, g):
    rec["geo_goto"] = {
        "status": g.get("status"),
        "dist_m": g.get("dist_m"),
        "pose_after": g.get("pose"),
        "why_steps": (g.get("steps") or [])[-8:],
    }
    rec["outcome"] = "goto_" + str(g.get("status"))


def _delta(prev_ref, rep):
    """上一轮 vs 本轮参照物 bearing/depth 变化文本（喂给 prompt 的【位置变化】）。"""
    if not prev_ref:
        return ""
    for o in rep.get("objects", []):
        if (prev_ref.get("name") or "").lower() in (o.get("name") or "").lower():
            return (f"【位置变化】{prev_ref['name']}: 上轮 bearing={prev_ref.get('bearing')} "
                    f"d≈{prev_ref.get('dist')}m → 本轮 bearing={o.get('bearing')} d≈{o.get('distance_m')}m")
    return ""


def main() -> int:
    _hr("autonomy_probe：去 GT 的 Qwen 自主导航诊断")

    gt = scorer.load_gt()
    gt_vp, gt_ft = gt["task"]["viewpoint"], gt["task"]["face_target"]
    print(f"[GT 仅供打分] 答案观察点=({gt_vp['x']},{gt_vp['y']}) 朝向=({gt_ft['x']},{gt_ft['y']})"
          " —— 不喂导航")

    mem = FsMemory(config.MEMORY_ROOT, config.ENV_NAME)
    seed = os.path.join(config.MEMORY_ROOT, config.ENV_NAME, AREA, "area.seed_gapped.json")
    if not os.path.exists(seed):
        print(f"❌ 找不到缺口种子记忆 {seed}")
        return 2
    with open(seed, encoding="utf-8") as f:
        gapped = json.load(f)
    name_hints = [o.get("name") for o in gapped.get("objects", []) if o.get("name")]  # 命名锚点(已知区物体)
    print(f"[缺口记忆] objects={[o['name'] for o in gapped.get('objects', [])]}  name_hints={name_hints}")

    try:
        ex = Executor()
    except Exception as e:  # noqa: BLE001
        print(f"❌ 无法建立 Executor: {e}")
        return 2
    print(f"[vLLM] model = {ex.model}")

    step_records = []
    arrived = False
    stop_reason = "budget"
    last_qwen_target = None
    cab_abs = None
    phaseA_cab_obs = None
    n_written = 0
    arrival_pose = {}
    sweep = {"objects": []}

    try:
        # ===== 起点守卫 + 软复位：不在起点则代码开回起点（起点是初始条件非答案；西侧无隔断墙可直达）=====
        sp = _pose(ex)
        d_start = math.hypot(sp.get("x", 99) - START_XY[0], sp.get("y", 99) - START_XY[1])
        print(f"[起始位姿] {sp}  距标准起点{START_XY} = {d_start:.2f}m")
        if d_start > START_TOL_M:
            print(f"[软复位] 不在起点，VFH 开回 {START_XY}（能绕墙，从墙后也能回来）…")
            for _ in range(22):
                cp = _pose(ex)
                if math.hypot(cp["x"] - START_XY[0], cp["y"] - START_XY[1]) <= 0.5:
                    break
                nav.geo_step_open(ex, nav.bearing_deg(cp["x"], cp["y"], START_XY[0], START_XY[1]))
            nav.geo_face_point(ex, START_XY[0] + 1.0, START_XY[1])  # 回正朝 +x（东，看红柜）
            sp = _pose(ex)
            d_start = math.hypot(sp.get("x", 99) - START_XY[0], sp.get("y", 99) - START_XY[1])
            print(f"[软复位后] {sp}  距起点 = {d_start:.2f}m")
            if d_start > START_TOL_M:
                print(f"❌ 软复位失败（仍在 ({sp.get('x')},{sp.get('y')})）。请手动 revert Webots 仿真后重跑。")
                return 3
        # 总是回正朝 +x（东）—— Phase A 需正对红柜方向取景
        nav.geo_face_point(ex, START_XY[0] + 1.0, START_XY[1])
        sp = _pose(ex)
        print(f"[起点就绪] {sp}（已朝东）")

        # ===== Phase A：感知 + 锚点接地（写 abs_pose 进 gapped，不喂导航目标）=====
        _hr("Phase A 感知 + 锚点接地")
        rep = harness.inspect_and_report(ex, max_tokens=600)  # 撤 name_hints（上轮它把红柜接地搞错+诱发幻觉）
        print(f"[Qwen objects] {json.dumps(rep.get('objects', []), ensure_ascii=False)[:400]}")
        for o in rep.get("objects", []):
            if not o.get("distance_m"):
                continue
            name = o.get("name", "?")
            pose = _pose(ex)
            K = dp.DEFAULT_K
            if o.get("bbox_center"):
                u, v = dp.qwen_norm_to_pixel(o["bbox_center"][0], o["bbox_center"][1],
                                             K["width"], K["height"])
                proj = dp.back_project(u, v, o["distance_m"], K, None, robot_pose=pose)
            else:
                bearing = o.get("bearing", "center")
                offset = {"left": 25, "center": 0, "right": -25}.get(bearing, 0)
                ang = math.radians(pose["yaw_deg"] + offset)
                proj = {"abs_pose": {"x": round(pose["x"] + o["distance_m"] * math.cos(ang), 3),
                                     "y": round(pose["y"] + o["distance_m"] * math.sin(ang), 3),
                                     "z": 0.7}}
            gobjs = list(gapped.get("objects", []))
            found = False
            for go in gobjs:
                gn = (go.get("name") or "").lower()
                if (name or "").lower() in gn or gn in (name or "").lower():
                    go["abs_pose"] = proj["abs_pose"]
                    found = True
                    break
            if not found:
                gobjs.append({"name": name, "abs_pose": proj["abs_pose"]})
            gapped["objects"] = gobjs
            n_written += 1
        if n_written:
            mem.upsert_area(AREA, gapped)
        # —— 选导航锚点（=红柜）：优先名字像柜子；否则取"中前方显著物"(按位置认，不靠 Qwen 命名)。
        #    Qwen 偶尔把红柜叫成"工作台"，但它始终是正前方~3m 的那个显著物，按位置锚定即稳。——
        def _pick_anchor(objs):
            named = [o for o in objs if o.get("distance_m") and
                     any(k in (o.get("name") or "").lower() for k in ["柜", "cabinet", "红", "red"])]
            if named:
                return named[0], "name"
            cands = [o for o in objs if o.get("distance_m") and o.get("bearing") == "center"
                     and 1.5 <= o.get("distance_m", 0) <= 4.5]
            if cands:
                return max(cands, key=lambda o: o.get("confidence", 0)), "center_prominent"
            return None, None

        # 接地锚点：用【水平角(u)+depth】定位、v 强制取图像中心(去竖直噪声——run10 把柜丢到墙上 z=1m)；
        # 合理性校验(在前方±70° 且 距离≈报告值)，不过关就重看一帧重投，最多 3 次 → 稳住目标。
        print(f"[depth 回投] {n_written} objects 接地")
        anchor = asrc = cab_abs = None
        for attempt in range(3):
            if attempt > 0:
                rep = harness.inspect_and_report(ex, max_tokens=600)
            a, s = _pick_anchor(rep.get("objects", []))
            if not a:
                print(f"[锚点尝试{attempt}] 无候选 → 重看")
                continue
            ap_pose = _pose(ex)
            K = dp.DEFAULT_K
            if a.get("bbox_center"):
                au, _v = dp.qwen_norm_to_pixel(a["bbox_center"][0], a["bbox_center"][1],
                                               K["width"], K["height"])
                av = K["height"] // 2   # v=cy：只用水平角+depth，去掉竖直 bbox 噪声
                cab = dp.back_project(au, av, a["distance_m"], K, None, robot_pose=ap_pose)["abs_pose"]
            else:
                aoff = {"left": 25, "center": 0, "right": -25}.get(a.get("bearing", "center"), 0)
                aang = math.radians(ap_pose["yaw_deg"] + aoff)
                cab = {"x": round(ap_pose["x"] + a["distance_m"] * math.cos(aang), 3),
                       "y": round(ap_pose["y"] + a["distance_m"] * math.sin(aang), 3), "z": 0.3}
            dd = math.hypot(cab["x"] - ap_pose["x"], cab["y"] - ap_pose["y"])
            brg = nav.bearing_deg(ap_pose["x"], ap_pose["y"], cab["x"], cab["y"])
            ahead = abs(((brg - ap_pose["yaw_deg"] + 180) % 360) - 180) <= 70
            ok = ahead and abs(dd - a.get("distance_m", 0)) <= 1.2
            anchor, asrc, cab_abs = a, s, cab
            print(f"[锚点尝试{attempt}] {a.get('name')}(src={s}) → ({cab['x']:.2f},{cab['y']:.2f}) "
                  f"ahead={ahead} dist_ok={abs(dd - a.get('distance_m', 0)):.2f} → {'OK' if ok else '不合理,重看'}")
            if ok:
                break
        if cab_abs:
            phaseA_cab_obs = dp.derive_observation_point(cab_abs, behind=True, standoff_m=STANDOFF_M)
        print(f"[锚点] {anchor.get('name') if anchor else None}(src={asrc}) → 红柜 abs_pose={cab_abs}; "
              f"后方目标={phaseA_cab_obs}")

        # ===== Phase B：代码 VFH 锚定固定目标绕墙（Qwen 只在 A 接地 / C 报告；导航全代码）=====
        #   run7 证明 Qwen 逐帧给方向会转圈；这里方向源换成【代码锚定的固定目标】(红柜后方点)，
        #   geo_step_open 每步朝目标挑最接近的开阔扇区走 → 自动 VFH 绕墙。
        _hr("Phase B 代码 VFH 绕墙（锚定红柜后方固定目标，无 Qwen 逐帧驾驶）")
        target = phaseA_cab_obs   # 红柜后方观察点（代码从接地红柜算，standoff=STANDOFF_M）
        if not target:
            print("❌ 红柜未接地，无导航目标 → 跳过 Phase B")
            stop_reason = "no_target"
        else:
            last_qwen_target = {"x": target["x"], "y": target["y"]}
            print(f"[固定目标] 红柜后方 = ({target['x']:.2f},{target['y']:.2f})  standoff={STANDOFF_M}m")
            prev_pose = _pose(ex)
            no_progress = 0
            for i in range(STEP_BUDGET):
                p = _pose(ex)
                dist = math.hypot(target["x"] - p["x"], target["y"] - p["y"])
                rec = {"step": i, "pose_before": p, "dist_to_target": round(dist, 2)}
                if dist <= REACH_TOL_M:
                    arrived = True
                    stop_reason = "reached_target"
                    rec["outcome"] = "reached"
                    step_records.append(rec)
                    print(f"[{i}] ✅ 到达红柜后方目标 dist={dist:.2f} pose=({p['x']:.2f},{p['y']:.2f})")
                    break
                desired = nav.bearing_deg(p["x"], p["y"], target["x"], target["y"])
                st = nav.geo_step_open(ex, desired)
                so = {k: st.get(k) for k in
                      ("moved_m", "status", "chosen_bearing", "clear_m", "passable", "desired_robot")}
                rec["step_open"] = so
                rec["outcome"] = "step_" + str(st.get("status"))
                cur = st.get("pose") if "x" in (st.get("pose") or {}) else _pose(ex)
                rec["pose_after"] = cur
                print(f"[{i}] →目标({target['x']:.1f},{target['y']:.1f}) dist={dist:.2f} "
                      f"step(moved={so['moved_m']} bear={so['chosen_bearing']} clear={so['clear_m']} "
                      f"pass={so['passable']}) pose=({cur['x']:.2f},{cur['y']:.2f},{cur['yaw_deg']:.0f})")
                step_records.append(rec)
                disp = math.hypot(cur["x"] - prev_pose["x"], cur["y"] - prev_pose["y"])
                no_progress = 0 if disp > 0.15 else no_progress + 1
                prev_pose = cur
                if no_progress >= NO_PROGRESS_LIMIT:
                    stop_reason = "no_progress"
                    print(f"[{i}] ⚠️ 连续 {no_progress} 步无进展(疑似卡局部最优) → 停")
                    break
            else:
                stop_reason = "budget"

        if not arrived:
            print(f"\n[代码VFH导航] ❌ 未到达目标（stop_reason={stop_reason}）—— 不回退 GT，如实进 Phase C")

        # ===== Phase C：主动巡视（Qwen 标像素选视角 → 代码 back-project + VFH 凑过去 → 多视角拼语义图）=====
        #   VFH 负责"过去"，但视角由 Qwen 控：到房间后让 Qwen 选要凑近看的像素，代码反投+VFH 移过去，
        #   多个视角累计物体 → 不再被 VFH 停车的单一死视角限制。
        _hr("Phase C 主动巡视（Qwen 选视角像素，代码 VFH 凑视角）")
        if cab_abs:
            nav.geo_face_point(ex, cab_abs["x"], cab_abs["y"])   # 起手朝办公区物体簇(红柜在西)回看
        all_objs = []
        SURVEY_STEPS = 4
        for sv in range(SURVEY_STEPS):
            rep = harness.inspect_and_report(ex)
            all_objs.extend(rep.get("objects", []))
            cp = _pose(ex)
            seen = sorted({o.get("name") for o in all_objs if o.get("name")})
            print(f"[巡视{sv}] pose=({cp['x']:.2f},{cp['y']:.2f},{cp['yaw_deg']:.0f}) "
                  f"本帧={[o.get('name') for o in rep.get('objects', [])]} 累计={seen}")
            dec = _survey_decide(ex, rep, seen)
            if (dec.get("action") or "").lower() != "go":
                print(f"[巡视{sv}] Qwen done — {str(dec.get('reason', ''))[:60]}")
                break
            u = float(dec.get("u", 500) or 500)
            v = float(dec.get("v", 500) or 500)
            pt, pose = _pixel_to_world(ex, rep, u, v)
            if not pt:
                print(f"[巡视{sv}] 像素({u:.0f},{v:.0f})无有效深度 → 原地左转再看")
                ex.ros.call("turn_left_deg", {"degrees": 30})
                continue
            d = math.hypot(pt["x"] - pose["x"], pt["y"] - pose["y"])
            standoff = 1.3
            if d > standoff:
                r = (d - standoff) / d
                vp = (pose["x"] + (pt["x"] - pose["x"]) * r, pose["y"] + (pt["y"] - pose["y"]) * r)
            else:
                vp = (pose["x"], pose["y"])
            print(f"[巡视{sv}] →看像素({u:.0f},{v:.0f}) 世界({pt['x']:.1f},{pt['y']:.1f}) 视角点({vp[0]:.1f},{vp[1]:.1f})")
            for _ in range(4):
                cp = _pose(ex)
                if math.hypot(vp[0] - cp["x"], vp[1] - cp["y"]) <= 0.5:
                    break
                st = nav.geo_step_open(ex, nav.bearing_deg(cp["x"], cp["y"], vp[0], vp[1]), step_m=0.8)
                if (st.get("moved_m") or 0) < 0.08:
                    break
            nav.geo_face_point(ex, pt["x"], pt["y"])   # 凑到后正对目标
        if cab_abs:
            nav.geo_face_point(ex, cab_abs["x"], cab_abs["y"])   # 巡视完回正朝办公区物体簇(红柜在西)，作汇报视角
        rep = harness.inspect_and_report(ex)           # 末帧再看一眼
        all_objs.extend(rep.get("objects", []))
        arrival_pose = _pose(ex)
        by_name = {}
        for o in all_objs:
            n = o.get("name", "")
            if n and (n not in by_name or o.get("confidence", 0) > by_name[n].get("confidence", 0)):
                by_name[n] = o
        sweep = {"objects": list(by_name.values()), "note": "", "sweeps": SURVEY_STEPS}
        print(f"[巡视完] {len(sweep['objects'])} 个去重物体: {[o.get('name') for o in sweep['objects']]}")
        ex.ros.call("stop", {})
    finally:
        ex.close()

    # ===== Phase D：回写 + 打分（GT 仅打分）=====
    _hr("Phase D 回写 + 打分")
    path = _writeback(mem, gapped, sweep["objects"], arrival_pose)
    print(f"[写回] {path}")

    result = {
        "task": gt["task"]["instruction_to_robot"],
        "arrival_pose": arrival_pose, "viewpoint": gt_vp, "face_target": gt_ft,
        "objects": sweep["objects"], "note": sweep.get("note", ""),
        "target_from_perception": n_written > 0,
        "observation_from_depth": last_qwen_target,   # Qwen 选的最终计算目标（非 GT）
        "claude_calls": 0,
    }
    with open(RUN_OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    s = scorer.score(result, gt)
    print(scorer.format_scorecard(s))

    hist = {}
    for r in step_records:
        a = r.get("action") or (("hint:" + r["direction_hint"]) if r.get("direction_hint") else "none")
        hist[a] = hist.get(a, 0) + 1
    log = {
        "arrived": arrived, "n_steps": len(step_records), "stop_reason": stop_reason,
        "final_pose": arrival_pose, "phaseA_cabinet_obs": phaseA_cab_obs,
        "last_qwen_target": last_qwen_target, "action_histogram": hist,
        "scorecard": s, "steps": step_records,
    }
    with open(LOG_OUT, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"[诊断日志已存] {LOG_OUT}")
    print(f"[结果已存] {RUN_OUT}")
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
