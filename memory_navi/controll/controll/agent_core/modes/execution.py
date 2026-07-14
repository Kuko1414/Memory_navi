"""执行模式（Mode 封装）：区域到区域的语义拓扑导航 + 末段靠近目标物。

组合复用：
- semantic_route 的骨架——Claude `plan_route` 读整理后地图规划语义路线(via+manner/门，不出坐标)，
  Qwen 逐段选地标实例，代码 `nav.apf_goto`(APF 点到点) 逐段避障推进。执行模式是点到点问题
  （起点+航点+目标坐标皆已知），用 APF（引力朝目标+斥力离障碍、天然穿缝）而非探索/补全的岔口-VFH。
- complete_target 的末段——到位 `geo_face_point` + `sweep_observe` 观测确认 + 身份接地确认门。

扩展（新执行模式核心）：
- 支持【有序多目标】：ctx.params["targets"] = [{name,x,y,manner}, …]，逐个"规划路线→逐段导航→
  末段靠近"，动作类比"从 A 区拿东西到 B 区"。
- 末段【安全兼容】到位：安全/导航层不可绕过(front_block 0.4m / clearance 0.4m / safety 0.15m 硬停)，
  物理到不了 0.1m。到达 = 表面距离 ≤ arrival_surface_tol(默认 0.45m)，表面距离 =
  hypot(car,obj) − car_half − obj_radius；每目标记录【实测最小表面距离】作为连续指标。

角色分工不变：几何/坐标归代码，语义(选哪个地标/该朝哪推进)归 Qwen，路线规划归 Claude。
memory_access='read'（只读地图，从不写）；uses_cloud=True（需 director.plan_route）。
"""
import math
import os
import shutil

from agent_core import config, harness
from agent_core import navigator as nav
from agent_core.geometry import depth_projection as dp
from agent_core.memory.fs_memory import match_object
from agent_core.modes.base import Mode, ModeResult, RunContext
from agent_core.modes.map_resolve import (
    _sa_center, _landmarks, _via_candidates, _manner_point, _bearing_from,
    _resolve_targets, leg_hits_target, leg_leads_away, door_leg_redundant,
)

_DEFAULTS = {
    "arrival_surface_tol_m": 0.45,     # 安全兼容到位阈值（表面距离）
    "car_half_m": 0.16,                # 小车半体尺（robomaster 约 0.32×0.24）
    "obj_radius_m": 0.20,              # 目标物近似半径（绿植/柜等，params 可调）
    "final_standoff_surf_m": 0.20,     # 末段停在离物体【表面】≥此值处(含车体半径)，不再扎到近触(0.04m)
    "leg_max_steps": 16,   # 步长从 1.2→0.8m，覆盖同样距离需更多步
    "final_manner": "near",
    # 兼容单目标语义路线（无 targets 时的回退，等价原 semantic_route 默认）
    "target_desc": "",
    "targets": None,
}


def _hr(t):
    print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


def _save_frame(report_dir, fname):
    """把 MCP 最近抓取的相机帧复制到实验目录（初期实验存图，best-effort）。"""
    if not report_dir:
        return None
    try:
        os.makedirs(report_dir, exist_ok=True)
        src = config.MCP_IMAGE_PATH
        if os.path.exists(src):
            dst = os.path.join(report_dir, fname)
            shutil.copyfile(src, dst)
            return dst
    except Exception:  # noqa: BLE001
        pass
    return None


def _pose_retry(ex, tries=8):
    import time
    for _ in range(tries):
        p = harness._pose(ex)
        if isinstance(p, dict) and "x" in p:
            return p
        time.sleep(0.5)
    return harness._pose(ex)


def _nav_to_subgoal(ex, sx, sy, *, goal_desc="", label="", tol=0.6, max_steps=16):
    """朝子目标【点到点 APF】推进（执行模式：起点+航点+目标坐标皆已知，用 APF 不用岔口-VFH）。

    薄封装 nav.apf_goto：引力朝子目标 + 斥力离障碍 → 合力航向逐小步开过去，天然穿缝、不把车导离/甩飞
    （区别于岔口-VFH"直路被堵就找最宽开口"）。卡死自动 geo_goto_around 兜底。导航环不再调 VLM——
    goal_desc 只是历史签名的占位，不再使用。
    """
    r = nav.apf_goto(ex, sx, sy, tol_m=tol, max_steps=max_steps)
    d = r.get("dist_m", 0.0)
    print(f"    [{label}] APF→ {r['status']} dist={d}m steps={len(r.get('steps') or [])}")
    return {"arrived": r.get("arrived", False), "dist": round(d, 2),
            "steps": len(r.get("steps") or []), "status": r["status"]}


def _roi_depth(ex, roi):
    """MCP depth_roi 取该 roi 的中位深度(米)；失败/无效 → None（交调用方退化）。"""
    import json as _json
    try:
        rx, ry, rw, rh = dp._roi_to_frac(roi)
        txt = ex.ros.call("depth_roi", {"rois_json": _json.dumps([{"x": rx, "y": ry, "w": rw, "h": rh}])}).text
        st = (_json.loads(txt).get("stats") or [{}])[0]
        med = st.get("median_m")
        return float(med) if isinstance(med, (int, float)) and med > 0 else None
    except Exception:  # noqa: BLE001
        return None


def _identify_target(ex, rec, tgt, tol=0.8):
    """末段身份确认门：面向目标后取【一帧】，把命中目标名的物体 bbox 接地→匹配记忆→比对期望 id。

    Qwen 只出框(它擅长的)，代码做 bbox→深度→世界坐标→match_object→比 id。返回
    {confirmed: True/False/None, confirmed_id, expected_id, note}。无框/无深度/没看到目标 →
    confirmed=None（未确认，不阻塞；确认率随感知线提升）。
    """
    expected = tgt.get("id")
    pose = _pose_retry(ex)
    try:
        rep = harness.inspect_and_report(ex)
    except Exception as e:  # noqa: BLE001
        return {"confirmed": None, "confirmed_id": None, "expected_id": expected,
                "note": f"观测失败({type(e).__name__})"}
    keys = _name_keys(tgt["name"])
    cands = [o for o in (rep.get("objects") or [])
             if o.get("roi") and any(k in (o.get("name") or "") for k in keys)]
    if not cands:
        return {"confirmed": None, "confirmed_id": None, "expected_id": expected, "note": "视野未框到目标"}
    roi = max(cands, key=lambda o: float((o["roi"].get("w") or 0)) * float((o["roi"].get("h") or 0)))["roi"]
    depth = _roi_depth(ex, roi) or max((o.get("distance_m") or 0) for o in cands) or None
    if not depth or depth <= 0:
        return {"confirmed": None, "confirmed_id": None, "expected_id": expected, "note": "无有效深度"}
    uc, vc = dp.roi_center_pixel(roi)
    abs_pose = dp.back_project(uc, vc, float(depth), robot_pose=pose).get("abs_pose") or {}
    matched = match_object(rec.get("objects", []), abs_pose, tgt["name"], tol=tol)
    mid = (matched or {}).get("id")
    d_to_tgt = math.hypot(abs_pose.get("x", 1e9) - tgt["x"], abs_pose.get("y", 1e9) - tgt["y"])
    if expected and mid:
        ok = (mid == expected)
    else:
        ok = d_to_tgt <= tol
    note = "确认" if ok else (f"接地未匹配记忆(dist目标{d_to_tgt:.1f}m)" if matched is None
                              else f"匹配到 {mid} 非期望 {expected}")
    return {"confirmed": bool(ok), "confirmed_id": mid, "expected_id": expected, "note": note}


class ExecutionMode(Mode):
    name = "execution"
    skill = "区域到区域语义拓扑导航 + 末段靠近目标物（Claude 规划路线 / Qwen 逐段 / 代码 VFH）"
    allowlist = config.ACTION_ALLOWLIST
    memory_access = "read"              # 只读地图，从不写
    uses_cloud = True                  # 需 director.plan_route

    def do_run(self, ctx: RunContext) -> ModeResult:
        p = {**_DEFAULTS, **(ctx.params or {})}
        ex, mem, area, director = ctx.ex, ctx.mem, ctx.area, ctx.director
        surf_tol = p["arrival_surface_tol_m"]
        car_half, obj_r = p["car_half_m"], p["obj_radius_m"]

        _hr(f"执行模式：区域到区域语义导航（area={area}）")
        rec = mem.load_area(area)
        if rec is None:
            return ModeResult(self.name, "NO_MAP", False,
                              summary=f"找不到地图 {area}", exit_code=2)
        targets = _resolve_targets(rec, p)
        if not targets:
            return ModeResult(self.name, "NO_TARGET", False,
                              summary="任务未解析出目标（缺 targets 或子区目标）", exit_code=1)
        seq = ["{}@({},{})".format(t["name"], t["x"], t["y"]) for t in targets]
        print(f"[目标序列] {seq}")

        per_target = []
        for ti, tgt in enumerate(targets):
            _hr(f"目标 {ti+1}/{len(targets)}：{tgt['name']} @({tgt['x']},{tgt['y']})")
            pose0 = _pose_retry(ex)
            print(f"[起点] pose={pose0}")

            # ---- 阶段1：Claude 读地图规划语义路线（不出坐标）----
            legs = []
            if director is not None:
                payload = {
                    "area": rec.get("area"), "type": rec.get("type"), "summary": rec.get("summary", ""),
                    "start_pose": {"x": round(pose0.get("x", 0), 2), "y": round(pose0.get("y", 0), 2),
                                   "yaw": round(pose0.get("yaw_deg", 0), 1)},
                    "target": tgt["desc"],
                    "sub_areas": [{"label": s.get("label"), "type": s.get("type"),
                                   "center": _sa_center(s), "summary": s.get("summary", "")}
                                  for s in rec.get("sub_areas", [])],
                    "landmarks": _landmarks(rec),
                    "doors": rec.get("doors", []),
                    "relations": [{"s": r.get("subject_id", "")[:6], "p": r.get("predicate"),
                                   "o": r.get("object_id", "")[:6]} for r in rec.get("relations", [])][:8],
                }
                try:
                    plan = director.plan_route(payload)
                    legs = plan.get("legs") if isinstance(plan, dict) else []
                    print(f"[语义路线] {plan.get('goal_note','') if isinstance(plan,dict) else ''}")
                    for i, lg in enumerate(legs or []):
                        print(f"   {i+1}. 到『{lg.get('via')}』的 {lg.get('manner')} —— {lg.get('note','')}")
                except Exception as e:  # noqa: BLE001
                    print(f"[语义路线] 规划失败({type(e).__name__})，退化为直接朝目标导航")
                    legs = []

            # ---- 阶段2：Qwen 逐段语义导航 + VFH（有 legs 才走；否则直接进末段）----
            for i, lg in enumerate(legs or []):
                via, manner = lg.get("via", ""), (lg.get("manner") or "near").lower()
                cands = _via_candidates(rec, via)
                if not cands:
                    print(f"[leg{i+1}] via『{via}』无匹配地标 → 跳过")
                    continue
                # leg5 根治：这条 leg 就是"去目标物本身" → 丢弃，交末段用已知目标坐标逼近
                # （杜绝 Qwen 在多同名实例里盲选目标）。
                if leg_hits_target(cands, tgt):
                    print(f"[leg{i+1}] via『{via}』即目标物本身 → 跳过，交末段(已知坐标)")
                    continue
                pose = harness._pose(ex)
                is_door = len(cands) == 1 and cands[0].get("src") == "door"
                chosen = cands[0]
                # 门 leg：机器人与目标已在门同侧 → 穿门无意义，跳过（治无谓绕路）
                if is_door and door_leg_redundant(pose, (chosen["x"], chosen["y"]), tgt):
                    print(f"[leg{i+1}] 门『{chosen['name']}』与目标已同侧 → 跳过(无需穿门)")
                    continue
                if len(cands) > 1:
                    chosen = self._qwen_pick_landmark(ex, pose, via, manner, lg, cands)
                if is_door:
                    sx, sy, ltol = chosen["x"], chosen["y"], 0.6       # 门：APF 直奔开口点穿缝(不 standoff)
                else:
                    sx, sy = _manner_point(chosen, manner, pose)
                    ltol = 0.6
                # 护栏：落点把车带离已知目标(误解析) → 跳过该 leg（门 leg 是正当中转，豁免）
                if not is_door and leg_leads_away(sx, sy, pose, tgt):
                    print(f"[leg{i+1}]『{chosen['name']}』落点({sx},{sy})离目标更远 → 判误解析跳过")
                    continue
                tag = "穿门" if is_door else manner
                print(f"[leg{i+1}] {'门' if is_door else 'Qwen 选地标'}『{chosen['name']}』"
                      f"@({chosen['x']},{chosen['y']}) → {tag} 落点({sx},{sy})；APF 点到点：")
                g = _nav_to_subgoal(ex, sx, sy, goal_desc=f"{via}({chosen['name']})", tol=ltol,
                                    label=f"t{ti+1}leg{i+1}", max_steps=p["leg_max_steps"])
                pf = harness._pose(ex)
                print(f"        → {g['status']} dist={g['dist']}m 到位({pf.get('x',0):.2f},{pf.get('y',0):.2f})")

            # ---- 阶段3：末段【体积感知】靠近 + 面向 + 环顾观测 ----
            _hr(f"目标 {ti+1} 末段：体积感知靠近 + 观测")
            # 停在离目标中心 car_half+obj_r+standoff_surf 处（表面留 ~standoff_surf），不再 geo_goto
            # 扎到近触(0.04m)——那会蹭目标/家具、触发 safety。沿 当前pose→目标 方向退开算停靠点，APF 开过去。
            pose = harness._pose(ex)
            approach_cd = car_half + obj_r + p["final_standoff_surf_m"]
            dxt, dyt = tgt["x"] - pose.get("x", 0.0), tgt["y"] - pose.get("y", 0.0)
            L = math.hypot(dxt, dyt) or 1e-9
            sx = round(tgt["x"] - approach_cd * dxt / L, 2)
            sy = round(tgt["y"] - approach_cd * dyt / L, 2)
            print(f"[目标{ti+1}末段] 体积感知停靠：离中心 {approach_cd:.2f}m(表面~{p['final_standoff_surf_m']}m)"
                  f" → 停靠点({sx},{sy})")
            gg = _nav_to_subgoal(ex, sx, sy, goal_desc=tgt["name"], tol=0.3,
                                 label=f"t{ti+1}末段", max_steps=p["leg_max_steps"])
            nav.geo_face_point(ex, tgt["x"], tgt["y"])
            # 身份确认门（面向目标后单帧接地匹配记忆 id；未确认不阻塞，几何到位仍成立）
            ident = _identify_target(ex, rec, tgt)
            rep = harness.sweep_observe(ex)
            seen = [o.get("name") for o in rep.get("objects", [])]
            frame = _save_frame(ctx.report_dir, f"target_{ti+1}_{tgt['name']}.jpg")
            pf = harness._pose(ex)
            center_d = math.hypot(pf.get("x", 0) - tgt["x"], pf.get("y", 0) - tgt["y"])
            surf_d = round(center_d - car_half - obj_r, 3)
            arrived = surf_d <= surf_tol
            saw = any(any(k in (o.get("name") or "") for k in _name_keys(tgt["name"]))
                      for o in rep.get("objects", []))
            print(f"[目标{ti+1}] 末段APF={gg.get('status')} 到位pose=({pf.get('x',0):.2f},{pf.get('y',0):.2f}) "
                  f"中心距={center_d:.2f}m 表面距={surf_d:.2f}m 到达={arrived} 视野见目标={saw} 看到{seen}")
            print(f"[目标{ti+1}] 身份确认: {ident['confirmed']}（{ident['note']}；"
                  f"匹配 id={ident['confirmed_id']} 期望={ident['expected_id']}）")
            per_target.append({
                "name": tgt["name"], "target_xy": [tgt["x"], tgt["y"]], "target_id": tgt.get("id"),
                "arrival_pose": {"x": round(pf.get("x", 0), 3), "y": round(pf.get("y", 0), 3),
                                 "yaw_deg": round(pf.get("yaw_deg", 0), 1)},
                "center_dist_m": round(center_d, 3), "min_surface_dist_m": surf_d,
                "arrived": arrived, "saw_target": saw, "objects_seen": seen,
                "identity_confirmed": ident["confirmed"], "confirmed_id": ident["confirmed_id"],
                "expected_id": ident["expected_id"], "identity_note": ident["note"],
                "final_nav_status": gg.get("status"), "frame": frame,
            })

        ex.ros.call("stop", {})

        _hr("结果")
        all_arrived = bool(per_target) and all(t["arrived"] for t in per_target)
        for t in per_target:
            print(f"  {t['name']}: 表面距={t['min_surface_dist_m']}m 到达={t['arrived']} 见目标={t['saw_target']}")
        status = "SUCCESS" if all_arrived else ("PARTIAL" if any(t["arrived"] for t in per_target)
                                                else "NOT_ARRIVED")
        print(f"判定: {status}（到达 {sum(t['arrived'] for t in per_target)}/{len(per_target)}）")
        return ModeResult(
            self.name, status, finish=all_arrived,
            summary="",                                # base 用 Qwen 补齐
            exit_code=0 if all_arrived else 1,
            metrics={"targets_total": len(per_target),
                     "targets_arrived": sum(t["arrived"] for t in per_target),
                     "surface_tol_m": surf_tol,
                     "per_target": per_target},
        )

    # ---- Qwen 段内地标消歧（多候选时）----
    def _qwen_pick_landmark(self, ex, pose, via, manner, lg, cands):
        cand_lines = []
        for k, c in enumerate(cands):
            br = _bearing_from(pose, c["x"], c["y"])
            d = math.hypot(c["x"] - pose["x"], c["y"] - pose["y"])
            side = "左" if br > 15 else ("右" if br < -15 else "前")
            cand_lines.append(f"  [{k}] {c['name']} 方位{side}(bearing {br:.0f}°) 距{d:.1f}m")
        look = ex.ros.call("look", {})
        img = look.images[0] if look.images else None
        user = (f"本段语义目标：到『{via}』的{manner}方位（{lg.get('note','')}）。\n"
                f"地图里同名/相关候选地标(相对你当前朝向的方位)：\n" + "\n".join(cand_lines) + "\n"
                f"结合画面选一个【这段该朝它推进】的地标编号。只输出 JSON：{{\"pick\":整数}}")
        content = [{"type": "text", "text": user}]
        if img:
            content.append(harness.to_openai_image_url(img))
        try:
            r = ex.client.chat.completions.create(
                model=ex.model, messages=[
                    {"role": "system", "content": harness.INSPECT_MEM_SYS},
                    {"role": "user", "content": content}],
                temperature=0.2, max_tokens=120, stream=False)
            dec = harness._extract_obj(r.choices[0].message.content or "")
            pk = int(dec.get("pick")) if isinstance(dec, dict) and dec.get("pick") is not None else 0
            if 0 <= pk < len(cands):
                return cands[pk]
        except Exception:  # noqa: BLE001
            return max(cands, key=lambda c: c["x"])
        return cands[0]


def _name_keys(name):
    """目标名的关键片段（用于视野命中判断，含常见同义）。"""
    base = [name]
    syn = {"绿植": ["绿植", "植", "plant", "盆栽"], "办公桌": ["桌", "desk", "table"],
           "办公椅": ["椅", "chair"], "显示器": ["显示", "屏", "monitor", "screen"],
           "红柜": ["柜", "红", "cabinet"]}
    for k, v in syn.items():
        if k in name:
            return v
    return base
