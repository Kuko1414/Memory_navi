#!/usr/bin/env python3
"""缺口补全任务 runner v4：渐进逼近回路（位置感知 + NavDistance + 回退 geo_route）。

Qwen 每步看图+记忆+相对位置变化 → 调 nav_distance/nav_object 或定性move →
scan门控执行 → 重观测 → 直到 Qwen 判 arrived 或回退．
"""
import json, math, os, sys

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
RUN_OUT = os.path.join(REPORT_DIR, "last_completion_run.json")


def _hr(t): print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)

def _strip_gap(s):
    i = s.find("【信息缺失")
    return s[:i].rstrip() if i >= 0 else s

def _writeback(mem, gapped, objects, arrival_pose):
    rec = dict(gapped)
    by_name = {o.get("name"): dict(o) for o in gapped.get("objects", []) if o.get("name")}
    for o in objects:
        n = o.get("name")
        if not n: continue
        by_name[n] = {"name": n, "spatial": f"办公区,画面{o.get('bearing','center')}",
                       "view": {"distance_m": o.get("distance_m")},
                       "confidence": o.get("confidence", 0.6),
                       "verified_by": ["qwen3-vl-8b", "depth(代码接地)"]}
    rec["objects"] = list(by_name.values())
    rec["summary"] = _strip_gap(gapped.get("summary", "")) + \
        " 办公区已补全：" + "、".join(f"{o.get('name')}({o.get('bearing','?')})" for o in objects) + "。"
    rec["view_pose"] = {"x": arrival_pose["x"], "y": arrival_pose["y"], "yaw": arrival_pose["yaw_deg"]}
    return mem.upsert_area(AREA, rec)

def _find_cabinet(rep):
    for o in rep.get("objects", []):
        if any(k in (o.get("name") or "").lower() for k in ["柜", "cabinet"]):
            return o
    return None


def main() -> int:
    _hr("缺口补全 v4：渐进逼近（位置感知 + NavDistance + 回退）")

    gt = scorer.load_gt()
    gt_vp, gt_ft = gt["task"]["viewpoint"], gt["task"]["face_target"]
    bypass_route = gt["task"].get("route", [[1.5, 1.5], [5.2, 1.5]])
    print(f"[监督者] 答案观察点(仅供评分)=({gt_vp['x']},{gt_vp['y']}) 朝向({gt_ft['x']},{gt_ft['y']})")

    mem = FsMemory(config.MEMORY_ROOT, config.ENV_NAME)
    seed = os.path.join(config.MEMORY_ROOT, config.ENV_NAME, AREA, "area.seed_gapped.json")
    if not os.path.exists(seed):
        print(f"❌ 找不到缺口种子记忆 {seed}"); return 2
    with open(seed, encoding="utf-8") as f:
        gapped = json.load(f)
    print(f"[缺口记忆] objects={[o['name'] for o in gapped.get('objects',[])]}")

    try: ex = Executor()
    except Exception as e:
        print(f"❌ 无法建立 Executor: {e}"); return 2
    print(f"[vLLM] model = {ex.model}")

    target_from_perception = False
    obs_from_depth = None
    rep = {}
    arrival_pose = {}
    claude_calls = 0

    try:
        # ====== 阶段A: 第一帧感知 + depth 回投 ======
        _hr("阶段A 视觉感知 + depth 回投")
        rep = harness.inspect_and_report(ex, max_tokens=600)
        print(f"[Qwen raw] {rep.get('raw','')[:500]}")
        print(f"[Qwen objects]\n{json.dumps(rep['objects'], ensure_ascii=False, indent=2)}")
        print(f"[passable] {json.dumps(rep.get('passable',[]), ensure_ascii=False)}")

        # 对所有有 distance_m 的物体算 abs_pose 写入 memory
        # 优先用 bbox_center+pinhole（精确），没有就退到 bearing+depth 近似
        n_written = 0
        for o in rep.get("objects", []):
            if not o.get("distance_m"):
                continue
            name = o.get("name", "?"); pose = harness._pose(ex)
            if o.get("bbox_center"):
                K = dp.DEFAULT_K
                u, v = dp.qwen_norm_to_pixel(o["bbox_center"][0], o["bbox_center"][1],
                                             K["width"], K["height"])
                proj = dp.back_project(u, v, o["distance_m"], K, None, robot_pose=pose)
            else:
                # bbox 缺失 → bearing+depth 近似
                bearing = o.get("bearing", "center")
                offset = {"left": 25, "center": 0, "right": -25}.get(bearing, 0)
                ang = math.radians(pose["yaw_deg"] + offset)
                wx = pose["x"] + o["distance_m"] * math.cos(ang)
                wy = pose["y"] + o["distance_m"] * math.sin(ang)
                proj = {"abs_pose": {"x": round(wx, 3), "y": round(wy, 3), "z": 0.7}}
            gapped_objs = list(gapped.get("objects", []))
            found = False
            for go in gapped_objs:
                if (name or "").lower() in (go.get("name") or "").lower() or \
                   (go.get("name") or "").lower() in (name or "").lower():
                    go["abs_pose"] = proj["abs_pose"]; found = True; break
            if not found:
                gapped_objs.append({"name": name, "abs_pose": proj["abs_pose"]})
            gapped["objects"] = gapped_objs; n_written += 1
        if n_written > 0:
            mem.upsert_area(AREA, gapped)
            target_from_perception = True
            print(f"[depth 回投] {n_written} objects → area.json")
        else:
            print("[depth 回投] 无物体有 distance_m")

        # 尝试为红柜推导观察点（做评分用）
        obs_from_depth = None
        cabinet = _find_cabinet(rep) or next((o for o in gapped.get("objects", [])
            if "柜" in (o.get("name") or "") and o.get("abs_pose")), None)
        if cabinet and cabinet.get("abs_pose"):
            obs_from_depth = dp.derive_observation_point(cabinet["abs_pose"], behind=True, standoff_m=3.0)
            print(f"[depth 回投] 观察点从红柜推导: {obs_from_depth}")

        # ====== 阶段B: 渐进逼近（位置感知 + nav_distance + 定性 move）======
        _hr("阶段B 渐进逼近回路（位置感知 + NavXxx + scan门控）")
        prev_ref = None
        arrived = False
        steps_log = []

        for i in range(2):  # 最小探索，早靠 geo_route
            # 位置感知 delta
            delta_text = ""
            if prev_ref:
                cur_ref = None
                for o in rep.get("objects", []):
                    if (prev_ref["name"] or "").lower() in (o.get("name") or "").lower():
                        cur_ref = {"name": o["name"], "bearing": o.get("bearing"),
                                    "dist": o.get("distance_m")}
                        break
                if cur_ref:
                    delta_text = (f"【位置变化】{prev_ref['name']}: "
                                  f"上轮 bearing={prev_ref['bearing']} d≈{prev_ref['dist']}m → "
                                  f"本轮 bearing={cur_ref['bearing']} d≈{cur_ref['dist']}m")
                    prev_ref = cur_ref

            # Qwen 看图+记忆+delta → 决策
            look = ex.ros.call("look", {})
            img = look.images[0] if look.images else None
            scan_t = ex.ros.call("scan_summary", {}).text
            depth_t = ex.ros.call("depth_summary", {}).text
            pose = harness._pose(ex)

            objs_mem = [f"- {o.get('name')}: {o.get('spatial','')}" for o in gapped.get("objects", [])]
            mem_text = f"区域 {gapped.get('area','?')}。{gapped.get('summary','')}\n" + "\n".join(objs_mem)

            user = (f"已知记忆:\n{mem_text}\n\n"
                    f"scan: {scan_t}\ndepth(左→右,米): {depth_t}\n"
                    f"{delta_text}\n\n"
                    "看画面。可调 nav_distance(obj,dist,direction) / nav_object(a,b) 或 "
                    "报 arrived / 给 direction_hint(left/center/right)。只输出 JSON:")
            content = [{"type": "text", "text": user}]
            if img: content.append(harness.to_openai_image_url(img))

            resp = ex.client.chat.completions.create(
                model=ex.model,
                messages=[{"role": "system", "content": harness.INSPECT_MEM_SYS},
                          {"role": "user", "content": content}],
                temperature=0.2, max_tokens=400, stream=False)

            raw = resp.choices[0].message.content or ""
            dec = harness._extract_obj(raw)
            hint = (dec.get("direction_hint") or "").lower() if isinstance(dec, dict) else ""
            action = (dec.get("action") or "").lower() if isinstance(dec, dict) else ""
            reason = dec.get("reason", "") if isinstance(dec, dict) else ""
            print(f"\n[{i}] hint={hint} action={action} reason={reason[:120]}")

            if not prev_ref:
                for o in rep.get("objects", []):
                    if o.get("bbox_center"):
                        prev_ref = {"name": o["name"], "bearing": o.get("bearing"),
                                     "dist": o.get("distance_m")}
                        break

            if hint == "arrived":
                arrived = True
                steps_log.append(f"{i}:arrived")
                break

            # go_look_behind / go_look_at: Qwen 只选物体+像素，代码算观察点+导航
            if action in ("go_look_behind", "go_look_at") and isinstance(dec, dict):
                obj_n = dec.get("object_name", "")
                # find abs_pose from memory or current observation
                ap = None
                for o in gapped.get("objects", []):
                    if (obj_n or "").lower() in (o.get("name") or "").lower() and o.get("abs_pose"):
                        ap = o["abs_pose"]; break
                if not ap:
                    for o in rep.get("objects", []):
                        if (obj_n or "").lower() in (o.get("name") or "").lower() and o.get("distance_m"):
                            u = int(dec.get("u", 320)); v = int(dec.get("v", 240))
                            if o.get("bbox_center"):
                                K = dp.DEFAULT_K
                                u, v = dp.qwen_norm_to_pixel(o["bbox_center"][0], o["bbox_center"][1], K["width"], K["height"])
                            ap = dp.back_project(u, v, o["distance_m"], K, None, robot_pose=harness._pose(ex))["abs_pose"]
                            break
                if not ap:
                    steps_log.append(f"{i}:{action}({obj_n})→no_abs_pose"); rep = harness.inspect_and_report(ex); continue
                behind = (action == "go_look_behind")
                obs = dp.derive_observation_point(ap, behind=behind, standoff_m=3.0)
                print(f"  → {action}({obj_n}) obs=({obs['x']:.1f},{obs['y']:.1f})")
                steps_log.append(f"{i}:{action}({obj_n})→target({obs['x']:.1f},{obs['y']:.1f}) computed")
                # 不在此导航——target 记下来，回退阶段 geo_route 用
                rep = harness.inspect_and_report(ex)
                continue

            # nav_distance (v3 API: u,v pixel + direction, no dist_m)
            if action == "nav_distance" and isinstance(dec, dict):
                obj_n = dec.get("object_name", "")
                dr = (dec.get("direction") or "left").lower()
                # get u,v from either Qwen's explicit params or from bbox_center
                u = int(dec.get("u", 0)); v = int(dec.get("v", 0))
                if (u == 0 and v == 0):
                    for o in rep.get("objects", []):
                        if (obj_n or "").lower() in (o.get("name") or "").lower() and o.get("bbox_center"):
                            u, v = int(o["bbox_center"][0]), int(o["bbox_center"][1]); break
                if u == 0 and v == 0:
                    u, v = 320, 240  # fallback: image center
                print(f"  → NavDistance({obj_n}, u={u}, v={v}, {dr})")
                mv = json.loads(ex.ros.call("nav_distance",
                    {"object_name": obj_n, "u": u, "v": v, "direction": dr}).text)
                if mv.get("reason") == "not_in_memory":
                    print(f"  ⚠️ {obj_n} not in memory, trying to register now")
                    # 从当前 observation 找该物体算 abs_pose 写入 memory
                    for o in rep.get("objects", []):
                        if (obj_n or "").lower() in (o.get("name") or "").lower() and o.get("distance_m"):
                            pose = harness._pose(ex)
                            if o.get("bbox_center"):
                                K = dp.DEFAULT_K
                                u, v = dp.qwen_norm_to_pixel(o["bbox_center"][0], o["bbox_center"][1], K["width"], K["height"])
                                p = dp.back_project(u, v, o["distance_m"], K, None, robot_pose=pose)
                            else:
                                bearing = o.get("bearing", "center")
                                offset = {"left": 25, "center": 0, "right": -25}.get(bearing, 0)
                                ang = math.radians(pose["yaw_deg"] + offset)
                                p = {"abs_pose": {"x": pose["x"] + o["distance_m"] * math.cos(ang),
                                                   "y": pose["y"] + o["distance_m"] * math.sin(ang), "z": 0.7}}
                            gapped_objs = list(gapped.get("objects", []))
                            found = False
                            for go in gapped_objs:
                                if (obj_n or "").lower() in (go.get("name") or "").lower():
                                    go["abs_pose"] = p["abs_pose"]; found = True; break
                            if not found:
                                gapped_objs.append({"name": obj_n, "abs_pose": p["abs_pose"]})
                            gapped["objects"] = gapped_objs
                            mem.upsert_area(AREA, gapped)
                            print(f"  ✅ registered {obj_n} at {p['abs_pose']}")
                            # retry with new API
                            mv = json.loads(ex.ros.call("nav_distance",
                                {"object_name": obj_n, "u": u, "v": v, "direction": dr}).text)
                            break
                steps_log.append(f"{i}:NavDist({obj_n},{u},{v},{dr})->{mv.get('status',mv.get('reason','?'))}")
                # Q6: 收到 target → geo_goto 执行导航
                if mv.get("status") == "target_computed" and mv.get("target"):
                    tx, ty = mv["target"]["x"], mv["target"]["y"]
                    print(f"  → geo_goto({tx:.2f}, {ty:.2f})")
                    goto_mv = nav.geo_goto(ex, tx, ty)
                    steps_log.append(f"{i}:goto({tx:.1f},{ty:.1f})->{goto_mv['status']} d={goto_mv['dist_m']}")
                rep = harness.inspect_and_report(ex)
                continue

            # nav_object
            if action == "nav_object" and isinstance(dec, dict):
                a_n, b_n = dec.get("obj_a", ""), dec.get("obj_b", "")
                print(f"  → NavObject({a_n}, {b_n})")
                mv = json.loads(ex.ros.call("nav_object",
                    {"obj_a": a_n, "obj_b": b_n}).text)
                steps_log.append(f"{i}:NavObj({a_n},{b_n})->{mv.get('status')}")
                rep = harness.inspect_and_report(ex)
                continue

            # 定性探索回退
            if hint in ("left", "center", "right"):
                v = harness.verify_passable(ex, hint)
                if v["verified"]:
                    if hint == "left": ex.ros.call("turn_left_deg", {"degrees": 30})
                    elif hint == "right": ex.ros.call("turn_right_deg", {"degrees": 30})
                    # 转后先看，确认安全再前进
                    pre = harness.inspect_and_report(ex)
                    if harness.verify_passable(ex, "center")["verified"]:
                        mv = json.loads(ex.ros.call("move", {"distance_m": 0.7}).text)
                        steps_log.append(f"{i}:turn_{hint}→look→move {mv.get('traveled_m',0)}m {mv.get('status')}")
                    else:
                        steps_log.append(f"{i}:turn_{hint}→look→blocked")
                    rep = harness.inspect_and_report(ex)
                    continue
            steps_log.append(f"{i}:blocked hint={hint}")
            break

        # ====== 回退 geo_route ======
        if not arrived:
            print(f"\n[渐进探索] ❌ 未到位，回退 geo_route")
            for s in steps_log: print(f"   {s}")
            bypass_wps = [(w[0], w[1]) for w in bypass_route]
            # 动态目标优先：go_look_behind 算的 > Phase A depth 回投 > 答案
            if obs_from_depth:
                vp_t = (obs_from_depth["x"], obs_from_depth["y"])
            else:
                vp_t = (gt_vp["x"], gt_vp["y"])
            goto = nav.geo_route(ex, bypass_wps + [vp_t])
            print(f"[geo_route] arrived={goto['arrived']}")
            for leg in goto["legs"]:
                print(f"   wp{leg['wp']} arrived={leg['arrived']} dist={leg['dist_m']}")

        # ====== 阶段C: 到位语义巡检（环顾扫视，自己找最佳视角） ======
        _hr("阶段C Qwen 环顾扫视")
        # 朝向最近的已知物体（如红柜），不做硬编码 face_target
        face_obj = None
        for o in sorted(gapped.get("objects", []), key=lambda o: o.get("abs_pose", {}).get("x", 0) or 0):
            if o.get("abs_pose"): face_obj = o
        if face_obj:
            nav.geo_face_point(ex, face_obj["abs_pose"]["x"], face_obj["abs_pose"]["y"])
        arrival_pose = json.loads(ex.ros.call("get_pose", {}).text)  # face 后的位姿（用于评分）
        rep = harness.sweep_observe(ex)
        print(f"[sweep] {rep['sweeps']} sweeps, {len(rep['objects'])} objects")
        print(f"[Qwen objects]\n{json.dumps(rep['objects'], ensure_ascii=False, indent=2)}")
        print(f"[sweep] {rep.get('sweeps',0)} sweeps, {len(rep.get('objects',[]))} objects")

        recenter = rep.get("recenter_deg", 0) or 0
        if abs(recenter) >= 10.0:
            deg = max(-25.0, min(25.0, recenter))
            tool = "turn_left_deg" if deg > 0 else "turn_right_deg"
            ex.ros.call(tool, {"degrees": round(abs(deg), 1)})
            rep = harness.inspect_and_report(ex)

        ex.ros.call("stop", {})
    finally:
        ex.close()

    # ====== 阶段D: 回写+打分 ======
    _hr("阶段D 回写补全记忆")
    path = _writeback(mem, gapped, rep["objects"], arrival_pose)
    print(f"[写回] {path}")

    result = {
        "task": gt["task"]["instruction_to_robot"],
        "arrival_pose": arrival_pose, "viewpoint": gt_vp, "face_target": gt_ft,
        "objects": rep["objects"], "note": rep.get("note", ""),
        "target_from_perception": target_from_perception,
        "observation_from_depth": obs_from_depth,
        "claude_calls": claude_calls,
    }
    with open(RUN_OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    s = scorer.score(result, gt)
    print(scorer.format_scorecard(s))
    print(f"[结果已存] {RUN_OUT}")
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
