#!/usr/bin/env python3
"""初期房间探索 v1：从零覆盖 + Claude C-format 语义标注。

与补足模式(autonomy_probe)的区别：不依赖任何先验语义/锚点，机器人从零在房间里【铺开覆盖】，
每到一个视角触发【云端记忆作者 Claude】产出 C-format 语义记录。粗边界 = 走过点 + 各扇区扫到的墙点。

角色分工：代码做执行/安全(geo_step_open)，Qwen 选下一个探索视角(标像素)，Claude 写语义记录。
v1 核心实验问：Qwen 逐帧驱动室内覆盖到底稳不稳（铺开 vs 转圈）——不稳再上代码 frontier/占据栅格。

运行：conda run -n vllm python explore_probe.py（需 ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL gateway）。
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
from agent_core.cloud.providers import LocalVLMProvider
from agent_core.cloud.memory_author import MemoryAuthor

AREA = "explore_room"          # 独立 area，不碰 autonomy_probe 的 break_room 记录
RUN_OUT = os.path.join(REPORT_DIR, "explore_run.json")
BUDGET = 8                     # 覆盖视角预算（turn/go 均消耗一步，但都有价值）
NO_PROGRESS_LIMIT = 3
START_XY = (-0.5, -1.0)        # 休息室起点（初始条件，非答案）
START_TOL_M = 1.5

EXPLORE_SYS = (
    "你是室内机器人，正在【从零探索、绘制一个未知房间】。给你：\n"
    " - 当前相机画面、可通行方向、激光/深度。\n"
    " - 已到达的坐标范围 → 判断【哪片还没去过】。\n\n"
    "每次只做【一个动作】，系统执行后你再看新画面决定：\n"
    " - go：去画面里某片没看过的区域/物体。给像素 u,v(0-1000)。\n"
    " - turn：原地旋转看周围（用于扫视环境找未探索方向）。给 degrees(正=左转，通常 60-120)。\n"
    " - done：已在多个方向多视角看过，主要物体看清 → 探索完成。\n\n"
    "探索策略：\n"
    "1) 每到一个新位置先 turn 扫一圈找未探索方向，再 go 过去。\n"
    "2) 开阔方向优先；被墙/家具挡了就 turn 换边。\n"
    "3) 如果同一位置已经看过多帧、走不动 → 果断 turn 换视角。\n"
    "4) 房间内至少覆盖 3-5 个不同位置（分散开），再 done。\n"
    "只输出 JSON：{\"action\":\"go|turn|done\",\"u\":500,\"v\":500,\"degrees\":90,\"reason\":\"\"}"
)


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


def _pixel_to_world(ex, rep, u_norm, v_norm):
    """Qwen 归一化像素 + 该列 depth → 世界点（看哪反投成哪）。复用 autonomy_probe 逻辑。"""
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


def _explore_decide(ex, rep, visited, wall_bbox):
    """Qwen 看图 → 下一个探索动作（go/turn/done）。"""
    dist = sorted({(round(p[0], 1), round(p[1], 1)) for p in visited})
    bbox_line = ""
    if wall_bbox:
        bbox_line = (f"已探边界 x[{wall_bbox['xmin']:.1f},{wall_bbox['xmax']:.1f}] "
                     f"y[{wall_bbox['ymin']:.1f},{wall_bbox['ymax']:.1f}]。")
    user = (f"已到达{len(dist)}点: {dist}。{bbox_line}\n"
            f"可通行={rep.get('passable', [])}\n"
            "下一步？go(像素u,v) / turn(degrees) / done。只输出 JSON：")
    content = [{"type": "text", "text": user}]
    img = rep.get("image")
    if img is not None:
        content.append(harness.to_openai_image_url(img))
    resp = ex.client.chat.completions.create(
        model=ex.model,
        messages=[{"role": "system", "content": EXPLORE_SYS}, {"role": "user", "content": content}],
        temperature=0.3, max_tokens=250, stream=False)
    return harness._extract_obj(resp.choices[0].message.content or "")


def main():
    _hr("explore_probe v1：从零覆盖 + Claude C-format 标注")
    ex = Executor()
    mem = FsMemory(config.MEMORY_ROOT, config.ENV_NAME)
    prov = LocalVLMProvider(base_url=config.VLLM_8B_BASE_URL)  # 复用 8B 做标注（Claude 网关今日限额满）
    print(f"[标注] LocalVLM (8B)  |  [执行] model = {ex.model}")
    author = MemoryAuthor(ex.ros, prov, mem)

    # 起点守卫：软复位到休息室起点（VFH 能从墙后绕回来）+ 朝东
    sp = _pose(ex)
    d_start = math.hypot(sp.get("x", 99) - START_XY[0], sp.get("y", 99) - START_XY[1])
    print(f"[起始位姿] {sp}  距起点 {d_start:.2f}m")
    if d_start > START_TOL_M:
        print(f"[软复位] VFH 开回 {START_XY}…")
        for _ in range(22):
            cp = _pose(ex)
            if math.hypot(cp["x"] - START_XY[0], cp["y"] - START_XY[1]) <= 0.5:
                break
            nav.geo_step_open(ex, nav.bearing_deg(cp["x"], cp["y"], START_XY[0], START_XY[1]))
    nav.geo_face_point(ex, START_XY[0] + 1.0, START_XY[1])   # 朝东起手

    visited, wall_points, vantage_records, steps_log = [], [], [], []
    prev_pose = _pose(ex)
    no_progress = 0
    stop_reason = "budget"

    _hr("覆盖回路（Qwen 选探索视角 → 代码 VFH 执行 → Claude 标注）")
    try:
        for i in range(BUDGET):
            rep = harness.inspect_and_report(ex)
            pose = _pose(ex)
            visited.append([round(pose["x"], 2), round(pose["y"], 2)])
            wall_points.extend(_scan_world_points(ex, pose))

            # —— Claude 标注本视角（云端记忆作者，C-format）——
            r = author.record(AREA, trigger="explore",
                              view_pose={"x": pose["x"], "y": pose["y"], "yaw": pose["yaw_deg"]})
            if r.ok:
                vantage_records.append(r.record)
                cobjs = [o.get("name") for o in r.record.get("objects", [])]
                csum = (r.record.get("summary") or "")[:60]
            else:
                cobjs, csum = [], f"FAIL:{r.error}"
            qobjs = [o.get("name") for o in rep.get("objects", [])]
            print(f"[巡{i}] pose=({pose['x']:.2f},{pose['y']:.2f},{pose['yaw_deg']:.0f}) "
                  f"Qwen看={qobjs}  Claude记={cobjs}  | {csum}")
            steps_log.append({"step": i, "pose": pose, "qwen_objects": qobjs, "claude_objects": cobjs})

            # —— Qwen 决定下一个探索视角 ——
            bbox = None
            allpts = wall_points + visited
            if allpts:
                bbox = {"xmin": min(p[0] for p in allpts), "xmax": max(p[0] for p in allpts),
                        "ymin": min(p[1] for p in allpts), "ymax": max(p[1] for p in allpts)}
            dec = _explore_decide(ex, rep, visited, bbox)
            action = (dec.get("action") or "").lower() if isinstance(dec, dict) else ""
            reason = str(dec.get("reason", "")[:80]) if isinstance(dec, dict) else ""

            # done
            if action == "done":
                stop_reason = "qwen_done"
                print(f"[巡{i}] Qwen done — {reason}")
                break

            # turn
            if action == "turn":
                deg = float(dec.get("degrees", 90) or 90)
                deg = max(30, min(180, abs(deg)))  # clamp
                print(f"[巡{i}] turn {deg}° — {reason}")
                ex.ros.call("turn_left_deg", {"degrees": deg})
                cur = _pose(ex)
                no_progress = 0  # turn counts as progress (new perspective)
                prev_pose = cur
                continue

            # go（默认；action!=turn/done 也当 go 处理）
            u = float(dec.get("u", 500) or 500) if isinstance(dec, dict) else 500
            v = float(dec.get("v", 500) or 500) if isinstance(dec, dict) else 500
            pt, _p = _pixel_to_world(ex, rep, u, v)
            if not pt:
                print(f"[巡{i}] 像素({u:.0f},{v:.0f})无有效深度 → 左转 60° 脱困")
                ex.ros.call("turn_left_deg", {"degrees": 60})
            else:
                d = math.hypot(pt["x"] - pose["x"], pt["y"] - pose["y"])
                standoff = 1.5
                if d > standoff:
                    rr = (d - standoff) / d
                    vp = (pose["x"] + (pt["x"] - pose["x"]) * rr, pose["y"] + (pt["y"] - pose["y"]) * rr)
                else:
                    vp = (pose["x"], pose["y"])
                print(f"[巡{i}] {action} →像素({u:.0f},{v:.0f}) 世界({pt['x']:.1f},{pt['y']:.1f}) "
                      f"视角点({vp[0]:.1f},{vp[1]:.1f}) — {reason}")
                for _ in range(3):
                    cp = _pose(ex)
                    if math.hypot(vp[0] - cp["x"], vp[1] - cp["y"]) <= 0.5:
                        break
                    st = nav.geo_step_open(ex, nav.bearing_deg(cp["x"], cp["y"], vp[0], vp[1]))
                    if (st.get("moved_m") or 0) < 0.08:
                        break

            cur = _pose(ex)
            disp = math.hypot(cur["x"] - prev_pose["x"], cur["y"] - prev_pose["y"])
            no_progress = 0 if disp > 0.15 else no_progress + 1
            prev_pose = cur
            if no_progress >= NO_PROGRESS_LIMIT:
                stop_reason = "no_progress"
                print(f"[巡{i}] ⚠️ 连续 {no_progress} 步无进展 → 停")
                break
        ex.ros.call("stop", {})
    finally:
        ex.close()

    # —— 合并各视角 Claude 记录的 objects（按名并集，留高 confidence）→ 写一份合并记录 ——
    by_name, rec_type = {}, "unknown"
    for rec in vantage_records:
        rec_type = rec.get("type") or rec_type
        for o in rec.get("objects", []):
            n = o.get("name")
            if not n:
                continue
            if n not in by_name or (o.get("confidence", 0) or 0) > (by_name[n].get("confidence", 0) or 0):
                by_name[n] = o
    merged = {
        "area": AREA, "type": rec_type,
        "summary": f"初期探索覆盖 {len(visited)} 视角；Claude 合并 {len(by_name)} 物体。",
        "view_pose": {"x": prev_pose["x"], "y": prev_pose["y"], "yaw": prev_pose["yaw_deg"]},
        "objects": list(by_name.values()),
        "hazards": [],
    }
    try:
        path = mem.upsert_area(AREA, merged)
        print(f"[写回合并记录] {path}")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 合并写回失败: {e}")

    allpts = wall_points + visited
    bbox = None
    if allpts:
        xs = [p[0] for p in allpts]
        ys = [p[1] for p in allpts]
        bbox = {"xmin": round(min(xs), 2), "xmax": round(max(xs), 2),
                "ymin": round(min(ys), 2), "ymax": round(max(ys), 2)}
    run = {
        "area": AREA, "n_vantages": len(visited), "stop_reason": stop_reason,
        "visited": visited, "coarse_bbox": bbox, "wall_points": wall_points[:400],
        "merged_objects": [o.get("name") for o in merged["objects"]],
        "claude_calls": len(vantage_records), "steps": steps_log,
    }
    with open(RUN_OUT, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)

    _hr("结果")
    print(f"覆盖视角={len(visited)}  停因={stop_reason}  粗边界bbox={bbox}")
    print(f"Claude 合并物体={[o.get('name') for o in merged['objects']]}")
    print(f"[run 工件] {RUN_OUT}  [记忆] {os.path.join(config.MEMORY_ROOT, config.ENV_NAME, AREA, 'area.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
