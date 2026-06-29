#!/usr/bin/env python3
"""客观打分：拿『答案』(break_room_ground_truth.json) 检验小车这次缺口补全是否合格。

纯函数、无 ROS/LLM 依赖，可被 completion_demo.py 复用，也可对着保存下来的结果 json 离线重跑：
  python Report/score_completion.py <run_result.json> [break_room_ground_truth.json]

判据为初期宽松版（见 break_room_map.md / ground_truth.scoring）；幻觉判负这条不放宽。
我（持答案）另读最终相机帧确认图文一致，本脚本只给客观的第一遍打分。
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GT = os.path.join(HERE, "break_room_ground_truth.json")

# 规范类 → 触发子串（小写匹配 reported name）。前 4 类是办公区家具（缺口召回的重点）。
OFFICE_CANON = {
    "办公桌": ["办公桌", "桌子", "书桌", "工位", "desk", "table", "桌"],
    "显示器": ["显示器", "屏幕", "显示屏", "monitor", "screen", "显示"],
    "办公椅": ["办公椅", "椅子", "座椅", "chair"],
    "键盘": ["键盘", "keyboard"],
    "隔断矮墙": ["隔断", "矮墙", "浅紫", "partition", "半墙"],
    "红柜": ["红色柜", "红柜", "红色储物", "红色矮柜", "red cabinet"],
}
# 现实里确实存在、非办公区目标但允许出现（不算幻觉、不计入缺口召回）。
OTHER_REAL = {
    "沙发": ["沙发", "sofa", "couch"],
    "绿植": ["绿植", "植物", "盆栽", "盆景", "plant", "tree"],
    "吊灯/灯": ["吊灯", "灯", "light", "lamp"],
    "地板/地面": ["地板", "地面", "floor"],
    "天花板": ["天花板", "顶", "ceiling"],
    "墙": ["墙", "wall"],
    "门": ["门", "door"],
    "窗": ["窗", "window"],
    "厨台/水槽": ["厨", "台面", "counter", "水槽", "sink"],
    "柜子(泛)": ["柜", "cabinet", "储物"],
    "橙色高柜": ["橙色", "橙柜", "orange"],
}
# 明确的幻觉黑名单（房间里根本没有的具体物品），命中即判负。
HALLUCINATION_BLACKLIST = [
    "人", "person", "people", "电视", "tv", "电视机", "书架", "bookshelf", "书柜",
    "床", "bed", "冰箱", "fridge", "refrigerator", "微波炉", "黑板", "白板", "投影",
    "汽车", "car", "楼梯", "stairs", "电梯",
]

OFFICE_FURNITURE = ("办公桌", "显示器", "办公椅", "键盘")


def _canon(name: str):
    """把上报名归一化到 (规范类, 是否办公家具, 是否现实物)。匹配不到返回 (None, ...)。"""
    n = (name or "").lower()
    for canon, toks in OFFICE_CANON.items():
        if any(t.lower() in n for t in toks):
            return canon, canon in OFFICE_FURNITURE, True
    for canon, toks in OTHER_REAL.items():
        if any(t.lower() in n for t in toks):
            return canon, False, True
    return None, False, False


def _is_blacklisted(name: str) -> bool:
    n = (name or "").lower()
    return any(b.lower() in n for b in HALLUCINATION_BLACKLIST)


def _office_truth_points(gt: dict) -> dict:
    """每个办公规范类 → 真实坐标点列表，用于算真值距离。"""
    pts = {k: [] for k in OFFICE_CANON}
    oa = gt.get("office_area_behind_red_cabinet", {})
    for o in oa.get("near_visible_objects", []) + oa.get("far_east_objects", []):
        canon, _, _ = _canon(o.get("name", ""))
        if canon in pts and "x" in o and "y" in o:
            pts[canon].append((o["x"], o["y"]))
    rc = gt.get("red_cabinet", {})
    if rc:
        pts["红柜"].append((rc["x"], rc["y"]))
    return pts


def _nearest_true_dist(ax, ay, points) -> float:
    return min((math.hypot(px - ax, py - ay) for px, py in points), default=None)


def _in_front(ax, ay, ayaw, px, py, half_fov=70.0) -> bool:
    """点 (px,py) 是否在机器人前方视野内(航向 ±half_fov)。相机看不到身后的物体。"""
    bearing = math.degrees(math.atan2(py - ay, px - ax))
    return abs(((bearing - ayaw + 180.0) % 360.0) - 180.0) <= half_fov


def _nearest_front_dist(ax, ay, ayaw, points):
    front = [(px, py) for px, py in points if _in_front(ax, ay, ayaw, px, py)]
    return _nearest_true_dist(ax, ay, front) if front else None


def _wall_dist(ax, ay) -> float:
    """到 wall(1) 隔断墙(x=2.65, y∈[-2.545,1.005]) 的近似距离。"""
    cy = max(-2.545, min(1.005, ay))
    return math.hypot(2.65 - ax, cy - ay)


def score(result: dict, gt: dict) -> dict:
    sc = gt.get("scoring", {})
    task = gt.get("task", {})
    vp = task.get("viewpoint", {})
    ft = task.get("face_target", {})
    arr = result.get("arrival_pose") or {}
    ax, ay = float(arr.get("x", 0.0)), float(arr.get("y", 0.0))
    ayaw = float(arr.get("yaw_deg", 0.0))

    # —— 到达：到观察点距离 且 必须真正绕到红柜后方(x > cabinet_x) ——
    arrival_tol = float(sc.get("arrival_tol_m", 1.0))
    min_x = float(sc.get("behind_cabinet_min_x", 2.62))
    d_vp = math.hypot(ax - float(vp.get("x", 4.6)), ay - float(vp.get("y", -0.5)))
    behind_cabinet = ax > min_x
    arrival_ok = (d_vp <= arrival_tol) and behind_cabinet

    # —— 朝向 —— 指向 face_target 的真值方位 ± heading_tol
    heading_tol = float(sc.get("heading_tol_deg", 20.0))
    want = math.degrees(math.atan2(float(ft.get("y", -0.1)) - ay, float(ft.get("x", 3.0)) - ax))
    heading_err = ((want - ayaw + 180.0) % 360.0) - 180.0
    heading_ok = abs(heading_err) <= heading_tol

    # —— 语义 / 幻觉 / 距离 ——
    truth_pts = _office_truth_points(gt)
    matched_office = set()
    hallucinations = []
    per_object = []
    dist_oks = []
    for o in result.get("objects", []) or []:
        name = o.get("name", "")
        canon, is_office, is_real = _canon(name)
        rep_d = o.get("distance_m")
        blacklisted = _is_blacklisted(name)
        if blacklisted or (not is_real):
            hallucinations.append(name)
        if is_office:
            matched_office.add(canon)
        # 距离核对（仅对能定位真值的物体）
        true_d = None
        if canon == "隔断矮墙":
            true_d = _wall_dist(ax, ay)
        elif canon and truth_pts.get(canon):
            # 只和【前方视野内】的同类真值比（相机看不到身后的桌椅）
            true_d = _nearest_front_dist(ax, ay, ayaw, truth_pts[canon])
        err_pct = None
        if true_d is not None and isinstance(rep_d, (int, float)) and rep_d > 0:
            err_pct = round(abs(rep_d - true_d) / true_d * 100.0, 0)
            dist_oks.append(err_pct <= float(sc.get("distance_tol_pct", 50.0)))
        per_object.append({
            "name": name, "canon": canon, "bearing": o.get("bearing"),
            "reported_m": rep_d, "true_m": round(true_d, 2) if true_d else None,
            "err_pct": err_pct, "in_vocab": is_real and not blacklisted,
        })

    n_office_furniture = len(matched_office & set(OFFICE_FURNITURE))
    semantic_ok = n_office_furniture >= 2 or (
        n_office_furniture >= 1 and ("隔断矮墙" in matched_office or "红柜" in matched_office))
    hallucination_ok = len(hallucinations) == 0
    distance_ok = (not dist_oks) or (sum(dist_oks) >= max(1, len(dist_oks) // 2))

    # 距离=代码从 depth 接地的真实测量(bearing→列中位，偏粗)，仅作参考不作硬门(用户："可粗糙一点")。
    # 硬门：到达(且在红柜后)、朝向、语义召回、无幻觉。
    passed = arrival_ok and heading_ok and semantic_ok and hallucination_ok

    # —— 新指标(v2)：目标是否来自感知、偏差、云调用 ——
    tgt_from_percep = bool(result.get("target_from_perception"))
    obs_d = result.get("observation_from_depth") or {}
    tgt_dev = None
    if obs_d and vp:
        tgt_dev = round(math.hypot(float(obs_d.get("x", 0)) - float(vp.get("x", 4.6)),
                                    float(obs_d.get("y", 0)) - float(vp.get("y", -0.5))), 2)
    claude_n = int(result.get("claude_calls", 0))

    return {
        "arrival": {"ok": arrival_ok, "dist_to_viewpoint_m": round(d_vp, 2),
                    "behind_cabinet": behind_cabinet, "min_x": min_x,
                    "pose": {"x": ax, "y": ay, "yaw_deg": ayaw}, "tol_m": arrival_tol},
        "heading": {"ok": heading_ok, "want_deg": round(want, 1), "got_deg": round(ayaw, 1),
                    "err_deg": round(heading_err, 1), "tol_deg": heading_tol},
        "semantic": {"ok": semantic_ok, "matched_office_classes": sorted(matched_office),
                     "office_furniture_hits": n_office_furniture},
        "hallucination": {"ok": hallucination_ok, "offending": hallucinations},
        "distance": {"ok": distance_ok, "checked": len(dist_oks), "within_tol": sum(dist_oks)},
        "objects": per_object,
        "target_from_perception": tgt_from_percep,
        "target_deviation_m": tgt_dev,
        "claude_calls": claude_n,
        "pass": passed,
    }


def format_scorecard(s: dict) -> str:
    def mark(b):
        return "✅" if b else "❌"
    lines = ["", "=" * 60, "  补全任务客观打分（初期宽松阈值；幻觉判负不放宽）", "=" * 60]
    a, h, se, hl, di = s["arrival"], s["heading"], s["semantic"], s["hallucination"], s["distance"]
    lines.append(f"{mark(a['ok'])} 到达：距观察点 {a['dist_to_viewpoint_m']}m (tol {a['tol_m']}m), "
                 f"在红柜后方(x>{a['min_x']})={a['behind_cabinet']}, 位姿={a['pose']}")
    lines.append(f"{mark(h['ok'])} 朝向：应朝 {h['want_deg']}°, 实际 {h['got_deg']}°, "
                 f"偏差 {h['err_deg']}° (tol ±{h['tol_deg']}°)")
    lines.append(f"{mark(se['ok'])} 语义召回：命中办公家具类 {se['office_furniture_hits']} 种, "
                 f"全部命中类={se['matched_office_classes']}")
    lines.append(f"{mark(hl['ok'])} 幻觉检查：" + ("无" if hl["ok"] else f"发现 {hl['offending']}"))
    lines.append(f"ℹ️ 距离(参考,不计入通过)：{di['within_tol']}/{di['checked']} 个在 ±50% 内 "
                 f"[depth 接地测量，bearing→列偏粗]")
    # 新指标(v2)
    tgt_p = s.get("target_from_perception", False)
    tgt_d = s.get("target_deviation_m")
    lines.append(f"{'✅' if tgt_p else '⚠️'} 目标来源={'感知(depth回投)' if tgt_p else '答案文件(硬编)'}"
                 + (f"  偏差={tgt_d}m vs答案观察点" if tgt_d is not None else ""))
    lines.append(f"☁️ Claude 援助调用：{s.get('claude_calls', 0)} 次")
    lines.append("-" * 60)
    lines.append("逐物体：")
    for o in s["objects"]:
        tag = "" if o["in_vocab"] else "  ⚠幻觉?"
        lines.append(f"  - {o['name']}({o['canon']}) bearing={o['bearing']} "
                     f"报={o['reported_m']}m 真≈{o['true_m']}m err={o['err_pct']}%{tag}")
    lines.append("=" * 60)
    lines.append(f"  本次任务：{'PASS ✅ 无幻觉完成' if s['pass'] else 'FAIL ❌ 见上'}")
    lines.append("=" * 60)
    return "\n".join(lines)


def load_gt(path: str = DEFAULT_GT) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    with open(sys.argv[1], encoding="utf-8") as f:
        result = json.load(f)
    gt = load_gt(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_GT)
    s = score(result, gt)
    print(format_scorecard(s))
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
