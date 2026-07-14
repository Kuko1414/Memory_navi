#!/usr/bin/env python3
"""执行模式客观打分（纯函数 + CLI 离线模式，模板参照 Report/score_completion.py）。

硬门：两目标均到达(表面距离达阀值) + Qwen 输出总结 + supervisor 回报 + 无幻觉。
连续指标(不作硬门，初期实验)：逐目标实测最小表面距离、碰撞次数、耗时。

run_result（由 run_execution_experiment.py 产出）关键字段：
  targets[]: {name, target_xy, arrival_pose, center_dist_m, min_surface_dist_m, arrived, saw_target, objects_seen}
  collisions: int（safety_node tripped 上升沿计数）
  duration_s: float
  qwen_summary: str
  supervisor_report: {phase, mode, finish, status, duration_s, summary, metrics}
可离线跑：python eval/score_execution.py <run_result.json> [gt.json]
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GT = os.path.join(HERE, "break_room_exec_ground_truth.json")


def _norm(s: str) -> str:
    return (s or "").strip().lower().replace("_", " ")


def _synonyms(gt) -> dict:
    return (gt.get("scoring", {}) or {}).get("target_synonyms", {}) or {}


def _name_match(name: str, target: str, syn: dict) -> bool:
    n, t = _norm(name), _norm(target)
    if not n:
        return False
    if t and (t in n or n in t):
        return True
    for a in syn.get(target, []):
        if _norm(a) in n or n in _norm(a):
            return True
    return False


def _is_hallucination(name: str, blacklist) -> bool:
    n = _norm(name)
    return any(_norm(b) in n for b in blacklist)


def score(result: dict, gt: dict) -> dict:
    sc = gt.get("scoring", {}) or {}
    surf_tol = float(sc.get("arrival_surface_tol_m", 0.45))
    car_half = float(sc.get("car_half_m", 0.16))
    obj_r = float(sc.get("obj_radius_m", 0.20))
    syn = _synonyms(gt)
    blacklist = sc.get("hallucination_blacklist", []) or []
    gt_targets = gt.get("targets", []) or []

    res_targets = result.get("targets", []) or []
    # 按【实际跑的目标数(attempted)】打分，避免单目标 smoke 被 2 目标 GT 打成假失败。
    attempted = len(res_targets)
    gt_total = len(gt_targets)
    per_target = []
    hallucinations = []
    for i, rt in enumerate(res_targets):
        gtt = gt_targets[i] if i < gt_total else {"name": rt.get("name"),
                                                  "x": (rt.get("target_xy") or [0, 0])[0],
                                                  "y": (rt.get("target_xy") or [0, 0])[1]}
        # 表面距离：优先 runner 已算的 min_surface_dist_m；否则用 arrival_pose 现算（打分自洽）
        surf_d = rt.get("min_surface_dist_m")
        arr = rt.get("arrival_pose") or {}
        if surf_d is None and arr:
            cd = math.hypot(float(arr.get("x", 0)) - float(gtt["x"]),
                            float(arr.get("y", 0)) - float(gtt["y"]))
            surf_d = round(cd - car_half - obj_r, 3)
        arrived = surf_d is not None and surf_d <= surf_tol
        saw = bool(rt.get("saw_target"))
        for nm in rt.get("objects_seen", []) or []:
            if _is_hallucination(nm, blacklist):
                hallucinations.append(nm)
        per_target.append({
            "name": gtt.get("name"), "target_xy": [gtt.get("x"), gtt.get("y")],
            "min_surface_dist_m": surf_d, "arrived": arrived, "saw_target": saw,
            "arrival_pose": arr,
            "identity_confirmed": rt.get("identity_confirmed"),
            "confirmed_id": rt.get("confirmed_id"), "expected_id": rt.get("expected_id"),
            "identity_note": rt.get("identity_note"),
        })

    n_arrived = sum(t["arrived"] for t in per_target)
    n_confirmed = sum(1 for t in per_target if t.get("identity_confirmed") is True)
    n_wrong_id = sum(1 for t in per_target if t.get("identity_confirmed") is False)
    all_arrived = attempted > 0 and n_arrived == attempted
    complete = attempted == gt_total          # 是否跑满 GT 全部目标（单目标 smoke=False）

    # Qwen 总结 / supervisor 回报
    qwen_summary = result.get("qwen_summary") or (result.get("supervisor_report") or {}).get("summary")
    summary_ok = bool(qwen_summary and str(qwen_summary).strip())
    rep = result.get("supervisor_report") or {}
    report_ok = bool(rep) and ("summary" in rep) and ("duration_s" in rep) and ("finish" in rep)
    hallucination_ok = len(hallucinations) == 0

    require_sum = sc.get("require_qwen_summary", True)
    require_rep = sc.get("require_supervisor_report", True)
    passed = (all_arrived and complete and hallucination_ok
              and (summary_ok or not require_sum)
              and (report_ok or not require_rep))

    return {
        "targets": per_target,
        "arrival": {"ok": all_arrived and complete, "arrived": n_arrived,
                    "attempted": attempted, "gt_total": gt_total, "complete": complete,
                    "surface_tol_m": surf_tol},
        "qwen_summary": {"ok": summary_ok, "text": qwen_summary},
        "supervisor_report": {"ok": report_ok, "phase": rep.get("phase"),
                              "mode": rep.get("mode"), "finish": rep.get("finish")},
        # 身份确认＝软指标(不计入 pass)：初期实验感知弱，确认率随 YOLO/感知线提升；未确认≠到达失败。
        "identity": {"confirmed": n_confirmed, "wrong_id": n_wrong_id, "attempted": attempted},
        "hallucination": {"ok": hallucination_ok, "offending": hallucinations},
        "collisions": int(result.get("collisions", 0)),
        "duration_s": result.get("duration_s"),
        "pass": passed,
    }


def format_scorecard(s: dict) -> str:
    def mark(b):
        return "✅" if b else "❌"
    lines = ["", "=" * 60, "  执行模式客观打分（安全兼容阀值；幻觉判负不放宽）", "=" * 60]
    a = s["arrival"]
    lines.append(f"{mark(a['ok'])} 到达：{a['arrived']}/{a['attempted']} 已跑目标达标 "
                 f"(表面距离 ≤ {a['surface_tol_m']}m)")
    if not a.get("complete", True):
        lines.append(f"    ⚠️ 只跑了 {a['attempted']}/{a['gt_total']} 个目标(未完整)——"
                     f"逐目标行若达标即为该目标 PASS，但整体任务未完整不算总 PASS")
    for t in s["targets"]:
        ic = t.get("identity_confirmed")
        ics = "✅确认" if ic is True else ("❌认错" if ic is False else "—未确认")
        lines.append(f"    · {t['name']}@{t['target_xy']}: 实测表面距={t['min_surface_dist_m']}m "
                     f"到达={mark(t['arrived'])} 视野见目标={mark(t['saw_target'])} 身份{ics}"
                     f"(id={t.get('confirmed_id')}/期望{t.get('expected_id')})")
    idn = s.get("identity", {})
    lines.append(f"ℹ️ 身份确认(软指标,不计入通过)：确认 {idn.get('confirmed',0)}/{idn.get('attempted',0)}"
                 f"，认错 {idn.get('wrong_id',0)}（感知弱时多为'未确认',几何到位仍成立）")
    su, rp, hl = s["qwen_summary"], s["supervisor_report"], s["hallucination"]
    lines.append(f"{mark(su['ok'])} Qwen 总结：" + (str(su["text"])[:50] if su["ok"] else "缺失"))
    lines.append(f"{mark(rp['ok'])} supervisor 回报：phase={rp['phase']} mode={rp['mode']} finish={rp['finish']}")
    lines.append(f"{mark(hl['ok'])} 幻觉检查：" + ("无" if hl["ok"] else f"发现 {hl['offending']}"))
    lines.append(f"ℹ️ 连续指标(不计入通过)：碰撞 {s['collisions']} 次，耗时 {s['duration_s']}s")
    lines.append("=" * 60)
    lines.append(f"  本次任务：{'PASS ✅' if s['pass'] else 'FAIL ❌ 见上（初期实验：停下评估）'}")
    lines.append("=" * 60)
    return "\n".join(lines)


def load_gt(path: str = DEFAULT_GT) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if len(sys.argv) < 2:
        print("用法：python eval/score_execution.py <run_result.json> [gt.json]")
        return 2
    with open(sys.argv[1], encoding="utf-8") as f:
        result = json.load(f)
    gt = load_gt(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_GT)
    s = score(result, gt)
    print(format_scorecard(s))
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
