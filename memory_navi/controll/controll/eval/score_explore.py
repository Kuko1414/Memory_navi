#!/usr/bin/env python3
"""探索模式记忆召回打分器（稳定门槛 = observable 物体召回 ≥ 0.70）。

用法：
    python memory_navi/controll/controll/eval/score_explore.py [area.json] [ground_truth.json]
默认 area = memory_navi/memory/sim/explore_room/area.json，GT = 同目录 explore_room_ground_truth.json。

打分逻辑（客观、可审计）：
  - 召回命中 = 某记录物体 同时满足 ①位置匹配（点物体 xy 距离 ≤ pos_tol；region 物体落在范围±margin 内）
    ②类别匹配（记录物体的 name 或任一 alias，归一化后 ∈ 该真值物体 match_names）。每条记录最多命中一个真值。
  - 召回率 = observable 真值命中数 / observable 真值总数。
  - observable=false 的真值（隔断后办公区/远区）命中算【加分】，不计入分母、记到也不算幻觉。
  - 未匹配的记录再分类：落在某 observable 真值 pos_tol 内但类别不符 = 【误标】；否则 = 【幻觉/噪声】。
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                         # …/controll/controll/eval
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))        # 仓库根 Kuko1414
DEFAULT_AREA = os.path.join(ROOT, "memory_navi/memory/sim/explore_room/area.json")
DEFAULT_GT = os.path.join(HERE, "explore_room_ground_truth.json")


def _norm(s):
    return re.sub(r"[\s_]+", " ", (s or "").strip().lower())


def _cand_names(obj):
    """记录物体的候选名集合（归一化）：name + aliases。"""
    names = [obj.get("name")] + list(obj.get("aliases") or [])
    return {_norm(n) for n in names if n}


def _xy(ap):
    if isinstance(ap, dict) and ap.get("x") is not None:
        return float(ap["x"]), float(ap.get("y") or 0.0)
    return None


def _pos_match(rec_xy, gt, pos_tol):
    """点物体：距离 ≤ pos_tol；region：落在范围±margin 内。"""
    if rec_xy is None:
        return False
    rx, ry = rec_xy
    if gt.get("type") == "region":
        m = gt.get("region_margin_m", 0.5)
        x0, x1 = gt["x_range"]
        y0, y1 = gt["y_range"]
        return (x0 - m) <= rx <= (x1 + m) and (y0 - m) <= ry <= (y1 + m)
    return math.hypot(rx - gt["x"], ry - gt["y"]) <= pos_tol


def _cat_match(rec, gt):
    gt_names = {_norm(n) for n in gt.get("match_names", [])}
    return bool(_cand_names(rec) & gt_names)


def score(area_path, gt_path):
    with open(area_path, encoding="utf-8") as f:
        area = json.load(f)
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    pos_tol = gt.get("pos_tol_m", 1.0)
    recorded = [o for o in area.get("objects", []) if isinstance(o, dict) and o.get("name")]
    gts = gt["objects"]
    obs = [g for g in gts if g.get("observable")]

    used = [False] * len(recorded)
    hits, misses, bonus = [], [], []

    def try_match(gt_obj):
        for i, rec in enumerate(recorded):
            if used[i]:
                continue
            if _pos_match(_xy(rec.get("abs_pose")), gt_obj, pos_tol) and _cat_match(rec, gt_obj):
                used[i] = True
                return rec
        return None

    for g in gts:
        m = try_match(g)
        if g.get("observable"):
            (hits if m else misses).append((g, m))
        elif m:
            bonus.append((g, m))

    # 未匹配的记录分类
    mislabels, halluc = [], []
    for i, rec in enumerate(recorded):
        if used[i]:
            continue
        rxy = _xy(rec.get("abs_pose"))
        near = None
        if rxy is not None:
            for g in obs:
                if _pos_match(rxy, g, pos_tol):
                    near = g
                    break
        (mislabels if near else halluc).append((rec, near))

    recall = len(hits) / len(obs) if obs else 0.0

    # ---- 报告 ----
    print(f"area   : {area_path}")
    print(f"GT     : {gt_path}   (pos_tol={pos_tol}m)")
    print(f"记录物体: {len(recorded)}   observable 真值: {len(obs)}")
    print("=" * 64)
    print(f"召回率 = {len(hits)}/{len(obs)} = {recall*100:.1f}%   门槛 70%  →  "
          + ("PASS ✅" if recall >= 0.70 else "FAIL ❌"))
    print("=" * 64)
    print(f"\n✓ 命中 observable 真值 ({len(hits)}):")
    for g, m in hits:
        print(f"    {g['name']:<14} ← 记录 '{m.get('name')}'"
              + (f" aliases={m.get('aliases')}" if m.get('aliases') else ""))
    print(f"\n✗ 漏记 observable 真值 ({len(misses)}):")
    for g, _ in misses:
        print(f"    {g['name']:<14} @({g.get('x')},{g.get('y')})  {g.get('note','')}")
    if bonus:
        print(f"\n＋ 加分：记到了隔断后/远区真值 ({len(bonus)}):")
        for g, m in bonus:
            print(f"    {g['name']:<14} ← 记录 '{m.get('name')}'")
    if mislabels:
        print(f"\n⚠ 误标（位置对、类别错）({len(mislabels)}):")
        for rec, g in mislabels:
            print(f"    记录 '{rec.get('name')}' @{_xy(rec.get('abs_pose'))} ≈ 真值 {g['name']}")
    if halluc:
        print(f"\n? 幻觉/噪声（不匹配任何真值）({len(halluc)}):")
        for rec, _ in halluc:
            print(f"    记录 '{rec.get('name')}' @{_xy(rec.get('abs_pose'))}")
    return recall


if __name__ == "__main__":
    area_p = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AREA
    gt_p = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_GT
    r = score(area_p, gt_p)
    sys.exit(0 if r >= 0.70 else 1)
