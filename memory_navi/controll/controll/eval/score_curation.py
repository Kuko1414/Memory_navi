#!/usr/bin/env python3
"""离线整理层三方对比打分器（不跑仿真）：真值 GT vs 整理前(raw) vs 整理后(curated)。

对每个现成实验产物(area_*.json)，只【重跑离线整理层】(ensure_ids→flag→Claude curate_area→gate→
apply_curation)，再用 score_explore 打分 raw 与 curated，检验：
  - 红线：curated 召回 ≥ raw 召回 − eps（整理【不许】把召回搞低——不误删/误改真物体）；
  - 期望：curated 噪声(幻觉)↓、误标↓。

整理层是纯文本后处理，脱离仿真即可测。需 ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL(gateway)；
缺则整理官降级为 OfflineDirector(空裁决)，curated==raw，只验管线不验语义。

用法：
    conda run -n vllm python .../eval/score_curation.py                # 跑默认产物集
    conda run -n vllm python .../eval/score_curation.py --artifacts A.json B.json
    conda run -n vllm python .../eval/score_curation.py --flag-only    # 不调 Claude，只看候选检出
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))                 # …/controll/controll/eval
PKG = os.path.dirname(HERE)                                       # …/controll/controll
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))   # 仓库根 Kuko1414
for p in (PKG, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent_core import config                                    # noqa: E402
from agent_core.cloud.providers import make_director             # noqa: E402
from agent_core.cloud.schema import validate_curated_record      # noqa: E402
from agent_core.memory.fs_memory import FsMemory                 # noqa: E402
import curate_memory as cm                                       # noqa: E402
import score_explore as se                                       # noqa: E402

GT = os.path.join(HERE, "explore_room_ground_truth.json")
EPS = 1e-6   # 红线容差：curated 召回不得低于 raw（浮点等值允许）

DEFAULT_ARTIFACTS = [
    "Report/hybrid_review_v3_claude/area_baseline.json",
    "Report/hybrid_review_v3_claude/area_hybrid.json",
    "Report/hybrid_review_v4/area_baseline.json",
    "Report/hybrid_review_v4/area_hybrid.json",
    "Report/hybrid_review_v5/area_baseline.json",
    "Report/hybrid_review_v5/area_hybrid.json",
    "Report/hybrid_review_v6/area_baseline.json",
    "Report/hybrid_review_v6/area_hybrid.json",
]


def _curate(rec, model, flag_only):
    """在内存里跑整理层，返回 (curated_rec_path_dict, meta)。不改动源文件。"""
    tmp = tempfile.mkdtemp()
    env, area = "sim", "explore_room"
    d = os.path.join(tmp, env, area)
    os.makedirs(d)
    with open(os.path.join(d, "area.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    mem = FsMemory(root=tmp, env_name=env)
    loaded = mem.load_area(area)
    cm._ensure_ids(loaded)
    all_ids = {o["id"] for o in loaded["objects"] if isinstance(o, dict) and o.get("id")}
    by_id = {o["id"]: o for o in loaded["objects"] if isinstance(o, dict) and o.get("id")}
    cands = cm._flag_candidates(loaded["objects"])
    flagged = {i for c in cands for i in c["ids"]}
    from collections import Counter
    kinds = Counter(c["kind"] for c in cands)
    if flag_only:
        return loaded, {"cands": len(cands), "kinds": dict(kinds), "verdicts": None, "gated": None}
    payload = {
        "area": loaded.get("area", area), "type": loaded.get("type"),
        "boundary": loaded.get("boundary", {}), "doors": loaded.get("doors", []),
        "objects": [{"id": o["id"], "name": o.get("name"), "aliases": o.get("aliases", []),
                     "abs_pose": o.get("abs_pose"), "size": o.get("size"),
                     "spatial": o.get("spatial"), "confidence": o.get("confidence")}
                    for o in loaded["objects"] if isinstance(o, dict) and o.get("id")],
        "sanity_candidates": cands,
    }
    verdicts = make_director(model=model).curate_area(payload)
    gated = cm._gate_verdicts(verdicts, flagged, all_ids, by_id=by_id, cands=cands)
    path = mem.apply_curation(area, gated, record=loaded, validator=validate_curated_record)
    return mem.load_area(area), {
        "cands": len(cands), "kinds": dict(kinds),
        "verdicts": {k: len(verdicts.get(k) or []) for k in
                     ("sub_areas", "corrections", "relations", "merges", "drops")},
        "gated": {k: len(gated[k]) for k in gated},
        "path": path,
    }


def _counts(area):
    with contextlib.redirect_stdout(io.StringIO()):
        r = se.score_counts(area, GT)
    return r


def main() -> int:
    global GT
    ap = argparse.ArgumentParser(description="离线整理层三方对比打分")
    ap.add_argument("--artifacts", nargs="*", default=DEFAULT_ARTIFACTS,
                    help="area_*.json 列表（相对仓库根或绝对路径）")
    ap.add_argument("--model", default=config.CURATE_MODEL)
    ap.add_argument("--flag-only", action="store_true", help="不调 Claude，只看候选检出(curated==raw)")
    ap.add_argument("--gt", default=GT)
    args = ap.parse_args()
    GT = args.gt

    rows = []
    print(f"{'artifact':<44} {'recall_raw→cur':>16} {'noise':>10} {'mislabel':>10} {'红线':>6}")
    print("-" * 92)
    for a in args.artifacts:
        path = a if os.path.isabs(a) else os.path.join(REPO, a)
        if not os.path.exists(path):
            print(f"{a:<44}  ❌ 缺文件")
            continue
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        raw = _counts(rec)
        try:
            curated, meta = _curate(rec, args.model, args.flag_only)
        except Exception as e:  # noqa: BLE001
            print(f"{a:<44}  ❌ 整理异常: {type(e).__name__}: {str(e)[:50]}")
            continue
        cur = _counts(curated)
        rr, cr = raw["recall"], cur["recall"]
        red_ok = cr + EPS >= rr
        name = a.replace("Report/", "").replace("/area", ":")
        print(f"{name:<44} "
              f"{rr*100:5.1f}%→{cr*100:5.1f}% {len(raw['halluc']):3d}→{len(cur['halluc']):<3d}    "
              f"{len(raw['mislabels']):3d}→{len(cur['mislabels']):<3d}   "
              f"{'PASS' if red_ok else 'FAIL':>6}")
        rows.append({
            "artifact": a, "recall_raw": rr, "recall_cur": cr, "red_ok": red_ok,
            "noise_raw": len(raw["halluc"]), "noise_cur": len(cur["halluc"]),
            "mis_raw": len(raw["mislabels"]), "mis_cur": len(cur["mislabels"]),
            "meta": meta,
        })

    if not rows:
        print("无有效产物。")
        return 1
    print("-" * 92)
    n = len(rows)
    red_fail = [r for r in rows if not r["red_ok"]]
    d_noise = sum(r["noise_cur"] - r["noise_raw"] for r in rows) / n
    d_mis = sum(r["mis_cur"] - r["mis_raw"] for r in rows) / n
    d_rec = sum(r["recall_cur"] - r["recall_raw"] for r in rows) / n
    print(f"聚合({n} 产物): Δrecall={d_rec*100:+.1f}pp  Δnoise={d_noise:+.1f}  Δmislabel={d_mis:+.1f}")
    print(f"红线(整理后召回≥整理前): {n - len(red_fail)}/{n} PASS"
          + ("" if not red_fail else "  ❌ 违红线: " + ", ".join(r["artifact"] for r in red_fail)))
    # 退出码：任一产物违红线 → 非 0
    return 0 if not red_fail else 2


if __name__ == "__main__":
    sys.exit(main())
