#!/usr/bin/env python3
"""记忆整理层（离线 pass）：Claude 整理官把探索产出的扁平 objects[] 组织成词典式结构。

角色分工（见 Report/proposal.md、Process.md §14.7/§14.8-A）：
- Qwen（探索作者）——逐帧命名+去重，产出扁平、带噪、无结构的 area.json（本脚本的输入原料）。
- Claude（整理官，本脚本）——纯文本、不读图，一次调用把物体聚成【区域→子区→物体】：
  A 功能子区、B 上下文纠错(改名)、C on/in/next_to 关系、D 对【代码检出】离谱候选给合并/丢弃裁决。
- 代码（本脚本 + fs_memory）——补稳定 id、检出离谱候选、护栏 gate、算子区 3D range、原子写回、校验。
  铁律：语义归 Claude、几何归代码——Claude 绝不输出/改动坐标；abs_pose 逐字保留。

运行：conda run -n vllm python curate_memory.py --area explore_room [--dry-run]
     （需 ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL gateway；缺则干净 SKIP、退出 0）
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPORT_DIR = os.path.join(REPO, "Report")
for p in (HERE, REPORT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent_core import config
from agent_core.cloud.providers import make_director
from agent_core.cloud.schema import validate_curated_record
from agent_core.memory.fs_memory import FsMemory, _abs_dist, _norm_name

# 不可能尺寸阈值（米）：只标【物理上确实不可能】的（如 10m 长桌）——尺寸估计本身噪声大，
# 阈值必须保守，否则会把稍大的真家具误标成候选。按类别子串匹配；未命中用 generic。
# 代码只【检出候选】，裁决交 Claude。
_GENERIC_MAX = 3.5
_SIZE_RULES = [
    # (name 子串关键词, {dim: max_m}) —— dim ∈ width_m/height_m/depth_m；任一超限即候选
    (("办公桌", "书桌", "桌", "desk", "table"), {"width_m": 3.5, "depth_m": 2.5}),
    (("显示器", "屏", "monitor", "screen"), {"width_m": 1.5, "height_m": 1.2}),
    (("椅", "chair", "stool"), {"width_m": 2.0, "height_m": 2.0, "depth_m": 2.0}),
    (("绿植", "植", "plant"), {"width_m": 2.5, "height_m": 3.0, "depth_m": 2.5}),
    (("柜", "cabinet", "shelf"), {"width_m": 3.5, "depth_m": 2.0}),
    (("沙发", "sofa", "couch"), {"width_m": 3.5, "depth_m": 2.0}),
]

# 近义类表（canonical class → 名字子串关键词）。用于：①按类聚近乎重合的重复(name_dup/synonym_dup)；
# ②检出 name↔alias 跨类冲突(alias_conflict)；③gate 里判"类翻转"改名是否有据。
# 只把【确属同一物】的叫法归一类——跨真类(桌 vs 椅 vs 显示器)绝不同类，护住"不并成一坨"红线。
SYNONYM_GROUPS = {
    "显示器": ("显示器", "显示屏", "屏幕", "屏", "monitor", "screen", "display", "tv", "电视"),
    "沙发": ("沙发", "couch", "sofa", "长椅"),
    "办公桌": ("办公桌", "书桌", "工作台", "桌子", "桌", "desk", "table"),
    "办公椅": ("办公椅", "椅子", "椅", "chair", "stool"),
    "柜子": ("柜子", "矮柜", "储物柜", "柜", "cabinet", "shelf", "cupboard"),
    "绿植": ("绿植", "盆栽", "植物", "植", "plant", "tree", "potted"),
    "厨台": ("厨台", "料理台", "操作台", "台面", "counter", "kitchen"),
    "水槽": ("水槽", "洗手池", "洗手台", "sink"),
    "门": ("门", "door", "开口", "通道"),
}


def _class_of(name):
    """物体名 → 归一化近义类（canonical class）；无法归类返回 None。子串匹配、大小写不敏感。

    注意关键词顺序：长/更具体的词先于短词，避免 '办公桌' 被泛 '桌' 抢先（这里 '办公桌' 与 '桌'
    在同一组 '办公桌' 内，不影响；跨组无重叠子串即可安全）。
    """
    n = _norm_name(name)
    if not n:
        return None
    for canon, keys in SYNONYM_GROUPS.items():
        if any(k.lower() in n for k in keys):
            return canon
    return None


# 合并候选半径（米）：只把【近乎重合=同一物跨视角】的同类物体作合并候选。
# 远小于成排家具间距(~0.6m)——避免把一排不同实例(南柜×4/一排椅)误判成一物合掉→丢召回(红线)。
MERGE_M = 0.45


def _obj_key(o: dict) -> str:
    ap = o.get("abs_pose") or {}
    return "{}|{}|{}".format(
        round(ap.get("x") or 0.0, 2), round(ap.get("y") or 0.0, 2), round(ap.get("z") or 0.0, 2))


def _ensure_ids(rec: dict) -> int:
    """给缺 id 的物体分配确定性位置哈希 id（名不入 key→改名不改身份）；已有 id 复用。

    共坐标碰撞（同 rounded 位置）加 _1/_2 后缀保唯一——正是 merge 候选。返回新分配个数。
    """
    objs = rec.get("objects") or []
    used = {o["id"] for o in objs if isinstance(o, dict) and o.get("id")}
    assigned = 0
    for o in objs:
        if not isinstance(o, dict) or o.get("id"):
            continue
        base = "o" + hashlib.sha1(_obj_key(o).encode()).hexdigest()[:8]
        cand, n = base, 1
        while cand in used:
            cand = f"{base}_{n}"
            n += 1
        o["id"] = cand
        used.add(cand)
        assigned += 1
    return assigned


def _size_candidate(o: dict):
    """物体尺寸对其类别是否物理不可能；是→返回 detail 串，否→None。size_unreliable/null 跳过。"""
    if o.get("size_unreliable"):
        return None
    size = o.get("size")
    if not isinstance(size, dict):
        return None
    dims = {k: size.get(k) for k in ("width_m", "height_m", "depth_m")}
    if all(v is None for v in dims.values()):
        return None
    name = o.get("name") or ""
    rules = None
    for keys, rule in _SIZE_RULES:
        if any(k in name for k in keys):
            rules = rule
            break
    for dim, v in dims.items():
        if v is None:
            continue
        limit = (rules or {}).get(dim, _GENERIC_MAX)
        if v > limit:
            return f"name={name} {dim}={v} 超阈值 {limit}m"
    return None


def _merge_key(o: dict) -> str:
    """合并聚簇键：有归一化近义类用类，否则退化到 norm 名。
    → 同类近义词(显示器/tv)共键可聚 synonym_dup；无类物体(如水槽)仍靠同名聚 name_dup。"""
    return _class_of(o.get("name")) or _norm_name(o.get("name")) or ""


def _flag_candidates(objects: list, merge_m: float = MERGE_M, dz_m: float = 0.5) -> list:
    """代码检出候选（几何归代码，只出候选不定结果；召回优先，半径收紧）：
    - name_dup：同名物体近乎重合(≤merge_m) → 疑同一物跨视角被记多条，供合并；
    - synonym_dup：不同名但【同近义类】近乎重合(≤merge_m，如 显示器+tv 紧挨) → 同物异名，供合并+统一名；
    - alias_conflict：物体 name 的类 ≠ 其某 alias 的类(内部误标证据，如 name=绿植 但 alias 含柜子) → 供纠名/丢弃；
    - impossible_size：尺寸对类别物理不可能。
    合并半径故意远小于成排家具间距——成排不同实例(南柜×4/一排椅)不会成合并候选，交 Claude 用 sub_area 归组而非并掉。
    【不再】做"近邻全异类→按邻居改名"的检出(去牙)：那会系统性抹掉少数类真物体、压低召回。
    """
    cands = []
    ided = [o for o in objects if isinstance(o, dict) and o.get("id")]

    # 近乎重合聚簇（按 merge_key：同近义类或同名 且 xy ≤ merge_m 且 |Δz| 小），≥2 成员 → 合并候选。
    # 同簇内名字若全同 → name_dup；若含 ≥2 种不同名(同类近义) → synonym_dup。
    used = set()
    for i, a in enumerate(ided):
        if a["id"] in used:
            continue
        ka = _merge_key(a)
        if not ka:
            continue
        ap_a = a.get("abs_pose") or {}
        group = [a]
        for b in ided[i + 1:]:
            if b["id"] in used or _merge_key(b) != ka:
                continue
            d = _abs_dist(ap_a, b.get("abs_pose"))
            if d is None or d > merge_m:
                continue
            za, zb = ap_a.get("z"), (b.get("abs_pose") or {}).get("z")
            if za is not None and zb is not None and abs(za - zb) > dz_m:
                continue
            group.append(b)
        if len(group) < 2:
            continue
        for g in group:
            used.add(g["id"])
        span = max(_abs_dist(group[0].get("abs_pose"), g.get("abs_pose")) or 0 for g in group)
        names = [g.get("name") for g in group]
        distinct = {_norm_name(n) for n in names}
        if len(distinct) >= 2:
            cands.append({
                "group_id": f"g{len(cands)}",
                "kind": "synonym_dup",
                "ids": [g["id"] for g in group],
                "detail": f"{len(group)} 个同类[{ka}]近义物近乎重合(≤{round(span,2)}m)：{names}；"
                          f"疑同一物被多视角记成异名；若确为一物→合并并统一规范名，若确为多个真实实例→保留",
            })
        else:
            cands.append({
                "group_id": f"g{len(cands)}",
                "kind": "name_dup",
                "ids": [g["id"] for g in group],
                "detail": f"{len(group)} 个同名 '{names[0]}' 近乎重合(≤{round(span,2)}m)，疑同一物跨视角被记多条；"
                          f"若确为一物→合并，若确为多个真实实例→保留",
            })

    # alias_conflict：物体自身 name 类与某 alias 类跨类冲突（内部误标证据，非邻居驱动）
    for o in ided:
        nc = _class_of(o.get("name"))
        if nc is None:
            continue
        conflicts = []
        for al in o.get("aliases") or []:
            ac = _class_of(al)
            if ac is not None and ac != nc:
                conflicts.append(al)
        if conflicts:
            cands.append({
                "group_id": f"g{len(cands)}",
                "kind": "alias_conflict",
                "ids": [o["id"]],
                "detail": f"'{o.get('name')}'(类={nc}) 的别名含异类 {conflicts}(类冲突)；疑被误标，"
                          f"请据邻域/尺寸判该处真类→改名，或确属幻觉→丢弃",
            })

    # 不可能尺寸
    for o in ided:
        detail = _size_candidate(o)
        if detail:
            cands.append({
                "group_id": f"g{len(cands)}",
                "kind": "impossible_size",
                "ids": [o["id"]],
                "detail": detail,
            })
    return cands


def _gate_verdicts(verdicts: dict, flagged_ids: set, all_ids: set,
                   by_id: dict = None, cands: list = None) -> dict:
    """护栏（代码强制红线）：
    - merges 只允许命中 flagged 候选 id（name_dup/synonym_dup，近乎重合的同类）；
    - drops【只】允许 impossible_size 候选（物理不可能=多半几何噪声）。alias_conflict 是【真物体被误标】，
      只准改名不准删——删一个真物体比留点噪声危害大（召回红线）。凭空幻觉的降噪不靠删候选实现。
    - corrections 的【类翻转】改名(old 类≠new 类，两者都可归类)只在有内部证据时放行——
      id 命中 alias_conflict/impossible_size 候选，或 new_name 的类 ∈ 该物体 aliases 的类集合；
      否则剔除、保留原名(防"按邻居改名"抹掉少数类真物体→压召回)。同类细化(桌→办公桌)永远放行。
    - corrections/relations/member_ids 引用未知 id 剔除。
    """
    out = {"sub_areas": [], "corrections": [], "relations": [], "merges": [], "drops": []}
    rejects = []
    by_id = by_id or {}
    # 可为"类翻转"改名提供内部证据的 id（误标类候选）。
    evidence_ids = {i for c in (cands or [])
                    if c.get("kind") in ("alias_conflict", "impossible_size")
                    for i in c.get("ids", [])}
    # 可删的 id：仅 impossible_size（若未传 cands 则退化到 flagged_ids，保持向后兼容）。
    droppable_ids = ({i for c in cands if c.get("kind") == "impossible_size" for i in c.get("ids", [])}
                     if cands is not None else set(flagged_ids))

    for m in verdicts.get("merges") or []:
        keep = (m.get("keep_id") or "").strip()
        drops = [d for d in (m.get("drop_ids") or []) if (d or "").strip() in all_ids]
        # keep + 所有 drop 都必须在 flagged 集合内（Claude 只能对代码检出候选合并）
        if keep in flagged_ids and keep in all_ids and drops and all(d in flagged_ids for d in drops):
            out["merges"].append({"keep_id": keep, "drop_ids": drops,
                                  "name": m.get("name"), "why": m.get("why", "")})
        else:
            rejects.append(f"merge keep={keep!r} drops={m.get('drop_ids')} 越权/未知 id")

    for d in verdicts.get("drops") or []:
        did = (d.get("id") or "").strip()
        if did in droppable_ids and did in all_ids:
            out["drops"].append({"id": did, "why": d.get("why", "")})
        else:
            rejects.append(f"drop id={did!r} 非可删候选(只 impossible_size 可删)/未知")

    for c in verdicts.get("corrections") or []:
        cid = (c.get("id") or "").strip()
        new_name = (c.get("new_name") or "").strip()
        if cid not in all_ids or not new_name:
            rejects.append(f"correction id={cid!r} 未知/无 new_name")
            continue
        o = by_id.get(cid) or {}
        old_cls, new_cls = _class_of(o.get("name")), _class_of(new_name)
        is_flip = old_cls is not None and new_cls is not None and old_cls != new_cls
        if is_flip:
            alias_cls = {_class_of(a) for a in (o.get("aliases") or [])}
            has_evidence = cid in evidence_ids or new_cls in alias_cls
            if not has_evidence:
                rejects.append(
                    f"correction id={cid!r} 类翻转 {old_cls}→{new_cls} 无内部证据(非误标候选/别名不含新类)，保留原名")
                continue
        out["corrections"].append({"id": cid, "new_name": new_name, "why": c.get("why", "")})

    for r in verdicts.get("relations") or []:
        s, t = (r.get("subject_id") or "").strip(), (r.get("object_id") or "").strip()
        if s in all_ids and t in all_ids and r.get("predicate"):
            out["relations"].append({"subject_id": s, "predicate": r["predicate"],
                                     "object_id": t, "why": r.get("why", "")})
        else:
            rejects.append(f"relation {s!r}-{r.get('predicate')}-{t!r} 未知 id")

    for sa in verdicts.get("sub_areas") or []:
        member_ids = [m for m in (sa.get("member_ids") or []) if (m or "").strip() in all_ids]
        if member_ids:
            out["sub_areas"].append({"label": sa.get("label", ""), "type": sa.get("type", "area"),
                                     "member_ids": member_ids, "summary": sa.get("summary", "")})
        else:
            rejects.append(f"sub_area label={sa.get('label')!r} 无有效成员")

    if rejects:
        print(f"⚠️ gate 剔除 {len(rejects)} 项越权/无效裁决：")
        for r in rejects[:20]:
            print(f"   - {r}")
    return out


def _print_dry_run(rec: dict, gated: dict) -> None:
    by_id = {o["id"]: o for o in rec["objects"] if isinstance(o, dict) and o.get("id")}
    print("\n===== DRY-RUN（不写盘）=====")
    print(f"merges({len(gated['merges'])}):")
    for m in gated["merges"]:
        names = [by_id.get(i, {}).get("name", "?") for i in [m["keep_id"]] + m["drop_ids"]]
        print(f"   keep {m['keep_id']}({by_id.get(m['keep_id'],{}).get('name')}) ← {m['drop_ids']}"
              f"  → {m.get('name')}  [{'/'.join(names)}]  因: {m.get('why')}")
    print(f"drops({len(gated['drops'])}):")
    for d in gated["drops"]:
        print(f"   {d['id']}({by_id.get(d['id'],{}).get('name')})  因: {d.get('why')}")
    print(f"renames({len(gated['corrections'])}):")
    for c in gated["corrections"]:
        print(f"   {c['id']}: {by_id.get(c['id'],{}).get('name')} → {c['new_name']}  因: {c.get('why')}")
    print(f"relations({len(gated['relations'])}):")
    for r in gated["relations"]:
        print(f"   {by_id.get(r['subject_id'],{}).get('name')} {r['predicate']} "
              f"{by_id.get(r['object_id'],{}).get('name')}")
    print(f"sub_areas({len(gated['sub_areas'])}):")
    for sa in gated["sub_areas"]:
        members = [by_id[i] for i in sa["member_ids"] if i in by_id]
        rng = FsMemory._sub_area_range(members)
        mnames = "、".join(by_id.get(i, {}).get("name", "?") for i in sa["member_ids"])
        print(f"   [{sa.get('type')}] {sa.get('label')}  range={rng}")
        print(f"       成员({len(sa['member_ids'])}): {mnames}")
        print(f"       summary: {sa.get('summary')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Claude 记忆整理官（离线词典式整理 pass）")
    ap.add_argument("--env", default=config.ENV_NAME)
    ap.add_argument("--area", default="explore_room")
    ap.add_argument("--model", default=config.CURATE_MODEL)
    ap.add_argument("--merge-m", type=float, default=MERGE_M,
                    help="合并候选半径（米）：只把近乎重合的同类物体作合并候选，默认收紧以护召回")
    ap.add_argument("--dry-run", action="store_true", help="只打印意图 diff，不写盘")
    args = ap.parse_args()

    mem = FsMemory(root=config.MEMORY_ROOT, env_name=args.env)
    rec = mem.load_area(args.area)
    if rec is None:
        print(f"❌ 区域记录不存在: {args.env}/{args.area}")
        return 1
    objects = rec.get("objects") or []
    if not objects:
        print(f"⚠️ {args.area} 无 objects，无需整理")
        return 0

    n_new = _ensure_ids(rec)
    all_ids = {o["id"] for o in rec["objects"] if isinstance(o, dict) and o.get("id")}
    by_id = {o["id"]: o for o in rec["objects"] if isinstance(o, dict) and o.get("id")}
    cands = _flag_candidates(rec["objects"], merge_m=args.merge_m)
    flagged_ids = {i for c in cands for i in c["ids"]}
    print(f"📋 {args.area}: {len(objects)} 物体（新分配 id {n_new}）；代码检出离谱候选 {len(cands)} 组"
          f"（涉及 {len(flagged_ids)} 物）")

    payload = {
        "area": rec.get("area", args.area),
        "type": rec.get("type"),
        "boundary": rec.get("boundary", {}),
        "doors": rec.get("doors", []),
        "objects": [
            {"id": o["id"], "name": o.get("name"), "aliases": o.get("aliases", []),
             "abs_pose": o.get("abs_pose"), "size": o.get("size"),
             "spatial": o.get("spatial"), "confidence": o.get("confidence")}
            for o in rec["objects"] if isinstance(o, dict) and o.get("id")
        ],
        "sanity_candidates": cands,
    }

    verdicts = make_director(model=args.model).curate_area(payload)
    if not verdicts:
        print("跳过整理（整理官返回空/失败）——文件未改动")
        return 0

    gated = _gate_verdicts(verdicts, flagged_ids, all_ids, by_id=by_id, cands=cands)

    if args.dry_run:
        _print_dry_run(rec, gated)
        print("\n(dry-run：未写盘)")
        return 0

    path = mem.apply_curation(args.area, gated, record=rec, validator=validate_curated_record)
    out = mem.load_area(args.area)
    print(f"✅ 写回 {path}")
    print(f"   sub_areas={len(out.get('sub_areas', []))}  relations={len(out.get('relations', []))}  "
          f"corrections={len(out.get('corrections', []))}  objects={len(out.get('objects', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
