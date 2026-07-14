"""记忆整理官候选检出/护栏 单测（纯函数、无网络、不连 ROS/LLM）。

覆盖召回优先重构：近乎重合才并(name_dup/synonym_dup)、成排不同实例不并、alias 内部证据纠名、
gate 拦截跨类 merge 与无据的"类翻转"改名。在含 anthropic 的 vllm 环境跑(curate_memory 顶层依赖它)。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import curate_memory as cm  # noqa: E402
from agent_core.memory.fs_memory import FsMemory  # noqa: E402


def _o(oid, name, x, y, z=0.5, aliases=None, size=None):
    return {"id": oid, "name": name, "abs_pose": {"x": x, "y": y, "z": z},
            "aliases": aliases or [], "size": size}


def _kinds_by_id(cands, oid):
    return {c["kind"] for c in cands if oid in c["ids"]}


# ---- _class_of ----
def test_class_of_synonyms_and_crossclass():
    assert cm._class_of("tv") == "显示器"
    assert cm._class_of("monitor") == "显示器"
    assert cm._class_of("办公桌") == "办公桌"
    assert cm._class_of("桌子") == "办公桌"
    assert cm._class_of("办公椅") == "办公椅"
    assert cm._class_of("柜子(南1)") == "柜子"
    assert cm._class_of("绿植") == "绿植"
    # 跨类不混：桌 ≠ 椅 ≠ 显示器 ≠ 柜
    assert len({cm._class_of("办公桌"), cm._class_of("办公椅"),
                cm._class_of("显示器"), cm._class_of("柜子")}) == 4


# ---- 第1层：近乎重合合并候选 ----
def test_two_coincident_sofas_one_name_dup():
    objs = [_o("o1", "沙发", 1.0, 1.0), _o("o2", "沙发", 1.05, 1.02)]
    cands = cm._flag_candidates(objs)
    dups = [c for c in cands if c["kind"] == "name_dup"]
    assert len(dups) == 1
    assert set(dups[0]["ids"]) == {"o1", "o2"}


def test_monitor_and_tv_adjacent_synonym_dup():
    objs = [_o("o1", "显示器", 2.0, 0.0), _o("o2", "tv", 2.1, 0.02)]
    cands = cm._flag_candidates(objs)
    syn = [c for c in cands if c["kind"] == "synonym_dup"]
    assert len(syn) == 1
    assert set(syn[0]["ids"]) == {"o1", "o2"}


def test_three_cabinets_in_row_not_merged():
    """成排间隔柜子(0.6m > MERGE_M)=不同实例，不应聚成合并候选(护召回)。"""
    assert cm.MERGE_M < 0.6
    objs = [_o("o1", "柜子", 0.0, 0.0), _o("o2", "柜子", 0.6, 0.0), _o("o3", "柜子", 1.2, 0.0)]
    cands = cm._flag_candidates(objs)
    merge_cands = [c for c in cands if c["kind"] in ("name_dup", "synonym_dup")]
    assert merge_cands == [], f"成排柜子不应被并: {merge_cands}"


# ---- 第2层：alias 内部证据纠名 / 尺寸 ----
def test_alias_conflict_flagged():
    objs = [_o("o1", "绿植", 1.0, 1.0, aliases=["柜子"])]
    cands = cm._flag_candidates(objs)
    assert _kinds_by_id(cands, "o1") == {"alias_conflict"}


def test_no_isolated_outlier_by_neighbor_majority():
    """去牙：孤立少数类物体(邻居全异类)但自身无内部冲突 → 不再被标(不按邻居改名)。"""
    objs = [_o("g", "柜子", 0.0, 0.0),
            _o("n1", "办公桌", 0.4, 0.0), _o("n2", "显示器", 0.0, 0.4), _o("n3", "办公椅", 0.4, 0.4)]
    cands = cm._flag_candidates(objs)
    assert _kinds_by_id(cands, "g") == set(), "孤立柜子不应被邻居驱动标记"


def test_impossible_size_flagged():
    objs = [_o("o1", "办公桌", 1.0, 1.0, size={"width_m": 5.0, "depth_m": 0.5})]
    cands = cm._flag_candidates(objs)
    assert _kinds_by_id(cands, "o1") == {"impossible_size"}


# ---- gate 护栏 ----
def _cands_synonym(o1, o2):
    return [{"group_id": "g0", "kind": "synonym_dup", "ids": [o1, o2], "detail": ""}]


def test_gate_synonym_merge_allowed():
    cands = _cands_synonym("o1", "o2")
    flagged, all_ids = {"o1", "o2"}, {"o1", "o2"}
    v = {"merges": [{"keep_id": "o1", "drop_ids": ["o2"], "name": "显示器", "why": ""}]}
    out = cm._gate_verdicts(v, flagged, all_ids, by_id={}, cands=cands)
    assert len(out["merges"]) == 1


def test_gate_crossclass_merge_rejected():
    """未 flagged 的 id 不能被合并(跨真类不会成候选)。"""
    flagged, all_ids = set(), {"o1", "o2"}
    v = {"merges": [{"keep_id": "o1", "drop_ids": ["o2"], "name": "办公桌", "why": ""}]}
    out = cm._gate_verdicts(v, flagged, all_ids, by_id={}, cands=[])
    assert out["merges"] == []


def test_gate_drop_alias_conflict_rejected():
    """alias_conflict=真物体被误标，只准改名不准删(护召回)。"""
    cands = [{"group_id": "g0", "kind": "alias_conflict", "ids": ["o1"], "detail": ""}]
    v = {"drops": [{"id": "o1", "why": "幻觉"}]}
    out = cm._gate_verdicts(v, {"o1"}, {"o1"}, by_id={}, cands=cands)
    assert out["drops"] == []


def test_gate_drop_impossible_size_allowed():
    cands = [{"group_id": "g0", "kind": "impossible_size", "ids": ["o1"], "detail": ""}]
    v = {"drops": [{"id": "o1", "why": "宽5m对桌不可能"}]}
    out = cm._gate_verdicts(v, {"o1"}, {"o1"}, by_id={}, cands=cands)
    assert len(out["drops"]) == 1


def test_gate_classflip_rename_without_evidence_rejected():
    by_id = {"o1": {"id": "o1", "name": "柜子", "aliases": []}}
    v = {"corrections": [{"id": "o1", "new_name": "办公桌", "why": "邻居都是桌"}]}
    out = cm._gate_verdicts(v, set(), {"o1"}, by_id=by_id, cands=[])
    assert out["corrections"] == [], "无据的类翻转应被拦截"


def test_gate_classflip_rename_with_alias_evidence_allowed():
    by_id = {"o1": {"id": "o1", "name": "绿植", "aliases": ["柜子"]}}
    cands = [{"group_id": "g0", "kind": "alias_conflict", "ids": ["o1"], "detail": ""}]
    v = {"corrections": [{"id": "o1", "new_name": "柜子", "why": "别名冲突"}]}
    out = cm._gate_verdicts(v, {"o1"}, {"o1"}, by_id=by_id, cands=cands)
    assert len(out["corrections"]) == 1 and out["corrections"][0]["new_name"] == "柜子"


def test_gate_sameclass_refine_allowed():
    """同类细化(桌子→办公桌)不是类翻转，永远放行。"""
    by_id = {"o1": {"id": "o1", "name": "桌子", "aliases": []}}
    v = {"corrections": [{"id": "o1", "new_name": "办公桌", "why": "规范名"}]}
    out = cm._gate_verdicts(v, set(), {"o1"}, by_id=by_id, cands=[])
    assert len(out["corrections"]) == 1


# ---- apply_curation：坐标不动 + 子区 range + 别名并集 ----
def test_apply_curation_coords_untouched_and_subarea(tmp_path):
    env, area = "sim", "explore_room"
    d = tmp_path / env / area
    d.mkdir(parents=True)
    rec = {"area": area, "type": "office", "objects": [
        _o("o1", "办公桌", 2.0, 0.0), _o("o2", "办公椅", 2.5, 0.5)]}
    (d / "area.json").write_text(__import__("json").dumps(rec, ensure_ascii=False), encoding="utf-8")
    mem = FsMemory(root=str(tmp_path), env_name=env)
    loaded = mem.load_area(area)
    verdicts = {"sub_areas": [{"label": "子办公区", "type": "office",
                               "member_ids": ["o1", "o2"], "summary": "桌+椅"}]}
    mem.apply_curation(area, verdicts, record=loaded, validator=None)
    out = mem.load_area(area)
    by = {o["id"]: o for o in out["objects"]}
    assert by["o1"]["abs_pose"] == {"x": 2.0, "y": 0.0, "z": 0.5}   # 坐标绝不动
    assert by["o2"]["abs_pose"] == {"x": 2.5, "y": 0.5, "z": 0.5}
    sa = out["sub_areas"][0]
    assert set(sa["member_ids"]) == {"o1", "o2"}
    assert sa["range"] == {"xmin": 2.0, "xmax": 2.5, "ymin": 0.0, "ymax": 0.5,
                           "zmin": 0.5, "zmax": 0.5}               # range 由代码从坐标算


def test_apply_curation_rename_keeps_old_name_in_aliases(tmp_path):
    env, area = "sim", "explore_room"
    d = tmp_path / env / area
    d.mkdir(parents=True)
    rec = {"area": area, "type": "office", "objects": [_o("o1", "绿植", 1.0, 1.0, aliases=["柜子"])]}
    (d / "area.json").write_text(__import__("json").dumps(rec, ensure_ascii=False), encoding="utf-8")
    mem = FsMemory(root=str(tmp_path), env_name=env)
    loaded = mem.load_area(area)
    verdicts = {"corrections": [{"id": "o1", "new_name": "柜子", "why": "别名冲突纠正"}]}
    mem.apply_curation(area, verdicts, record=loaded, validator=None)
    out = mem.load_area(area)
    o = out["objects"][0]
    assert o["name"] == "柜子"
    assert "绿植" in (o.get("aliases") or [])   # 旧名并入别名(并集不覆盖)
    assert o.get("old_name") == "绿植"
