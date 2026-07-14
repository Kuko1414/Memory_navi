"""fs_memory.match_object / match_index 单测（纯函数，无仿真/无 LLM/无 mcp）。

只读身份接地匹配：给 abs_pose(+name / id) 返回记忆里匹配的物体。id 优先、否则同名 xy≤tol 取最近、
同名更远＝不同实例(不匹配)、缺 pose 退化按名、无匹配 None。与 upsert_object 写路径共用同一语义。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core.memory.fs_memory import match_object, match_index  # noqa: E402


def _objs():
    return [
        {"id": "gt_00", "name": "绿植", "abs_pose": {"x": 6.23, "y": 1.42}},
        {"id": "gt_01", "name": "绿植", "abs_pose": {"x": -5.78, "y": -2.17}},
        {"id": "gt_02", "name": "沙发", "abs_pose": {"x": 1.0, "y": 0.0}},
        {"id": "gt_03", "name": "椅子"},                 # 无 abs_pose
    ]


def test_match_by_id():
    assert match_object(_objs(), obj_id="gt_01")["id"] == "gt_01"


def test_id_given_but_absent_is_no_match():
    # 给了 id 但没命中 → 视为新实例，不退化按名
    assert match_object(_objs(), {"x": 6.23, "y": 1.42}, "绿植", obj_id="zzz") is None


def test_match_by_name_pos_nearest_within_tol():
    m = match_object(_objs(), {"x": 6.2, "y": 1.4}, "绿植")
    assert m["id"] == "gt_00"                            # 取最近的那株，不是远的 gt_01


def test_same_name_outside_tol_is_no_match():
    # 两株绿植都离查询点很远(>0.8m) → 判不同实例，无匹配
    assert match_object(_objs(), {"x": 100.0, "y": 100.0}, "绿植") is None


def test_missing_pose_falls_back_to_name():
    assert match_object(_objs(), None, "椅子")["id"] == "gt_03"


def test_no_name_no_id_returns_none():
    assert match_object(_objs(), {"x": 0, "y": 0}) is None


def test_unknown_name_returns_none():
    assert match_object(_objs(), {"x": 1.0, "y": 0.0}, "床") is None


def test_match_index_matches_object():
    objs = _objs()
    idx = match_index(objs, name="绿植", abs_pose={"x": -5.7, "y": -2.1})
    assert objs[idx]["id"] == "gt_01"
