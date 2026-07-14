"""detect_phase 真值表单测（纯函数、无仿真、无 LLM）。

无 area.json → explore；有区域无目标物 → completion；有区域且目标物在图 → execution。
只依赖 agent_core.phase + FsMemory（轻依赖），可在 base pytest 直接跑。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core.memory.fs_memory import FsMemory  # noqa: E402
from agent_core.phase import detect_phase, target_resolvable  # noqa: E402


def _mem(tmp_path, objects=None):
    m = FsMemory(str(tmp_path), "sim")
    if objects is not None:
        m.upsert_area("break_room_ideal", {"area": "break_room_ideal", "type": "lounge",
                                            "objects": objects})
    return m


# ---- 三分支真值表 ----
def test_no_area_json_is_explore(tmp_path):
    m = _mem(tmp_path)                                 # 没写任何区域
    assert detect_phase(m, "break_room_ideal", "去绿植") == "explore"


def test_area_without_target_is_completion(tmp_path):
    m = _mem(tmp_path, objects=[{"name": "沙发", "abs_pose": {"x": 1, "y": 0, "z": 0.3}}])
    assert detect_phase(m, "break_room_ideal", "去红柜后办公区的绿植") == "completion"


def test_area_with_target_is_execution(tmp_path):
    m = _mem(tmp_path, objects=[{"name": "绿植", "abs_pose": {"x": 6.23, "y": 1.42, "z": 0.3}},
                                {"name": "沙发", "abs_pose": {"x": 1, "y": 0, "z": 0.3}}])
    assert detect_phase(m, "break_room_ideal", "去绿植那里") == "execution"


def test_explicit_targets_drive_resolution(tmp_path):
    m = _mem(tmp_path, objects=[{"name": "绿植", "abs_pose": {"x": 6.23, "y": 1.42, "z": 0.3}}])
    targets = [{"name": "绿植", "x": 6.23, "y": 1.42}, {"name": "绿植", "x": -5.78, "y": -2.17}]
    assert detect_phase(m, "break_room_ideal", "从A区到B区", targets=targets) == "execution"


# ---- 别名匹配（非 embedding，但覆盖同义/英文别名）----
def test_alias_match_resolves_target(tmp_path):
    m = _mem(tmp_path, objects=[{"name": "绿植", "aliases": ["potted plant", "盆栽"],
                                 "abs_pose": {"x": 6.23, "y": 1.42, "z": 0.3}}])
    assert target_resolvable(m.load_area("break_room_ideal"), "go to the potted plant") is True
    assert target_resolvable(m.load_area("break_room_ideal"), "去沙发") is False


def test_detect_is_pure_no_side_effect(tmp_path):
    m = _mem(tmp_path, objects=[{"name": "绿植", "abs_pose": {"x": 6.23, "y": 1.42, "z": 0.3}}])
    p1 = detect_phase(m, "break_room_ideal", "去绿植")
    p2 = detect_phase(m, "break_room_ideal", "去绿植")
    assert p1 == p2 == "execution"                     # 幂等、无副作用
