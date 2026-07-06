"""自由空间反证单测（_free_space_phantoms）：反投坐标落在车已走过位姿的车体半径内 → 判幻觉丢弃。

治近距离幻觉、零召回代价：真物体不在车走过的点上。纯几何、代码执行，不连 ROS/LLM。
在含 openai/anthropic 的 vllm 环境跑（explore_probe 顶层依赖它们）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import explore_probe as ep  # noqa: E402


def _obj(name, x, y):
    return {"name": name, "abs_pose": {"x": x, "y": y, "z": 0.5}, "distance_m": 1.2}


def test_object_on_path_dropped():
    """物体反投坐标落在某已走过位姿半径内 → 车穿过该处 → 判幻觉丢弃。"""
    visited = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    objs = [_obj("幻影柜子", 1.0, 0.05)]        # 距 (1,0) 仅 0.05m < 0.3m
    kept, dropped = ep._free_space_phantoms(objs, visited)
    assert kept == [] and len(dropped) == 1 and dropped[0]["name"] == "幻影柜子"


def test_object_off_path_kept():
    """物体远离所有走过位姿 → 无从反证 → 保留（真墙边家具）。"""
    visited = [[0.0, 0.0], [1.0, 0.0]]
    objs = [_obj("红柜", 2.6, 0.2)]             # 距最近走过点 >1.5m
    kept, dropped = ep._free_space_phantoms(objs, visited)
    assert [o["name"] for o in kept] == ["红柜"] and dropped == []


def test_boundary_radius_kept():
    """恰在半径外(> radius) → 保留（仅 ≤ radius 才判穿过）。"""
    visited = [[0.0, 0.0]]
    objs = [_obj("绿植", ep.FREE_SPACE_R_M + 0.05, 0.0)]
    kept, dropped = ep._free_space_phantoms(objs, visited)
    assert kept and not dropped


def test_no_trajectory_keeps_all():
    """无轨迹（visited 空）→ 无从反证 → 全保留，不误杀。"""
    objs = [_obj("沙发", 0.0, 0.0)]
    kept, dropped = ep._free_space_phantoms(objs, [])
    assert kept == objs and dropped == []


def test_no_abs_pose_kept():
    """无 abs_pose 的物体无从反证 → 保留。"""
    visited = [[0.0, 0.0]]
    objs = [{"name": "显示器"}]                 # 无坐标
    kept, dropped = ep._free_space_phantoms(objs, visited)
    assert [o["name"] for o in kept] == ["显示器"] and dropped == []


def test_mixed_partition():
    """混合：轨迹上的丢、轨迹外的留 —— 只删压在自身轨迹上的近距离幻觉。"""
    visited = [[3.0, -1.0]]
    objs = [_obj("幻觉A", 3.05, -1.0), _obj("真物体B", 0.0, 2.0)]
    kept, dropped = ep._free_space_phantoms(objs, visited)
    assert [o["name"] for o in kept] == ["真物体B"]
    assert [o["name"] for o in dropped] == ["幻觉A"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_free_space_phantom: all PASS")
