"""occ 自由空间/视线反证单测（_occ_free_phantom / _occupancy_phantoms / _backfill LOS）。

治「离轨迹的空地板幻觉」(1a) 与「钉在墙后的幻觉」(1b)：用 occ 占据栅格(已扫自由空间+墙体几何)
反证反投坐标。精度优先——贴墙真家具豁免、无坐标/occ 空一律保留(不误杀)。纯几何、脱 ROS/LLM。
在含 openai/anthropic 的 vllm 环境跑（explore_probe 顶层依赖它们）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import explore_probe as ep  # noqa: E402
from agent_core.geometry import occupancy as oc  # noqa: E402


def _grid_open_floor():
    """一片已扫自由空间：以 (0,0) 为中心 5x5 格全 FREE，四周无 occupied（空旷地板）。"""
    g = oc.OccGrid(res_m=0.4)
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            g.mark((ix, iy), oc.FREE)
    return g


def _obj(name, x, y):
    return {"name": name, "abs_pose": {"x": x, "y": y, "z": 0.5}}


def test_open_floor_phantom_dropped():
    """abs_pose 落在空旷自由格、邻域无 occupied → 空旷地板幻觉 → 丢。"""
    g = _grid_open_floor()
    assert ep._occ_free_phantom(g, 0.0, 0.0) is True
    kept, dropped = ep._occupancy_phantoms([_obj("幻影柜子", 0.0, 0.0)], g)
    assert kept == [] and [o["name"] for o in dropped] == ["幻影柜子"]


def test_near_wall_object_kept():
    """自由格但邻域内有 occupied（贴墙真家具）→ 豁免保留（精度护栏）。"""
    g = _grid_open_floor()
    g.mark((2, 0), oc.OCCUPIED)                 # (0.8,0) 处一堵墙
    # (0.4,0)=格(1,0) 是自由格，但 clear 半径 0.6m(≈2格) 内有 occupied → 豁免
    assert ep._occ_free_phantom(g, 0.4, 0.0) is False
    kept, dropped = ep._occupancy_phantoms([_obj("柜子", 0.4, 0.0)], g)
    assert [o["name"] for o in kept] == ["柜子"] and dropped == []


def test_occupied_cell_kept():
    """abs_pose 落在 occupied 格本身（钉在墙面上）→ 不由本门处理 → 保留。"""
    g = _grid_open_floor()
    g.mark((2, 0), oc.OCCUPIED)
    assert ep._occ_free_phantom(g, 0.8, 0.0) is False


def test_unknown_cell_kept():
    """abs_pose 落在未观测格 → 无从反证 → 保留。"""
    g = _grid_open_floor()
    assert ep._occ_free_phantom(g, 10.0, 10.0) is False


def test_empty_grid_keeps_all():
    """occ 空 / None → 全保留，不误杀。"""
    objs = [_obj("沙发", 0.0, 0.0)]
    assert ep._occupancy_phantoms(objs, oc.OccGrid())[0] == objs
    assert ep._occupancy_phantoms(objs, None)[0] == objs


def test_no_abs_pose_kept():
    """无 abs_pose 的物体无从反证 → 保留。"""
    g = _grid_open_floor()
    objs = [{"name": "显示器"}]
    kept, dropped = ep._occupancy_phantoms(objs, g)
    assert [o["name"] for o in kept] == ["显示器"] and dropped == []


def test_los_through_wall_detected():
    """观测位姿→abs_pose 线段穿过 occupied 墙 = 相机隔墙看不到 → line_free 判 False（1b 依据）。"""
    g = oc.OccGrid(res_m=0.4)
    g.mark((3, 0), oc.OCCUPIED)                 # (1.2,0) 一堵墙
    # 从 (0,0) 看 (2.0,0)：中途穿过 (1.2,0) 墙 → 视线被挡
    assert oc.line_free(g, 0.0, 0.0, 2.0, 0.0) is False
    # 墙前的物体 (0.8,0)：视线不穿墙 → 通行
    assert oc.line_free(g, 0.0, 0.0, 0.8, 0.0) is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_occupancy_phantom: all PASS")
