"""occupancy 栅格单测（geometry.occupancy）：纯几何、可脱 ROS/LLM。

覆盖：状态优先级不降级、射线标 free/命中标 occupied、无命中射线标 free 到量程、
frontier=自由邻接 unknown、低覆盖格、直线穿障判定。在 vllm 环境跑（与其它 test/ 统一）。
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core.geometry import occupancy as oc  # noqa: E402


def test_cell_roundtrip():
    g = oc.OccGrid(res_m=0.4)
    assert g.cell_of(0.0, 0.0) == (0, 0)
    assert g.cell_of(0.41, -0.39) == (1, -1)
    cx, cy = g.center((2, -3))
    assert math.isclose(cx, 0.8) and math.isclose(cy, -1.2)


def test_mark_priority_no_downgrade():
    """visited>occupied>free：free 不擦 occupied，visited 纠正 occupied。"""
    g = oc.OccGrid(0.4)
    g.mark((0, 0), oc.OCCUPIED)
    g.mark((0, 0), oc.FREE)                 # 更低优先级 → 不覆盖
    assert g.state((0, 0)) == oc.OCCUPIED
    g.mark((0, 0), oc.VISITED)              # 更高优先级 → 覆盖(真到过=非墙)
    assert g.state((0, 0)) == oc.VISITED
    g.mark((0, 0), oc.OCCUPIED)             # visited 后 occupied 不再降级
    assert g.state((0, 0)) == oc.VISITED
    assert g.state((9, 9)) == oc.UNKNOWN


def test_beam_hit_marks_free_then_occupied():
    """朝东(bearing 0, yaw 0)命中 2.0m：沿途标 free，2.0m 处标 occupied。"""
    g = oc.OccGrid(0.4)
    pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    n_occ = oc.update_from_beams(g, pose, [[0.0, 2.0]], range_max_m=8.0, max_mark_m=6.0)
    assert n_occ == 1
    assert g.state((5, 0)) == oc.OCCUPIED            # 2.0/0.4 = 5
    assert g.state((2, 0)) == oc.FREE                # 途中
    assert g.state((3, 0)) == oc.FREE


def test_beam_no_hit_marks_free_to_max():
    """无命中射线(dist=-1)朝北：标 free 到 max_mark，不产 occupied。"""
    g = oc.OccGrid(0.5)
    pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    n_occ = oc.update_from_beams(g, pose, [[90.0, -1]], range_max_m=8.0, max_mark_m=3.0)
    assert n_occ == 0
    assert g.state((0, 4)) == oc.FREE                # 北向 2.0m 处
    assert oc.OCCUPIED not in g.cells.values()


def test_frontier_free_adjacent_unknown():
    """孤立一个 free 格四邻全 unknown → 它是 frontier；被 free 包住的格不是。"""
    g = oc.OccGrid(0.4)
    g.mark((0, 0), oc.FREE)
    assert (0, 0) in oc.frontier_cells(g)
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        g.mark(d, oc.FREE)
    # (0,0) 现在四邻皆 free → 不再是 frontier（但四邻各自邻 unknown → 是 frontier）
    fr = oc.frontier_cells(g)
    assert (0, 0) not in fr
    assert (1, 0) in fr


def test_occupied_not_frontier():
    """occupied 格永不作 frontier 基底（墙不是可去的边界）。"""
    g = oc.OccGrid(0.4)
    g.mark((0, 0), oc.OCCUPIED)
    assert (0, 0) not in oc.frontier_cells(g)


def test_low_coverage_far_from_visited():
    """远离 visited 的自由格判低覆盖；近 visited 的不算；无 visited → 空。"""
    g = oc.OccGrid(0.4)
    assert oc.low_coverage_cells(g) == set()
    g.mark((0, 0), oc.VISITED)
    g.mark((1, 0), oc.FREE)                 # 0.4m，近
    g.mark((10, 0), oc.FREE)               # 4.0m，远
    low = oc.low_coverage_cells(g, radius_m=1.2)
    assert (10, 0) in low and (1, 0) not in low


def test_line_free_blocks_on_occupied():
    g = oc.OccGrid(0.4)
    # (0,0)->(2.0,0) 中间放一堵墙格
    g.mark((3, 0), oc.OCCUPIED)
    assert oc.line_free(g, 0.0, 0.0, 2.0, 0.0) is False
    assert oc.line_free(g, 0.0, 0.0, 0.0, 2.0) is True


def test_coverage_summary_counts():
    g = oc.OccGrid(0.4)
    g.mark((0, 0), oc.VISITED)
    g.mark((1, 0), oc.FREE)
    g.mark((5, 0), oc.OCCUPIED)
    s = oc.coverage_summary(g)
    assert s["visited"] == 1 and s["free"] == 1 and s["occupied"] == 1
    assert s["frontier"] >= 1


def _fill_free(g, x0, x1, y0, y1):
    """把矩形格区间标 free（含端点）。"""
    for ix in range(x0, x1 + 1):
        for iy in range(y0, y1 + 1):
            g.mark((ix, iy), oc.FREE)


def test_astar_straight_free():
    g = oc.OccGrid(0.4)
    _fill_free(g, 0, 5, 0, 0)
    path = oc.astar(g, (0, 0), (5, 0))
    assert path is not None and path[0] == (0, 0) and path[-1] == (5, 0)


def test_astar_detours_around_wall_through_gap():
    """一堵竖墙 x=3 挡在中间、仅 y=2 处有缝：A* 必须绕经缝到达东侧。"""
    g = oc.OccGrid(0.4)
    _fill_free(g, 0, 6, 0, 3)                # 一片自由区
    for iy in range(0, 4):                   # x=3 竖墙，仅 y=2 留缝(不标occupied=射线穿过处)
        if iy != 2:
            g.mark((3, iy), oc.OCCUPIED)
    path = oc.astar(g, (0, 0), (6, 0))
    assert path is not None
    assert (3, 2) in path                    # 必经缝
    # 墙格不在路径里
    assert all(g.state(c) != oc.OCCUPIED for c in path)


def test_astar_none_when_walled_off():
    """目标被墙完全封死(无缝) → 不可达 None。"""
    g = oc.OccGrid(0.4)
    _fill_free(g, 0, 6, 0, 3)
    for iy in range(0, 4):
        g.mark((3, iy), oc.OCCUPIED)         # 无缝整墙
    assert oc.astar(g, (0, 0), (6, 0)) is None


def test_astar_none_when_goal_unknown():
    g = oc.OccGrid(0.4)
    _fill_free(g, 0, 3, 0, 0)
    assert oc.astar(g, (0, 0), (9, 9)) is None   # goal 未知


def test_path_waypoints_straight_collapses():
    """直线走廊 → 简化成单个终点 waypoint。"""
    g = oc.OccGrid(0.4)
    _fill_free(g, 0, 5, 0, 0)
    path = oc.astar(g, (0, 0), (5, 0))
    wps = oc.path_waypoints(g, path)
    assert len(wps) == 1
    assert math.isclose(wps[-1][0], 2.0) and math.isclose(wps[-1][1], 0.0)


def test_path_waypoints_keeps_turn_around_wall():
    """绕墙路径有拐点 → waypoints 保留至少一个中转点（非单点直达）。"""
    g = oc.OccGrid(0.4)
    _fill_free(g, 0, 6, 0, 3)
    for iy in range(0, 4):
        if iy != 2:
            g.mark((3, iy), oc.OCCUPIED)
    path = oc.astar(g, (0, 0), (6, 0))
    wps = oc.path_waypoints(g, path)
    assert len(wps) >= 2                      # 需绕行，非直达
    assert wps[-1] == (round(6 * 0.4, 2), 0.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_occupancy: all PASS")
