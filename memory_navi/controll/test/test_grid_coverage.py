"""均匀网格覆盖目标生成单测（_uncovered_grid_targets）：bbox 内均匀铺格、过滤占用/已覆盖/blocked。纯函数。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import explore_probe as ep  # noqa: E402
from agent_core.geometry import occupancy as oc  # noqa: E402

BBOX = {"xmin": -3.0, "xmax": 3.0, "ymin": -3.0, "ymax": 3.0}


def test_empty_bbox_no_targets():
    assert ep._uncovered_grid_targets(None, oc.OccGrid(), [], set()) == []


def test_uniform_grid_spans_bbox_including_south():
    occ = oc.OccGrid()
    targets = ep._uncovered_grid_targets(BBOX, occ, [], set())
    assert len(targets) > 0
    xs = [t[0] for t in targets]
    ys = [t[1] for t in targets]
    # 均匀铺满：既有北(y>1)也有南(y<-1)的目标（治"南带无候选"）
    assert any(y < -1.0 for y in ys), "南带应有网格目标"
    assert any(y > 1.0 for y in ys), "北带应有网格目标"
    assert min(xs) < -1.0 and max(xs) > 1.0, "东西两侧都应有目标"


def test_covered_cells_excluded():
    """某格中心 COVER_RADIUS 内有过 vantage → 不再作目标。"""
    occ = oc.OccGrid()
    all_t = ep._uncovered_grid_targets(BBOX, occ, [], set())
    # 用第一个目标当作"已观测点" → 它自身应被排除
    vx, vy = all_t[0]
    left = ep._uncovered_grid_targets(BBOX, occ, [(vx, vy)], set())
    assert (vx, vy) not in left
    assert len(left) < len(all_t)


def test_occupied_cells_excluded():
    """占用格(墙)不作观测目标。"""
    occ = oc.OccGrid()
    # 把某个网格中心所在格标 occupied
    all_t = ep._uncovered_grid_targets(BBOX, occ, [], set())
    tx, ty = all_t[0]
    occ.mark(occ.cell_of(tx, ty), oc.OCCUPIED)
    left = ep._uncovered_grid_targets(BBOX, occ, [], set())
    assert (tx, ty) not in left


def test_blocked_grid_excluded():
    occ = oc.OccGrid()
    all_t = ep._uncovered_grid_targets(BBOX, occ, [], set())
    tx, ty = all_t[0]
    left = ep._uncovered_grid_targets(BBOX, occ, [], {(round(tx, 1), round(ty, 1))})
    assert (tx, ty) not in left


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_grid_coverage: all PASS")
