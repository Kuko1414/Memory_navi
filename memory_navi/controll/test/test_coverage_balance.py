"""覆盖均衡单测（_least_covered_quad / _balanced_pick）：候选优先落在【访问最少的象限】，
破 frontier 近邻贪心的方向漂移（治"预算被一个方向吃光、别的区整片漏"的覆盖方差）。纯函数。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import explore_probe as ep  # noqa: E402

BBOX = {"xmin": -4.0, "xmax": 4.0, "ymin": -4.0, "ymax": 4.0}   # 中点(0,0)


def _c(cid, x, y):
    return {"id": cid, "x": x, "y": y}


def test_quad_visit_counts():
    vc = ep._quad_visit_counts([[2, 2], [3, 3], [-2, 2], [2, -2]], BBOX)
    assert vc == {"NE": 2, "NW": 1, "SE": 1, "SW": 0}


def test_balanced_pick_avoids_overvisited_quadrant():
    """NE 被扎了 5 次、其余 0 → 候选各象限一个 → 不选 NE。"""
    visited = [[2, 2], [3, 3], [2, 3], [3, 2], [2.5, 2.5]]   # 全 NE
    cands = [_c("ne", 2, 2), _c("nw", -2, 2), _c("se", 2, -2), _c("sw", -2, -2)]
    tgt = ep._balanced_pick(cands, BBOX, visited)
    q = ep._quad_of(tgt["x"], tgt["y"], 0.0, 0.0)
    assert q != "NE", f"应避开过度访问的 NE，实际选了 {q}"


def test_balanced_pick_nearest_within_quadrant():
    """同一欠覆盖象限里取近优先（cands 已按近排序 → 取第一个）。"""
    visited = [[2, 2]]   # NE 有 1
    # NW 两个候选，近的在前（模拟 _occ_candidates 的近优先）
    cands = [_c("nw_near", -1, 1), _c("nw_far", -3.5, 3.5), _c("ne", 2, 2)]
    tgt = ep._balanced_pick(cands, BBOX, visited)
    assert tgt["id"] == "nw_near"


def test_balance_filter_restricts_to_least_covered():
    visited = [[2, 2], [3, 3], [3, 2]]   # NE=3
    cands = [_c("ne1", 2, 2), _c("ne2", 3, 3), _c("sw1", -2, -2)]
    sub = ep._balance_filter(cands, BBOX, visited)
    assert all(ep._quad_of(c["x"], c["y"], 0.0, 0.0) == "SW" for c in sub)


def test_no_bbox_returns_first():
    cands = [_c("a", 1, 1), _c("b", 2, 2)]
    assert ep._balanced_pick(cands, None, [])["id"] == "a"
    assert ep._balance_filter(cands, None, []) == cands


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_coverage_balance: all PASS")
