"""物体多视角一致性投票单测（_dedup_objects 的 min_views 门）：孤帧幻觉丢弃、多视角保留。

投票由确定性代码执行（比对反投世界坐标数"几个独立 vantage 报到"），Qwen 不参与——本测直接
喂构造好的 vantage_records，断言：单视角物体在 min_views=2 时被丢进 dropped_out，双视角保留，
min_views=1 时全保留（向后兼容），且返回物体不残留内部 _views 记账字段。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import explore_probe as ep  # noqa: E402


def _obj(name, x, y, **kw):
    """构造一条落在可信高度带、界内的可信观测。"""
    o = {"name": name, "abs_pose": {"x": x, "y": y, "z": 0.5}, "distance_m": 1.2,
         "confidence": 0.8}
    o.update(kw)
    return o


BBOX = {"xmin": -5.0, "xmax": 5.0, "ymin": -5.0, "ymax": 5.0}


def test_single_vantage_dropped_as_phantom():
    """只被单一 vantage 报到的可信物体 → min_views=2 时判孤帧幻觉丢弃，收进 dropped_out。"""
    records = [
        {"objects": [_obj("柜子", 1.0, 1.0)]},        # vantage 0：真物体，两视角都见
        {"objects": [_obj("柜子", 1.05, 0.95)]},      # vantage 1：同位置(≤DEDUP_M)→合并=2视角
        {"objects": [_obj("幻影柜子", 3.0, -2.0)]},   # vantage 2：仅此一帧→幻觉
    ]
    dropped = []
    kept = ep._dedup_objects(records, BBOX, min_views=2, dropped_out=dropped)
    names = [o["name"] for o in kept]
    assert "柜子" in names, f"双视角真物体应保留：{names}"
    assert "幻影柜子" not in names, f"单视角幻觉应被丢：{names}"
    assert len(dropped) == 1 and dropped[0]["name"] == "幻影柜子"


def test_min_views_1_keeps_all():
    """min_views=1（默认，中途覆盖统计用）→ 不投票，孤帧也保留（向后兼容）。"""
    records = [{"objects": [_obj("绿植", 2.0, 2.0)]}]
    kept = ep._dedup_objects(records, BBOX, min_views=1)
    assert [o["name"] for o in kept] == ["绿植"]


def test_default_is_no_vote():
    """不传 min_views → 默认 1 → 单视角保留（保证 _objects_xy 中途取全量不受影响）。"""
    records = [{"objects": [_obj("沙发", -1.0, -1.0)]}]
    assert len(ep._dedup_objects(records, BBOX)) == 1


def test_no_views_field_leaks():
    """返回物体不得残留内部记账字段 _views（非 schema，勿写盘）。"""
    records = [
        {"objects": [_obj("桌子", 0.0, 0.0)]},
        {"objects": [_obj("桌子", 0.1, 0.0)]},
    ]
    kept = ep._dedup_objects(records, BBOX, min_views=2)
    assert kept and all("_views" not in o for o in kept)


def test_two_distinct_names_same_spot_each_single():
    """同位置但不同类各自成条（_name_compat 不合并）→ 各仅 1 视角 → min_views=2 全丢。"""
    records = [
        {"objects": [_obj("柜子", 1.0, 1.0)]},
        {"objects": [_obj("绿植", 1.0, 1.0)]},   # 同位不同类 → 不合并 → 各单视角
    ]
    dropped = []
    kept = ep._dedup_objects(records, BBOX, min_views=2, dropped_out=dropped)
    assert kept == [] and len(dropped) == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_multiview_vote: all PASS")
