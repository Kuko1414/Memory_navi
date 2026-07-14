"""map_resolve 纯函数单测：门解析、多实例、目标 id 反查、leg 丢弃/护栏判据。

无 mcp 依赖（map_resolve 只依赖 math/fs_memory/depth_projection），base pytest 直测。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core.modes import map_resolve as mr  # noqa: E402


def _rec():
    return {
        "objects": [
            {"id": "gt_25", "name": "绿植", "abs_pose": {"x": 6.23, "y": 1.42}},
            {"id": "gt_27", "name": "绿植", "abs_pose": {"x": -5.78, "y": -2.17}},
            {"id": "gt_00", "name": "沙发", "abs_pose": {"x": 1.0, "y": 0.0}},
        ],
        "sub_areas": [{"id": "sa_lounge_0", "label": "起始休息区",
                       "range": {"xmin": -2.0, "xmax": 2.0, "ymin": -2.0, "ymax": 2.0}}],
        "doors": [
            {"id": "door_0", "label": "红柜隔断北口", "pose": {"x": 2.65, "y": 1.30}},
            {"id": "door_1", "label": "西隔断北口", "pose": {"x": -2.65, "y": 2.70}},
        ],
    }


# ---- _via_candidates ----
def test_via_resolves_door_by_label():
    c = mr._via_candidates(_rec(), "西隔断北口")
    assert len(c) == 1 and c[0]["src"] == "door"
    assert (c[0]["x"], c[0]["y"]) == (-2.65, 2.7)


def test_via_resolves_door_by_id():
    c = mr._via_candidates(_rec(), "door_0")
    assert len(c) == 1 and c[0]["src"] == "door" and c[0]["x"] == 2.65


def test_via_resolves_subarea_center():
    c = mr._via_candidates(_rec(), "起始休息区")
    assert len(c) == 1 and c[0]["src"] == "sub_area"
    assert c[0]["x"] == 0.0 and c[0]["y"] == 0.0


def test_via_object_name_returns_all_instances():
    c = mr._via_candidates(_rec(), "绿植")
    assert len(c) == 2 and all(x["src"] == "object" for x in c)


# ---- _resolve_targets 补 id ----
def test_resolve_targets_backfills_memory_id():
    tgts = mr._resolve_targets(_rec(), {"targets": [
        {"name": "绿植", "x": 6.23, "y": 1.42},
        {"name": "绿植", "x": -5.78, "y": -2.17}]})
    assert [t["id"] for t in tgts] == ["gt_25", "gt_27"]


# ---- leg 丢弃 / 护栏 ----
def test_leg_hits_target_true_for_same_instance():
    tgt = {"name": "绿植", "x": 6.23, "y": 1.42}
    assert mr.leg_hits_target([{"name": "绿植", "x": 6.2, "y": 1.4, "src": "object"}], tgt) is True


def test_leg_hits_target_false_for_other_name_or_far():
    tgt = {"name": "绿植", "x": 6.23, "y": 1.42}
    assert mr.leg_hits_target([{"name": "沙发", "x": 6.2, "y": 1.4}], tgt) is False
    # 另一株绿植(远)不算"就是目标物本身"
    assert mr.leg_hits_target([{"name": "绿植", "x": -5.78, "y": -2.17}], tgt) is False


def test_leg_leads_away():
    tgt = {"name": "绿植", "x": 6.23, "y": 1.42}
    pose = {"x": 5.3, "y": -0.5}                          # 离目标 ~2m
    assert mr.leg_leads_away(-5.0, -1.0, pose, tgt) is True    # 落点离目标 ~11m ≫ 2m → 误解析
    assert mr.leg_leads_away(6.0, 1.0, pose, tgt) is False     # 落点更靠近目标 → 正常


# ---- door_leg_redundant（门同侧跳过护栏；竖隔断 x 分侧）----
def test_door_redundant_same_side():
    # 上次失败：起点(4.03)与目标(6.23)都在门(2.65)以东 → 同侧 → 穿门无意义
    assert mr.door_leg_redundant({"x": 4.03, "y": 0.39}, (2.65, 1.3), {"x": 6.23, "y": 1.42}) is True


def test_door_not_redundant_opposite_side():
    # 起点(-0.5)在门西、目标(6.23)在门东 → 异侧 → 该穿门
    assert mr.door_leg_redundant({"x": -0.5, "y": -1.0}, (2.65, 1.3), {"x": 6.23, "y": 1.42}) is False


def test_door_redundant_edge_too_close_to_judge():
    # 机器人贴门(|Δx|<0.3) → 无从判侧，保守不跳过
    assert mr.door_leg_redundant({"x": 2.75, "y": 1.3}, (2.65, 1.3), {"x": 6.23, "y": 1.42}) is False
