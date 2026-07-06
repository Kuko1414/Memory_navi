"""势场(APF)观测点选择纯函数单测：障碍投点 / 势场 / 候选筛(含 LOS) / ASCII。

不连 ROS/LLM——potential_field 全是显式注入的纯函数。可 pytest，也可直接 python 运行。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core.geometry import potential_field as pf  # noqa: E402


def test_obstacle_points_project_to_world():
    """扇区最近距离 → 世界障碍点：正前(sec_0)在车前方 +X，左侧(sec_90)在 +Y。"""
    pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    pts = pf.obstacle_points_from_scan({"sec_0": 1.0, "sec_90": 2.0, "bad": None}, pose)
    assert (round(pts[0][0], 2), round(pts[0][1], 2)) == (1.0, 0.0)
    assert (round(pts[1][0], 2), round(pts[1][1], 2)) == (0.0, 2.0)   # +左=+Y


def test_repulsion_high_near_obstacle():
    """近障碍格势能(斥力)显著高于开阔格。"""
    pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    field = pf.build_field([(0.3, 0.0)], pose, target_xy=None, half_size_m=1.5, res_m=0.25)
    near = pf._clear_at(field, 0.25, 0.0)
    far = pf._clear_at(field, 0.0, 1.5)
    assert near < far   # 近障碍格 clearance 更小


def test_candidates_in_standoff_band_and_reachable():
    """候选观测点 clearance 落在站位带内，且不选障碍背后(LOS 剔除)。"""
    pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    # 正前 0.35m 一堵墙(会钻桌底)，两侧开阔
    sectors = {"sec_0": 0.35, "sec_10": 0.4, "sec_-10": 0.4,
               "sec_90": 2.0, "sec_-90": 2.0, "sec_180": 2.5}
    payload = pf.build_viewpoint_payload(sectors, pose, target_xy=(1.0, 0.0))
    assert payload["candidates"], "应能找到安全站位观测点"
    for c in payload["candidates"]:
        assert 0.5 <= c["clearance_m"] <= 1.5, c
    # 墙后(x>0.4, |y|<0.3)的格不可达 → 不应入候选
    behind = [c for c in payload["candidates"] if c["x"] > 0.4 and abs(c["y"]) < 0.3]
    assert not behind, f"LOS 应剔除障碍背后候选，却有 {behind}"


def test_line_of_sight_blocks_through_obstacle():
    """LOS：直线穿过近障碍格 → 判不通；开阔方向 → 通。"""
    pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    field = pf.build_field([(0.4, 0.0)], pose, half_size_m=1.5, res_m=0.25)
    assert pf._line_of_sight(field, 0.0, 0.0, 1.2, 0.0) is False   # 正前穿墙
    assert pf._line_of_sight(field, 0.0, 0.0, 0.0, 1.2) is True    # 侧向开阔


def test_no_candidates_when_fully_boxed():
    """四面贴身障碍(全格 clearance 都太小) → 无候选，调用方回退默认目标。"""
    pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    sectors = {f"sec_{a}": 0.15 for a in range(-170, 190, 20)}   # 四周 0.15m 全贴身
    payload = pf.build_viewpoint_payload(sectors, pose, target_xy=(0.5, 0.0))
    assert payload["candidates"] == []


def test_ascii_marks_robot_and_target():
    """ASCII 图含车 R、目标 T 与候选数字标记。"""
    pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    sectors = {"sec_0": 1.5, "sec_90": 1.5, "sec_-90": 1.5, "sec_180": 1.5}
    payload = pf.build_viewpoint_payload(sectors, pose, target_xy=(0.5, 0.0))
    art = payload["ascii_field"]
    assert "R" in art and "T" in art
    if payload["candidates"]:
        assert payload["candidates"][0]["id"][-1] in art


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_potential_field: all PASS")
