"""APF 点到点导航方向核单测（纯函数，无仿真/无 LLM/无 mcp）。

potential_field.apf_heading：引力朝目标 + 斥力离障碍 → 合力世界航向；navigator._fwd_clear：前向最近障。
验证：无障=朝目标、近障侧偏离、对称缝穿中不侧偏、超 d0/无返回束忽略、world 航向不受 yaw 影响。
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core.geometry import potential_field as pf  # noqa: E402
from agent_core import navigator as nav  # noqa: E402


def _wrap(a):
    return ((a + 180.0) % 360.0) - 180.0


P0 = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}


# ---- apf_heading ----
def test_no_obstacle_heads_to_goal():
    assert abs(_wrap(pf.apf_heading(P0, 5.0, 0.0, []))) < 1e-6           # 目标正东 → 航向 0


def test_goal_north_bearing_90():
    assert abs(_wrap(pf.apf_heading(P0, 0.0, 5.0, []) - 90.0)) < 1e-6


def test_world_heading_independent_of_yaw():
    # 车头朝北(yaw90)，目标在东；引力用世界目标 → 世界航向仍 0（非体系）
    pose = {"x": 0.0, "y": 0.0, "yaw_deg": 90.0}
    assert abs(_wrap(pf.apf_heading(pose, 5.0, 0.0, []))) < 1e-6


def test_close_obstacle_on_right_veers_left():
    # 障在前偏右(体系 -30°)、近 → 航向左偏(>0)
    h = pf.apf_heading(P0, 5.0, 0.0, [[-30.0, 0.5]])
    assert h > 5.0


def test_close_obstacle_on_left_veers_right():
    h = pf.apf_heading(P0, 5.0, 0.0, [[30.0, 0.5]])
    assert h < -5.0


def test_symmetric_gap_threads_straight():
    # 缝两侧对称(±60° 等距)→ 侧向斥力对消 → 不侧偏，仍朝目标穿中
    h = pf.apf_heading(P0, 5.0, 0.0, [[60.0, 0.8], [-60.0, 0.8]])
    assert abs(_wrap(h)) < 5.0


def test_far_and_invalid_beams_ignored():
    # 超 d0(1.0) 的远障 + dist=-1(无返回) 均不产生斥力 → 等价无障
    h = pf.apf_heading(P0, 5.0, 0.0, [[-30.0, 1.5], [45.0, -1.0]])
    assert abs(_wrap(h)) < 1e-6


def test_obstacle_directly_behind_no_veer():
    # 障正后(体系 180°)→ 斥力沿 +x（朝目标）→ 航向仍 0，不侧偏
    h = pf.apf_heading(P0, 5.0, 0.0, [[180.0, 0.5]])
    assert abs(_wrap(h)) < 1e-6


# ---- _fwd_clear ----
def test_fwd_clear_picks_nearest_in_cone():
    beams = [[0.0, 2.0], [30.0, 0.5], [-30.0, 3.0]]
    assert nav._fwd_clear(beams, 0.0) == 2.0            # ±25° 内只有 0° 束
    assert nav._fwd_clear(beams, 30.0) == 0.5           # 朝 30° 时取那束


def test_fwd_clear_min_within_cone():
    assert nav._fwd_clear([[10.0, 1.0], [0.0, 2.0]], 0.0) == 1.0


def test_fwd_clear_none_and_invalid():
    assert nav._fwd_clear([], 0.0) is None
    assert nav._fwd_clear([[0.0, -1.0]], 0.0) is None   # -1 无返回跳过
