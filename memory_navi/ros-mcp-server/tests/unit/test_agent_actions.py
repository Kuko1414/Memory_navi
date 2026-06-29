"""Unit tests for ros_mcp/tools/agent_actions.py pure helpers (no ROS)."""

import pytest

from ros_mcp.tools.agent_actions import (
    quat_to_yaw_deg,
    shortest_angle_diff,
    yaw_deg_to_quat,
)


class TestShortestAngleDiff:
    def test_simple(self):
        assert shortest_angle_diff(10.0, 0.0) == pytest.approx(10.0)
        assert shortest_angle_diff(0.0, 10.0) == pytest.approx(-10.0)

    def test_wraparound_positive(self):
        # 170 -> -170 is +20 (crossing +180), not -340
        assert shortest_angle_diff(-170.0, 170.0) == pytest.approx(20.0)

    def test_wraparound_negative(self):
        # -170 -> 170 is -20
        assert shortest_angle_diff(170.0, -170.0) == pytest.approx(-20.0)

    def test_within_range(self):
        for a, b in [(0, 0), (90, -90), (45, 30), (-120, 120)]:
            d = shortest_angle_diff(a, b)
            assert -180.0 <= d <= 180.0

    def test_accumulation_full_turn(self):
        # 模拟逐步累积转角：从 0 转到 +90，分多步，abs 累积应≈90
        yaws = [0, 20, 40, 60, 80, 90]
        acc = 0.0
        for prev, cur in zip(yaws, yaws[1:]):
            acc += abs(shortest_angle_diff(cur, prev))
        assert acc == pytest.approx(90.0)

    def test_accumulation_across_wrap(self):
        # 跨 ±180 绕回累积：170 -> 175 -> -180(=180) -> -175，每步 5°，共 15°
        yaws = [170, 175, 180, -175]
        acc = 0.0
        for prev, cur in zip(yaws, yaws[1:]):
            acc += abs(shortest_angle_diff(cur, prev))
        assert acc == pytest.approx(15.0)


class TestQuatYaw:
    def test_identity_is_zero(self):
        assert quat_to_yaw_deg(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_ninety_deg(self):
        q = yaw_deg_to_quat(90.0)
        assert quat_to_yaw_deg(q["x"], q["y"], q["z"], q["w"]) == pytest.approx(90.0)

    def test_roundtrip_various(self):
        for yaw in (-179.0, -90.0, -1.0, 0.0, 1.0, 45.0, 135.0, 179.0):
            q = yaw_deg_to_quat(yaw)
            back = quat_to_yaw_deg(q["x"], q["y"], q["z"], q["w"])
            assert back == pytest.approx(yaw, abs=1e-6)

    def test_yaw_quat_is_unit_and_z_only(self):
        q = yaw_deg_to_quat(30.0)
        assert q["x"] == 0.0 and q["y"] == 0.0
        norm = q["z"] ** 2 + q["w"] ** 2
        assert norm == pytest.approx(1.0)
