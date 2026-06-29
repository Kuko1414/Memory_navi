"""Unit tests for ros_mcp/tools/perception.py::summarize_scan (pure, no ROS)."""

import math

from ros_mcp.tools.perception import summarize_scan


def _ranges_for(n, value=10.0):
    return [value] * n


class TestSummarizeScan:
    def test_empty_when_all_invalid(self):
        ranges = [float("nan"), float("inf"), -1.0]
        out = summarize_scan(ranges, angle_min=-math.pi, angle_increment=math.pi)
        assert out["nearest"] is None
        assert out["valid_points"] == 0
        assert out["total_points"] == 3

    def test_nearest_dist_and_bearing_front(self):
        # 360 rays over full circle; put a close obstacle straight ahead (index at 0 rad).
        n = 360
        inc = 2 * math.pi / n
        amin = -math.pi
        ranges = _ranges_for(n, 10.0)
        front_idx = int(round((0.0 - amin) / inc))  # ray pointing to 0deg
        ranges[front_idx] = 0.3
        out = summarize_scan(ranges, angle_min=amin, angle_increment=inc, range_max=20.0)
        assert out["nearest"]["dist_m"] == 0.3
        assert abs(out["nearest"]["bearing_deg"]) <= 1.0  # ~front

    def test_invalid_filtering(self):
        ranges = [float("nan"), 0.005, 0.5, float("inf"), 100.0]
        out = summarize_scan(
            ranges, angle_min=0.0, angle_increment=0.1, range_min=0.05, range_max=10.0
        )
        # nan dropped, 0.005 < range_min dropped, inf dropped, 100 > range_max dropped
        assert out["valid_points"] == 1
        assert out["nearest"]["dist_m"] == 0.5

    def test_clear_path_true_when_front_open(self):
        n = 360
        inc = 2 * math.pi / n
        amin = -math.pi
        ranges = _ranges_for(n, 5.0)  # everything far
        out = summarize_scan(ranges, angle_min=amin, angle_increment=inc, clear_thresh_m=0.5)
        assert out["clear_path"] is True

    def test_clear_path_false_when_obstacle_ahead(self):
        n = 360
        inc = 2 * math.pi / n
        amin = -math.pi
        ranges = _ranges_for(n, 5.0)
        front_idx = int(round((0.0 - amin) / inc))
        ranges[front_idx] = 0.2  # blocking obstacle dead ahead
        out = summarize_scan(
            ranges, angle_min=amin, angle_increment=inc, range_max=20.0, clear_thresh_m=0.5
        )
        assert out["clear_path"] is False
        assert out["front_min_m"] == 0.2

    def test_eight_sectors_labeled(self):
        n = 360
        inc = 2 * math.pi / n
        amin = -math.pi
        ranges = _ranges_for(n, 4.0)
        out = summarize_scan(ranges, angle_min=amin, angle_increment=inc, sectors=8)
        assert set(out["sectors"]) == {
            "front", "front_left", "left", "rear_left",
            "rear", "rear_right", "right", "front_right",
        }

    def test_left_obstacle_bearing_positive(self):
        # REP-103: +90deg == left. Place obstacle at +90deg, expect positive bearing.
        n = 360
        inc = 2 * math.pi / n
        amin = -math.pi
        ranges = _ranges_for(n, 8.0)
        left_idx = int(round((math.pi / 2 - amin) / inc))
        ranges[left_idx] = 0.4
        out = summarize_scan(ranges, angle_min=amin, angle_increment=inc, range_max=20.0)
        assert out["nearest"]["dist_m"] == 0.4
        assert 80 <= out["nearest"]["bearing_deg"] <= 100
        assert out["sectors"]["left"] == 0.4
