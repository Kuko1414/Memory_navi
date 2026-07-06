"""Unit tests for ros_mcp/tools/perception.py::downsample_rays (pure, no ROS).

Dense per-ray downsampling for occupancy/frontier mapping: keeps bearing+distance,
encodes no-return (nan/inf/out-of-range) as dist=-1 (free to range_max).
"""

import math

from ros_mcp.tools.perception import downsample_rays


def _ranges_for(n, value=5.0):
    return [value] * n


class TestDownsampleRays:
    def test_empty(self):
        assert downsample_rays([], angle_min=0.0, angle_increment=0.1) == []

    def test_downsamples_to_max_beams(self):
        n = 360
        inc = 2 * math.pi / n
        beams = downsample_rays(_ranges_for(n, 4.0), angle_min=-math.pi,
                                angle_increment=inc, range_max=20.0, max_beams=90)
        assert 0 < len(beams) <= 90
        # every kept beam is [bearing_deg, dist]
        assert all(len(b) == 2 for b in beams)

    def test_no_return_encoded_as_neg1(self):
        # nan / inf / below range_min / above range_max → -1 (free to max)
        ranges = [float("nan"), float("inf"), 0.005, 100.0, 3.0]
        beams = downsample_rays(ranges, angle_min=0.0, angle_increment=0.1,
                                range_min=0.05, range_max=10.0, max_beams=180)
        dists = [d for _, d in beams]
        assert dists[:4] == [-1.0, -1.0, -1.0, -1.0]
        assert dists[4] == 3.0

    def test_bearing_front_is_zero(self):
        # a ray at angle 0 rad → bearing ~0deg (front)
        n = 360
        inc = 2 * math.pi / n
        amin = -math.pi
        ranges = _ranges_for(n, 4.0)
        beams = downsample_rays(ranges, angle_min=amin, angle_increment=inc,
                                range_max=20.0, max_beams=360)
        # find beam nearest bearing 0
        b0 = min(beams, key=lambda b: abs(b[0]))
        assert abs(b0[0]) <= 1.0

    def test_bearing_left_positive(self):
        # REP-103: +90deg == left
        n = 360
        inc = 2 * math.pi / n
        amin = -math.pi
        ranges = _ranges_for(n, 8.0)
        left_idx = int(round((math.pi / 2 - amin) / inc))
        ranges[left_idx] = 0.4
        beams = downsample_rays(ranges, angle_min=amin, angle_increment=inc,
                                range_max=20.0, max_beams=360)
        hit = min(beams, key=lambda b: abs(b[1] - 0.4) if b[1] > 0 else 1e9)
        assert 80 <= hit[0] <= 100 and hit[1] == 0.4
