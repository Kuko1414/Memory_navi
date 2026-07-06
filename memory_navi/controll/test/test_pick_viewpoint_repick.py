"""_pick_viewpoint 观测点重选环单测：到位 scan 不合格 → 回 APF 重选；全不合格 → 就地兜底(None)。

确定性、不连真实 ROS/LLM：假 ros 只供 get_pose；APF 候选生成与导航 monkeypatch 掉，只保留真实
质量门 pf.scan_obs_quality（本测的验证核心）。在 vllm 环境跑（explore_probe 顶层依赖 openai/anthropic）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import explore_probe as ep  # noqa: E402

BAD = {f"sec_{i * 30}": 0.3 for i in range(6)}          # 全 0.3 → min<0.5 → 不合格
GOOD = {f"sec_{i * 30}": 2.0 for i in range(6)}         # 全 ~2.0 → 均匀开阔 → 合格


class _Res:
    def __init__(self, text):
        self.text = text


class _FakeRos:
    """假 ros：get_pose 恒 (0,0)；其余(move/turn/scan)返回空——scan 由 monkeypatch 的 _read_sectors 供。"""
    def call(self, name, args=None):
        import json
        if name == "get_pose":
            return _Res(json.dumps({"x": 0.0, "y": 0.0, "yaw_deg": 0.0}))
        return _Res(json.dumps({}))


class _FakeEx:
    def __init__(self):
        self.ros = _FakeRos()


class _FakeDirector:
    """假调度官：pick_viewpoint 恒选 v0。"""
    def pick_viewpoint(self, plan_in):
        return {"target_id": "v0", "rationale": "test"}


def _patch(sector_script, cand_coords):
    """装配：_read_sectors 按脚本逐次返回；build_viewpoint_payload 按尝试次给不同候选；导航 no-op。"""
    calls = {"i": 0, "attempt": 0}

    def fake_read(ex, n=36):
        i = calls["i"]
        calls["i"] += 1
        return sector_script[i] if i < len(sector_script) else GOOD

    def fake_payload(sectors, pose, tgt, **kw):
        a = calls["attempt"]
        calls["attempt"] += 1
        cx, cy = cand_coords[a] if a < len(cand_coords) else cand_coords[-1]
        c = {"id": "v0", "x": cx, "y": cy, "clearance_m": 0.9, "potential": 0.1,
             "_ix": 0, "_iy": 0}
        return {"ascii_field": "", "target_xy": None, "pose": pose,
                "candidates": [dict(c)], "_candidates_full": [c]}

    ep._read_sectors = fake_read
    ep.pf.build_viewpoint_payload = fake_payload
    ep.nav.geo_goto_around = lambda *a, **k: {"arrived": True}
    return calls


def test_repick_then_success():
    """首次到位 scan 坏 → 重选 → 次次到位 scan 好 → 返回该(第二个)观测点。"""
    # 每尝试 _read_sectors 调 2 次(build 忽略 + 到位复核)：坏、好交替
    _patch([GOOD, BAD, GOOD, GOOD], [(1.0, 0.0), (2.0, 0.0)])
    xy = ep._pick_viewpoint(_FakeEx(), _FakeDirector(), target_xy=(3.0, 0.0))
    assert xy == (2.0, 0.0), f"应重选到第二个合格点，实得 {xy}"


def test_all_bad_returns_none():
    """所有到位 scan 都不合格 → 重选用尽 → 返回 None(调用方就地尽力观测)。"""
    _patch([GOOD, BAD, GOOD, BAD, GOOD, BAD, GOOD, BAD],
           [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
    xy = ep._pick_viewpoint(_FakeEx(), _FakeDirector(), target_xy=(5.0, 0.0))
    assert xy is None, f"全不合格应返回 None，实得 {xy}"


def test_first_ok_no_repick():
    """首次到位就合格 → 直接返回，不触发重选。"""
    _patch([GOOD, GOOD], [(1.0, 0.0)])
    xy = ep._pick_viewpoint(_FakeEx(), _FakeDirector(), target_xy=(3.0, 0.0))
    assert xy == (1.0, 0.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_pick_viewpoint_repick: all PASS")
