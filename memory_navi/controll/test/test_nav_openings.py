"""navigator.scan_openings 单测（纯函数，stub 掉 _scan_sectors，无仿真/无 LLM）。

验证单帧 scan → 离散开口聚簇：单走廊=1、丁字/十字≥2、被隔断分簇、死胡同=0；
以及执行模式导航的开口判定辅助（_opening_for_bearing / _nearest_opening）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core import navigator as nav  # noqa: E402


def _stub_sectors(monkeypatch, mapping):
    monkeypatch.setattr(nav, "_scan_sectors", lambda ex, sectors=12: dict(mapping))


# ---- scan_openings 聚簇 ----
def test_single_corridor_one_opening(monkeypatch):
    # 只有正前方通，两侧堵 → 1 个开口
    _stub_sectors(monkeypatch, {0: 2.5, 30: 0.4, -30: 0.4, 60: 0.3, -60: 0.3})
    ops = nav.scan_openings(None)
    assert len(ops) == 1 and abs(ops[0]["center_deg"]) < 1


def test_junction_two_openings(monkeypatch):
    # 前方堵、左簇(30~60)与右簇(-30~-60)各一 → 2 个开口，被前方 0° 堵隔开
    _stub_sectors(monkeypatch, {0: 0.3, 30: 2.0, 60: 2.2, -30: 2.1, -60: 2.0})
    ops = nav.scan_openings(None)
    assert len(ops) == 2
    centers = sorted(o["center_deg"] for o in ops)
    assert centers[0] < 0 < centers[1]           # 一左一右


def test_deadend_zero_openings(monkeypatch):
    _stub_sectors(monkeypatch, {0: 0.3, 30: 0.4, -30: 0.4, 60: 0.3, -60: 0.3})
    assert nav.scan_openings(None) == []


def test_open_area_merges_into_one(monkeypatch):
    _stub_sectors(monkeypatch, {0: 3, 30: 3, 60: 3, -30: 3, -60: 3})
    ops = nav.scan_openings(None)
    assert len(ops) == 1 and ops[0]["width_deg"] >= 120     # 宽开口


def test_none_sectors_skipped(monkeypatch):
    # None(无返回)当作不可通行的隔断
    _stub_sectors(monkeypatch, {0: 2.0, 30: None, 60: 2.0, -30: 0.3, -60: 0.3})
    ops = nav.scan_openings(None)
    assert len(ops) == 2                          # 0° 与 60° 被 30°=None 隔成两簇


def test_pass_min_threshold(monkeypatch):
    _stub_sectors(monkeypatch, {0: 0.65, 30: 0.65})   # 都 < 0.7 → 不通
    assert nav.scan_openings(None, pass_min_m=0.7) == []
    assert len(nav.scan_openings(None, pass_min_m=0.6)) == 1


def test_min_clear_reported(monkeypatch):
    _stub_sectors(monkeypatch, {0: 2.5, 30: 1.2, 60: 3.0})
    ops = nav.scan_openings(None)
    assert len(ops) == 1 and ops[0]["min_clear_m"] == 1.2   # 簇内最近障碍


# ---- 开口判定辅助（navigator，纯几何，不拉 mcp 栈）----
def test_opening_for_bearing_and_nearest():
    openings = [{"center_deg": -45, "angles": [-60, -30]},
                {"center_deg": 45, "angles": [30, 60]}]
    # 目标方位 -40 落在左开口([-60,-30]±15)内
    assert nav.opening_for_bearing(openings, -40) is openings[0]
    # 目标方位 0 不在任何开口内(前方被堵) → None
    assert nav.opening_for_bearing(openings, 0) is None
    # 最近开口：目标 40 → 右开口
    assert nav.nearest_opening(openings, 40) is openings[1]
    assert nav.nearest_opening([], 0) is None
