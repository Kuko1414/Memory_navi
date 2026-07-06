"""观测点质量门单测（potential_field.scan_obs_quality）：最近障 ≥0.5m 且四周均匀(CV≤0.6)才算合格。

纯几何、代码执行，不连 ROS/LLM。在含 openai/anthropic 的 vllm 环境跑（explore_probe 顶层依赖它们，
但本文件只导 potential_field，理论上可脱离；与其它测试统一放 test/ 下）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core.geometry import potential_field as pf  # noqa: E402


def _sectors(dists):
    """把一串距离铺成 sec_<角度> 扇区字典。"""
    return {f"sec_{i * 30}": d for i, d in enumerate(dists)}


def test_open_uniform_ok():
    """四周开阔且均匀(全 ~2.0) → 合格观测点。"""
    q = pf.scan_obs_quality(_sectors([2.0, 2.1, 1.9, 2.0, 2.2, 1.8]))
    assert q["ok"] is True and q["uniform"] is True and q["min_clear_m"] >= 0.5


def test_under_table_rejected_by_min():
    """桌底/四面贴障(全 ~0.3) → 最小 <0.5 → 拒(即使均匀)。"""
    q = pf.scan_obs_quality(_sectors([0.3, 0.32, 0.28, 0.31, 0.3, 0.29]))
    assert q["ok"] is False and q["min_clear_m"] < 0.5


def test_extreme_lopsided_rejected_by_cv():
    """极偏斜/窄道(三侧 0.5、三侧 4.0)：最小恰过 0.5 门，但 CV 很高(≈0.78) → 均匀性门拒。

    注：门刻意宽松(CV≤0.6)——只剔这种极端箱式站位；温和贴墙(退后看墙)不误杀，交由 min 门保安全。
    """
    q = pf.scan_obs_quality(_sectors([0.5, 0.5, 0.5, 4.0, 4.0, 4.0]))
    assert q["min_clear_m"] >= 0.5 and q["uniform"] is False and q["ok"] is False


def test_standoff_from_wall_ok():
    """退后看墙(一侧墙~1.0m、其余开阔~2.5) → 合格(正当观测位不误杀)。"""
    q = pf.scan_obs_quality(_sectors([1.0, 1.1, 2.5, 2.6, 2.4, 2.5]))
    assert q["ok"] is True, f"退后看墙应合格：{q}"


def test_empty_sectors_not_ok():
    """无有效距离 → ok=False（无从判断，不误判为好点）。"""
    assert pf.scan_obs_quality({})["ok"] is False
    assert pf.scan_obs_quality({"sec_0": None, "sec_30": -1})["ok"] is False


def test_min_boundary():
    """最小恰等于阈值 OBS_MIN_CLEAR_M 且均匀 → 合格(≥ 阈值)。"""
    m = pf.OBS_MIN_CLEAR_M
    q = pf.scan_obs_quality(_sectors([m, m + 0.02, m + 0.05, m, m + 0.03, m + 0.01]))
    assert q["min_clear_m"] == m and q["ok"] is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_scan_obs_quality: all PASS")
