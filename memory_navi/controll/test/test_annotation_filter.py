"""标注过滤谓词 _annotation_ok 单测（Task 3：远距离过报 / 视野过小 → 丢弃）。

只测纯谓词，不连 ROS/LLM。需在含 openai/anthropic 的 vllm 环境跑（explore_probe 顶层依赖它们）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import explore_probe as ep  # noqa: E402

# _annotation_ok 现在受 FILTER_ROI_LOS 门控（默认关=v7）；本文件测的就是过滤逻辑本身 → 强制打开。
ep.FILTER_ROI_LOS = True


def test_far_object_dropped():
    """distance_m > FAR_LABEL_M → 丢弃(远距离过报)。"""
    ok, why = ep._annotation_ok({"name": "柜子", "distance_m": ep.FAR_LABEL_M + 0.5,
                                 "roi": {"x": 400, "y": 400, "w": 200, "h": 200}})
    assert ok is False and "dist" in why


def test_good_object_kept():
    """适中距离 + 合理大小 ROI → 保留。"""
    ok, _ = ep._annotation_ok({"name": "沙发", "distance_m": 1.2,
                               "roi": {"x": 400, "y": 400, "w": 200, "h": 200}})
    assert ok is True


def test_tiny_roi_dropped():
    """ROI 面积过小(视野过小/团在一起) → 丢弃。"""
    ok, why = ep._annotation_ok({"name": "显示器", "distance_m": 1.5,
                                 "roi": {"x": 500, "y": 500, "w": 20, "h": 20}})
    assert ok is False and "roi" in why


def test_thin_roi_dropped():
    """ROI 最短边过小(细长/极小框) → 丢弃。"""
    ok, why = ep._annotation_ok({"name": "键盘", "distance_m": 1.0,
                                 "roi": {"x": 100, "y": 500, "w": 800, "h": 30}})
    assert ok is False and "roi" in why


def test_no_distance_no_roi_kept():
    """缺距离与 ROI 时无从判断 → 不误杀(保留)。"""
    ok, _ = ep._annotation_ok({"name": "绿植"})
    assert ok is True


def test_boundary_distance_kept():
    """distance_m 恰等于 FAR_LABEL_M → 不丢(仅 > 才丢)。"""
    ok, _ = ep._annotation_ok({"name": "办公桌", "distance_m": ep.FAR_LABEL_M,
                               "roi": {"x": 300, "y": 300, "w": 300, "h": 300}})
    assert ok is True


def test_forbidden_names_dropped():
    """建筑面/门/机器人自身等禁记类 → 硬过滤(不入物体库)。含中英大小写。"""
    for n in ("地板", "门", "机器人", "墙面", "天花板", "floor", "Door", "WALL", "robot"):
        assert ep._is_forbidden_name(n) is True, n


def test_real_furniture_not_forbidden():
    """真家具名不被禁记类误杀。"""
    for n in ("柜子", "沙发", "办公桌", "办公椅", "显示器", "绿植", "键盘", "水槽", "厨台"):
        assert ep._is_forbidden_name(n) is False, n


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_annotation_filter: all PASS")
