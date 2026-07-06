"""单帧共现反合并单测（_dedup_objects）：密集同名家具(间距≈坐标噪声)靠"同一帧共现"当铁证保住实例数。

纯几何、代码执行，不连 ROS/LLM（在 vllm 环境跑，explore_probe 顶层依赖 openai/anthropic）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import explore_probe as ep  # noqa: E402

BBOX = {"xmin": -5.0, "xmax": 5.0, "ymin": -5.0, "ymax": 5.0}


def _o(name, x, y, frame, conf=0.8):
    return {"name": name, "confidence": conf, "_frame": frame,
            "abs_pose": {"x": x, "y": y, "z": 0.5}}


def test_same_frame_distinct_instances_kept_separate():
    """同一帧两个同名框、间距 0.5m(>COOCCUR_EPS, <DEDUP_M) → 感知已分辨=两个实例 → 不合并。"""
    vr = [{"objects": [_o("显示器", 0.0, 0.0, "F1"), _o("显示器", 0.5, 0.0, "F1")]}]
    out = ep._dedup_objects(vr, BBOX)
    mons = [o for o in out if o["name"] == "显示器"]
    assert len(mons) == 2, f"应保住 2 个显示器，实际 {len(mons)}: {out}"
    assert all("_frame" not in o and "_frames" not in o for o in out)   # 内部字段已剥


def test_cross_frame_same_object_merged():
    """不同帧的同名近物(0.5m) = 同一物体多视角重复 → 合并为 1。"""
    vr = [{"objects": [_o("显示器", 0.0, 0.0, "F1")]},
          {"objects": [_o("显示器", 0.5, 0.0, "F2")]}]
    out = ep._dedup_objects(vr, BBOX)
    assert len([o for o in out if o["name"] == "显示器"]) == 1


def test_same_frame_same_position_merged_dual_source():
    """同一帧同名、位置几乎重合(0.1m<COOCCUR_EPS) = 同物双源(YOLO+Qwen) → 仍合并为 1。"""
    vr = [{"objects": [_o("显示器", 0.0, 0.0, "F1"), _o("显示器", 0.1, 0.0, "F1")]}]
    out = ep._dedup_objects(vr, BBOX)
    assert len([o for o in out if o["name"] == "显示器"]) == 1


def test_east_monitor_row_survives():
    """东排 3 台显示器(y=0.36/-0.18/-1.79)一帧共现：前两台仅 0.54m，旧逻辑会并 → 现应保 3 台。"""
    vr = [{"objects": [_o("显示器", 2.88, 0.36, "F1"), _o("显示器", 2.88, -0.18, "F1"),
                       _o("显示器", 2.88, -1.79, "F1")]}]
    out = ep._dedup_objects(vr, BBOX)
    assert len([o for o in out if o["name"] == "显示器"]) == 3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_cooccur_dedup: all PASS")
