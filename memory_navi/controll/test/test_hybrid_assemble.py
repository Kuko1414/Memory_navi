"""混合装配器纯函数测试：YOLO 出框 + Qwen 命名/弃权 → 记忆物体。无 Ultralytics/vLLM/ROS。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core.perception import yoloe  # noqa: E402


def test_raw_detections_to_boxes_label_agnostic_and_conf_gated():
    raw = [
        {"label": "cabinet", "confidence": 0.8, "bbox_xyxy": [10, 20, 110, 220]},
        {"label": "wall clutter", "confidence": 0.06, "bbox_xyxy": [0, 0, 60, 60]},
        {"label": "chair", "confidence": 0.02, "bbox_xyxy": [0, 0, 50, 50]},   # <conf 丢
        {"label": "bad", "confidence": 0.9, "bbox_xyxy": [1, 2, 3]},           # 坏 bbox 丢
    ]
    boxes = yoloe.raw_detections_to_boxes(raw, 640, 480, conf_thres=0.05)
    assert len(boxes) == 2                       # 只有前两个过 conf 且 bbox 合法
    assert [b["idx"] for b in boxes] == [1, 2]   # 顺序编号从 1
    # 词表外标签("wall clutter")也保留（label-agnostic，命名交给 Qwen）
    assert boxes[1]["detector_label"] == "wall clutter"
    assert set(boxes[0]) == {"idx", "roi", "bbox_center", "confidence", "detector_label"}
    assert all(k in boxes[0]["roi"] for k in ("x", "y", "w", "h"))


def _boxes():
    return [
        {"idx": 1, "roi": {"x": 10, "y": 20, "w": 100, "h": 200},
         "bbox_center": [60, 120], "confidence": 0.11, "detector_label": "chair"},
        {"idx": 2, "roi": {"x": 5, "y": 5, "w": 50, "h": 50},
         "bbox_center": [30, 30], "confidence": 0.4, "detector_label": "cabinet"},
        {"idx": 3, "roi": {"x": 0, "y": 0, "w": 30, "h": 30},
         "bbox_center": [15, 15], "confidence": 0.3, "detector_label": "wall"},
        {"idx": 4, "roi": {"x": 0, "y": 0, "w": 40, "h": 40},
         "bbox_center": [20, 20], "confidence": 0.5, "detector_label": "desk"},
    ]


def test_assemble_keeps_only_complete_kept_named():
    judg = {
        1: {"name": "办公椅", "completeness": "完整", "keep": True},     # 留
        2: {"name": None, "completeness": "部分", "keep": False},        # 弃权
        3: {"name": None, "completeness": "完整", "keep": False},        # 完整但无名 → 丢
        4: {"name": "办公桌", "completeness": "部分", "keep": True},     # 非完整 → 丢
    }
    out = yoloe.assemble_hybrid_objects(_boxes(), judg)
    assert len(out) == 1
    o = out[0]
    assert o["name"] == "办公椅"                       # Qwen 命名
    assert o["roi"] == {"x": 10, "y": 20, "w": 100, "h": 200}   # 保留 YOLO 干净 ROI
    assert o["bbox_center"] == [60, 120]              # 保留 YOLO 中心
    assert o["confidence"] == 0.11                    # 保留 YOLO 置信度
    assert o["verified_by"] == ["hybrid"]
    assert o["completeness"] == "完整"
    assert "abs_pose" not in o                        # 绝不产坐标（代码下游回填）


def test_assemble_empty_when_all_abstain():
    judg = {b["idx"]: {"name": None, "completeness": "部分", "keep": False}
            for b in _boxes()}
    assert yoloe.assemble_hybrid_objects(_boxes(), judg) == []


if __name__ == "__main__":
    test_raw_detections_to_boxes_label_agnostic_and_conf_gated()
    test_assemble_keeps_only_complete_kept_named()
    test_assemble_empty_when_all_abstain()
    print("test_hybrid_assemble: OK")
