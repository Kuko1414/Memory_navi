"""YOLOE conversion helpers: pure tests, no Ultralytics import required."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core.memory.fs_memory import FsMemory  # noqa: E402
from agent_core.perception import yoloe  # noqa: E402


def test_class_mapping_to_chinese_names():
    assert yoloe.canonical_name("cabinet") == "柜子"
    assert yoloe.canonical_name("office_chair") == "办公椅"
    assert yoloe.canonical_name("potted plant") == "绿植"
    assert yoloe.canonical_name("unknown thing") is None


def test_bbox_to_roi_and_center_are_clipped():
    roi = yoloe.bbox_xyxy_to_roi([-10, 48, 330, 300], 640, 480)
    assert roi == {"x": 0, "y": 100, "w": 516, "h": 525}
    assert yoloe.bbox_xyxy_to_center([64, 48, 320, 240], 640, 480) == [300, 300]


def test_normalize_detections_filters_low_conf_and_doors_by_default():
    raw = [
        {"label": "cabinet", "confidence": 0.8, "bbox_xyxy": [10, 20, 110, 220]},
        {"label": "chair", "confidence": 0.1, "bbox_xyxy": [0, 0, 50, 50]},
        {"label": "door", "confidence": 0.9, "bbox_xyxy": [200, 0, 260, 300]},
    ]
    objs = yoloe.normalize_detections(raw, 640, 480, conf_thres=0.25)
    assert len(objs) == 1
    assert objs[0]["name"] == "柜子"
    assert objs[0]["verified_by"] == ["yoloe"]
    assert objs[0]["detector_label"] == "cabinet"


def test_doors_can_be_included_for_diagnostics():
    raw = [{"label": "door", "confidence": 0.9, "bbox_xyxy": [200, 0, 260, 300]}]
    objs = yoloe.normalize_detections(raw, 640, 480, include_doors=True)
    assert len(objs) == 1
    assert objs[0]["name"] == "门"
    assert objs[0]["is_door"] is True


def test_fs_memory_accepts_yolo_object(tmp_path):
    mem = FsMemory(str(tmp_path), "sim")
    obj = yoloe.raw_detection_to_object(
        {"label": "desk", "confidence": 0.76, "bbox_xyxy": [100, 120, 300, 320]},
        640,
        480,
    )
    obj["abs_pose"] = {"x": 1.0, "y": 2.0, "z": 0.0}
    mem.upsert_object("yolo_test", obj)
    rec = mem.load_area("yolo_test")
    assert rec["objects"][0]["name"] == "办公桌"
    assert rec["objects"][0]["verified_by"] == ["yoloe"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if name == "test_fs_memory_accepts_yolo_object":
                import tempfile
                from pathlib import Path

                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
    print("test_yoloe_perception: all PASS")
