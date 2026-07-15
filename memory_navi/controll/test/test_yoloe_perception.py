"""YOLOE conversion helpers: pure tests, no Ultralytics import required."""
import json
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


def test_mask_polygon_depth_probe_sits_inside_mask_not_full_bbox():
    polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]

    probe = yoloe.mask_polygon_to_depth_probe_roi(polygon, 640, 480)

    center_x = probe["x"] + probe["w"] / 2
    center_y = probe["y"] + probe["h"] / 2
    assert 100 / 640 * 1000 < center_x < 300 / 640 * 1000
    assert 100 / 480 * 1000 < center_y < 300 / 480 * 1000
    assert 0 < probe["w"] < 30
    assert 0 < probe["h"] < 30


def test_mask_polygon_depth_probe_rejects_degenerate_polygon():
    assert yoloe.mask_polygon_to_depth_probe_roi([[1, 1], [2, 2]], 640, 480) is None


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


def test_same_class_duplicate_boxes_are_suppressed():
    raw = [
        {"label": "kitchen_counter", "confidence": 0.91, "bbox_xyxy": [10, 10, 200, 40]},
        {"label": "kitchen_counter", "confidence": 0.54, "bbox_xyxy": [11, 10, 201, 40]},
        {"label": "cabinet", "confidence": 0.88, "bbox_xyxy": [10, 10, 200, 40]},
    ]

    kept = yoloe.deduplicate_raw_detections(raw, iou_threshold=0.85)

    assert [(item["label"], item["confidence"]) for item in kept] == [
        ("kitchen_counter", 0.91),
        ("cabinet", 0.88),
    ]


def test_class_thresholds_and_disabled_sink_gate_publication():
    raw = [
        {"label": "monitor", "confidence": 0.74, "bbox_xyxy": [10, 20, 80, 100]},
        {"label": "sink", "confidence": 0.99, "bbox_xyxy": [100, 100, 130, 120]},
    ]

    objects = yoloe.normalize_detections(
        raw,
        640,
        480,
        conf_thres=0.25,
        class_conf_thresholds={"monitor": 0.8},
        disabled_labels={"sink"},
        source="low_view_furniture_v1",
    )

    assert objects == []


def test_partition_keeps_unsupported_and_low_score_detections_as_candidates():
    raw = [
        {"label": "cabinet", "confidence": 0.8, "bbox_xyxy": [10, 20, 80, 100]},
        {"label": "desk", "confidence": 0.3, "bbox_xyxy": [90, 20, 180, 100]},
        {"label": "monitor", "confidence": 0.95, "bbox_xyxy": [190, 20, 230, 80]},
    ]

    confirmed, candidates = yoloe.partition_detections(
        raw,
        640,
        480,
        conf_thres=0.05,
        class_conf_thresholds={"cabinet": 0.4, "desk": 0.44},
        candidate_only_labels={"monitor"},
        source="low_view_furniture_v1",
    )

    assert [obj["detector_label"] for obj in confirmed] == ["cabinet"]
    assert confirmed[0]["memory_status"] == "confirmed"
    assert [obj["detector_label"] for obj in candidates] == ["desk", "monitor"]
    assert candidates[0]["candidate_reason"] == "below_publish_threshold"
    assert candidates[1]["candidate_reason"] == "class_candidate_only"


def test_trained_detection_preserves_score_polygon_and_version():
    polygon = [[10.0, 20.0], [80.0, 20.0], [80.0, 100.0]]
    obj = yoloe.raw_detection_to_object(
        {
            "label": "cabinet",
            "confidence": 0.93,
            "bbox_xyxy": [10, 20, 80, 100],
            "mask_polygon": polygon,
        },
        640,
        480,
        source="low_view_furniture_v1",
    )

    assert obj["detector_score"] == 0.93
    assert obj["mask_polygon"] == polygon
    assert obj["model_version"] == "low_view_furniture_v1"


def test_inference_profile_normalizes_thresholds_and_candidate_labels(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({
            "model_version": "low_view_furniture_v1",
            "weights": "~/models/furniture.pt",
            "candidate_conf": 0.05,
            "class_conf_thresholds": {"Office_Chair": 0.23, "CABINET": 0.4},
            "candidate_only": ["Kitchen_Counter", "SINK"],
        }),
        encoding="utf-8",
    )

    profile = yoloe.load_inference_profile(profile_path)

    assert profile["class_conf_thresholds"] == {
        "office chair": 0.23,
        "cabinet": 0.4,
    }
    assert profile["candidate_only"] == ["kitchen counter", "sink"]
    assert profile["weights"].endswith("/models/furniture.pt")


def test_inference_profile_rejects_out_of_range_threshold(tmp_path):
    profile_path = tmp_path / "bad.json"
    profile_path.write_text(
        json.dumps({
            "model_version": "bad",
            "weights": "bad.pt",
            "class_conf_thresholds": {"cabinet": 1.1},
        }),
        encoding="utf-8",
    )

    try:
        yoloe.load_inference_profile(profile_path)
    except ValueError as error:
        assert "cabinet" in str(error)
    else:
        raise AssertionError("invalid threshold was accepted")


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
