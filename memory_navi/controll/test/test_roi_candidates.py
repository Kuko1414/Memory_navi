"""Pure tests for ROI candidate snapping and late-fusion helpers."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core.perception import roi_candidates as rc  # noqa: E402


def test_depth_snap_picks_probe_matching_qwen_distance():
    obj = {
        "name": "绿植",
        "roi": {"x": 100, "y": 100, "w": 180, "h": 180},
        "distance_m": 1.5,
    }
    rois = [
        {"x": 100, "y": 100, "w": 180, "h": 180},
        {"x": 180, "y": 130, "w": 120, "h": 140},
    ]
    stats = [
        {"median_m": 3.4, "near_min_m": 3.2, "near_max_m": 3.5, "n_valid": 500},
        {"median_m": 1.55, "near_min_m": 1.45, "near_max_m": 1.65, "n_valid": 420},
    ]
    picked = rc.choose_depth_snap_roi(obj, rois, stats, depth_tol_m=0.8)
    assert picked["status"] == rc.GEOMETRY_OK
    assert picked["roi"] == rois[1]
    assert picked["roi_quality"]["depth_error_m"] == 0.05


def test_depth_snap_rejects_when_best_probe_still_mismatches_depth():
    obj = {
        "name": "柜子",
        "roi": {"x": 100, "y": 100, "w": 180, "h": 180},
        "distance_m": 1.2,
    }
    picked = rc.choose_depth_snap_roi(
        obj,
        [{"x": 100, "y": 100, "w": 180, "h": 180}],
        [{"median_m": 3.1, "near_min_m": 3.0, "near_max_m": 3.2, "n_valid": 300}],
        depth_tol_m=0.8,
    )
    assert picked["status"] == rc.GEOMETRY_DEPTH_MISMATCH
    assert picked["roi"] is None


def test_yolo_match_prefers_validated_roi_for_qwen_object():
    qwen = {
        "name": "办公椅",
        "roi": {"x": 300, "y": 300, "w": 120, "h": 120},
        "distance_m": 1.6,
    }
    yolo = [
        {
            "idx": 1,
            "roi": {"x": 310, "y": 305, "w": 110, "h": 115},
            "distance_m": 1.55,
            "abs_pose": {"x": 1.0, "y": 2.0},
            "geometry_status": rc.GEOMETRY_OK,
        },
        {
            "idx": 2,
            "roi": {"x": 800, "y": 100, "w": 80, "h": 90},
            "distance_m": 1.5,
            "abs_pose": {"x": 3.0, "y": 4.0},
            "geometry_status": rc.GEOMETRY_OK,
        },
    ]
    picked = rc.match_yolo_candidate(qwen, yolo)
    assert picked["idx"] == 1

    fused = rc.apply_candidate_to_object(qwen, picked, roi_source="yoloe", semantic_source="qwen_full")
    assert fused["name"] == "办公椅"
    assert fused["roi"] == yolo[0]["roi"]
    assert fused["abs_pose"] == {"x": 1.0, "y": 2.0}
    assert fused["roi_source"] == "yoloe"


def test_high_confusion_name_detection():
    assert rc.is_high_confusion_name("显示器") is True
    assert rc.is_high_confusion_name("办公桌") is True
    assert rc.is_high_confusion_name("绿植") is False


def test_generate_depth_snap_rois_from_bbox_center_when_roi_missing():
    rois = rc.generate_depth_snap_rois({"bbox_center": [500, 400], "distance_m": 1.0})
    assert rois
    assert all(set(r) == {"x", "y", "w", "h"} for r in rois)


def test_south_wall_allowed_names_filter_rejects_off_list_labels():
    parsed = {
        1: {"name": "水槽", "completeness": "完整", "keep": True},
        2: {"name": "显示器", "completeness": "完整", "keep": True},
    }
    out = rc.filter_judgments_by_allowed_names(parsed, ["厨台", "水槽", "柜子", "办公桌"])
    assert out[1]["keep"] is True
    assert out[1]["name"] == "水槽"
    assert out[2]["keep"] is False
    assert out[2]["name"] is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_roi_candidates: OK")
