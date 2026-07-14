"""ROI candidate helpers for Qwen/Yolo late fusion.

The functions in this module are deliberately pure: they do not call ROS, YOLO,
or an LLM.  The exploration runner supplies depth_roi stats and receives back a
validated ROI choice that can be projected by the existing geometry pipeline.
"""
from __future__ import annotations

from collections import Counter
import math

from agent_core.geometry import depth_projection as dp


GEOMETRY_OK = "ok"
GEOMETRY_DEPTH_MISMATCH = "depth_mismatch"
GEOMETRY_LOS_BLOCKED = "los_blocked"
GEOMETRY_NO_DEPTH = "no_depth"
GEOMETRY_SIZE_UNRELIABLE = "size_unreliable"
GEOMETRY_OUT_OF_BOUNDS = "out_of_bounds"

HIGH_CONFUSION_NAMES = {"显示器", "办公桌", "办公椅"}
DEFAULT_DEPTH_MISMATCH_M = 0.8
DEFAULT_SANE_MAX_M = 2.5
DEFAULT_CENTER_MATCH = 180.0


def norm_name(name: str | None) -> str:
    return str(name or "").strip().lower()


def is_high_confusion_name(name: str | None) -> bool:
    return norm_name(name) in {norm_name(n) for n in HIGH_CONFUSION_NAMES}


def roi_center_norm(roi: dict | None) -> tuple[float, float] | None:
    if not isinstance(roi, dict):
        return None
    try:
        x, y, w, h = dp._roi_to_frac(roi)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return ((x + w / 2.0) * 1000.0, (y + h / 2.0) * 1000.0)


def roi_iou(a: dict | None, b: dict | None) -> float:
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return 0.0
    try:
        ax, ay, aw, ah = dp._roi_to_frac(a)
        bx, by, bw, bh = dp._roi_to_frac(b)
    except (TypeError, ValueError):
        return 0.0
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _bearing_of_roi(roi: dict | None) -> str | None:
    c = roi_center_norm(roi)
    if c is None:
        return None
    u, _v = c
    if u < 333:
        return "left"
    if u > 666:
        return "right"
    return "center"


def _same_bearing(obj: dict, box: dict) -> bool:
    b = str(obj.get("spatial") or obj.get("bearing") or "").lower()
    if not b:
        return False
    roi_b = _bearing_of_roi(box.get("roi"))
    if roi_b is None:
        return False
    return ("left" in b or "左" in b) and roi_b == "left" or (
        ("right" in b or "右" in b) and roi_b == "right"
    ) or (("center" in b or "中" in b) and roi_b == "center")


def geometry_ok(item: dict | None) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("geometry_status") not in (None, GEOMETRY_OK):
        return False
    ap = item.get("abs_pose") or {}
    return isinstance(ap, dict) and ap.get("x") is not None


def match_yolo_candidate(qwen_obj: dict, yolo_boxes: list[dict] | None,
                         *, center_match: float = DEFAULT_CENTER_MATCH,
                         depth_tol_m: float = DEFAULT_DEPTH_MISMATCH_M) -> dict | None:
    """Pick a validated YOLO ROI for a Qwen full-scene object.

    Matching is intentionally semantic-light: YOLO supplies localization only.
    We match by ROI overlap/center or by same bearing with compatible depth.
    """
    if not yolo_boxes:
        return None
    q_roi = qwen_obj.get("roi")
    q_center = roi_center_norm(q_roi)
    q_dist = qwen_obj.get("distance_m")
    ranked = []
    for box in yolo_boxes:
        if not geometry_ok(box):
            continue
        iou = roi_iou(q_roi, box.get("roi"))
        b_center = roi_center_norm(box.get("roi"))
        cdist = 1e9
        if q_center and b_center:
            cdist = math.hypot(q_center[0] - b_center[0], q_center[1] - b_center[1])
        depth_err = None
        if isinstance(q_dist, (int, float)) and isinstance(box.get("distance_m"), (int, float)):
            depth_err = abs(float(q_dist) - float(box["distance_m"]))
        bearing_match = _same_bearing(qwen_obj, box)
        if iou <= 0 and cdist > center_match and not (
            bearing_match and depth_err is not None and depth_err <= depth_tol_m
        ):
            continue
        score = 0.0
        score += iou * 4.0
        score += max(0.0, (center_match - min(cdist, center_match)) / center_match)
        if depth_err is not None:
            score += max(0.0, (depth_tol_m - min(depth_err, depth_tol_m)) / depth_tol_m)
        if bearing_match:
            score += 0.25
        ranked.append((score, box))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def _clamp_roi_1000(x: float, y: float, w: float, h: float) -> dict | None:
    w = max(30.0, min(900.0, float(w)))
    h = max(30.0, min(900.0, float(h)))
    x = max(0.0, min(1000.0 - w, float(x)))
    y = max(0.0, min(1000.0 - h, float(y)))
    if w <= 0 or h <= 0:
        return None
    return {"x": round(x, 1), "y": round(y, 1), "w": round(w, 1), "h": round(h, 1)}


def _base_roi_for_object(obj: dict) -> dict | None:
    roi = obj.get("roi")
    if isinstance(roi, dict):
        try:
            x, y, w, h = dp._roi_to_frac(roi)
        except (TypeError, ValueError):
            return None
        if w > 0 and h > 0:
            return {
                "x": x * 1000.0,
                "y": y * 1000.0,
                "w": w * 1000.0,
                "h": h * 1000.0,
            }
    bc = obj.get("bbox_center")
    if isinstance(bc, (list, tuple)) and len(bc) == 2:
        try:
            u, v = float(bc[0]), float(bc[1])
        except (TypeError, ValueError):
            return None
        return {"x": u - 80.0, "y": v - 80.0, "w": 160.0, "h": 160.0}
    return None


def generate_depth_snap_rois(obj: dict, *, max_rois: int = 25) -> list[dict]:
    """Generate small ROI probes around a Qwen coarse ROI/center."""
    base = _base_roi_for_object(obj)
    if not base:
        return []
    cx = base["x"] + base["w"] / 2.0
    cy = base["y"] + base["h"] / 2.0
    scales = (0.55, 0.75, 1.0, 1.25)
    shifts = (-0.28, 0.0, 0.28)
    out, seen = [], set()
    for scale in scales:
        w = base["w"] * scale
        h = base["h"] * scale
        for dx in shifts:
            for dy in shifts:
                roi = _clamp_roi_1000(cx - w / 2.0 + dx * base["w"],
                                      cy - h / 2.0 + dy * base["h"], w, h)
                if not roi:
                    continue
                key = tuple(roi[k] for k in ("x", "y", "w", "h"))
                if key in seen:
                    continue
                seen.add(key)
                out.append(roi)
                if len(out) >= max_rois:
                    return out
    return out


def _size_reasonable(roi: dict, stats: dict, *, sane_max_m: float) -> tuple[bool, dict]:
    sz = dp.roi_to_size(roi, stats)
    dims = [sz.get("width_m"), sz.get("height_m"), sz.get("depth_m")]
    ok = not any(v is not None and v > sane_max_m for v in dims)
    return ok, sz


def choose_depth_snap_roi(obj: dict, rois: list[dict], stats_list: list[dict | None],
                          *, depth_tol_m: float = DEFAULT_DEPTH_MISMATCH_M,
                          sane_max_m: float = DEFAULT_SANE_MAX_M) -> dict:
    """Choose the best depth-consistent ROI probe for a Qwen object."""
    target = obj.get("distance_m")
    if not rois or not stats_list:
        return {"status": GEOMETRY_NO_DEPTH, "roi": None, "roi_quality": {"reason": "no_probes"}}
    candidates = []
    for roi, stats in zip(rois, stats_list):
        if not stats:
            continue
        med = stats.get("median_m")
        if not (isinstance(med, (int, float)) and med > 0):
            continue
        if isinstance(target, (int, float)) and target > 0:
            depth_err = abs(float(med) - float(target))
        else:
            depth_err = 0.0
        size_ok, size = _size_reasonable(roi, stats, sane_max_m=sane_max_m)
        if not size_ok:
            continue
        n_valid = int(stats.get("n_valid") or 0)
        area = (roi["w"] * roi["h"]) / 1_000_000.0
        score = depth_err + max(0.0, 0.005 - area) * 20.0 - min(n_valid, 500) / 5000.0
        candidates.append((score, depth_err, roi, stats, size))
    if not candidates:
        return {"status": GEOMETRY_NO_DEPTH, "roi": None, "roi_quality": {"reason": "no_valid_depth"}}
    candidates.sort(key=lambda x: x[0])
    _score, depth_err, roi, stats, size = candidates[0]
    if isinstance(target, (int, float)) and target > 0 and depth_err > depth_tol_m:
        return {
            "status": GEOMETRY_DEPTH_MISMATCH,
            "roi": None,
            "roi_quality": {
                "reason": "best_depth_mismatch",
                "depth_error_m": round(depth_err, 3),
                "best_roi": roi,
            },
        }
    return {
        "status": GEOMETRY_OK,
        "roi": roi,
        "depth_stats": stats,
        "roi_quality": {
            "source": "qwen_depth_snap",
            "depth_error_m": round(depth_err, 3),
            "n_valid": int(stats.get("n_valid") or 0),
            "size": size,
        },
    }


def apply_candidate_to_object(obj: dict, candidate: dict, *, roi_source: str,
                              semantic_source: str = "qwen_full") -> dict:
    out = dict(obj)
    for k in ("roi", "bbox_center", "abs_pose", "distance_m", "size", "geometry_status",
              "depth_stats", "roi_quality", "detector_label", "confidence"):
        if k in candidate and candidate[k] is not None:
            out[k] = candidate[k]
    out["roi_source"] = roi_source
    out["semantic_source"] = semantic_source
    if "roi" in out:
        c = roi_center_norm(out["roi"])
        if c is not None:
            out["bbox_center"] = [round(c[0]), round(c[1])]
    return out


def roi_source_counts(objects: list[dict]) -> dict:
    return dict(Counter(o.get("roi_source") or "unknown" for o in objects or []))


def filter_judgments_by_allowed_names(parsed: dict, allowed_names: list | None) -> dict:
    """Force box judgments into a task-specific class set."""
    if not allowed_names:
        return parsed
    allowed_set = {str(n).strip() for n in allowed_names if str(n).strip()}
    out = {}
    for idx, j in (parsed or {}).items():
        jj = dict(j)
        if jj.get("keep") and jj.get("name") not in allowed_set:
            jj.update({"name": None, "keep": False, "completeness": "部分"})
        out[idx] = jj
    return out
