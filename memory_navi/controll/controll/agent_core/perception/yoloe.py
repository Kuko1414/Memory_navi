"""YOLOE detection/segmentation helpers.

This module keeps the heavy Ultralytics dependency behind function boundaries so
the normal vLLM/ROS control environment can import the pure conversion helpers
without installing YOLO.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "yoloe-26n-seg.pt"
DEFAULT_CONF = 0.25
DEFAULT_TEXT_CLASSES = [
    "cabinet",
    "desk",
    "table",
    "office chair",
    "chair",
    "monitor",
    "tv",
    "sofa",
    "couch",
    "potted plant",
    "plant",
    "sink",
    "countertop",
    "kitchen counter",
    "keyboard",
    "door",
]

CANONICAL_NAME_BY_CLASS = {
    "cabinet": "柜子",
    "desk": "办公桌",
    "table": "办公桌",
    "dining table": "办公桌",
    "office chair": "办公椅",
    "chair": "办公椅",
    "monitor": "显示器",
    "tv": "显示器",
    "television": "显示器",
    "sofa": "沙发",
    "couch": "沙发",
    "potted plant": "绿植",
    "plant": "绿植",
    "sink": "水槽",
    "countertop": "厨台",
    "kitchen counter": "厨台",
    "counter": "厨台",
    "keyboard": "键盘",
    "door": "门",
}
DOOR_LABELS = {"door"}


def load_inference_profile(profile_path: str | os.PathLike) -> dict:
    """Load and validate a versioned detector publication profile.

    The profile deliberately keeps model selection and publication thresholds outside
    the exploration code. Switching backends is therefore an environment/config
    change, and labels that failed held-out precision can remain visible in raw
    detections without being published into memory.
    """
    path = Path(profile_path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        profile = json.load(stream)
    if not isinstance(profile, dict):
        raise ValueError("inference profile must be a JSON object")

    required = {"model_version", "weights", "class_conf_thresholds"}
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"inference profile missing keys: {', '.join(missing)}")

    thresholds = profile["class_conf_thresholds"]
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("class_conf_thresholds must be a non-empty JSON object")
    normalized_thresholds = {}
    for label, threshold in thresholds.items():
        value = float(threshold)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid confidence threshold for {label}: {value}")
        normalized_thresholds[normalize_label(label)] = value

    profile["profile_path"] = str(path)
    profile["weights"] = str(Path(profile["weights"]).expanduser().resolve())
    profile["model_version"] = str(profile["model_version"])
    profile["class_conf_thresholds"] = normalized_thresholds
    profile["candidate_only"] = [
        normalize_label(label) for label in profile.get("candidate_only", [])
    ]
    candidate_conf = float(profile.get("candidate_conf", 0.05))
    if not 0.0 <= candidate_conf <= 1.0:
        raise ValueError("candidate_conf must be in [0, 1]")
    profile["candidate_conf"] = candidate_conf
    return profile


def normalize_label(label: str) -> str:
    """Normalize detector labels for mapping and comparison."""
    return " ".join(str(label or "").strip().lower().replace("_", " ").split())


def canonical_name(label: str) -> str | None:
    """Map an English detector class to the memory system's Chinese object name."""
    return CANONICAL_NAME_BY_CLASS.get(normalize_label(label))


def is_door_label(label: str) -> bool:
    """Return whether the detector label represents a door/topology cue."""
    return normalize_label(label) in DOOR_LABELS


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bbox_xyxy_to_roi(bbox_xyxy, width: int, height: int, scale: int = 1000) -> dict:
    """Convert pixel xyxy bbox to the existing normalized {x,y,w,h} 0..1000 ROI."""
    if width <= 0 or height <= 0:
        raise ValueError("image width/height must be positive")
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1 = clamp(x1, 0.0, float(width))
    x2 = clamp(x2, 0.0, float(width))
    y1 = clamp(y1, 0.0, float(height))
    y2 = clamp(y2, 0.0, float(height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return {
        "x": int(round(x1 / width * scale)),
        "y": int(round(y1 / height * scale)),
        "w": int(round((x2 - x1) / width * scale)),
        "h": int(round((y2 - y1) / height * scale)),
    }


def bbox_xyxy_to_center(bbox_xyxy, width: int, height: int, scale: int = 1000) -> list:
    """Convert pixel xyxy bbox center to the existing 0..1000 center convention."""
    roi = bbox_xyxy_to_roi(bbox_xyxy, width, height, scale=scale)
    return [int(round(roi["x"] + roi["w"] / 2)), int(round(roi["y"] + roi["h"] / 2))]


def mask_polygon_to_depth_probe_roi(
    polygon,
    width: int,
    height: int,
    *,
    scale: int = 1000,
    scanlines: int = 15,
    min_probe_px: float = 5.0,
    max_probe_px: float = 12.0,
) -> dict | None:
    """Return a small normalized ROI guaranteed to sit inside a mask polygon.

    A segmentation bbox often includes floor, background, or a neighbouring object.
    Sampling the full bbox (especially its nearest depth) can therefore project a valid
    detection to the wrong world position.  This helper finds the widest polygon
    interior interval across several scanlines and places a small depth probe there.
    """
    if (
        width <= 0
        or height <= 0
        or scanlines <= 0
        or min_probe_px <= 0
        or max_probe_px < min_probe_px
    ):
        return None
    try:
        points = [(float(point[0]), float(point[1])) for point in polygon]
    except (TypeError, ValueError, IndexError):
        return None
    if len(points) < 3:
        return None
    ys = [point[1] for point in points]
    ymin, ymax = min(ys), max(ys)
    if ymax <= ymin:
        return None

    best = None
    for index in range(scanlines):
        y = ymin + (index + 0.5) / scanlines * (ymax - ymin)
        intersections = []
        for start, end in zip(points, points[1:] + points[:1]):
            x1, y1 = start
            x2, y2 = end
            if (y1 <= y < y2) or (y2 <= y < y1):
                intersections.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        intersections.sort()
        for left, right in zip(intersections[::2], intersections[1::2]):
            span = right - left
            if span > 0 and (best is None or span > best[0]):
                best = (span, (left + right) / 2.0, y)
    if best is None:
        return None

    span, center_x, center_y = best
    probe_width = max(min_probe_px, min(max_probe_px, span * 0.25))
    probe_height = max(min_probe_px, min(max_probe_px, (ymax - ymin) * 0.08))
    x1 = clamp(center_x - probe_width / 2.0, 0.0, float(width))
    x2 = clamp(center_x + probe_width / 2.0, 0.0, float(width))
    y1 = clamp(center_y - probe_height / 2.0, 0.0, float(height))
    y2 = clamp(center_y + probe_height / 2.0, 0.0, float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "x": int(round(x1 / width * scale)),
        "y": int(round(y1 / height * scale)),
        "w": max(1, int(round((x2 - x1) / width * scale))),
        "h": max(1, int(round((y2 - y1) / height * scale))),
    }


def raw_detection_to_object(
    det: dict,
    width: int,
    height: int,
    *,
    conf_thres: float = DEFAULT_CONF,
    include_doors: bool = False,
    source: str = "yoloe",
) -> dict | None:
    """Convert one raw YOLO detection into the current memory-compatible object dict."""
    label = normalize_label(det.get("label") or det.get("class_name") or "")
    name = canonical_name(label)
    if not name:
        return None
    conf = float(det.get("confidence", det.get("conf", 0.0)) or 0.0)
    if conf < conf_thres:
        return None
    if is_door_label(label) and not include_doors:
        return None
    bbox = det.get("bbox_xyxy") or det.get("xyxy")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None

    obj = {
        "name": name,
        "confidence": round(conf, 4),
        "detector_score": round(conf, 4),
        "roi": bbox_xyxy_to_roi(bbox, width, height),
        "bbox_center": bbox_xyxy_to_center(bbox, width, height),
        "verified_by": [source],
        "detector_label": label,
        "detector_source": source,
        "model_version": source,
    }
    if is_door_label(label):
        obj["is_door"] = True
    if det.get("mask_area_px") is not None:
        obj["mask_area_px"] = int(det["mask_area_px"])
    if det.get("mask_polygon") is not None:
        obj["mask_polygon"] = det["mask_polygon"]
    return obj


def normalize_detections(
    detections: list[dict],
    width: int,
    height: int,
    *,
    conf_thres: float = DEFAULT_CONF,
    include_doors: bool = False,
    source: str = "yoloe",
    class_conf_thresholds: dict[str, float] | None = None,
    disabled_labels: set[str] | None = None,
) -> list[dict]:
    """Convert a list of raw detections to memory-compatible objects."""
    thresholds = {
        normalize_label(label): float(threshold)
        for label, threshold in (class_conf_thresholds or {}).items()
    }
    disabled = {normalize_label(label) for label in (disabled_labels or set())}
    out = []
    for det in detections or []:
        label = normalize_label(det.get("label") or det.get("class_name") or "")
        if label in disabled:
            continue
        obj = raw_detection_to_object(
            det,
            width,
            height,
            conf_thres=thresholds.get(label, conf_thres),
            include_doors=include_doors,
            source=source,
        )
        if obj is not None:
            out.append(obj)
    return out


def partition_detections(
    detections: list[dict],
    width: int,
    height: int,
    *,
    conf_thres: float = DEFAULT_CONF,
    include_doors: bool = False,
    source: str = "yoloe",
    class_conf_thresholds: dict[str, float] | None = None,
    candidate_only_labels: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Split detections into publishable objects and non-navigable candidates.

    Candidate-only classes and known detections below their class publish threshold
    remain available for multi-view/Qwen review. They are never mixed into the
    confirmed ``objects`` list.
    """
    thresholds = {
        normalize_label(label): float(threshold)
        for label, threshold in (class_conf_thresholds or {}).items()
    }
    candidate_only = {
        normalize_label(label) for label in (candidate_only_labels or set())
    }
    confirmed: list[dict] = []
    candidates: list[dict] = []
    for det in detections or []:
        label = normalize_label(det.get("label") or det.get("class_name") or "")
        score = float(det.get("confidence", det.get("conf", 0.0)) or 0.0)
        if score < conf_thres:
            continue
        publish_threshold = thresholds.get(label, conf_thres)
        is_candidate_only = label in candidate_only
        below_publish_threshold = score < publish_threshold
        obj = raw_detection_to_object(
            det,
            width,
            height,
            conf_thres=conf_thres,
            include_doors=include_doors,
            source=source,
        )
        if obj is None:
            continue
        if is_candidate_only or below_publish_threshold:
            obj["memory_status"] = "candidate"
            obj["candidate_reason"] = (
                "class_candidate_only" if is_candidate_only else "below_publish_threshold"
            )
            obj["publish_threshold"] = round(float(publish_threshold), 4)
            candidates.append(obj)
        else:
            obj["memory_status"] = "confirmed"
            obj["publish_threshold"] = round(float(publish_threshold), 4)
            confirmed.append(obj)
    return confirmed, candidates


def raw_detections_to_boxes(
    raw_detections: list[dict],
    width: int,
    height: int,
    *,
    conf_thres: float = 0.05,
) -> list[dict]:
    """混合后端定位源：把 YOLO 低 conf 的 raw_detections 转成【只带 ROI/编号、不带名字】的框。

    与 normalize_detections 的区别：**不套 canonical_name**（命名交给 Qwen 弃权门），
    label-agnostic 保留所有 >=conf 的框（含开集词表外的），只丢无效 bbox。
    返回 [{"idx": 1, "roi": {x,y,w,h 0..1000}, "bbox_center": [x,y],
           "confidence": float, "detector_label": str}]，idx 从 1 顺序编号。
    """
    boxes = []
    idx = 0
    for det in raw_detections or []:
        conf = float(det.get("confidence", det.get("conf", 0.0)) or 0.0)
        if conf < conf_thres:
            continue
        bbox = det.get("bbox_xyxy") or det.get("xyxy")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        idx += 1
        boxes.append({
            "idx": idx,
            "roi": bbox_xyxy_to_roi(bbox, width, height),
            "bbox_center": bbox_xyxy_to_center(bbox, width, height),
            "confidence": round(conf, 4),
            "detector_label": normalize_label(
                det.get("label") or det.get("class_name") or ""),
        })
    return boxes


def assemble_hybrid_objects(
    boxes: list[dict],
    judgments: dict,
    *,
    source: str = "hybrid",
) -> list[dict]:
    """混合装配：YOLO 干净 ROI + Qwen 命名/弃权 → rep 形状的物体列表。

    boxes:     raw_detections_to_boxes 的输出（带 idx/roi/bbox_center/confidence）。
    judgments: name_boxes 的输出 {idx: {"name","completeness","keep"}}。
    仅保留 keep and completeness=='完整' and name 的框；
    输出保留 **YOLO 的 roi/bbox_center/confidence** + **Qwen 的 name** + verified_by=[source]。
    绝不产坐标（守"神经网络不直出米制坐标"铁律，几何回填由代码在下游做）。纯函数。
    """
    out = []
    for b in boxes or []:
        j = (judgments or {}).get(b.get("idx")) or {}
        name = j.get("name")
        if not (j.get("keep") and j.get("completeness") == "完整" and name):
            continue
        obj = {
            "name": name,
            "confidence": b.get("confidence"),
            "roi": b.get("roi"),
            "bbox_center": b.get("bbox_center"),
            "verified_by": [source],
            "detector_source": source,
            "completeness": "完整",
        }
        if b.get("detector_label"):
            obj["detector_label"] = b["detector_label"]
        out.append(obj)
    return out


def _to_list(x: Any) -> list:
    """Best-effort tensor/array/list conversion without importing numpy/torch here."""
    if x is None:
        return []
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _names_get(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def bbox_iou(first, second) -> float:
    """Return intersection-over-union for two pixel ``xyxy`` boxes."""
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def deduplicate_raw_detections(
    detections: list[dict],
    *,
    iou_threshold: float = 0.85,
) -> list[dict]:
    """Suppress near-identical same-class YOLO26 candidates before memory insertion.

    Different labels are deliberately preserved so the downstream semantic gate can
    resolve a genuine class conflict.  This only removes the one-to-one/one-to-many head
    duplicates observed with almost identical boxes and masks.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    ordered = sorted(
        detections or [],
        key=lambda item: float(item.get("confidence", item.get("conf", 0.0)) or 0.0),
        reverse=True,
    )
    kept: list[dict] = []
    for detection in ordered:
        label = normalize_label(
            detection.get("label") or detection.get("class_name") or ""
        )
        bbox = detection.get("bbox_xyxy") or detection.get("xyxy")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            kept.append(detection)
            continue
        duplicate = any(
            label
            == normalize_label(existing.get("label") or existing.get("class_name") or "")
            and bbox_iou(bbox, existing.get("bbox_xyxy") or existing.get("xyxy"))
            >= iou_threshold
            for existing in kept
            if isinstance(existing.get("bbox_xyxy") or existing.get("xyxy"), (list, tuple))
            and len(existing.get("bbox_xyxy") or existing.get("xyxy")) == 4
        )
        if not duplicate:
            kept.append(detection)
    return kept


def result_to_raw_detections(result, *, dedup_iou: float | None = 0.85) -> list[dict]:
    """Convert an Ultralytics Results object to small JSON-safe raw detections."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxys = _to_list(getattr(boxes, "xyxy", None))
    confs = _to_list(getattr(boxes, "conf", None))
    clss = _to_list(getattr(boxes, "cls", None))
    names = getattr(result, "names", None) or {}
    masks = getattr(result, "masks", None)
    mask_areas = []
    mask_polygons = []
    if masks is not None and getattr(masks, "data", None) is not None:
        for m in _to_list(masks.data):
            try:
                mask_areas.append(int(sum(sum(row) for row in m)))
            except TypeError:
                mask_areas.append(None)
        for polygon in getattr(masks, "xy", []) or []:
            mask_polygons.append(
                [[round(float(x), 2), round(float(y), 2)] for x, y in _to_list(polygon)]
            )

    detections = []
    for i, bbox in enumerate(xyxys):
        cls_id = int(clss[i]) if i < len(clss) else -1
        det = {
            "label": _names_get(names, cls_id),
            "confidence": float(confs[i]) if i < len(confs) else 0.0,
            "bbox_xyxy": [round(float(v), 2) for v in bbox],
        }
        if i < len(mask_areas) and mask_areas[i] is not None:
            det["mask_area_px"] = mask_areas[i]
        if i < len(mask_polygons):
            det["mask_polygon"] = mask_polygons[i]
        detections.append(det)
    if dedup_iou is None:
        return detections
    return deduplicate_raw_detections(detections, iou_threshold=dedup_iou)


def load_model(model_name: str = DEFAULT_MODEL, text_classes: list[str] = None):
    """Load a YOLO/YOLOE model. Imports Ultralytics only inside this function."""
    text_classes = list(text_classes or DEFAULT_TEXT_CLASSES)
    if "yoloe" in Path(model_name).name.lower():
        from ultralytics import YOLOE

        model = YOLOE(model_name)
        if hasattr(model, "get_text_pe"):
            model.set_classes(text_classes, model.get_text_pe(text_classes))
        else:
            model.set_classes(text_classes)
        return model

    from ultralytics import YOLO

    return YOLO(model_name)


def predict_image(
    model,
    image_path: str | os.PathLike,
    *,
    device: str = "cpu",
    conf: float = DEFAULT_CONF,
    imgsz: int | None = None,
    include_doors: bool = False,
    source: str = "yoloe",
    class_conf_thresholds: dict[str, float] | None = None,
    disabled_labels: set[str] | None = None,
    dedup_iou: float | None = 0.85,
) -> dict:
    """Run one image through a loaded model and return raw + normalized detections."""
    from PIL import Image

    image_path = str(image_path)
    with Image.open(image_path) as img:
        width, height = img.size
    kwargs = {"device": device, "conf": conf, "verbose": False}
    if imgsz:
        kwargs["imgsz"] = imgsz
    t0 = time.perf_counter()
    results = model.predict(image_path, **kwargs)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    result = results[0]
    raw = result_to_raw_detections(result, dedup_iou=dedup_iou)
    objects, candidate_objects = partition_detections(
        raw,
        width,
        height,
        conf_thres=conf,
        include_doors=include_doors,
        source=source,
        class_conf_thresholds=class_conf_thresholds,
        candidate_only_labels=disabled_labels,
    )
    return {
        "image": image_path,
        "width": width,
        "height": height,
        "elapsed_ms": elapsed_ms,
        "raw_detections": raw,
        "objects": objects,
        "candidate_objects": candidate_objects,
        "_result": result,
    }
