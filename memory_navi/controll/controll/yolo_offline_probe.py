#!/usr/bin/env python3
"""Offline YOLOE probe over saved camera images.

Run in the isolated yolo environment, for example:
  conda run -n yolo python memory_navi/controll/controll/yolo_offline_probe.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(HERE), str(REPO / "Report")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent_core.perception import yoloe  # noqa: E402


def _image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in exts)


def _write_readme(out_dir: Path, model: str, device: str, rows: list[dict], by_label: dict):
    lat = [float(r["elapsed_ms"]) for r in rows]
    median = statistics.median(lat) if lat else None
    status = "PASS" if median is not None and median <= 1500.0 else "REVIEW"
    lines = [
        "# YOLOE Offline Probe",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- model: `{model}`",
        f"- device: `{device}`",
        f"- images: {len(rows)}",
        f"- median_latency_ms: {median}",
        f"- latency_gate: {status} (target <= 1500 ms/image on CPU)",
        f"- accepted_by_label: {json.dumps(by_label, ensure_ascii=False, sort_keys=True)}",
        "",
        "Manual check still required: inspect `annotated/`, especially ROI placement on "
        "`err_roi_displaced.jpg` and the `cap*.jpg` frames.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(REPO / "Report" / "qwen_labeling_issue"))
    ap.add_argument("--out", default=str(REPO / "Report" / "yolo_probe"))
    ap.add_argument("--model", default=yoloe.DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=yoloe.DEFAULT_CONF)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--include-doors", action="store_true")
    args = ap.parse_args(argv)

    img_root = Path(args.images).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    ann_dir = out_dir / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    images = _image_paths(img_root)
    if not images:
        raise SystemExit(f"No images found at {img_root}")

    model = yoloe.load_model(args.model)
    rows = []
    detections = {
        "model": args.model,
        "device": args.device,
        "conf": args.conf,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "images": [],
    }
    by_label = {}

    for path in images:
        item = yoloe.predict_image(
            model,
            path,
            device=args.device,
            conf=args.conf,
            imgsz=args.imgsz,
            include_doors=args.include_doors,
        )
        result = item.pop("_result")
        ann_path = ann_dir / f"{path.stem}_{Path(args.model).stem}.jpg"
        try:
            result.save(filename=str(ann_path))
        except Exception as e:  # noqa: BLE001
            print(f"warning: failed to save annotated image for {path.name}: {e}")
            ann_path = None

        for obj in item["objects"]:
            by_label[obj["name"]] = by_label.get(obj["name"], 0) + 1
        rows.append({
            "image": path.name,
            "elapsed_ms": item["elapsed_ms"],
            "accepted": len(item["objects"]),
            "raw": len(item["raw_detections"]),
        })
        item["image"] = path.name
        item["annotated"] = str(ann_path.relative_to(out_dir)) if ann_path else None
        detections["images"].append(item)
        print(f"{path.name}: {item['elapsed_ms']} ms, accepted={len(item['objects'])}, raw={len(item['raw_detections'])}")

    with (out_dir / "detections.json").open("w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)
    with (out_dir / "latency.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "elapsed_ms", "accepted", "raw"])
        writer.writeheader()
        writer.writerows(rows)
    _write_readme(out_dir, args.model, args.device, rows, by_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

