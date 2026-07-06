#!/usr/bin/env python3
"""One-frame live YOLOE probe: look() -> YOLOE -> depth_roi -> optional memory write.

This script runs in the normal vLLM/agent_core environment and delegates the
heavy detector to the isolated yolo environment via `yolo_offline_probe.py`.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from agent_core import config  # noqa: E402
from agent_core.geometry import depth_projection as dp  # noqa: E402
from agent_core.mcp_bridge import McpBridge, RosTools  # noqa: E402
from agent_core.memory.fs_memory import FsMemory  # noqa: E402
from agent_core.perception import yoloe  # noqa: E402


def _call_json(ros: RosTools, name: str, args: dict) -> dict:
    out = ros.call(name, args)
    try:
        return json.loads((out.text or "").strip())
    except Exception:  # noqa: BLE001
        return {"ok": False, "raw": out.text, "is_error": out.is_error}


def _run_yolo(image_path: Path, out_dir: Path, args) -> list[dict]:
    probe = HERE / "yolo_offline_probe.py"
    cmd = [
        args.detector_python,
        str(probe),
        "--images",
        str(image_path),
        "--out",
        str(out_dir),
        "--model",
        args.model,
        "--device",
        args.device,
        "--conf",
        str(args.conf),
    ]
    if args.include_doors:
        cmd.append("--include-doors")
    subprocess.run(cmd, check=True, timeout=args.detector_timeout_s)
    with (out_dir / "detections.json").open(encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images") or []
    return (images[0].get("objects") if images else []) or []


def _backfill_live_geometry(ros: RosTools, objects: list[dict], pose: dict) -> None:
    rois = [o["roi"] for o in objects if isinstance(o.get("roi"), dict)]
    if not rois:
        return
    data = _call_json(ros, "depth_roi", {"rois_json": json.dumps(rois, ensure_ascii=False)})
    if not data.get("ok"):
        return
    stats_list = data.get("stats") or []
    p = {"x": float(pose["x"]), "y": float(pose["y"]),
         "yaw_deg": float(pose.get("yaw_deg", pose.get("yaw", 0.0)))}
    for obj, stats in zip(objects, stats_list):
        if not stats:
            continue
        med = stats.get("median_m")
        nmin = stats.get("near_min_m")
        dist = nmin if isinstance(nmin, (int, float)) and nmin > 0 else med
        if isinstance(dist, (int, float)) and dist > 0:
            obj["distance_m"] = round(float(dist), 3)
            obj["distance_src"] = "depth_roi"
        try:
            obj["size"] = dp.roi_to_size(obj["roi"], stats)
        except Exception:  # noqa: BLE001
            pass
        if isinstance(dist, (int, float)) and dist > 0:
            try:
                uc, vc = dp.roi_center_pixel(obj["roi"])
                obj["abs_pose"] = dp.back_project(uc, vc, float(dist), robot_pose=p)["abs_pose"]
            except Exception:  # noqa: BLE001
                obj["abs_pose"] = None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "Report" / "yolo_live_probe"))
    ap.add_argument("--detector-python", default="/home/kuko/miniconda3/envs/yolo/bin/python")
    ap.add_argument("--model", default=yoloe.DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=yoloe.DEFAULT_CONF)
    ap.add_argument("--include-doors", action="store_true")
    ap.add_argument("--detector-timeout-s", type=float, default=120.0)
    ap.add_argument("--write-area", default="")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bridge = McpBridge(config.MCP_URL, connect_timeout=config.MCP_CONNECT_TIMEOUT)
    ros = RosTools(bridge, call_timeout=120.0)
    try:
        ros.connect(config.ROSBRIDGE_IP, config.ROSBRIDGE_PORT)
        look = ros.call("look", {})
        if not look.images:
            raise SystemExit("look() returned no image")
        image_path = out_dir / "live_frame.jpg"
        image_path.write_bytes(base64.b64decode(look.images[0].b64))

        pose = _call_json(ros, "get_pose", {})
        if "x" not in pose:
            pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0, "_pose_error": pose}
        objects = _run_yolo(image_path, out_dir / "offline", args)
        _backfill_live_geometry(ros, objects, pose)

        if args.write_area:
            mem = FsMemory(config.MEMORY_ROOT, config.ENV_NAME)
            for obj in objects:
                if obj.get("is_door"):
                    continue
                mem.upsert_object(args.write_area, obj)

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pose": pose,
            "image": str(image_path),
            "objects": objects,
            "write_area": args.write_area or None,
        }
        (out_dir / "live_probe.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

