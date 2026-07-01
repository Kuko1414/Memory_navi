"""Perception summary tools for ROS MCP.

把高维传感器原始消息（如 LaserScan ~400 ranges）在 MCP 层就**全精度计算成决策标量**，
只把少量标量回灌给 LLM，从根上避免对话历史被原始数组撑爆（上下文膨胀根因）。

设计原则（见 CLAUDE.md / 记忆 context-bloat-supervisor-design）：
- 全精度计算、少 token 输出 —— 不降精度、不死板裁剪。
- 过滤逻辑与 safety_node 保持一致（剔 nan / inf / < range_min）。
- 单机项目：命名空间从环境变量 ROS_NAMESPACE 读，默认 agent0（不让 LLM 传参）。
"""

import base64
import json
import math
import os
import time

import numpy as np
from fastmcp import FastMCP

from ros_mcp.utils.websocket import WebSocketManager, parse_input

#: 默认命名空间（单机项目；多机时改环境变量）。
DEFAULT_NAMESPACE = os.environ.get("ROS_NAMESPACE", "agent0")

#: 8 扇区（REP-103：x 前、y 左、yaw 逆时针为正）中心角(度) → 人类可读标签。
_SECTOR8_LABELS = {
    0: "front",
    45: "front_left",
    90: "left",
    135: "rear_left",
    180: "rear",
    -180: "rear",        # _norm_deg(180) 落到 -180
    -135: "rear_right",
    -90: "right",
    -45: "front_right",
}


def _norm_deg(deg: float) -> float:
    """归一化到 [-180, 180)。"""
    return ((deg + 180.0) % 360.0) - 180.0


def summarize_scan(
    ranges,
    angle_min: float,
    angle_increment: float,
    range_min: float = 0.0,
    range_max: float = float("inf"),
    sectors: int = 8,
    front_halfwidth_deg: float = 15.0,
    clear_thresh_m: float = 0.5,
) -> dict:
    """把一条 LaserScan 的全精度 ranges 压成决策标量（纯函数，可脱离 ROS 单测）。

    Args:
        ranges: 原始距离数组（米）。
        angle_min: 第一个 ray 的角度（弧度）。
        angle_increment: 相邻 ray 角度增量（弧度）。
        range_min/range_max: 有效距离窗（米）；越界与 nan/inf 视为无效。
        sectors: 均分扇区数（8 时给人类可读标签，否则用 "sec_<中心角度>"）。
        front_halfwidth_deg: 正前方畅通判定的半宽（度）。
        clear_thresh_m: 正前方最近障碍 > 该值则 clear_path=True。

    Returns:
        dict: {nearest, sectors, clear_path, valid_points, total_points}
              全部距离 round(2)；无任何有效点时 nearest 为 None。
    """
    rmin = max(range_min, 0.01)
    bin_size = 360.0 / sectors

    nearest_d = float("inf")
    nearest_deg = 0.0
    # 每扇区最近距离
    sector_min = [float("inf")] * sectors
    front_min = float("inf")
    valid = 0

    for i, r in enumerate(ranges):
        if r is None or r != r:                 # None / nan
            continue
        if r < rmin or r == float("inf") or r > range_max:
            continue
        valid += 1
        deg = _norm_deg(math.degrees(angle_min + i * angle_increment))

        if r < nearest_d:
            nearest_d = r
            nearest_deg = deg

        # 扇区归并：bin 0 以正前方(0°)为中心
        idx = int(round(deg / bin_size)) % sectors
        if r < sector_min[idx]:
            sector_min[idx] = r

        if abs(deg) <= front_halfwidth_deg and r < front_min:
            front_min = r

    # 扇区结果整理为标签字典
    sector_out = {}
    for idx in range(sectors):
        center = _norm_deg(idx * bin_size)
        label = _SECTOR8_LABELS.get(int(center)) if sectors == 8 else None
        if label is None:
            label = f"sec_{int(center)}"
        d = sector_min[idx]
        sector_out[label] = None if d == float("inf") else round(d, 2)

    nearest = None
    if nearest_d != float("inf"):
        nearest = {"dist_m": round(nearest_d, 2), "bearing_deg": round(nearest_deg, 1)}

    return {
        "nearest": nearest,
        "sectors": sector_out,
        "clear_path": front_min > clear_thresh_m,
        "front_min_m": None if front_min == float("inf") else round(front_min, 2),
        "valid_points": valid,
        "total_points": len(ranges),
    }


def register_perception_tools(mcp: FastMCP, ws_manager: WebSocketManager) -> None:
    """注册感知摘要工具到 FastMCP 实例（仿 register_topic_tools）。"""

    @mcp.tool(
        description=(
            "Get a compact obstacle summary from the robot's LiDAR (full-precision "
            "computed server-side, returns only ~scalars instead of ~400 raw ranges). "
            "Use this INSTEAD of subscribing to /scan directly — it avoids flooding the "
            "conversation with raw arrays.\n"
            "Returns: nearest {dist_m, bearing_deg} (0=front, +=left, -=right), per-sector "
            "min distances, and clear_path (is the path straight ahead open?)."
        ),
    )
    def scan_summary(sectors: int = 8, timeout: float = 2.0, clear_thresh_m: float = 0.5) -> dict:
        """订阅一次 /<ns>/scan 并返回决策标量摘要。

        Args:
            sectors: 均分扇区数（默认 8）。
            timeout: 等待一帧的超时（秒）。
            clear_thresh_m: 正前方畅通阈值（米）。
        """
        ns = DEFAULT_NAMESPACE
        topic = f"/{ns}/scan"
        msg_type = "sensor_msgs/msg/LaserScan"

        try:
            sectors = int(sectors)
            if sectors < 1:
                return {"error": "sectors must be an integer >= 1"}
        except (ValueError, TypeError):
            return {"error": "sectors must be an integer"}
        try:
            timeout = float(timeout)
            if timeout <= 0:
                return {"error": "timeout must be > 0"}
        except (ValueError, TypeError):
            return {"error": "timeout must be a number"}

        subscribe_msg = {
            "op": "subscribe",
            "topic": topic,
            "type": msg_type,
            "queue_length": 1,
            "throttle_rate": 0,
        }

        with ws_manager:
            send_error = ws_manager.send(subscribe_msg)
            if send_error:
                return {"error": f"Failed to subscribe: {send_error}"}

            end_time = time.time() + timeout
            try:
                while time.time() < end_time:
                    response = ws_manager.receive(timeout=0.5)
                    if response is None:
                        continue
                    msg_data, _ = parse_input(response, False)
                    if not msg_data:
                        continue
                    if msg_data.get("op") == "status" and msg_data.get("level") == "error":
                        return {"error": f"Rosbridge error: {msg_data.get('msg', 'Unknown error')}"}
                    if msg_data.get("op") == "publish" and msg_data.get("topic") == topic:
                        scan = msg_data.get("msg", {})
                        ranges = scan.get("ranges") or []
                        if not ranges:
                            return {"error": f"{topic} produced an empty scan"}
                        summary = summarize_scan(
                            ranges,
                            angle_min=float(scan.get("angle_min", 0.0)),
                            angle_increment=float(scan.get("angle_increment", 0.0)),
                            range_min=float(scan.get("range_min", 0.0)),
                            range_max=float(scan.get("range_max", float("inf"))),
                            sectors=sectors,
                            clear_thresh_m=clear_thresh_m,
                        )
                        summary["topic"] = topic
                        return summary
                return {"error": f"Timeout waiting for a message from {topic}"}
            finally:
                ws_manager.send({"op": "unsubscribe", "topic": topic})

    @mcp.tool(
        description=(
            "Get a compact depth reading straight ahead from the robot's depth camera, as "
            "per-column median distances (meters) ordered LEFT->RIGHT across the field of view. "
            "Use this together with the camera image to estimate how far away objects are: an "
            "object on the left of the image is roughly the leftmost column's distance, center "
            "object the middle column, etc. Returns {cols, depths_m, min_m, valid_cols}; a column "
            "value of -1 means no valid depth there (too near/far or sky). "
            "NOTE: reads the small /<ns>/camera/depth/columns topic produced by depth_summary_relay.py "
            "(the raw depth image is too large for rosbridge); if you get a timeout, that relay is not running."
        ),
    )
    def depth_summary(timeout: float = 2.0) -> dict:
        """订阅一次 /<ns>/camera/depth/summary（Float32MultiArray，每列中位深度）返回标量摘要。"""
        ns = DEFAULT_NAMESPACE
        topic = f"/{ns}/camera/depth/columns"
        msg_type = "std_msgs/msg/Float32MultiArray"
        try:
            timeout = float(timeout)
            if timeout <= 0:
                return {"error": "timeout must be > 0"}
        except (ValueError, TypeError):
            return {"error": "timeout must be a number"}

        subscribe_msg = {
            "op": "subscribe", "topic": topic, "type": msg_type,
            "queue_length": 1, "throttle_rate": 0,
        }
        with ws_manager:
            send_error = ws_manager.send(subscribe_msg)
            if send_error:
                return {"error": f"Failed to subscribe: {send_error}"}
            end_time = time.time() + timeout
            try:
                while time.time() < end_time:
                    response = ws_manager.receive(timeout=0.5)
                    if response is None:
                        continue
                    msg_data, _ = parse_input(response, False)
                    if not msg_data:
                        continue
                    if msg_data.get("op") == "status" and msg_data.get("level") == "error":
                        return {"error": f"Rosbridge error: {msg_data.get('msg', 'Unknown error')}"}
                    if msg_data.get("op") == "publish" and msg_data.get("topic") == topic:
                        data = (msg_data.get("msg", {}) or {}).get("data") or []
                        depths = [round(float(d), 2) for d in data]
                        valid = [d for d in depths if d >= 0]
                        return {
                            "topic": topic,
                            "order": "left_to_right",
                            "cols": len(depths),
                            "depths_m": depths,
                            "min_m": round(min(valid), 2) if valid else None,
                            "valid_cols": len(valid),
                        }
                return {"error": f"Timeout waiting for {topic} (is depth_summary_relay.py running?)"}
            finally:
                ws_manager.send({"op": "unsubscribe", "topic": topic})

    @mcp.tool(
        description=(
            "Read ONE full depth frame and compute per-ROI depth stats in meters, for a batch of "
            "normalized bounding boxes. Use this to let CODE size up objects a vision model marked "
            "with an ROI (the model only outputs the box; geometry is computed here). "
            "rois_json is a JSON list like [{\"x\":0.1,\"y\":0.2,\"w\":0.3,\"h\":0.4}, ...] with "
            "x,y = top-left and w,h = width,height, all normalized 0~1 (0~1000 also accepted). "
            "Returns {ok, n, stats:[{median_m,min_m,max_m,near_min_m,near_max_m,n_valid}, ...]} aligned "
            "to the input order; near_min/near_max is the depth cluster the ROI center sits in (background "
            "separated by a gap is excluded), so near_max-near_min ~= object thickness. A null stat means "
            "no valid depth in that ROI. Reads /<ns>/camera/depth/image (32FC1 meters)."
        ),
    )
    def depth_roi(rois_json: str, timeout: float = 2.5, gap_thresh_m: float = 0.30) -> dict:
        """读一帧 /<ns>/camera/depth/image，对一批归一化 ROI 批量算深度统计（方法B尺寸的底层）。"""
        ns = DEFAULT_NAMESPACE
        topic = f"/{ns}/camera/depth/image"
        msg_type = "sensor_msgs/msg/Image"
        try:
            rois = json.loads(rois_json) if isinstance(rois_json, str) else rois_json
        except (ValueError, TypeError):
            return {"ok": False, "error": "rois_json must be a JSON list of {x,y,w,h}"}
        if not isinstance(rois, list) or not rois:
            return {"ok": False, "error": "rois_json must be a non-empty JSON list"}
        try:
            timeout = float(timeout)
        except (ValueError, TypeError):
            timeout = 2.5

        def _frac(r):
            x = float(r.get("x", 0) or 0); y = float(r.get("y", 0) or 0)
            w = float(r.get("w", 0) or 0); h = float(r.get("h", 0) or 0)
            if max(abs(x), abs(y), abs(w), abs(h)) > 1.5:   # 0~1000 量纲容错
                x, y, w, h = x / 1000.0, y / 1000.0, w / 1000.0, h / 1000.0
            return x, y, w, h

        def _stats_for(depth, H, W, r):
            x, y, w, h = _frac(r)
            u0 = max(0, min(W - 1, int(round(x * W))))
            u1 = max(u0 + 1, min(W, int(round((x + w) * W))))
            v0 = max(0, min(H - 1, int(round(y * H))))
            v1 = max(v0 + 1, min(H, int(round((y + h) * H))))
            patch = depth[v0:v1, u0:u1].ravel()
            vals = patch[(patch > 0.05) & (patch < 9.5)]
            if vals.size == 0:
                return None
            vals = np.sort(vals)
            median = float(np.median(vals))
            # 1) 按 gap 切簇，取包含 median 的那簇 → 剔掉与前景明显分离的远背景斑块
            if vals.size == 1:
                band = vals
            else:
                splits = np.where(np.diff(vals) > gap_thresh_m)[0]
                clusters = np.split(vals, splits + 1)
                band = next((c for c in clusters if c[0] <= median <= c[-1]),
                            max(clusters, key=len))
            # 2) 簇内再用稳健分位数 p15/p85 修剪连续地面/墙的深度斜坡尾巴（否则厚度=整跨度虚大）
            near_min = float(np.percentile(band, 15))
            near_max = float(np.percentile(band, 85))
            # median 也改用簇内中位（更贴物体表面，少受远背景拉偏）
            median = float(np.median(band))
            return {
                "median_m": round(median, 3),
                "min_m": round(float(vals[0]), 3),
                "max_m": round(float(vals[-1]), 3),
                "near_min_m": round(near_min, 3),
                "near_max_m": round(near_max, 3),
                "n_valid": int(vals.size),
            }

        with ws_manager:
            if ws_manager.send({"op": "subscribe", "topic": topic, "type": msg_type,
                                "queue_length": 1, "throttle_rate": 0}):
                return {"ok": False, "error": f"Failed to subscribe {topic}"}
            end = time.time() + timeout
            try:
                while time.time() < end:
                    response = ws_manager.receive(timeout=0.5)
                    if response is None:
                        continue
                    md, _ = parse_input(response, False)
                    if not md or md.get("op") != "publish" or md.get("topic") != topic:
                        continue
                    msg = md.get("msg", {}) or {}
                    data = msg.get("data", "")
                    if not data:
                        return {"ok": False, "error": "empty depth frame"}
                    raw = np.frombuffer(base64.b64decode(data), dtype=np.float32)
                    H, W = int(msg["height"]), int(msg["width"])
                    if raw.size < H * W:
                        return {"ok": False, "error": f"depth size {raw.size} < {H}x{W}"}
                    depth = raw[:H * W].reshape(H, W)
                    stats = [_stats_for(depth, H, W, r) if isinstance(r, dict) else None
                             for r in rois]
                    return {"ok": True, "n": len(stats), "width": W, "height": H, "stats": stats}
                return {"ok": False, "error": f"Timeout waiting for {topic} (is the depth camera publishing?)"}
            finally:
                ws_manager.send({"op": "unsubscribe", "topic": topic})
