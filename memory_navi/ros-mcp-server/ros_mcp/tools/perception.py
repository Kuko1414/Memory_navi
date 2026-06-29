"""Perception summary tools for ROS MCP.

把高维传感器原始消息（如 LaserScan ~400 ranges）在 MCP 层就**全精度计算成决策标量**，
只把少量标量回灌给 LLM，从根上避免对话历史被原始数组撑爆（上下文膨胀根因）。

设计原则（见 CLAUDE.md / 记忆 context-bloat-supervisor-design）：
- 全精度计算、少 token 输出 —— 不降精度、不死板裁剪。
- 过滤逻辑与 safety_node 保持一致（剔 nan / inf / < range_min）。
- 单机项目：命名空间从环境变量 ROS_NAMESPACE 读，默认 agent0（不让 LLM 传参）。
"""

import math
import os
import time

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
