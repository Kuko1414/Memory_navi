"""High-level robot action/perception tools for ROS MCP.

机器人专属"意图层"，薄薄坐在通用 ROS 桥（topics.py/actions.py）之上：把命名空间、话题名、
消息构造全部写死/隐藏，只给 LLM 暴露少量高层意图（move/turn_left_deg/turn_right_deg/stop/look/get_pose）。
内部委托给同一套 advertise/publish/subscribe 机制，不重复逻辑、不改动上游通用工具。

设计原则（见 CLAUDE.md / 记忆 sim-motion-under-actuation、context-bloat-supervisor-design）：
- 单机项目：命名空间从环境变量 ROS_NAMESPACE 读，默认 agent0（不让 LLM 传参）。
- **闭环运动**：mecanum 麦轮打滑 + 仿真 <1× 实时使开环按墙钟计时严重欠走，故 move/turn 改成
  边动边读**真值位姿**（gps 真值 x,y + imu 真值 yaw），到目标位移/航向即停 → "走多远=多远、转多少=多少"。
- 安全由独立的 safety_node 兜底；本层运动循环额外读 /dev/shm/agent_safety_<ns>.json，tripped 立即停。
- 与 perception.py 一致：新文件 additive，不 fork vendored 上游。
"""

import json
import math
import os
import time

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent
from PIL import Image as PILImage

from ros_mcp.tools.images import _encode_image_to_imagecontent
from ros_mcp.utils.websocket import WebSocketManager, parse_input

#: 默认命名空间（单机项目；多机时改环境变量）。
DEFAULT_NAMESPACE = os.environ.get("ROS_NAMESPACE", "agent0")

#: 默认运动参数。
DEFAULT_LIN_SPEED = 0.6      # m/s（提速：闭环读真值位姿故距离仍准，只缩短时间；mecanum 欠驱动~0.19×）
DEFAULT_ANG_SPEED = 1.5      # rad/s（提速：闭环读 imu 故转角仍准，只缩短时间；欠驱动~⅓）
#: 位姿 receive 超时：必须**大于**位姿话题周期（gps 10Hz=0.1s, imu 20Hz=0.05s），否则
#: ws_manager.receive 超时会 close() 连接、丢掉订阅 → 闭环读不到位姿。0.5s 给足余量。
POSE_RECV_TIMEOUT = 0.5

#: 抓图存盘路径（与 topics.py::subscribe_once 一致，相对 MCP server cwd）。
_IMAGE_PATH = "./camera/received_image.jpeg"


# ---------------------------------------------------------------------------
# 纯函数（可脱离 ROS 单测）
# ---------------------------------------------------------------------------
def _twist(vx: float = 0.0, wz: float = 0.0) -> dict:
    """构造 geometry_msgs/msg/Twist 的 rosbridge dict。"""
    return {
        "linear": {"x": float(vx), "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": float(wz)},
    }


def quat_to_yaw_deg(x: float, y: float, z: float, w: float) -> float:
    """四元数 → yaw（度，[-180,180]）。"""
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_deg_to_quat(yaw_deg: float) -> dict:
    """yaw（度）→ 仅绕 z 的单位四元数。"""
    half = math.radians(yaw_deg) / 2.0
    return {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)}


def shortest_angle_diff(a_deg: float, b_deg: float) -> float:
    """a-b 的最短带符号角差，归一化到 [-180,180]（处理 ±180 绕回）。"""
    return ((a_deg - b_deg + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# 注册（@mcp.tool 委托给 ws_manager）
# ---------------------------------------------------------------------------
def register_agent_action_tools(mcp: FastMCP, ws_manager: WebSocketManager) -> None:
    """注册高层动作/感知工具到 FastMCP 实例（仿 register_perception_tools）。"""
    ns = DEFAULT_NAMESPACE
    cmd_vel = f"/{ns}/cmd_vel"
    twist_type = "geometry_msgs/msg/Twist"
    gps_topic = f"/{ns}/gps"
    imu_topic = f"/{ns}/imu"
    shm_path = f"/dev/shm/agent_safety_{ns}.json"

    def _safety_tripped() -> bool:
        """读 safety_node 写的 shm，tripped=True 则需立即停。"""
        try:
            with open(shm_path) as f:
                return bool(json.load(f).get("tripped"))
        except Exception:  # noqa: BLE001
            return False

    def _closed_loop(subs, twist: dict, is_done, timeout: float) -> tuple:
        """闭环：advertise cmd_vel + subscribe subs，循环发 twist 直到 is_done(latest)/safety/超时；末尾零速。

        subs: [(topic, msg_type), ...]，其最新消息累积进 latest 供 is_done 读。
        返回 (status, latest, elapsed_s)。status ∈ done|safety_stop|timeout|error_*。
        """
        latest = {}
        t0 = time.time()

        def _pump():
            """读一条位姿消息更新 latest（超时返回 False，不更新）。"""
            r = ws_manager.receive(timeout=POSE_RECV_TIMEOUT)
            if not r:
                return False
            md, _ = parse_input(r, False)
            if md and md.get("op") == "publish" and md.get("topic"):
                latest[md["topic"]] = md.get("msg", {})
                return True
            return False

        with ws_manager:
            if ws_manager.send({"op": "advertise", "topic": cmd_vel, "type": twist_type}):
                return ("error_advertise", latest, 0.0)
            for topic, mtype in subs:
                ws_manager.send({"op": "subscribe", "topic": topic, "type": mtype,
                                 "queue_length": 1, "throttle_rate": 0})
            try:
                # 等首批位姿读数（最多 ~3s）
                warmup = time.time() + 3.0
                while time.time() < warmup and not all(t in latest for t, _ in subs):
                    _pump()
                if not all(t in latest for t, _ in subs):
                    return ("error_no_pose", latest, round(time.time() - t0, 2))

                status = "timeout"
                deadline = t0 + timeout
                while time.time() < deadline:
                    if _safety_tripped():
                        status = "safety_stop"
                        break
                    if is_done(latest):
                        status = "done"
                        break
                    ws_manager.send({"op": "publish", "topic": cmd_vel, "msg": twist})
                    _pump()  # 读最新位姿（周期内必有消息，故不会超时 close 连接）
                ws_manager.send({"op": "publish", "topic": cmd_vel, "msg": _twist()})
            finally:
                for topic, _ in subs:
                    ws_manager.send({"op": "unsubscribe", "topic": topic})
                ws_manager.send({"op": "unadvertise", "topic": cmd_vel})
        return (status, latest, round(time.time() - t0, 2))

    @mcp.tool(
        description=(
            "Drive the robot straight by distance_m meters (closed-loop on ground-truth pose; "
            "+forward, -backward). Stops exactly at the target distance, or earlier if the safety "
            "node trips on a near obstacle. Example: move(0.5) goes 0.5m forward."
        ),
    )
    def move(distance_m: float, speed: float = DEFAULT_LIN_SPEED) -> dict:
        """闭环直行：读 gps 真值位移，到 distance_m 即停。"""
        try:
            distance_m = float(distance_m)
            speed = float(speed)
            if speed <= 0:
                return {"error": "speed must be > 0"}
        except (ValueError, TypeError):
            return {"error": "distance_m, speed must be numbers"}
        if distance_m == 0:
            return {"error": "distance_m is 0: nothing to do"}

        target = abs(distance_m)
        vx = speed if distance_m > 0 else -speed
        start = {}
        prog = {"traveled": 0.0}

        def is_done(latest):
            g = latest[gps_topic]
            p = g.get("point", g)
            x, y = float(p.get("x", 0.0)), float(p.get("y", 0.0))
            if "x" not in start:
                start["x"], start["y"] = x, y
            prog["traveled"] = math.hypot(x - start["x"], y - start["y"])
            return prog["traveled"] >= target

        timeout = (target / speed) * 6.0 + 6.0
        status, _, elapsed = _closed_loop(
            [(gps_topic, "geometry_msgs/msg/PointStamped")], _twist(vx, 0.0), is_done, timeout)
        return {
            "ok": status in ("done", "safety_stop"),
            "status": status,
            "requested_m": round(distance_m, 3),
            "traveled_m": round(prog["traveled"], 3),
            "elapsed_s": elapsed,
        }

    def _turn(target_deg: float, sign: int) -> dict:
        """闭环转向：读 imu yaw 累积转角，到 target_deg 即停。sign=+1 左/CCW，-1 右/CW。"""
        try:
            target_deg = float(target_deg)
        except (ValueError, TypeError):
            return {"error": "degrees must be a number"}
        if target_deg <= 0:
            return {"error": "degrees must be > 0"}

        st = {"prev": None, "acc": 0.0}

        def is_done(latest):
            q = latest[imu_topic].get("orientation", {})
            yaw = quat_to_yaw_deg(
                float(q.get("x", 0.0)), float(q.get("y", 0.0)),
                float(q.get("z", 0.0)), float(q.get("w", 1.0)))
            if st["prev"] is not None:
                st["acc"] += abs(shortest_angle_diff(yaw, st["prev"]))
            st["prev"] = yaw
            return st["acc"] >= target_deg

        wz = sign * DEFAULT_ANG_SPEED
        timeout = (math.radians(target_deg) / DEFAULT_ANG_SPEED) * 6.0 + 6.0
        status, _, elapsed = _closed_loop(
            [(imu_topic, "sensor_msgs/msg/Imu")], _twist(0.0, wz), is_done, timeout)
        return {
            "ok": status in ("done", "safety_stop"),
            "status": status,
            "requested_deg": round(target_deg, 1),
            "turned_deg": round(st["acc"], 1),
            "direction": "left" if sign > 0 else "right",
            "elapsed_s": elapsed,
        }

    @mcp.tool(
        description=(
            "Turn the robot LEFT (counter-clockwise) by `degrees` (closed-loop on IMU heading). "
            "Stops at the target angle. Example: turn_left_deg(90) turns 90 degrees left."
        ),
    )
    def turn_left_deg(degrees: float) -> dict:
        """左转（CCW）degrees 度。"""
        return _turn(degrees, +1)

    @mcp.tool(
        description=(
            "Turn the robot RIGHT (clockwise) by `degrees` (closed-loop on IMU heading). "
            "Stops at the target angle. Example: turn_right_deg(90) turns 90 degrees right."
        ),
    )
    def turn_right_deg(degrees: float) -> dict:
        """右转（CW）degrees 度。"""
        return _turn(degrees, -1)

    @mcp.tool(
        description="Immediately stop the robot (publish a single zero velocity to cmd_vel).",
    )
    def stop() -> dict:
        """急停：发单条零 Twist。"""
        with ws_manager:
            if ws_manager.send({"op": "advertise", "topic": cmd_vel, "type": twist_type}):
                return {"error": "failed to advertise cmd_vel"}
            try:
                ws_manager.send({"op": "publish", "topic": cmd_vel, "msg": _twist()})
            finally:
                ws_manager.send({"op": "unadvertise", "topic": cmd_vel})
        return {"ok": True, "topic": cmd_vel}

    @mcp.tool(
        description=(
            "Capture one frame from the robot's camera and return it as an image so you can "
            "see what's in front of the robot. Use this to observe the scene."
        ),
    )
    def look(timeout: float = 3.0):
        """抓一帧压缩相机图，返回 ImageContent。"""
        try:
            timeout = float(timeout)
            if timeout <= 0:
                return {"error": "timeout must be > 0"}
        except (ValueError, TypeError):
            return {"error": "timeout must be a number"}

        topic = f"/{ns}/camera/image_color/compressed"
        msg_type = "sensor_msgs/msg/CompressedImage"
        subscribe_msg = {
            "op": "subscribe", "topic": topic, "type": msg_type,
            "queue_length": 1, "throttle_rate": 0,
        }
        with ws_manager:
            if ws_manager.send(subscribe_msg):
                return {"error": f"Failed to subscribe: {topic}"}
            end = time.time() + timeout
            try:
                while time.time() < end:
                    response = ws_manager.receive(timeout=0.5)
                    if response is None:
                        continue
                    msg_data, was_image = parse_input(response, True)
                    if not msg_data:
                        continue
                    if msg_data.get("op") == "status" and msg_data.get("level") == "error":
                        return {"error": f"Rosbridge error: {msg_data.get('msg', 'Unknown error')}"}
                    if msg_data.get("op") == "publish" and msg_data.get("topic") == topic:
                        if was_image and os.path.exists(_IMAGE_PATH):
                            img = PILImage.open(_IMAGE_PATH)
                            return ToolResult(content=[
                                _encode_image_to_imagecontent(img),
                                TextContent(type="text", text=f"Camera frame from {topic}."),
                            ])
                        return {"error": "frame received but image not found on disk"}
                return {"error": f"Timeout waiting for a frame from {topic}"}
            finally:
                ws_manager.send({"op": "unsubscribe", "topic": topic})

    @mcp.tool(
        description=(
            "Get the robot's current pose in the map frame. Returns {x, y, yaw_deg} "
            "(yaw 0=facing +x/east, +=counter-clockwise)."
        ),
    )
    def get_pose(timeout: float = 2.0) -> dict:
        """订一次 gps(x,y) + imu(yaw) 返回当前位姿。"""
        try:
            timeout = float(timeout)
            if timeout <= 0:
                return {"error": "timeout must be > 0"}
        except (ValueError, TypeError):
            return {"error": "timeout must be a number"}

        def _grab(topic: str, msg_type: str):
            sub = {"op": "subscribe", "topic": topic, "type": msg_type,
                   "queue_length": 1, "throttle_rate": 0}
            with ws_manager:
                if ws_manager.send(sub):
                    return None
                end = time.time() + timeout
                try:
                    while time.time() < end:
                        response = ws_manager.receive(timeout=0.5)
                        if response is None:
                            continue
                        msg_data, _ = parse_input(response, False)
                        if not msg_data:
                            continue
                        if msg_data.get("op") == "publish" and msg_data.get("topic") == topic:
                            return msg_data.get("msg", {})
                    return None
                finally:
                    ws_manager.send({"op": "unsubscribe", "topic": topic})

        gps = _grab(gps_topic, "geometry_msgs/msg/PointStamped")
        imu = _grab(imu_topic, "sensor_msgs/msg/Imu")
        if gps is None or imu is None:
            return {"error": "timeout reading gps/imu"}
        point = gps.get("point", gps)
        q = imu.get("orientation", {})
        yaw = quat_to_yaw_deg(
            float(q.get("x", 0.0)), float(q.get("y", 0.0)),
            float(q.get("z", 0.0)), float(q.get("w", 1.0)),
        )
        return {
            "x": round(float(point.get("x", 0.0)), 3),
            "y": round(float(point.get("y", 0.0)), 3),
            "yaw_deg": round(yaw, 1),
        }

    @mcp.tool(
        description=(
            "Send a navigation goal to Nav2 (publishes a PoseStamped to the goal_pose topic). "
            "REQUIRES Nav2 to be running; if it is not up yet, use move()/turn_*() for short hops."
        ),
    )
    def navigate_to(x: float, y: float, yaw_deg: float = 0.0) -> dict:
        """薄封装 Nav2 简单目标：发 PoseStamped 到 /<ns>/goal_pose。"""
        try:
            x = float(x)
            y = float(y)
            yaw_deg = float(yaw_deg)
        except (ValueError, TypeError):
            return {"error": "x, y, yaw_deg must be numbers"}
        topic = f"/{ns}/goal_pose"
        msg_type = "geometry_msgs/msg/PoseStamped"
        msg = {
            "header": {"frame_id": "map"},
            "pose": {
                "position": {"x": x, "y": y, "z": 0.0},
                "orientation": yaw_deg_to_quat(yaw_deg),
            },
        }
        with ws_manager:
            if ws_manager.send({"op": "advertise", "topic": topic, "type": msg_type}):
                return {"error": "failed to advertise goal_pose"}
            try:
                ws_manager.send({"op": "publish", "topic": topic, "msg": msg})
            finally:
                ws_manager.send({"op": "unadvertise", "topic": topic})
        return {"ok": True, "topic": topic, "goal": {"x": x, "y": y, "yaw_deg": yaw_deg}}
