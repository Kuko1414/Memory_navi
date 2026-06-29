#!/usr/bin/env python3
"""底层安全节点（独立进程，不依赖 vLLM / MCP / agent_core）。

两职责：
1) 传感器存活性校验（高频、短窗，快速发现掉线）：以 ~50Hz 校验一组关键 topic 的
   最近消息时戳；任一超其短 staleness 阈值 → sensor fault。用小消息（scan/imu/gps/
   camera_info）做存活信号，避免订阅大图。
2) 安全距离硬停：订 /<ns>/scan，最小有效距离 < stop_distance(默认0.15m) → 持续发零
   Twist 到 /<ns>/cmd_vel（绕过 MCP/LLM），目标 <10ms 反应。

状态输出：/<ns>/safety/status (std_msgs/String JSON) + /dev/shm/agent_safety_<ns>.json
（供未来 Supervisor 的 safety_iface 读）。rule0 永远在代码里，绝不交给模型。

运行（ROS2 系统 py3.10，先剔除 miniconda）：
  source /opt/ros/humble/setup.bash
  python3 memory_navi/safety_node.py --ros-args -p namespace:=agent0
"""
import json

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Imu, LaserScan
from std_msgs.msg import String


class SafetyNode(Node):
    def __init__(self):
        super().__init__("safety_node")
        self.ns = self.declare_parameter("namespace", "agent0").value
        self.stop_dist = self.declare_parameter("stop_distance", 0.15).value
        # 迟滞：trip 后需退到 stop_dist+margin 才解除，避免阈值附近 trip/untrip 抖动
        self.release_dist = self.stop_dist + self.declare_parameter("release_margin", 0.05).value
        check_hz = self.declare_parameter("check_hz", 50.0).value
        ns = self.ns

        # 关键传感器存活性：topic -> 短 staleness 阈值(s)（取各自发布周期的小倍数）
        self.watch = {
            f"/{ns}/scan": 0.4,                       # lidar @10Hz
            f"/{ns}/imu": 0.3,                        # @20Hz
            f"/{ns}/gps": 0.4,                        # @10Hz
            f"/{ns}/camera/camera_info": 0.3,         # color 存活（小）
            f"/{ns}/camera/depth/camera_info": 0.6,   # depth 存活（@5Hz，小）
        }
        self.last = {t: 0.0 for t in self.watch}

        q = qos_profile_sensor_data
        self.create_subscription(LaserScan, f"/{ns}/scan", self._scan_cb, q)
        self.create_subscription(Imu, f"/{ns}/imu", lambda m: self._stamp(f"/{ns}/imu"), q)
        self.create_subscription(PointStamped, f"/{ns}/gps", lambda m: self._stamp(f"/{ns}/gps"), q)
        self.create_subscription(CameraInfo, f"/{ns}/camera/camera_info",
                                 lambda m: self._stamp(f"/{ns}/camera/camera_info"), q)
        self.create_subscription(CameraInfo, f"/{ns}/camera/depth/camera_info",
                                 lambda m: self._stamp(f"/{ns}/camera/depth/camera_info"), q)

        self.cmd_pub = self.create_publisher(Twist, f"/{ns}/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, f"/{ns}/safety/status", 10)
        self.shm_path = f"/dev/shm/agent_safety_{ns}.json"

        self.lidar_min = float("inf")
        self.tripped = False
        self._dist_trip = False   # 距离触发的迟滞锁存
        self._warned = False
        self.create_timer(1.0 / check_hz, self._tick)
        self.get_logger().info(
            f"safety_node ns={ns} stop<{self.stop_dist}m check@{check_hz}Hz watch={list(self.watch)}"
        )

    # ---- helpers ----
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _stamp(self, topic: str):
        self.last[topic] = self._now()

    def _scan_cb(self, msg: LaserScan):
        self._stamp(f"/{self.ns}/scan")
        rmin = max(msg.range_min, 0.01)
        mn = float("inf")
        for r in msg.ranges:
            if r != r:                       # nan
                continue
            if r < rmin or r == float("inf"):
                continue
            if r < mn:
                mn = r
        self.lidar_min = mn

    # ---- main safety loop ----
    def _tick(self):
        now = self._now()
        stale = [t for t, win in self.watch.items() if (now - self.last[t]) > win]
        sensors_ok = not stale
        # 距离迟滞：< stop_dist 触发；> release_dist 才解除；之间保持上次状态
        if self.lidar_min < self.stop_dist:
            self._dist_trip = True
        elif self.lidar_min > self.release_dist:
            self._dist_trip = False
        near = self._dist_trip
        tripped = near or (not sensors_ok)

        if tripped:
            self.cmd_pub.publish(Twist())   # 零速硬停，持续覆盖任何运动指令
        self.tripped = tripped

        reason = "near_obstacle" if near else ("sensor_fault" if not sensors_ok else "ok")
        status = {
            "tripped": tripped,
            "lidar_min_m": None if self.lidar_min == float("inf") else round(self.lidar_min, 3),
            "sensors_ok": sensors_ok,
            "stale": stale,
            "reason": reason,
            "ts": round(now, 3),
        }
        self.status_pub.publish(String(data=json.dumps(status, ensure_ascii=False)))
        try:
            with open(self.shm_path, "w") as f:
                json.dump(status, f)
        except Exception:  # noqa: BLE001
            pass

        if tripped and not self._warned:
            self.get_logger().warn(f"SAFETY TRIP: {reason} {status}")
            self._warned = True
        elif not tripped:
            self._warned = False


def main():
    rclpy.init()
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
