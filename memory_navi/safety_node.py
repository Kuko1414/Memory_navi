#!/usr/bin/env python3
"""底层安全节点（独立进程，不依赖 vLLM / MCP / agent_core）。

两职责：
1) 传感器存活性校验（高频、短窗，快速发现掉线）：以 ~50Hz 校验一组关键 topic 的
   最近消息时戳；任一超其短 staleness 阈值 → sensor fault。用小消息（scan/imu/gps/
   camera_info）做存活信号，避免订阅大图。
2) **有向安全滤波（twist_mux 模式，本节点是 /<ns>/cmd_vel 的唯一发布者）**：
   导航层(agent_actions)把意图发到 /<ns>/cmd_vel_nav；本节点订阅它，按 scan 的**方向性**
   最近障碍过滤后中继到 /<ns>/cmd_vel（Webots 机器人订阅端）：
   - 前进(linear.x>0) 且 front_min<stop → 线速置0；后退(x<0) 且 rear_min<stop → 线速置0；
   - **角速(旋转)永远放行**、背离障碍的平移放行 → 机器人能自己转身/后退脱离死区（旧版全零会钉死）。
   - 传感器失活 → 发零（急停）；cmd_vel_nav 过期(>cmd_stale_s) → 发零（导航空闲即停）。
   目标 <10ms 反应；本节点挂掉则无 cmd_vel，机器人经 Webots 驱动 1.5s 超时归零（互锁）。

状态输出：/<ns>/safety/status (std_msgs/String JSON) + /dev/shm/agent_safety_<ns>.json（原子写）
（含 blocked_forward/blocked_rear/front_min_m/rear_min_m，供 MCP 运动层有向判停 + Supervisor 读）。
rule0 永远在代码里，绝不交给模型。

运行（ROS2 系统 py3.10，先剔除 miniconda）：
  source /opt/ros/humble/setup.bash
  python3 memory_navi/safety_node.py --ros-args -p namespace:=agent0
"""
import json
import math
import os

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
        # 有向滤波：前/后向锥半角；cmd_vel_nav 过期阈值（导航空闲即停）
        self.front_cone = float(self.declare_parameter("front_cone_deg", 45.0).value)
        self.rear_cone = float(self.declare_parameter("rear_cone_deg", 45.0).value)
        self.cmd_stale_s = float(self.declare_parameter("cmd_stale_s", 0.3).value)
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

        # 导航意图入口（唯一由 agent_actions 发布）；本节点过滤后中继到 cmd_vel
        self.create_subscription(Twist, f"/{ns}/cmd_vel_nav", self._cmd_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, f"/{ns}/cmd_vel", 10)   # 唯一 cmd_vel 发布者
        self.status_pub = self.create_publisher(String, f"/{ns}/safety/status", 10)
        self.shm_path = f"/dev/shm/agent_safety_{ns}.json"

        self.lidar_min = float("inf")
        self.front_min = float("inf")     # 前向锥内最近障碍
        self.rear_min = float("inf")      # 后向锥内最近障碍
        self._empty_scans = 0             # 连续空/全无效扫描帧计数（Risk-B 软兜底）
        self._last_cmd = Twist()          # 最近一次导航意图
        self._last_cmd_t = 0.0
        self.tripped = False
        self._dist_trip = False   # 距离触发的迟滞锁存（全向，供 tripped 兼容字段）
        self._fwd_trip = False    # 前向阻挡迟滞锁存
        self._rear_trip = False   # 后向阻挡迟滞锁存
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

    @staticmethod
    def _norm_deg(a: float) -> float:
        return ((a + 180.0) % 360.0) - 180.0

    def _cmd_cb(self, msg: Twist):
        """存最近导航意图 + 时戳（_tick 有向过滤后中继到 cmd_vel）。"""
        self._last_cmd = msg
        self._last_cmd_t = self._now()

    def _scan_cb(self, msg: LaserScan):
        self._stamp(f"/{self.ns}/scan")
        rmin = max(msg.range_min, 0.01)
        rmax = msg.range_max if getattr(msg, "range_max", 0.0) and msg.range_max > 0 else float("inf")
        mn = front = rear = float("inf")
        n_valid = 0
        for i, r in enumerate(msg.ranges):
            if r != r:                       # nan
                continue
            if r < rmin or r == float("inf") or r > rmax:
                continue
            n_valid += 1
            if r < mn:
                mn = r
            deg = self._norm_deg(math.degrees(msg.angle_min + i * msg.angle_increment))
            if abs(deg) <= self.front_cone and r < front:      # 前向锥（0°=正前）
                front = r
            if abs(self._norm_deg(deg - 180.0)) <= self.rear_cone and r < rear:  # 后向锥（±180°）
                rear = r
        self.lidar_min = mn
        self.front_min = front
        self.rear_min = rear
        # Risk-B：连续多帧无任何有效点 → 扫描异常（_tick 折进 sensor fault，保守停）
        self._empty_scans = 0 if n_valid > 0 else self._empty_scans + 1

    def _hyst(self, value: float, latched: bool) -> bool:
        """迟滞：<stop_dist 触发，>release_dist 才解除，之间保持。"""
        if value < self.stop_dist:
            return True
        if value > self.release_dist:
            return False
        return latched

    # ---- main safety loop（本节点是 /<ns>/cmd_vel 唯一发布者，每 tick 都发）----
    def _tick(self):
        now = self._now()
        stale = [t for t, win in self.watch.items() if (now - self.last[t]) > win]
        scan_bad = self._empty_scans >= 5          # 连续空扫描 → 当传感器异常（保守停）
        sensors_ok = (not stale) and (not scan_bad)

        # 方向性 + 全向（后者仅供 tripped 兼容字段）迟滞锁存
        self._fwd_trip = self._hyst(self.front_min, self._fwd_trip)
        self._rear_trip = self._hyst(self.rear_min, self._rear_trip)
        self._dist_trip = self._hyst(self.lidar_min, self._dist_trip)

        # —— 输出 Twist：中继导航意图，只否决"朝近障碍的平移"，角速永远放行 ——
        out = Twist()
        cmd_fresh = (now - self._last_cmd_t) <= self.cmd_stale_s
        if sensors_ok and cmd_fresh:               # 否则保持全零（急停 / 导航空闲即停）
            vx = self._last_cmd.linear.x
            if vx > 0.0 and self._fwd_trip:
                vx = 0.0                           # 否决前进撞近障碍
            elif vx < 0.0 and self._rear_trip:
                vx = 0.0                           # 否决后退撞近障碍
            out.linear.x = vx
            out.angular.z = self._last_cmd.angular.z   # 旋转永远放行（脱困关键）
        self.cmd_pub.publish(out)

        blocked_forward, blocked_rear = self._fwd_trip, self._rear_trip
        tripped = blocked_forward or blocked_rear or (not sensors_ok)
        self.tripped = tripped

        reason = ("sensor_fault" if not sensors_ok
                  else ("near_obstacle" if (blocked_forward or blocked_rear) else "ok"))
        status = {
            "tripped": tripped,
            "lidar_min_m": None if self.lidar_min == float("inf") else round(self.lidar_min, 3),
            "front_min_m": None if self.front_min == float("inf") else round(self.front_min, 3),
            "rear_min_m": None if self.rear_min == float("inf") else round(self.rear_min, 3),
            "blocked_forward": blocked_forward,
            "blocked_rear": blocked_rear,
            "sensors_ok": sensors_ok,
            "stale": stale,
            "reason": reason,
            "ts": round(now, 3),
        }
        self.status_pub.publish(String(data=json.dumps(status, ensure_ascii=False)))
        try:
            tmp = self.shm_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(status, f)
            os.replace(tmp, self.shm_path)          # 原子写，消除 MCP 半读竞态
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
