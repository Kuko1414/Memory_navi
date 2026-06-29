#!/usr/bin/env python3
"""
独立测试脚本：验证 odom 位置和 yaw 方向的关系。

核心逻辑：
  订阅 {{ODOM_TOPIC}}，实时打印机器人的 (x, y, yaw) 信息，
  并在每次更新时计算 yaw 对应的"前方"方向向量，
  帮助判断 odom 坐标系的 x/y 方向与实际空间的对应关系。

使用方法：
  1. 确保机器人底盘和 odom 发布者已启动
  2. 运行: python3 scripts/test_odom_yaw.py
  3. 手动推动/遥控小车，观察输出：
     - 向前推：看 x 和 y 哪个变化
     - 向左转：看 yaw 是增大还是减小
  4. 按 Ctrl+C 退出，会打印汇总信息

输出说明：
  - yaw=0° 时，cos(yaw)=1, sin(yaw)=0 → 前方向量 = (+1, 0)，即 odom x 正方向
  - yaw=90° 时，cos(yaw)=0, sin(yaw)=1 → 前方向量 = (0, +1)，即 odom y 正方向
  - 如果实际前方与向量不一致，说明 yaw 或 odom 坐标系有偏差

注意: 使用前请确认 {{ODOM_TOPIC}} 话题名与实际 odom 发布者匹配。
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math
import time


class OdomYawTester(Node):
    def __init__(self):
        super().__init__('odom_yaw_tester')

        self.sub_odom = self.create_subscription(
            Odometry, '{{ODOM_TOPIC}}', self.odom_callback, 10)

        self.history = []  # 记录 (time, x, y, yaw)
        self.start_time = time.monotonic()
        self._first_msg = True
        self._last_print_time = 0.0

        self.get_logger().info("=" * 60)
        self.get_logger().info("  Odom & Yaw 测试工具")
        self.get_logger().info("=" * 60)
        self.get_logger().info("请手动推动/遥控小车，观察以下信息：")
        self.get_logger().info("  1. 向前推小车 → 看 x 和 y 哪个变化更大")
        self.get_logger().info("  2. 向左转小车 → 看 yaw 是增大还是减小")
        self.get_logger().info("  3. 按 Ctrl+C 退出查看汇总")
        self.get_logger().info("=" * 60)

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        # 提取 yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        now = time.monotonic()
        elapsed = now - self.start_time

        self.history.append((elapsed, p.x, p.y, yaw))

        # 每 0.5 秒打印一次
        if now - self._last_print_time >= 0.5 or self._first_msg:
            self._first_msg = False
            self._last_print_time = now

            yaw_deg = math.degrees(yaw)

            # 计算 yaw 对应的前方向量
            fwd_x = math.cos(yaw)
            fwd_y = math.sin(yaw)

            # 计算位移（相对于上一次打印）
            dx_str = ""
            dy_str = ""
            if len(self.history) > 10:
                old = self.history[-10]
                dx = p.x - old[1]
                dy = p.y - old[2]
                dx_str = f"  Δx={dx:+.4f}  Δy={dy:+.4f}"

            self.get_logger().info(
                f"[{elapsed:6.1f}s] "
                f"pos=({p.x:+.4f}, {p.y:+.4f})  "
                f"yaw={yaw_deg:+7.2f}°  "
                f"前方向量=({fwd_x:+.3f}, {fwd_y:+.3f})"
                f"{dx_str}"
            )

            # 四元数原始值（用于调试）
            if self._first_msg or (len(self.history) % 20 == 0):
                self.get_logger().info(
                    f"  [四元数] w={q.w:.4f}, x={q.x:.4f}, y={q.y:.4f}, z={q.z:.4f}")

    def print_summary(self):
        """退出时打印汇总信息。"""
        if len(self.history) < 2:
            self.get_logger().info("数据不足，无法汇总。")
            return

        first = self.history[0]
        last = self.history[-1]

        total_dx = last[1] - first[1]
        total_dy = last[2] - first[2]
        total_dist = math.hypot(total_dx, total_dy)

        yaw_first = math.degrees(first[3])
        yaw_last = math.degrees(last[3])
        yaw_change = yaw_last - yaw_first

        # 计算实际移动方向角
        if total_dist > 0.01:
            move_angle = math.degrees(math.atan2(total_dy, total_dx))
        else:
            move_angle = float('nan')

        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info("  汇总")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  起点: ({first[1]:+.4f}, {first[2]:+.4f}), yaw={yaw_first:+.2f}°")
        self.get_logger().info(f"  终点: ({last[1]:+.4f}, {last[2]:+.4f}), yaw={yaw_last:+.2f}°")
        self.get_logger().info(f"  总位移: Δx={total_dx:+.4f}, Δy={total_dy:+.4f}, 距离={total_dist:.4f}m")
        self.get_logger().info(f"  实际移动方向角: {move_angle:+.2f}°")
        self.get_logger().info(f"  yaw 变化: {yaw_first:+.2f}° → {yaw_last:+.2f}° (Δ={yaw_change:+.2f}°)")
        self.get_logger().info("")
        self.get_logger().info("  【判断方法】")
        self.get_logger().info("  如果你向前推了小车：")
        self.get_logger().info(f"    - 实际移动方向角 = {move_angle:+.2f}°")
        self.get_logger().info(f"    - yaw 指示的前方 = {yaw_first:+.2f}°")
        if not math.isnan(move_angle):
            diff = move_angle - yaw_first
            # 归一化到 [-180, 180]
            diff = (diff + 180) % 360 - 180
            self.get_logger().info(f"    - 差值 = {diff:+.2f}°")
            if abs(diff) < 30:
                self.get_logger().info("    → ✅ yaw 与实际前方基本一致")
            elif abs(diff - 180) < 30 or abs(diff + 180) < 30:
                self.get_logger().info("    → ❌ yaw 与实际前方相差 ~180°（方向反了！）")
            elif abs(diff - 90) < 30:
                self.get_logger().info("    → ❌ yaw 与实际前方相差 ~+90°（逆时针偏了 90°）")
            elif abs(diff + 90) < 30:
                self.get_logger().info("    → ❌ yaw 与实际前方相差 ~-90°（顺时针偏了 90°）")
            else:
                self.get_logger().info(f"    → ⚠️ yaw 与实际前方有 {diff:+.1f}° 偏差")
        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = OdomYawTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
