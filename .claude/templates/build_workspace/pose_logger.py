#!/usr/bin/env python3
"""
位姿记录器：每 0.5s 记录小车位姿，Ctrl+C 退出时保存为 log 文件。

核心逻辑：
  订阅 {{ODOM_TOPIC}}，每 0.5s 采样一次 (x, y, yaw)，
  退出时将完整轨迹写入 data/log/pose_taskN.log。

使用方法：
  python3 scripts/pose_logger.py              # 自动使用下一个可用编号
  python3 scripts/pose_logger.py --task 3     # 指定 task 编号
  # 小车运动过程中持续记录
  # Ctrl+C 退出后自动保存日志

注意: 使用前请确认 {{ODOM_TOPIC}} 话题名与实际 odom 发布者匹配。
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math
import time
import os
import argparse
import glob
from datetime import datetime


LOG_DIR = '{{DATA_DIR}}/log'


def get_next_task_id():
    """自动找到下一个可用的 task 编号。"""
    existing = glob.glob(os.path.join(LOG_DIR, 'pose_task*.log'))
    if not existing:
        return 0
    ids = []
    for f in existing:
        basename = os.path.basename(f)  # pose_task3.log
        try:
            num = int(basename.replace('pose_task', '').replace('.log', ''))
            ids.append(num)
        except ValueError:
            continue
    return max(ids) + 1 if ids else 0


class PoseLogger(Node):
    def __init__(self, task_id):
        super().__init__('pose_logger')

        self.task_id = task_id

        self.sub_odom = self.create_subscription(
            Odometry, '{{ODOM_TOPIC}}', self.odom_callback, 10)

        self.records = []           # [(elapsed, x, y, yaw_deg)]
        self.start_time = time.monotonic()
        self._last_record_time = 0.0
        self._record_interval = 0.5  # 采样间隔（秒）

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        os.makedirs(LOG_DIR, exist_ok=True)

        self.get_logger().info("=" * 50)
        self.get_logger().info(f"  位姿记录器已启动 — Task {self.task_id}")
        self.get_logger().info(f"  采样间隔: {self._record_interval}s")
        self.get_logger().info(f"  保存路径: {LOG_DIR}/pose_task{self.task_id}.log")
        self.get_logger().info("  按 Ctrl+C 退出并保存日志")
        self.get_logger().info("=" * 50)

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.current_x = p.x
        self.current_y = p.y
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        now = time.monotonic()
        if now - self._last_record_time >= self._record_interval:
            self._last_record_time = now
            elapsed = now - self.start_time
            yaw_deg = math.degrees(self.current_yaw)

            self.records.append((elapsed, self.current_x, self.current_y, yaw_deg))

            # 每 5 秒打印一次状态
            if len(self.records) % 10 == 1:
                self.get_logger().info(
                    f"[{elapsed:6.1f}s] pos=({self.current_x:+.4f}, {self.current_y:+.4f}) "
                    f"yaw={yaw_deg:+.1f}°  已记录 {len(self.records)} 个点")

    def save_log(self):
        """保存位姿记录到 log 文件。"""
        if len(self.records) == 0:
            self.get_logger().info("无记录数据，跳过保存。")
            return

        log_path = os.path.join(LOG_DIR, f'pose_task{self.task_id}.log')

        first = self.records[0]
        last = self.records[-1]
        total_dx = last[1] - first[1]
        total_dy = last[2] - first[2]
        total_dist = math.hypot(total_dx, total_dy)
        duration = last[0] - first[0]

        lines = []
        lines.append(f"=== 位姿记录日志 (Task {self.task_id}) ===")
        lines.append(f"时间: {datetime.now().isoformat()}")
        lines.append(f"记录点数: {len(self.records)}")
        lines.append(f"采样间隔: {self._record_interval}s")
        lines.append(f"持续时间: {duration:.1f}s")
        lines.append(f"")
        lines.append(f"--- 汇总 ---")
        lines.append(f"起点: ({first[1]:+.4f}, {first[2]:+.4f}), yaw={first[3]:+.1f}°")
        lines.append(f"终点: ({last[1]:+.4f}, {last[2]:+.4f}), yaw={last[3]:+.1f}°")
        lines.append(f"总位移: Δx={total_dx:+.4f}, Δy={total_dy:+.4f}, 距离={total_dist:.4f}m")
        if total_dist > 0.01:
            move_angle = math.degrees(math.atan2(total_dy, total_dx))
            lines.append(f"实际移动方向角: {move_angle:+.1f}°")
        lines.append(f"")
        lines.append(f"--- 详细轨迹 ({len(self.records)} 个点) ---")
        lines.append(f"{'时间(s)':>8}  {'x':>10}  {'y':>10}  {'yaw(°)':>8}")
        for elapsed, x, y, yaw_deg in self.records:
            lines.append(f"{elapsed:8.1f}  {x:+10.4f}  {y:+10.4f}  {yaw_deg:+8.1f}")
        lines.append(f"")
        lines.append(f"=== END ===")

        with open(log_path, 'w') as f:
            f.write('\n'.join(lines))

        self.get_logger().info(f"")
        self.get_logger().info(f"📝 位姿日志已保存: {log_path}")
        self.get_logger().info(f"   共 {len(self.records)} 个点, 持续 {duration:.1f}s, 移动 {total_dist:.2f}m")


def main(args=None):
    parser = argparse.ArgumentParser(description='位姿记录器')
    parser.add_argument('--task', type=int, default=None,
                        help='指定 task 编号（默认自动递增）')
    parsed_args, ros_args = parser.parse_known_args()

    task_id = parsed_args.task if parsed_args.task is not None else get_next_task_id()

    rclpy.init(args=ros_args)
    node = PoseLogger(task_id)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_log()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
