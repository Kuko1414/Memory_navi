#!/usr/bin/env python3
"""深度摘要中继：raw depth Image (32FC1) -> 每列中位深度小数组 Float32MultiArray。

为什么要中继（同 image_jpeg_relay.py 的理由）：Webots RangeFinder 的深度图是 32FC1
640x480x4 ≈ 1.23MB/帧，和 raw 彩色图一样大，无法在 rosbridge 的 JSON 窗口内可靠传完。
本节点在 ROS 端原生订阅深度图（无 rosbridge 体积问题），把它压成「左→右 N 列的中位有效深度」
这么个极小数组（默认 9 个 float）republish 到 /<ns>/camera/depth/summary，
MCP 的 depth_summary 工具再去订这个小话题（payload 极小，过 rosbridge 稳）。

只取画面【地平线略上方】的行带（默认竖向 25%~55%）求每列中位：相机低(0.21m)且仅上仰~8.6°，
更低的行会大量看到近处地板（距离失真），故取地平线上方聚焦红柜/矮隔断/显示器等物体高度。
无效深度（nan/inf/<min/>max）剔除；整列无效则该列填 -1。

运行（ROS2 / 系统 py3.10，先剔除 miniconda 避免冲突）：
  source /opt/ros/humble/setup.bash
  python3 memory_navi/depth_summary_relay.py /agent0/camera/depth/image /agent0/camera/depth/columns
"""
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


class DepthSummaryRelay(Node):
    def __init__(self, in_topic: str, out_topic: str, cols: int = 9,
                 row_band=(0.15, 0.55), dmin: float = 0.06, dmax: float = 9.5):
        super().__init__("depth_summary_relay")
        self.cols = int(cols)
        self.row_band = row_band
        self.dmin, self.dmax = float(dmin), float(dmax)
        self.pub = self.create_publisher(Float32MultiArray, out_topic, 5)
        self.sub = self.create_subscription(Image, in_topic, self._cb, 5)
        self.n = 0
        self.get_logger().info(
            f"depth summary relay {in_topic} -> {out_topic} ({cols} cols, band {row_band})"
        )

    def _decode(self, msg: Image) -> np.ndarray | None:
        """把 sensor_msgs/Image 解成 float32 米制深度 (h,w)。仅处理 32FC1（Webots RangeFinder）。"""
        enc = (msg.encoding or "").lower()
        if enc not in ("32fc1", "32fc1 "):
            self.get_logger().warn(f"unexpected depth encoding {msg.encoding!r}, expect 32FC1")
            # 仍尝试按 float32 解
        try:
            arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
            return arr.reshape(msg.height, msg.width)
        except (ValueError, TypeError) as e:
            self.get_logger().warn(f"decode failed: {e}")
            return None

    def _cb(self, msg: Image):
        depth = self._decode(msg)
        if depth is None:
            return
        h, w = depth.shape
        r0, r1 = int(h * self.row_band[0]), int(h * self.row_band[1])
        band = depth[r0:r1, :]
        out = []
        edges = np.linspace(0, w, self.cols + 1, dtype=int)
        for c in range(self.cols):
            col = band[:, edges[c]:edges[c + 1]].ravel()
            valid = col[np.isfinite(col) & (col >= self.dmin) & (col <= self.dmax)]
            out.append(float(round(np.median(valid), 3)) if valid.size else -1.0)
        self.pub.publish(Float32MultiArray(data=out))
        self.n += 1
        if self.n % 30 == 1:
            self.get_logger().info(f"relayed {self.n}; cols L->R = {out}")


def main():
    in_topic = sys.argv[1] if len(sys.argv) > 1 else "/agent0/camera/depth/image"
    out_topic = sys.argv[2] if len(sys.argv) > 2 else "/agent0/camera/depth/columns"
    rclpy.init()
    node = DepthSummaryRelay(in_topic, out_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
