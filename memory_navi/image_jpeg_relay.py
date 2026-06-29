#!/usr/bin/env python3
"""JPEG 中继：raw Image -> CompressedImage。

Webots 相机发布的是 raw bgra8（640x480x4 ≈ 1.23MB/帧），这么大的 raw 图
无法在 rosbridge 的 JSON 通道里于 subscribe 窗口内传完（camera_info 等小消息正常）。
本节点把它转成 JPEG CompressedImage（~30-80KB），便于经 rosbridge/MCP 传输；
MCP server 已内置 CompressedImage 处理，订到即用。

运行（ROS2 / 系统 py3.10，先剔除 miniconda 避免冲突）：
  source /opt/ros/humble/setup.bash
  python3 memory_navi/image_jpeg_relay.py /agent0/camera/image_color /agent0/camera/image_color/compressed
"""
import sys

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


class JpegRelay(Node):
    def __init__(self, in_topic: str, out_topic: str, quality: int = 80):
        super().__init__("jpeg_relay")
        self.bridge = CvBridge()
        self.quality = quality
        self.pub = self.create_publisher(CompressedImage, out_topic, 5)
        self.sub = self.create_subscription(Image, in_topic, self._cb, 5)
        self.n = 0
        self.get_logger().info(f"relay {in_topic} -> {out_topic} (jpeg q{quality})")

    def _cb(self, msg: Image):
        try:
            # cv_bridge 把 bgra8 转成 bgr8（丢 alpha），JPEG 不支持 alpha
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"convert failed: {e}")
            return
        ok, buf = cv2.imencode(".jpg", cv, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        out = CompressedImage()
        out.header = msg.header
        out.format = "jpeg"
        out.data = buf.tobytes()
        self.pub.publish(out)
        self.n += 1
        if self.n % 30 == 1:
            self.get_logger().info(f"relayed {self.n} frames, last {len(out.data)} bytes")


def main():
    in_topic = sys.argv[1] if len(sys.argv) > 1 else "/agent0/camera/image_color"
    out_topic = sys.argv[2] if len(sys.argv) > 2 else "/agent0/camera/image_color/compressed"
    rclpy.init()
    node = JpegRelay(in_topic, out_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
