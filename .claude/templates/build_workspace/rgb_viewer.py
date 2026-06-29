#!/usr/bin/env python3
"""
独立脚本：订阅 RGB 话题，接收一帧 RGB 图像后保存到 data/image/ 目录，然后退出。

用法: python3 scripts/rgb_viewer.py
注意: 使用前请确认 {{RGB_TOPIC}} 话题名与实际相机匹配。
"""

import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

import numpy as np
import cv2


class RGBViewer(Node):
    def __init__(self):
        super().__init__('rgb_viewer')

        # 使用 RELIABLE QoS 以匹配相机发布端
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image,
            '{{RGB_TOPIC}}',
            self.rgb_callback,
            qos
        )
        self.get_logger().info('等待 RGB 图像消息 {{RGB_TOPIC}} ...')
        self.saved = False

    def rgb_callback(self, msg: Image):
        if self.saved:
            return
        self.saved = True

        self.get_logger().info(
            f'收到 RGB 图像: {msg.width}x{msg.height}, encoding={msg.encoding}'
        )

        # ---------- 将 ROS Image 转为 numpy 数组 ----------
        if msg.encoding == 'rgb8':
            img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
            # OpenCV 使用 BGR 格式
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'bgr8':
            img_bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
        elif msg.encoding == 'rgba8':
            img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4
            )
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif msg.encoding == 'bgra8':
            img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4
            )
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
        else:
            self.get_logger().error(f'不支持的 encoding: {msg.encoding}')
            rclpy.shutdown()
            return

        # ---------- 保存图片 ----------
        save_dir = '{{DATA_DIR}}/image'
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'rgb_{timestamp}.png'
        filepath = os.path.join(save_dir, filename)

        cv2.imwrite(filepath, img_bgr)
        self.get_logger().info(f'RGB 图像已保存: {filepath}')

        rclpy.shutdown()


def main():
    rclpy.init()
    node = RGBViewer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
