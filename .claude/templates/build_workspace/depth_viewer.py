#!/usr/bin/env python3
"""
独立脚本：订阅深度话题，接收一帧深度图后用 colormap 可视化并保存到 data/image/ 目录，然后退出。

用法: python3 scripts/depth_viewer.py
注意: 使用前请确认 {{DEPTH_TOPIC}} 话题名与实际相机匹配。
"""

import os
import sys
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

import numpy as np
import cv2


class DepthViewer(Node):
    def __init__(self):
        super().__init__('depth_viewer')

        # 使用 RELIABLE QoS 以匹配相机发布端
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image,
            '{{DEPTH_TOPIC}}',
            self.depth_callback,
            qos
        )
        self.get_logger().info('等待深度图消息 {{DEPTH_TOPIC}} ...')
        self.saved = False

    def depth_callback(self, msg: Image):
        if self.saved:
            return
        self.saved = True

        self.get_logger().info(
            f'收到深度图: {msg.width}x{msg.height}, encoding={msg.encoding}'
        )

        # ---------- 将 ROS Image 转为 numpy 数组 ----------
        if msg.encoding == '16UC1' or msg.encoding == 'mono16':
            # mono16 / 16UC1: 每像素 2 字节，单位通常为 mm
            depth_array = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.width
            )
        elif msg.encoding == '32FC1':
            # 32FC1: 每像素 4 字节浮点，单位通常为 m
            depth_array = np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.width
            )
            # 转为 mm 以统一后续处理
            depth_array = (depth_array * 1000.0).astype(np.uint16)
        else:
            self.get_logger().error(f'不支持的 encoding: {msg.encoding}')
            rclpy.shutdown()
            return

        # ---------- 统计信息 ----------
        valid_mask = depth_array > 0
        valid_count = np.count_nonzero(valid_mask)
        total = depth_array.size
        self.get_logger().info(
            f'深度统计: 有效像素 {valid_count}/{total} '
            f'({valid_count / total * 100:.1f}%), '
            f'min={depth_array[valid_mask].min() if valid_count else 0} mm, '
            f'max={depth_array[valid_mask].max() if valid_count else 0} mm'
        )

        # ---------- 归一化 + colormap 可视化 ----------
        depth_float = depth_array.astype(np.float64)
        depth_float[depth_array == 0] = np.nan

        if valid_count > 0:
            d_min = np.nanmin(depth_float)
            d_max = np.nanmax(depth_float)
            normalized = np.zeros_like(depth_float)
            if d_max > d_min:
                normalized = (depth_float - d_min) / (d_max - d_min) * 255.0
            normalized = np.nan_to_num(normalized, nan=0.0).astype(np.uint8)
        else:
            normalized = np.zeros((msg.height, msg.width), dtype=np.uint8)

        # 应用 JET colormap（近处蓝色，远处红色）
        color_depth = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        # 无效深度区域设为黑色
        color_depth[depth_array == 0] = [0, 0, 0]

        # ---------- 保存图片 ----------
        save_dir = '{{DATA_DIR}}/image'
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'depth_{timestamp}.png'
        filepath = os.path.join(save_dir, filename)

        cv2.imwrite(filepath, color_depth)
        self.get_logger().info(f'深度图已保存: {filepath}')

        # 同时保存原始深度图（灰度，方便后续分析）
        raw_filename = f'depth_raw_{timestamp}.png'
        raw_filepath = os.path.join(save_dir, raw_filename)
        cv2.imwrite(raw_filepath, depth_array)
        self.get_logger().info(f'原始深度图已保存: {raw_filepath}')

        rclpy.shutdown()


def main():
    rclpy.init()
    node = DepthViewer()
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
