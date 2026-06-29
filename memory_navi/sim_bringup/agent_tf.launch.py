#!/usr/bin/env python3
"""agentN TF 链：map -> base_link（GPS+IMU 真值）+ 静态 base_link->{lidar_link,camera_link}。

仿真原本无 TF（无 odom/robot_state_publisher 有效描述）。本 launch 补齐 agent0 的
可用 TF 树，供后续几何投影/导航使用。多机器人(agent1)与传感器消息 frame_id 命名空间化为后续。

运行（ROS2 系统 py3.10，先剔除 miniconda）：
  source /opt/ros/humble/setup.bash
  ros2 launch memory_navi/sim_bringup/agent_tf.launch.py
"""
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

NS = "agent0"
HERE = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    return LaunchDescription([
        # 静态：base_link -> lidar_link（LDS-01 安装 0 0 0.141）
        Node(
            package="tf2_ros", executable="static_transform_publisher", name="tf_base_lidar",
            # x y z yaw pitch roll parent child
            arguments=["0", "0", "0.141", "0", "0", "0", "base_link", "lidar_link"],
        ),
        # 静态：base_link -> camera_link（相机安装 0.17 0 0.13，俯仰 0.15rad）
        Node(
            package="tf2_ros", executable="static_transform_publisher", name="tf_base_camera",
            arguments=["0.17", "0", "0.13", "0", "0.15", "0", "base_link", "camera_link"],
        ),
        # 动态：map -> base_link（GPS 位置 + IMU 朝向 真值）
        ExecuteProcess(
            cmd=["python3", os.path.join(HERE, "webots_pose_tf.py"),
                 "--ros-args", "-p", f"namespace:={NS}"],
            output="screen",
        ),
    ])
