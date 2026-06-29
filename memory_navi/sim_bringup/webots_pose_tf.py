#!/usr/bin/env python3
"""map -> <ns>/base_link 地面真值 TF（Webots 仿真真值）。

仿真没有 odom/TF，这里用机器人自带的 GPS（位置）+ IMU（朝向）合成
map->base_link 的真值变换。配合 agent_tf.launch.py 里的静态
base_link->{lidar_link,camera_link} 一起构成可用的 TF 链。

参数：namespace(默认 agent0)、map_frame(map)、base_frame(base_link)、rate_hz(50)。
GPS 话题类型先按 geometry_msgs/PointStamped 处理；若实际是 NavSatFix，改订阅类型即可。

运行（ROS2 系统 py3.10，先剔除 miniconda）：
  source /opt/ros/humble/setup.bash
  python3 memory_navi/sim_bringup/webots_pose_tf.py --ros-args -p namespace:=agent0
"""
import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


class WebotsPoseTf(Node):
    def __init__(self):
        super().__init__("webots_pose_tf")
        ns = self.declare_parameter("namespace", "agent0").value
        self.map_frame = self.declare_parameter("map_frame", "map").value
        self.base_frame = self.declare_parameter("base_frame", "base_link").value
        rate = self.declare_parameter("rate_hz", 50.0).value

        self._pos = None   # (x, y, z) from GPS
        self._ori = None   # (x, y, z, w) from IMU
        self._br = TransformBroadcaster(self)
        self.create_subscription(PointStamped, f"/{ns}/gps", self._gps_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, f"/{ns}/imu", self._imu_cb, qos_profile_sensor_data)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"webots_pose_tf: {self.map_frame} -> {self.base_frame}  from /{ns}/gps + /{ns}/imu"
        )

    def _gps_cb(self, m: PointStamped):
        self._pos = (m.point.x, m.point.y, m.point.z)

    def _imu_cb(self, m: Imu):
        q = m.orientation
        self._ori = (q.x, q.y, q.z, q.w)

    def _tick(self):
        if self._pos is None or self._ori is None:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = self._pos
        (t.transform.rotation.x, t.transform.rotation.y,
         t.transform.rotation.z, t.transform.rotation.w) = self._ori
        self._br.sendTransform(t)


def main():
    rclpy.init()
    node = WebotsPoseTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
