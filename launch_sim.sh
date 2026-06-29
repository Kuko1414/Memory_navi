#!/bin/bash
# 启动 Small House World + TurtleBot3 Waffle Pi 仿真
set -e

# ROS2 Humble 需要 Python 3.10，先剔除 conda 的 Python 3.13 避免冲突
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV

source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
source /home/kuko/Kuko1414/humble_ws/install/setup.bash 2>/dev/null || true

export TURTLEBOT3_MODEL=waffle_pi
export GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models:${GAZEBO_MODEL_PATH}
export GAZEBO_MODEL_PATH=/home/kuko/Kuko1414/aws-robomaker-small-house-world/models:${GAZEBO_MODEL_PATH}

WORLD=/home/kuko/Kuko1414/aws-robomaker-small-house-world/worlds/small_house.world

# 清理旧进程
pkill -f "gzserver" 2>/dev/null || true
pkill -f "gzclient" 2>/dev/null || true
# 确保端口释放
while lsof -ti:11345 2>/dev/null; do
    kill -9 $(lsof -ti:11345) 2>/dev/null || true
    sleep 1
done
sleep 2

echo "[1/3] Starting Gazebo (with ROS2 system plugins)..."
# -s 加载 ROS2 系统插件（init + factory + force_system）
gzserver -s libgazebo_ros_init.so -s libgazebo_ros_factory.so -s libgazebo_ros_force_system.so --verbose "$WORLD" &
sleep 5
gzclient --verbose &
sleep 3

echo "[2/3] Waiting for /spawn_entity service..."
for i in $(seq 1 30); do
    if ros2 service list 2>/dev/null | grep -q "/spawn_entity"; then
        echo "       /spawn_entity ready"
        break
    fi
    sleep 1
done

echo "[3/3] Spawning TurtleBot3 Waffle Pi..."
ros2 run gazebo_ros spawn_entity.py \
    -entity turtlebot3_waffle_pi \
    -file /opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf \
    -x 0.0 -y 0.0 -z 0.01

echo ""
echo "=== Done ==="
echo "Topics: /cmd_vel  /odom  /scan  /imu  /camera/image_raw  /joint_states"
