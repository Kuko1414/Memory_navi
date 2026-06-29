For AI coding agents. This is the task we need to accomplish in this workspace and the design concept for each step. Read before making changes.

## 1. Overview

All the goals that this workspace aims to achieve and each of the work steps are written in 'PROCESS.md'. You can refer to this file to obtain the 
current work goals. Only when I set the goal for the next stage of work can I start the next stage.

When generating the required code, here are all the ROS2-related topics that you need to use or will use in the future.

## 2. ROS2 Topics

The topic for obtaining the current position of the robot:
    - 'PoseStamped, {{POSE_TOPIC}}' (include: position and orientation)
    - 'Odometry, {{ODOM_TOPIC}}' (include: pose with position+orientation, twist with linear+angular velocity)

The topic for sending the speed to the robot:
    - 'Twist, {{CMD_VEL_TOPIC}}' (include: linear x,y,z and angular x,y,z)

The topic for obtaining the robot's orientation (yaw):
    - '{{IMU_TOPIC}}' (yaw/IMU data for orientation)

The topic for sending the path points to the robot:
    - 'Path, {{PATH_TOPIC}}' (include: header and poses)

The topic for depth camera and vision:
    - 'Image, {{RGB_TOPIC}}' (RGB image from depth camera)
    - 'CameraInfo, {{DEPTH_CAMERA_INFO_TOPIC}}' (depth camera_info for pixel to physical conversion)
    - 'Image, {{DEPTH_TOPIC}}' (the depth image data for obstacle detection and path planning)
    - 'CameraInfo, {{RGB_CAMERA_INFO_TOPIC}}' (RGB camera intrinsic parameters)

The topic for coordinate transforms:
    - 'TFMessage, /tf' (dynamic coordinate transform tree)
    - 'TFMessage, /tf_static' (static coordinate transforms, e.g. camera mount position relative to base_link)

## 3. ROS2 Nodes

<!-- TODO: Document each ROS2 node here. Include: node name, purpose, subscribed/published topics, key design decisions. -->

## 4. Workspace Structure

The workspace is structured as follows:

```
src/
    (source files for workspace, strictly no runtime generated data like images should be saved here to avoid polluting source code)

data/
    image/    (runtime captured images for debugging and VLM)
    log/      (debug logs, pose logs, conversion logs)

future/
    future_work.md  (functions that MAY be applied but wait to be identified)

scripts/
    (debug/utility scripts: depth_viewer, pose_logger, rgb_viewer, test_odom_yaw)
    (adjust ROS2 topic subscriptions in these scripts after copying to new workspace)

build/ install/ log/
    (standard ROS2 directories, created by colcon build)
```
