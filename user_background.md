# 用户学术背景摘要

> 此文件供 AI 会话快速了解用户的研究背景，方便在课程推荐、学业规划、职场建议等场景下提供精准建议。

---

## 基本信息

- **学历**：硕士研究生
- **核心方向**：机器人学、具身智能（Embodied AI）、VLM 驱动的自主导航
- **开发环境**：ROS2 Humble (Ubuntu 22.04 / WSL2) + Python

---

## 研究方向

### VLM 驱动的轮式机器人视觉导航

正在开发一个完整的**"感知 → 推理 → 规划 → 控制"端到端闭环系统**，核心思想是用多模态大模型（Gemini）替代传统路径规划模块，实现"看 → 想 → 走"的自主导航。

### 系统架构

```
深度相机 RGB ──→ Gemini 2.5 Flash API ──→ 像素路径点
                                                │
深度相机 Depth ──→ 针孔模型反投影 + TF2 变换 ──→ 3D 路径点(/path)
                                                │
                                          Pure Pursuit + PID ──→ /cmd_vel
```

---

## 已掌握的技术栈

### ROS2 机器人开发
- ROS2 Humble 节点开发、话题/Service/Action 通信、Launch 文件
- TF2 坐标变换、多线程执行器（MultiThreadedExecutor）、QoS 策略（RELIABLE vs BEST_EFFORT）
- 自定义消息/服务包管理

### 计算机视觉与传感器
- 使用过多种深度相机：**Orbbec Aurora（结构光）、Intel RealSense D435、Orbbec Gemini 2L（双目）**
- 精通针孔相机模型：内参矩阵（fx, fy, cx, cy）→ 2D 像素 → 3D 空间反投影
- 深度图处理：窗口滤波、中值采样、空洞处理、地面平面 fallback
- 低视角（~11cm）下地面材质镜面反射导致深度图失效的实战经验

### 大模型/多模态模型应用（VLM）
- 使用 **Google Gemini Robotics ER 1.6 Preview** 多模态 API 做端到端视觉路径规划
- **Skill 系统设计**：YAML 热插拔提示词框架，零代码切换模型行为/角色
- **Function Calling 架构**：让 Gemini 作为 Agent 主动调用 ROS2 工具函数（获取位姿、图像、障碍物距离、发布路径等）
- **渐进式场景认知架构**：Scout（粗粒度区域探索）→ Inspector（细粒度物体标注）→ Navigator（基于语义地图的精准导航）
- Prompt Engineering：归一化坐标 vs 绝对像素、透视约束、图像尺寸注入、输出格式约束

### 控制算法
- 实现 **Pure Pursuit + PID 控制** 做路径跟踪
- PID 调参实战：抗振荡（低通滤波、余弦衰减减速）、角速度/线速度协调
- 前视距离、到达阈值、障碍物安全距离等参数调优

### 硬件与部署
- 目标部署平台：**NVIDIA Jetson Orin Nano (8GB)**，了解 TensorRT / ONNX 模型部署流程
- 真实轮式机器人调试：IMU yaw 启动温漂、DDS 发送队列饱和导致相机停发、相机安装高度对感知影响

### 工具链
- WSL2 跨平台开发（Windows IDE + Linux ROS2）
- rosbridge + ros-mcp-server 远程调试
- Git 版本管理、colcon 构建、rosdep 依赖管理
- OpenCV、cv_bridge、NumPy 视觉处理

---

## 关键项目经验

1. **端到端 VLM 导航闭环**：实现了"Gemini 看图出路径点 → 深度反投影 → PID 跟踪 → 触发重规划"的完整闭环，成功引导小车到达走廊门口
2. **多款深度相机适配**：因地面材质问题从结构光相机切换到双目相机，积累了不同传感器性能评估经验
3. **Agent 架构设计**：设计了完整的 Function Calling 工具调用架构和渐进式场景认知 Skill 系统，虽尚未全部落地但体系完整
4. **PID 振荡问题解决**：从 ±60° 剧烈振荡调参到 ±3° 平稳跟踪，积累了控制参数整定经验

---

## 适合向用户提供的建议类型

用户在以下方面可能需要建议：
- 机器人/具身智能方向的研究课题选择与论文发表
- VLM + 机器人系统的架构设计与实验设计
- 从原型开发到学术论文的方法论
- 求职方向（机器人感知/规划、具身智能、自动驾驶等）
- 技能栈扩展（强化学习、SLAM、模型部署优化、传感器融合等）
- 学术写作和项目展示
