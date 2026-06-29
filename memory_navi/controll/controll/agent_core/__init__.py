"""agent_core — 薄 Agent 核心（执行者 Qwen ↔ MCP/ROS ↔ 记忆作者 Claude）。

设计见 plan：
- 不 import rclpy；经 MCP(HTTP) 碰 ROS、HTTP 碰 vLLM/Claude，整套住 py3.11 vLLM 环境。
- 顶层确定性调度（Supervisor，下一步）；本模块先打通垂直切片所需的接口。
"""
