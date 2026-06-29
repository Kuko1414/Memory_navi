"""集中配置：endpoints / 模型 / 端口 / 路径 / 阈值。

只放常量与少量环境读取；不含逻辑。路径以 plan 与现有部署为准。
"""
import os

# ---- vLLM（执行者 / 本地 VLM）----
VLLM_8B_BASE_URL = os.environ.get("VLLM_8B_BASE_URL", "http://localhost:8000/v1")
VLLM_4B_BASE_URL = os.environ.get("VLLM_4B_BASE_URL", "http://localhost:8001/v1")
# 模型 id 优先在运行时从 /v1/models 解析（见 llm_client.resolve_model）；以下为回退值。
VLLM_8B_MODEL_FALLBACK = "/home/kuko/.cache/huggingface/hub/qwen/Qwen3-VL-8B-Instruct"
VLLM_4B_MODEL_FALLBACK = "/home/kuko/.cache/huggingface/hub/Qwen/Qwen3-VL-4B-Instruct"

# ---- MCP server（streamable-http）----
# FastMCP streamable-http 默认挂在 /mcp。server 启动：
#   uv run server.py --transport streamable-http --host 127.0.0.1 --port 9000
MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:9000/mcp")
MCP_CONNECT_TIMEOUT = 30.0       # 建立 ClientSession 的超时
MCP_CALL_TIMEOUT = 30.0          # 单次 call_tool 超时

# ---- rosbridge（MCP server 后端）----
ROSBRIDGE_IP = os.environ.get("ROSBRIDGE_IP", "127.0.0.1")
ROSBRIDGE_PORT = int(os.environ.get("ROSBRIDGE_PORT", "9090"))

# ---- 机器人 / 话题 ----
AGENT_NS = os.environ.get("AGENT_NS", "agent0")
CAMERA_TOPIC = f"/{AGENT_NS}/camera/image_color"
CAMERA_MSG_TYPE = "sensor_msgs/msg/Image"
# 原始帧是 raw bgra8 ~1.23MB，太大无法经 rosbridge 在 subscribe 窗口内传完；
# 用 memory_navi/image_jpeg_relay.py 转出 JPEG CompressedImage（~30KB）后由记忆作者订阅。
CAMERA_COMPRESSED_TOPIC = f"/{AGENT_NS}/camera/image_color/compressed"
CAMERA_COMPRESSED_MSG_TYPE = "sensor_msgs/msg/CompressedImage"

# ---- 图像采集 / 降采样 ----
# MCP server 抓图会先存盘到 <mcp-cwd>/camera/received_image.jpeg；以 server 目录为基准。
MCP_SERVER_DIR = os.environ.get(
    "MCP_SERVER_DIR", "/home/kuko/Kuko1414/memory_navi/ros-mcp-server"
)
MCP_IMAGE_PATH = os.path.join(MCP_SERVER_DIR, "camera", "received_image.jpeg")
IMG_MAX_EDGE = int(os.environ.get("IMG_MAX_EDGE", "896"))  # 发 Claude 前长边上限
IMG_QUALITY = int(os.environ.get("IMG_QUALITY", "80"))

# ---- 记忆作者（云端大模型）----
# Anthropic model id；默认 Sonnet 4.6（视觉强、快/省），更高质量可设 claude-opus-4-8。
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "1500"))

# ---- 文件系统式 STG 记忆 ----
MEMORY_ROOT = os.environ.get("MEMORY_ROOT", "/home/kuko/Kuko1414/memory_navi/memory")
ENV_NAME = os.environ.get("ENV_NAME", "sim")  # 记忆按环境分目录 MEMORY_ROOT/<env>

# ---- tool loop ----
TOOL_LOOP_MAX_ITERS = 6
TOOL_LOOP_TEMPERATURE = 0.2
TOOL_LOOP_MAX_TOKENS = 1024
TOOL_LOOP_MAX_IMAGE_TURNS = 2   # 回灌图像轮上限，护住 8B 4096 上下文

# ---- 执行器（route-B 短上下文动作循环）----
# 只把这组精选高层工具暴露给 Qwen（裸底层 MCP 工具保留但不进 allowlist）。
ACTION_ALLOWLIST = {
    "move", "turn_left_deg", "turn_right_deg", "stop",
    "look", "get_pose", "navigate_to", "scan_summary",
    "depth_summary",
    "if_in_memory", "nav_distance", "nav_object", "record_area",
}
# 执行器默认 system prompt：强调单步执行、用感知确认、不要自由发挥裸话题。
ACTION_SYSTEM_PROMPT = (
    "你是一个室内机器人的执行器。只用提供的工具完成被交代的这一小步任务，不要规划长远目标。\n"
    "- 移动用 move(distance_m, turn_deg)；急停用 stop()。+turn_deg 左转、+distance_m 前进。\n"
    "- 用 scan_summary() 查最近障碍/前方是否畅通；用 look() 观察场景；用 get_pose() 查当前位姿。\n"
    "- 已知的目标/路线会在用户消息里给出，按它执行；动作前后用感知工具确认，避免盲动。\n"
    "- 完成或受阻就用简短文字说明结果，不要再调用工具。"
)
