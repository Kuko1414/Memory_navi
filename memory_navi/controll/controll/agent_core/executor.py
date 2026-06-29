"""可复用执行器（route-B 短上下文动作循环）。

把 slice_demo 里一次性的接线提成一个可被 supervisor 反复调用的对象：
  Executor.run(brief) → 用精选动作工具跑一轮短上下文 tool-loop。

全部复用现有件（make_vllm_client / resolve_model / McpBridge / RosTools / run_tool_loop），
几乎无新逻辑。`brief` 是给第三块 supervisor 的注入口：远导航填拓扑切片、近任务填 sub_region
语义切片；执行器本身不碰记忆。

依赖可注入（client/model/ros），便于不连真实 vLLM/MCP 的接线单测。
"""
from agent_core import config
from agent_core.llm_client import make_vllm_client, resolve_model
from agent_core.mcp_bridge import McpBridge, RosTools
from agent_core.tool_loop import run_tool_loop


class Executor:
    """驱动 Qwen 用精选工具执行单步任务的薄封装。"""

    def __init__(
        self,
        *,
        base_url: str = config.VLLM_8B_BASE_URL,
        mcp_url: str = config.MCP_URL,
        allowlist=None,
        model_fallback: str = config.VLLM_8B_MODEL_FALLBACK,
        connect_ros: bool = True,
        call_timeout: float = 120.0,   # 闭环 move/turn 在慢仿真里可达数十秒，需大于 MCP 默认 30s
        client=None,
        model: str = None,
        ros=None,
        bridge=None,
    ):
        """构建执行器；未注入的依赖按 config 建立真实连接。"""
        self.bridge = bridge
        if ros is None:
            self.bridge = bridge or McpBridge(
                mcp_url, connect_timeout=config.MCP_CONNECT_TIMEOUT
            )
            ros = RosTools(self.bridge, call_timeout=call_timeout)
            if connect_ros:
                ros.connect(config.ROSBRIDGE_IP, config.ROSBRIDGE_PORT)
        self.ros = ros

        self.client = client or make_vllm_client(base_url)
        self.model = model or resolve_model(self.client, model_fallback)

        self.allowlist = config.ACTION_ALLOWLIST if allowlist is None else allowlist
        self.tools = self.ros.list_openai_tools(allowlist=self.allowlist)

    def run(
        self,
        brief: str,
        *,
        system: str = config.ACTION_SYSTEM_PROMPT,
        max_iters: int = None,
        tool_choice="auto",
    ):
        """跑一轮短上下文 tool-loop；brief 为 supervisor 注入的最小任务切片。

        tool_choice="required"（或指定函数）可强制本轮必须调用工具，杜绝"只叙述不调用"。
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": brief},
        ]
        return run_tool_loop(
            self.client,
            self.model,
            messages,
            self.tools,
            dispatch=self.ros.call,
            max_iters=max_iters or config.TOOL_LOOP_MAX_ITERS,
            temperature=config.TOOL_LOOP_TEMPERATURE,
            max_tokens=config.TOOL_LOOP_MAX_TOKENS,
            max_image_turns=config.TOOL_LOOP_MAX_IMAGE_TURNS,
            tool_choice=tool_choice,
        )

    def close(self):
        """关闭 MCP 桥（若由本对象建立）。"""
        if self.bridge is not None:
            self.bridge.close()
