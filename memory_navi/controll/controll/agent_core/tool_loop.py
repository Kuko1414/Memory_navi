"""同步 ReAct tool loop（执行者 Qwen ↔ MCP 工具）。

chat.completions.create(tools=..., tool_choice="auto") 非流式
  → 有 tool_calls 就逐个 dispatch、回灌 role:"tool"
  → 工具返回图像时作为后续 user image turn 注入（OpenAI/vLLM 的 tool 消息无图像通道），
    并对图像轮数封顶，护住 8B 4096 上下文。
非流式：规避 vLLM hermes parser 流式下返回 raw 文本的已知 bug。
"""
import json
from dataclasses import dataclass, field

from .image_utils import to_openai_image_url


@dataclass
class LoopResult:
    text: str = ""
    messages: list = field(default_factory=list)
    tool_trace: list = field(default_factory=list)
    truncated: bool = False


def run_tool_loop(
    client,
    model: str,
    messages: list,
    tools: list,
    dispatch,                      # (name:str, args:dict) -> ToolOutput
    *,
    max_iters: int = 6,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    max_image_turns: int = 2,
    tool_choice="auto",            # "auto"|"required"|{"type":"function",...} 透传给 vLLM
) -> LoopResult:
    tool_trace: list = []
    image_turns = 0

    for _ in range(max_iters):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        msg = resp.choices[0].message

        assistant = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant)

        if not msg.tool_calls:
            return LoopResult(
                text=msg.content or "", messages=messages, tool_trace=tool_trace, truncated=False
            )

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            out = dispatch(name, args)
            tool_trace.append(
                {"name": name, "args": args, "is_error": out.is_error, "text": (out.text or "")[:500]}
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": out.text or ("[error]" if out.is_error else "[ok]"),
            })
            for img in out.images:
                if image_turns >= max_image_turns:
                    messages.append({
                        "role": "user",
                        "content": f"[image from {name} omitted: image-turn budget reached]",
                    })
                    break
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"[image returned by {name}]"},
                        to_openai_image_url(img),
                    ],
                })
                image_turns += 1

    return LoopResult(text="", messages=messages, tool_trace=tool_trace, truncated=True)
