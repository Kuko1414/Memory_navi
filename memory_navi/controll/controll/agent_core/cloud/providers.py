"""记忆作者的 provider（可替换）。

AnthropicProvider —— 主：Claude Messages API + tool-use 强制合法 JSON（最稳的结构化输出）。
LocalVLMProvider —— 回退：本地 vLLM 4B，best-effort 解析 JSON。

provider.annotate(image: ImagePart, context: dict) -> dict（符合 MEMORY_RECORD_SCHEMA 的原始记录）

注：Anthropic Messages API + image block + tool_use 写法稳定；若版本差异有问题，
用 `claude-api` skill 复核最新 API 与 model id。
"""
import json
import re

import anthropic

from .. import config
from ..image_utils import to_anthropic_image_block, to_openai_image_url
from ..llm_client import make_vllm_client, resolve_model
from .schema import MEMORY_RECORD_SCHEMA

RECORD_TOOL = {
    "name": "record_memory",
    "description": (
        "把当前相机视角看到的区域语义记录成结构化记忆。只填你能从这张 RGB 图可靠判断的字段。"
        "**objects 是必填数组**：画面里每一个能辨认的显著物体都各记一条（至少 name；尽量给定性 spatial 相对位置、"
        "归一化 ROI bbox x,y 左上角 w,h 宽高 0~1、state、affordance、confidence）。"
        "不要只把物体写进 summary —— summary 是概述，objects 才是结构化清单，二者都要给。"
        "不要臆造绝对坐标 abs_pose 与精确 distance_m —— 留 null，它们由几何管线（深度/TF）回填。"
    ),
    "input_schema": MEMORY_RECORD_SCHEMA,
}


SYSTEM_RECORD = (
    "你是室内机器人的『记忆作者』。你只能通过 record_memory 工具输出，不要输出自由文本。"
    "硬性要求：objects 必须是非空数组，把画面中每一个能辨认的显著物体各列一条（每条至少有 name）。"
    "严禁把物体仅写进 summary 而留空 objects——summary 是概述，objects 是结构化清单，两者都必须给。"
    "abs_pose / abs_pose_delta_m / view.distance_m 一律填 null（不要从单帧 RGB 猜测米制坐标/距离）。"
)


def _build_prompt(context: dict) -> str:
    area = context.get("area_hint") or "未知"
    trigger = context.get("trigger") or "manual"
    known = context.get("known_areas") or []
    lines = [
        "你是室内具身机器人的『记忆作者』。下面是机器人当前相机的一帧 RGB 图。",
        f"当前区域提示：{area}；触发原因：{trigger}。",
        f"已知区域：{', '.join(known) if known else '（暂无）'}。",
        "请通过 record_memory 工具记录：区域类型 type、简要 summary、**objects 数组（必填）**、可能的 hazards。",
        "objects 必须把画面中可辨认的主要物体逐个列出（每个一条，name 必填）。",
        "例如看到桌子和杯子就输出 objects:[{\"name\":\"desk\",\"spatial\":\"正前方桌台\",\"confidence\":0.8},{\"name\":\"cup\",\"spatial\":\"桌面右侧\",\"confidence\":0.6}]。",
        "务必：abs_pose / abs_pose_delta_m / view.distance_m 一律留 null（不要从单帧 RGB 猜测米制坐标/距离）。",
    ]
    if context.get("schema_error"):
        lines.append(f"上次输出不符合 schema（{context['schema_error']}），请修正后重新输出。")
    return "\n".join(lines)


class AnthropicProvider:
    """Claude 记忆作者（主）。"""

    def __init__(self, model: str = None, api_key: str = None, max_tokens: int = None):
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._model = model or config.ANTHROPIC_MODEL
        self._max_tokens = max_tokens or config.ANTHROPIC_MAX_TOKENS

    def annotate(self, image, context: dict) -> dict:
        kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": SYSTEM_RECORD,
            "tools": [RECORD_TOOL],
            "messages": [{
                "role": "user",
                "content": [to_anthropic_image_block(image), {"type": "text", "text": _build_prompt(context)}],
            }],
        }
        # 网关代理模型常拒绝 tool_choice="auto" 字符串（需 internally tagged enum）或强制 tool。
        # 策略：先不传 tool_choice（默认 auto），失败/无 tool_use 再先后试 forced 与无 tools。
        for tries, (tc, use_tools) in enumerate((
            (None, True),                                    # 默认 auto
            ({"type": "tool", "name": "record_memory"}, True),  # 强制（原生 Claude）
            (None, False),                                   # 无工具（纯 text）
        )):
            try:
                call_kw = dict(kwargs)
                if tc is not None:
                    call_kw["tool_choice"] = tc
                if not use_tools:
                    call_kw.pop("tools", None)
                    call_kw["system"] = SYSTEM_RECORD + "\n你的输出必须是一份合法 JSON 对象。不要加任何解释，只输出 JSON。"
                resp = self._client.messages.create(**call_kw)
                for block in resp.content:
                    if getattr(block, "type", None) == "tool_use" and block.name == "record_memory":
                        return dict(block.input)
                # 无 tool_use → 从 text 里抠 JSON
                for block in resp.content:
                    if getattr(block, "type", None) == "text":
                        m = re.search(r"\{.*\}", (block.text or ""), re.DOTALL)
                        if m:
                            try:
                                return json.loads(m.group(0))
                            except (json.JSONDecodeError, TypeError):
                                continue
                if tries == 0:
                    continue  # auto 没调工具→ 下轮 forced
            except Exception:  # noqa: BLE001
                if tries < 2:
                    continue
                raise
        raise RuntimeError("Anthropic annotate 三路尝试均失败")

    def strategic_guidance(self, payload: dict) -> dict:
        """Claude 不读图、纯文本战略援助（Qwen 卡住/矛盾时低频触发）。

        payload: {objects, passable, scan, depth, pose, history_steps, gap_memory, stuck_reason}
        返回 {action, direction, waypoint, rationale} —— 代码据此调整导航。
        delay ~1-8s，只在 Qwen 连续 N 次无进展时触发（不在常规控制回路里）。
        """
        text = _build_guidance_prompt(payload)
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=500,
            system=(
                "你是室内机器人的战略援助者。你【看不到图像】——只收到机器人本地传来的结构化感知数据。"
                "机器人卡住了(连续几次探索无进展/感知与传感器矛盾)。"
                "请根据这些文本数据给出简洁的下一步建议(只输出 JSON,不要其他文字):"
            ),
            messages=[{"role": "user", "content": text}],
        )
        raw = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                raw += block.text
        return _extract_json(raw or "")


def _build_guidance_prompt(payload: dict) -> str:
    lines = [
        "=== 机器人当前结构化感知 ===",
        f"卡住原因: {payload.get('stuck_reason', '未知')}",
        f"物体清单: {json.dumps(payload.get('objects', []), ensure_ascii=False)}",
        f"可通行方向: {json.dumps(payload.get('passable', []), ensure_ascii=False)}",
        f"scan 摘要: {payload.get('scan', '')}",
        f"depth 摘要: {payload.get('depth', '')}",
        f"当前位姿: {json.dumps(payload.get('pose', {}), ensure_ascii=False)}",
        f"最近动作: {json.dumps(payload.get('history_steps', []), ensure_ascii=False)}",
        f"已知区域记忆(缺口): {json.dumps(payload.get('gap_memory', {}), ensure_ascii=False)[:800]}",
        "请给出: {action:'move'/'turn'/'reshoot','direction':'left'/'center'/'right',"
        "waypoint:[x,y](可选),'rationale':'...'}。只输出 JSON。",
    ]
    return "\n".join(lines)


class LocalVLMProvider:
    """本地 vLLM 4B 记忆作者（回退，best-effort 解析 JSON）。"""

    def __init__(self, base_url: str = None, model: str = None):
        self._client = make_vllm_client(base_url or config.VLLM_4B_BASE_URL)
        self._model = model or resolve_model(self._client, config.VLLM_4B_MODEL_FALLBACK)

    def annotate(self, image, context: dict) -> dict:
        instruction = (
            _build_prompt(context)
            + "\n\n只输出一个 JSON 对象，字段遵循：area,type,summary,objects[{name,spatial,roi,state,affordance,confidence}],hazards。"
            + "不要输出 JSON 以外的任何文本。abs_pose/distance_m 一律 null。"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": [to_openai_image_url(image), {"type": "text", "text": instruction}]}],
            temperature=0.2,
            max_tokens=1024,
            stream=False,
        )
        text = resp.choices[0].message.content or ""
        return _extract_json(text)


def _extract_json(text: str) -> dict:
    """从模型文本里抠出第一个 JSON 对象；确保返回值是 dict。"""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return _sanitize_null_arrays(obj)
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return _sanitize_null_arrays(obj)
    raise RuntimeError(f"无法从本地 VLM 输出解析 JSON：{text[:200]}")


def _sanitize_null_arrays(d: dict) -> dict:
    """把 dict 里 schema 期望 array 但值是 None/string 的字段规范化。"""
    _ARRAY_KEYS = {"objects", "hazards", "affordance", "verified_by"}
    for k, v in list(d.items()):
        if k in _ARRAY_KEYS:
            if v is None:
                d[k] = []
            elif isinstance(v, str):
                d[k] = [v]
    # recurse into objects
    for obj in (d.get("objects") or []):
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if k in _ARRAY_KEYS:
                    if v is None:
                        obj[k] = []
                    elif isinstance(v, str):
                        obj[k] = [v]
    return d
