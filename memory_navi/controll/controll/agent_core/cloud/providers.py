"""记忆作者的 provider（可替换）。

AnthropicProvider —— 主：Claude Messages API + tool-use 强制合法 JSON（最稳的结构化输出）。
LocalVLMProvider —— 回退：本地 vLLM 4B，best-effort 解析 JSON。

provider.annotate(image: ImagePart, context: dict) -> dict（符合 MEMORY_RECORD_SCHEMA 的原始记录）

注：Anthropic Messages API + image block + tool_use 写法稳定；若版本差异有问题，
用 `claude-api` skill 复核最新 API 与 model id。
"""
import json
import math
import re

try:
    import anthropic
except ImportError:  # pragma: no cover - optional for offline GPT review helpers
    anthropic = None
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional in non-vLLM test envs
    OpenAI = None

from .. import config
from ..image_utils import to_anthropic_image_block, to_openai_image_url
from ..llm_client import make_vllm_client, resolve_model
from .schema import MEMORY_RECORD_SCHEMA, TOOL_RECORD_SCHEMA

RECORD_TOOL = {
    "name": "record_memory",
    "description": (
        "把当前相机视角看到的区域语义记录成结构化记忆。只填你能从这张 RGB 图可靠判断的字段。"
        "**objects 是必填数组**：画面里每一个能辨认的显著物体都各记一条（至少 name；尽量给定性 spatial 相对位置、"
        "归一化 ROI bbox x,y 左上角 w,h 宽高 0~1、state、affordance、confidence）。"
        "不要只把物体写进 summary —— summary 是概述，objects 才是结构化清单，二者都要给。"
        "不要臆造绝对坐标 abs_pose 与精确 distance_m —— 留 null，它们由几何管线（深度/TF）回填。"
    ),
    "input_schema": TOOL_RECORD_SCHEMA,   # 严格版：objects required + minItems，逼模型列物体
}


SYSTEM_RECORD = (
    "你是室内机器人的『记忆作者』。你只能通过 record_memory 工具输出，不要输出自由文本。"
    "硬性要求：objects 必须是非空数组，把画面中每一个能辨认的显著物体各列一条（每条至少有 name）。"
    "严禁把物体仅写进 summary 而留空 objects——summary 是概述，objects 是结构化清单，两者都必须给。"
    "abs_pose / abs_pose_delta_m / view.distance_m 一律填 null（不要从单帧 RGB 猜测米制坐标/距离）。"
)

# 纯 JSON 路径专用 system（不提工具——实测网关用 tool_use 会回空 objects，纯 JSON 才列得全）。
SYSTEM_RECORD_JSON = (
    "你是室内机器人的『记忆作者』。看这张相机图，把画面里【独立的家具/设备/可移动物体】逐个列出。\n"
    "只输出一个合法 JSON 对象（不要代码块标记、不要任何解释）：\n"
    "{\"area\":\"\",\"type\":\"区域类型如 office/lounge\",\"summary\":\"一句话概述\","
    "\"objects\":[{\"name\":\"\",\"spatial\":\"定性相对位置\",\"roi\":{\"x\":0,\"y\":0,\"w\":0,\"h\":0},"
    "\"confidence\":0.8,\"abs_pose\":null}]}\n"
    "【只记离散物体】：桌、椅、沙发、柜、显示器、绿植、灯具、门 等。\n"
    "【不要记】：墙面/地板/天花板/踢脚线/梁/转角等建筑表面；阴影/反光/光斑等视觉假象；"
    "以及机器人【自身】可见的轮子/机身部件（名字含 robot/wheel/self 的一律不列）。\n"
    "【命名规范】：用规范单数通用名（多台显示器统一都叫 monitor，不要拆成 monitor_left/monitor_center）；"
    "靠 spatial 区分位置，不要靠名字后缀把同类物体拆成多条。\n"
    "硬性要求：objects 必须非空；roi 用归一化 0~1；abs_pose 一律 null（不要猜米制坐标，由几何管线回填）。"
)


BOX_ARBITRATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["boxes"],
    "properties": {
        "boxes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["idx", "keep", "name", "roi_ok", "completeness", "reason"],
                "properties": {
                    "idx": {"type": "integer"},
                    "keep": {"type": "boolean"},
                    "name": {"type": ["string", "null"]},
                    "roi_ok": {"type": "boolean"},
                    "completeness": {"type": "string", "enum": ["完整", "部分", "勉强"]},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}

BOX_ARBITRATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "box_arbitration",
        "strict": True,
        "schema": BOX_ARBITRATION_SCHEMA,
    },
}

TARGET_CONFIRM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["seen", "confidence", "reason"],
    "properties": {
        "seen": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
}

TARGET_CONFIRM_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "target_confirmation",
        "strict": True,
        "schema": TARGET_CONFIRM_SCHEMA,
    },
}

SYSTEM_BOX_ARBITER = (
    "你是室内机器人视觉标注仲裁员。图像里已经画好带编号的蓝框；你只裁决这些编号框。"
    "不要新增框、移动框、输出坐标或距离。每个框输出 keep/name/roi_ok/completeness/reason。"
    "只有框住的是完整、清晰、独立的家具/设备/可移动物体时 keep=true。"
    "墙/地板/天花板/踢脚线/梁/门/通道/机器人自身/只是一块平面/半截物/远处糊成一团都 keep=false。"
    "命名用规范中文通用单数名：沙发/显示器/办公桌/办公椅/柜子/绿植/厨台/水槽/键盘/吊灯等。"
    "若不确定，宁可 keep=false。只输出符合 schema 的 JSON。"
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
    known_objs = context.get("known_objects") or []
    if known_objs:
        lines.append(
            "【已记录物体】（本房间此前视角已登记，避免重复）："
            + "、".join(str(n) for n in known_objs[:40]) + "。")
        lines.append(
            "只补充画面里【尚未记录】的新物体；上面已记录过的同类同位物体不要再重复列出"
            "（除非这次能看得更清、提供更准的 spatial/roi）。这样可避免同一物体被反复登记。")
    if context.get("schema_error"):
        lines.append(f"上次输出不符合 schema（{context['schema_error']}），请修正后重新输出。")
    return "\n".join(lines)


def _parse_box_judgments_obj(obj: dict, box_ids: list) -> dict:
    """Normalize box-arbitration JSON into {idx: judgment}; missing ids abstain."""
    items = obj.get("boxes", []) if isinstance(obj, dict) else []
    parsed = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            idx = int(it.get("idx"))
        except (TypeError, ValueError):
            continue
        comp = it.get("completeness")
        comp = comp if comp in ("完整", "部分", "勉强") else "部分"
        name = it.get("name")
        name = str(name).strip() if isinstance(name, str) and name.strip() else None
        roi_ok = bool(it.get("roi_ok"))
        keep = bool(it.get("keep")) and roi_ok and comp == "完整" and name is not None
        parsed[idx] = {
            "name": name if keep else None,
            "completeness": comp,
            "keep": keep,
            "roi_ok": roi_ok,
            "reason": str(it.get("reason") or "")[:120],
        }
    for bid in box_ids or []:
        parsed.setdefault(int(bid), {
            "name": None,
            "completeness": "部分",
            "keep": False,
            "roi_ok": False,
            "reason": "missing judgment",
        })
    return parsed


def parse_openai_box_judgments(raw: str | dict, box_ids: list) -> dict:
    """Parse GPT box arbitration output; kept boxes must be complete, named, and roi_ok."""
    if isinstance(raw, dict):
        return _parse_box_judgments_obj(raw, box_ids)
    obj = _extract_json(raw or "")
    return _parse_box_judgments_obj(obj, box_ids)


def parse_target_confirmation(raw: str | dict) -> dict:
    """Parse GPT target confirmation output into {seen, confidence, reason}."""
    obj = raw if isinstance(raw, dict) else _extract_json(raw or "")
    if not isinstance(obj, dict):
        return {"seen": False, "confidence": 0.0, "reason": "invalid output"}
    try:
        conf = float(obj.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "seen": bool(obj.get("seen")),
        "confidence": max(0.0, min(1.0, conf)),
        "reason": str(obj.get("reason") or "")[:160],
    }


class OpenAIProvider:
    """GPT visual arbitrator for YOLO ROI + Qwen naming experiments.

    This provider is intentionally narrow: it judges numbered image boxes and
    returns schema-bound labels/abstentions. It does not plan routes, alter
    coordinates, or replace Claude's text-only director/curator roles.
    """

    def __init__(self, model: str = None, api_key: str = None,
                 base_url: str = None, max_tokens: int = None):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url or config.OPENAI_BASE_URL:
            kwargs["base_url"] = base_url or config.OPENAI_BASE_URL
        self._client = OpenAI(**kwargs)
        self._model = model or config.OPENAI_ARBITER_MODEL
        self._max_tokens = max_tokens or config.OPENAI_ARBITER_MAX_TOKENS

    def judge_boxes(self, image, boxes: list, *, qwen_judgments: dict = None,
                    name_hints: list = None, context: str = "") -> dict:
        """Judge numbered boxes in an image and return {idx: judgment}.

        image must be an ImagePart, normally the numbered-box image already
        produced by draw_dual_boxes. Coordinates remain owned by depth/TF code.
        """
        box_ids = [int(b.get("idx")) for b in (boxes or []) if b.get("idx") is not None]
        if not box_ids:
            return {}
        user = _build_box_arbitration_prompt(
            boxes, qwen_judgments=qwen_judgments, name_hints=name_hints, context=context)
        content = [{"type": "text", "text": user}, to_openai_image_url(image)]
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_BOX_ARBITER},
                    {"role": "user", "content": content},
                ],
                temperature=0,
                max_tokens=self._max_tokens,
                response_format=BOX_ARBITRATION_RESPONSE_FORMAT,
                stream=False,
            )
        except TypeError:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_BOX_ARBITER},
                    {"role": "user", "content": content},
                ],
                temperature=0,
                max_tokens=self._max_tokens,
                stream=False,
            )
        raw = resp.choices[0].message.content or ""
        return parse_openai_box_judgments(raw, box_ids)

    def confirm_target(self, image, target_name: str, *, context: str = "") -> dict:
        """Low-frequency visual confirmation for ARRIVED_NO_TARGET cases."""
        user = (
            f"目标物体：{target_name}\n"
            "请只判断这张当前相机图里是否能清楚看到目标物体。"
            "不要猜测画外/记忆里的目标；看不清就 seen=false。\n"
            f"上下文：{context[:800] if context else '无'}"
        )
        content = [{"type": "text", "text": user}, to_openai_image_url(image)]
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "你是机器人到位后的目标可见性复核员，只输出 JSON。"},
                    {"role": "user", "content": content},
                ],
                temperature=0,
                max_tokens=300,
                response_format=TARGET_CONFIRM_RESPONSE_FORMAT,
                stream=False,
            )
        except TypeError:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "你是机器人到位后的目标可见性复核员，只输出 JSON。"},
                    {"role": "user", "content": content},
                ],
                temperature=0,
                max_tokens=300,
                stream=False,
            )
        return parse_target_confirmation(resp.choices[0].message.content or "")


def _build_box_arbitration_prompt(boxes: list, *, qwen_judgments: dict = None,
                                  name_hints: list = None, context: str = "") -> str:
    rows = []
    for b in boxes or []:
        idx = b.get("idx")
        qj = (qwen_judgments or {}).get(idx) or (qwen_judgments or {}).get(str(idx)) or {}
        rows.append({
            "idx": idx,
            "detector_label": b.get("detector_label"),
            "detector_confidence": b.get("confidence"),
            "qwen": {
                "keep": qj.get("keep"),
                "name": qj.get("name"),
                "completeness": qj.get("completeness"),
            } if qj else None,
        })
    lines = [
        "请裁决图中这些蓝框编号。只判断给出的 idx，不要新增框或坐标。",
        "候选框元数据:",
        json.dumps(rows, ensure_ascii=False),
    ]
    if name_hints:
        lines.append("本区域常见/已知名称，可沿用但不要被其诱导: " + "、".join(map(str, name_hints[:40])))
    if context:
        lines.append("上下文: " + str(context)[:1000])
    lines.append("每个 idx 都必须输出；看不清、框不准、非家具设备、门/墙/地板/机器人自身均 keep=false。")
    return "\n".join(lines)


class OfflineDirector:
    """Deterministic fallback for Claude quota outages.

    It keeps the same narrow interfaces used by explore/semantic_route:
    choose one candidate id, skip optional semantic consolidation, and produce a
    conservative landmark route from the already-curated map. It never calls a
    cloud model and never emits coordinates for route legs.
    """

    def plan_coverage(self, payload: dict) -> dict:
        cands = payload.get("candidates") or []
        if not cands:
            return {"done": True, "target_id": None, "rationale": "offline: no candidates"}
        return {"done": False, "target_id": cands[0].get("id"), "rationale": "offline: first reachable frontier"}

    def pick_viewpoint(self, payload: dict) -> dict:
        cands = payload.get("candidates") or []
        if not cands:
            return {"target_id": None, "rationale": "offline: no candidates"}
        chosen = min(cands, key=lambda c: (
            float(c.get("potential", 0.0) or 0.0),
            -float(c.get("clearance_m", 0.0) or 0.0),
        ))
        return {"target_id": chosen.get("id"), "rationale": "offline: min potential + max clearance"}

    def consolidate_memory(self, records: list) -> dict:
        return {}

    def curate_area(self, payload: dict) -> dict:
        return {}

    def plan_route(self, payload: dict) -> dict:
        return _offline_route_plan(payload)


def make_director(model: str = None):
    """Return AnthropicProvider unless Claude is disabled/unavailable."""
    disabled = bool(getattr(config, "DISABLE_CLAUDE", False))
    # Keep env compatibility without adding another config constant to older callers.
    import os
    disabled = disabled or os.environ.get("DISABLE_CLAUDE", "").lower() in ("1", "true", "yes", "on")
    disabled = disabled or os.environ.get("CLOUD_DIRECTOR", "").lower() in ("offline", "none", "local")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if disabled or not has_key:
        print("[云端规划] Claude disabled/unavailable → using OfflineDirector")
        return OfflineDirector()
    try:
        return AnthropicProvider(model=model) if model else AnthropicProvider()
    except Exception as e:  # noqa: BLE001
        print(f"[云端规划] AnthropicProvider unavailable({type(e).__name__}: {str(e)[:80]}) → OfflineDirector")
        return OfflineDirector()


def _offline_route_plan(payload: dict) -> dict:
    target_text = str(payload.get("target") or "")
    start = payload.get("start_pose") or {}
    sx, sy = float(start.get("x", 0.0) or 0.0), float(start.get("y", 0.0) or 0.0)
    landmarks = [lm for lm in (payload.get("landmarks") or []) if lm.get("name")]
    sub_areas = [sa for sa in (payload.get("sub_areas") or []) if sa.get("label")]

    target_landmarks = [lm for lm in landmarks if str(lm.get("name")) in target_text]
    target_lm = None
    if target_landmarks:
        if "最东" in target_text or "东" in target_text:
            target_lm = max(target_landmarks, key=lambda lm: float(lm.get("x", sx) or sx))
        elif "最西" in target_text or "西" in target_text:
            target_lm = min(target_landmarks, key=lambda lm: float(lm.get("x", sx) or sx))
        else:
            target_lm = min(target_landmarks, key=lambda lm: math.hypot(
                float(lm.get("x", sx) or sx) - sx, float(lm.get("y", sy) or sy) - sy))
    tx = float((target_lm or {}).get("x", sx) or sx)

    legs = []
    named_target_areas = [sa for sa in sub_areas if str(sa.get("label")) in target_text]
    target_area = named_target_areas[0] if named_target_areas else None
    target_label = target_area.get("label") if target_area else None

    bridges = []
    for sa in sub_areas:
        label = sa.get("label")
        center = sa.get("center") or {}
        if not label or label == target_label or center.get("x") is None:
            continue
        x = float(center.get("x"))
        if (sx <= x <= tx) or (tx <= x <= sx):
            bridges.append(sa)
    bridges.sort(key=lambda sa: abs(float((sa.get("center") or {}).get("x", sx)) - sx))
    for sa in bridges[:2]:
        legs.append({"via": sa["label"], "manner": "at", "note": "offline中转子区"})

    # 门到门：目标跨隔断时，插一条"最近门"桥接（沿 start→target 的 x 跨度选带 label 的门，确定性）。
    span_doors = []
    for d in (payload.get("doors") or []):
        pose = d.get("pose") or {}
        label = d.get("label") or d.get("id")
        dx = pose.get("x")
        if not label or dx is None:
            continue
        if (sx <= float(dx) <= tx) or (tx <= float(dx) <= sx):
            span_doors.append((abs(float(dx) - sx), label))
    if span_doors:
        span_doors.sort()
        legs.append({"via": span_doors[0][1], "manner": "at", "note": "offline穿隔断开口"})

    if target_label:
        legs.append({"via": target_label, "manner": "at", "note": "offline进入目标子区"})

    if target_lm:
        if "左" in target_text or "left" in target_text.lower():
            manner = "left"
        elif "右" in target_text or "right" in target_text.lower():
            manner = "right"
        elif "后" in target_text or "behind" in target_text.lower():
            manner = "behind"
        else:
            manner = "near"
        legs.append({"via": target_lm["name"], "manner": manner, "note": "offline目标收尾"})

    if not legs and landmarks:
        nearest = min(landmarks, key=lambda lm: math.hypot(
            float(lm.get("x", sx) or sx) - sx, float(lm.get("y", sy) or sy) - sy))
        legs.append({"via": nearest["name"], "manner": "near", "note": "offline最近地标兜底"})

    # Deduplicate adjacent equal via/manner pairs while preserving order.
    deduped = []
    seen = set()
    for leg in legs:
        key = (leg.get("via"), leg.get("manner"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(leg)
    return {"legs": deduped, "goal_note": "offline deterministic route"}


class AnthropicProvider:
    """Claude 记忆作者（主）。"""

    def __init__(self, model: str = None, api_key: str = None, max_tokens: int = None):
        if anthropic is None:
            raise RuntimeError("anthropic package is not installed")
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._model = model or config.ANTHROPIC_MODEL
        self._max_tokens = max_tokens or config.ANTHROPIC_MAX_TOKENS

    def annotate(self, image, context: dict) -> dict:
        """主路径：纯文本 JSON（实测网关用 tool_use 会回空 objects，纯 JSON 能列全物体）。
        次路径：tool_use（原生 Claude 环境）。"""
        msg = [{
            "role": "user",
            "content": [to_anthropic_image_block(image), {"type": "text", "text": _build_prompt(context)}],
        }]
        # —— 主：纯 JSON（不带 tools）——
        try:
            resp = self._client.messages.create(
                model=self._model, max_tokens=self._max_tokens,
                system=SYSTEM_RECORD_JSON, messages=msg)
            txt = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            obj = _extract_json(txt)
            if isinstance(obj, dict) and obj.get("objects"):
                return obj
        except Exception:  # noqa: BLE001
            obj = None
        # —— 次：tool_use（原生 Claude 强制结构化）——
        try:
            resp = self._client.messages.create(
                model=self._model, max_tokens=self._max_tokens, system=SYSTEM_RECORD,
                tools=[RECORD_TOOL], tool_choice={"type": "tool", "name": "record_memory"}, messages=msg)
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use" and block.name == "record_memory":
                    return dict(block.input)
        except Exception:  # noqa: BLE001
            pass
        # 兜底：返回主路径拿到的（可能 objects 空）或抛错
        if isinstance(obj, dict):
            return obj
        raise RuntimeError("AnthropicProvider.annotate 未能产出 JSON")
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

    def plan_coverage(self, payload: dict) -> dict:
        """Claude 象限调度官：探索覆盖规划（纯文本、不读图、结构性触发，非常规控制回路）。

        代码递给它一张紧凑符号地图（bbox + 四象限覆盖统计 + 物体世界坐标 + **代码筛好的候选格**），
        它只【从候选里选一个 id】指出下一步该去补哪个欠覆盖象限——绝不自己编坐标（防幻觉）。
        返回 {done, target_quadrant, target_id, reopen, rationale}；任何异常/超时 → 返回 {} 当作跳过本轮。
        """
        try:
            text = _build_plan_prompt(payload)
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=400,
                system=SYSTEM_PLAN_COVERAGE,
                messages=[{"role": "user", "content": text}],
            )
            raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            obj = _extract_json(raw or "")
            return obj if isinstance(obj, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def pick_viewpoint(self, payload: dict) -> dict:
        """Claude 观测点选择官（Task 2，每个 vantage 前一次，纯文本、看势场 ASCII 不读图）。

        代码用雷达势场筛好候选观测点，Claude 只【从候选选一个 id】站得离障碍适中、朝目标视野好——
        绝不产坐标（防幻觉，同 plan_coverage）。返回 {target_id, rationale}；异常/超时/空 → {}。
        """
        try:
            text = _build_viewpoint_prompt(payload)
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=300,
                system=SYSTEM_PICK_VIEWPOINT,
                messages=[{"role": "user", "content": text}],
            )
            raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            obj = _extract_json(raw or "")
            return obj if isinstance(obj, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def consolidate_memory(self, records: list) -> dict:
        """Claude 记忆整理官（Role-2，探索写盘前一次，纯文本、不读图）。

        records: [{id, name, x, y}, ...] —— 代码去重后仍可能同一物体被跨帧标成不同名。
        Claude 只按 id【分组 + 规范命名 + 标幻觉】：同一物体的不同叫法(桌子/办公桌/工位)并一组、
        不同类(桌子 vs 椅子)分开、明显不存在的放 drop。【绝不改坐标】——坐标由代码按组聚合。
        返回 {groups:[{name, ids:[...]}], drop:[ids]}；任何异常/超时 → 返回 {} 当作跳过（安全降级）。
        """
        try:
            text = _build_consolidate_prompt(records)
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=8192,   # 每物体一条 id 分组输出较长；给足以免截断成非法 JSON
                system=SYSTEM_CONSOLIDATE,
                messages=[{"role": "user", "content": text}],
            )
            if getattr(resp, "stop_reason", None) == "max_tokens":
                print("⚠️ consolidate_memory 输出被 max_tokens 截断 → JSON 不完整，跳过整理")
            raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            obj = _extract_json(raw or "")
            return obj if isinstance(obj, dict) else {}
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ consolidate_memory 失败({type(e).__name__}: {str(e)[:80]}) → 跳过整理")
            return {}

    def curate_area(self, payload: dict) -> dict:
        """Claude 记忆整理官（离线整理 pass，一次调用、纯文本、不读图）。

        把探索产出的扁平 objects[] 组织成词典式结构：A 功能子区、B 上下文纠错(改名)、
        C on/in/next_to 关系、D 对【代码检出的】离谱候选(共坐标/不可能尺寸)给合并/丢弃裁决。
        【绝不输出/改动坐标】——子区 range 由代码从成员 abs_pose 算；只按给定 id 引用。
        返回 {sub_areas,corrections,relations,merges,drops}；任何异常/超时/截断 → {} 当跳过。
        """
        try:
            text = _build_curate_prompt(payload)
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=config.CURATE_MAX_TOKENS,
                system=SYSTEM_CURATE,
                messages=[{"role": "user", "content": text}],
            )
            if getattr(resp, "stop_reason", None) == "max_tokens":
                print("⚠️ curate_area 输出被 max_tokens 截断 → JSON 不完整，跳过整理")
                return {}
            raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            obj = _extract_json(raw or "")
            return obj if isinstance(obj, dict) else {}
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ curate_area 失败({type(e).__name__}: {str(e)[:80]}) → 跳过整理")
            return {}

    def plan_route(self, payload: dict) -> dict:
        """Claude 语义路线规划官（读自己整理后的词典式地图，规划到目标的语义路线；不读图、不出坐标）。

        输入：整理后地图(sub_areas+landmarks+relations+doors) + 起点 + 目标描述。
        输出：有序语义 legs（每段 via=地标名/子区名 + manner=near/at/behind/left/right/front + note），
        由 Qwen 逐段语义导航、代码把 via+manner 解析成坐标并 VFH 避障。**绝不输出坐标**。
        返回 {legs:[...], goal_note}；任何失败/截断 → {}。
        """
        try:
            text = _build_route_prompt(payload)
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=1500,
                system=SYSTEM_ROUTE,
                messages=[{"role": "user", "content": text}],
            )
            if getattr(resp, "stop_reason", None) == "max_tokens":
                print("⚠️ plan_route 输出被 max_tokens 截断 → 跳过")
                return {}
            raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            obj = _extract_json(raw or "")
            return obj if isinstance(obj, dict) else {}
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ plan_route 失败({type(e).__name__}: {str(e)[:80]}) → 跳过")
            return {}


SYSTEM_PLAN_COVERAGE = (
    "你是室内机器人的【探索覆盖调度官】。你【看不到图像】——只收到代码算出的紧凑符号地图："
    "房间 bbox、occupancy 覆盖计数(free/occupied/visited/frontier)、按 NE/NW/SE/SW 切的四象限覆盖统计、"
    "已记录物体的世界坐标、以及一份【代码筛好的候选目标点】。\n"
    "候选是 **occupancy frontier 可达点**——已探明自由空间与【未知区】的交界；代码已确保每个都能 A* 走到"
    "（含穿隔断墙缺口进被遮挡区）。\n"
    "你的唯一任务：从 candidates 里【选一个 id】作为机器人下一个补扫目标，优先把覆盖推进到【未探索 / 被隔断"
    "遮挡】的方向（frontier 密、物体少的欠覆盖象限）。\n"
    "硬性规则：\n"
    "1. target_id 必须是 candidates 里真实存在的 id；【绝不自己编造坐标或不存在的 id】。\n"
    "2. 优先补【coverage 最低且 under_covered=true】的象限；被遮挡的凹区(物体少但可能藏东西)也值得去。\n"
    "3. frontier 已耗尽 / 候选都落在已充分覆盖处时，返回 done:true。\n"
    "只输出一个合法 JSON(不要代码块标记、不要解释)："
    "{\"done\":false,\"target_id\":\"c3\",\"rationale\":\"≤120字\"}"
)


SYSTEM_PICK_VIEWPOINT = (
    "你是室内机器人的【观测点选择官】。你【看不到图像】——只收到代码用雷达算出的局部【势场 ASCII 图】"
    "和一份【代码筛好的候选观测点】(每个含 id、世界坐标 x,y、clearance=离最近障碍距离米)。\n"
    "ASCII 图例：R=车当前位置 T=想观测的目标 数字=候选观测点 #=贴障碍(危险) +=近障碍带 .=开阔；北在上、东在右。\n"
    "你的唯一任务：从 candidates 里【选一个 id】作为机器人下一步要去站的观测点，要求：\n"
    "1. 离障碍适中(clearance 别太小=会贴墙/钻桌底看不清，也别太大=离目标太远)，站位开阔、朝 T 视野好；\n"
    "2. target_id 必须是 candidates 里真实存在的 id；【绝不自己编造坐标或不存在的 id】；\n"
    "3. 候选为空 / 都不理想时返回 {\"target_id\":null}，代码会用默认目标点兜底。\n"
    "只输出一个合法 JSON(不要代码块、不要解释)："
    "{\"target_id\":\"v2\",\"rationale\":\"≤80字\"}"
)


def _build_viewpoint_prompt(payload: dict) -> str:
    return "\n".join([
        "=== 局部势场图(北上/东右) ===",
        str(payload.get("ascii_field", "")),
        f"车位姿: {json.dumps(payload.get('pose', {}), ensure_ascii=False)}",
        f"目标点 target_xy(想观测的对象/区域): {json.dumps(payload.get('target_xy'), ensure_ascii=False)}",
        f"候选观测点(只能从这里选 id): {json.dumps(payload.get('candidates', []), ensure_ascii=False)}",
        "从候选里选一个站位最好的观测点 id（离障碍适中、朝目标视野好）。候选空/都不好则 target_id:null。只输出 JSON。",
    ])


SYSTEM_CONSOLIDATE = (
    "你是室内机器人的【记忆整理官】。你【看不到图像】。代码已经把探索记录**按位置聚成若干『簇』**，"
    "每簇含 id、中心坐标(x,y 米)、该位置多视角给出的**名字列表**（常互相矛盾，如同一张桌子被标成 "
    "桌子/办公桌/柜子）、以及**代表尺寸 size=[宽,高]米**（可能为 null=尺寸不可信）。\n"
    "你的任务：对**每个簇**判定『这个位置是什么物体』，给【一个】规范中文名：\n"
    "1. 每个簇就是一个物体的多视角重复——名字列表里互相矛盾正是同物误标。选出/纠正为最准确的【一个】规范中文名。\n"
    "2. 统一用规范名：办公桌/显示器/办公椅/柜子/沙发/绿植/门 等。\n"
    "3. 放进 drop 的两种情况：①整簇明显是幻觉/不存在；②**尺寸对判定的类别明显离谱**——用常识判断，"
    "如 办公桌 宽 <0.3m 或 >2.5m、显示器 宽 >1.2m、沙发 宽 <0.6m、办公椅/绿植 >1.5m 之类明显不合理。"
    "（size 为 null 时尺寸未知，**不要**因此 drop。）\n"
    "硬性规则：**每个簇只给一个 name**（同簇名字再矛盾也归并为一个物体，绝不因名字不同拆成多个）；"
    "只按簇 id 判定，【绝不输出坐标】。\n"
    "只输出一个合法 JSON(不要代码块、不要解释)："
    "{\"clusters\":[{\"id\":0,\"name\":\"办公桌\"},{\"id\":1,\"name\":\"显示器\"}],\"drop\":[7]}"
)


def _build_consolidate_prompt(clusters: list) -> str:
    lines = ["=== 位置簇清单(id | 中心 x,y 米 | 该位置多视角名字 | 代表尺寸[宽,高]米,null=不可信) ===",
             json.dumps(clusters, ensure_ascii=False),
             "对每个簇判定该位置是什么物体、给【一个】规范中文名(同簇名字矛盾也归并为一个)；"
             "整簇幻觉、或尺寸对类别明显离谱的 → 放 drop(size=null 不作删依据)。绝不输出坐标。只输出 JSON。"]
    return "\n".join(lines)


SYSTEM_CURATE = (
    "你是室内机器人的【记忆整理官】。你【看不到图像】——只收到代码算好的结构化物体清单"
    "(每条含 id、name、aliases、世界坐标 abs_pose{x,y,z} 米、尺寸 size 米、定性 spatial、confidence)、"
    "房间 boundary、doors，以及一份【代码检出的离谱候选 sanity_candidates】。\n"
    "探索作者(Qwen)只做了逐帧命名+去重，产出是一份【扁平、带噪、无结构】的清单。你要把它组织成"
    "『区域→子区→物体』的词典式记忆，让机器人之后能按功能区找到目标物体。\n"
    "\n"
    "★★ 最高红线：整理【绝不能让真实物体消失或被改错类】。宁可留一条低置信记录、也不要误删/误改。"
    "机器人靠这张地图找东西，抹掉一个真柜子/真显示器，比留一点噪声危害大得多。★★\n"
    "\n"
    "做四件事（按下面三层判据）：\n"
    "\n"
    "【第1层 归并同物】只把【近乎重合=同一物被多视角记成多条】的合并成一条：\n"
    "  · 两块『沙发』贴在同一处 → 合成 1 个沙发；\n"
    "  · 『显示器』与『tv』(或 monitor)紧挨同一处 → 同一物异名，合并并统一成规范名『显示器』。\n"
    "  判据：**只有位置近乎重合(代码已按 ≤合并半径 圈好并放进 sanity_candidates)才是同物**。\n"
    "\n"
    "【第2层 纠名(去牙，谨慎) + 同类成组】\n"
    "  · 纠名【只在有内部证据时做】：①别名冲突——name 与它自己的 aliases 指向不同类(如 name=绿植 但 "
    "aliases 含 柜子)，据其真实所在判定正确类；②尺寸对类别物理不可能。\n"
    "  · ❌【严禁】仅因某物体孤立在一堆异类邻居中，就把它改成邻居的多数类！办公区里的一个柜子【仍是柜子】，"
    "一排南墙柜子、隔断后的一片显示器都必须【原样保留】——按邻居改名会系统性抹掉少数类真物体、是最大的错。\n"
    "  · 【紧邻同类成排=多个真实实例，不要合并】：三个柜子并排、一排四把椅子，是不同的物体，各自坐标不同、"
    "各自都要能被找到。**把它们保留为独立物体，用第3层的 sub_area 归到同一子区**，绝不 merge 掉。\n"
    "\n"
    "【第3层 划分子区】把 object id 按空间位置+功能聚成若干子区(办公区/休息区/绿植角/门厅/储物区…)。"
    "每个子区给：中文 label、英文 type、member_ids、一句话 summary。桌+椅→子办公区；沙发/柜子围一起→休息/中心区。"
    "尽量让每个物体归属恰好一个子区；无法归类的 id 可不放。\n"
    "\n"
    "【关系 C】对明确成立的物体关系给 {subject_id, predicate, object_id}，predicate 只用 "
    "on/in/next_to/under/above(如 显示器 on 办公桌)。\n"
    "\n"
    "【候选裁决 D】只针对 sanity_candidates 里的 id 表态，按 kind：\n"
    "  · kind=name_dup / synonym_dup(一组近乎重合的同类物体)：判断是【同一物多视角】还是【多个真实实例】。"
    "同一物→merges:{keep_id, drop_ids[], name, why}(synonym_dup 的 name 用统一规范名)；"
    "确为多个真实实例→**不要合并**，可不表态(它们会各自留存并进子区)。\n"
    "  · kind=alias_conflict(name 与自身别名跨类冲突)：这是【真物体被误标】——据证据判该处真类→corrections 改名。"
    "**不要 drop**(删一个真物体比留点噪声危害大)；代码只允许改名不允许删它。\n"
    "  · kind=impossible_size(尺寸物理不可能=多半几何噪声) → 可 drops:{id, why}，或改成尺寸合理的类。"
    "【只有】impossible_size 候选可删。\n"
    "  · 你【只能】对 sanity_candidates 里出现过的 id 做 merge；drop 只对 impossible_size 候选。改名(corrections)"
    "应基于内部证据，【不要】凭邻居多数类翻转一个物体的类别(代码会拦截无据的类翻转)。\n"
    "\n"
    "硬性规则：\n"
    "1. 【绝不输出或修改任何坐标】——子区 range 由代码从成员 abs_pose 算，你只给 member_ids。\n"
    "2. 只用清单里【真实存在的 id】引用物体；你输出的每个 id 都必须在输入里出现过。\n"
    "3. 跨真类绝不合并(桌/椅/显示器/柜子彼此不同物)；有疑就【保留】不删不改。\n"
    "4. 只输出一个合法 JSON(不要代码块标记、不要任何解释)。\n"
    "输出格式：{\"sub_areas\":[{\"label\":\"办公区\",\"type\":\"office\",\"member_ids\":[\"o1\",\"o2\"],"
    "\"summary\":\"东侧办公桌+显示器+办公椅工位群\"}],"
    "\"corrections\":[{\"id\":\"o5\",\"new_name\":\"显示器\",\"why\":\"name=tv与别名显示器同物,统一规范名\"}],"
    "\"relations\":[{\"subject_id\":\"o3\",\"predicate\":\"on\",\"object_id\":\"o5\",\"why\":\"显示器立于桌面\"}],"
    "\"merges\":[{\"keep_id\":\"o7\",\"drop_ids\":[\"o8\"],\"name\":\"显示器\",\"why\":\"显示器与tv近乎重合为同一屏\"}],"
    "\"drops\":[{\"id\":\"o11\",\"why\":\"宽3.1m对办公桌物理不可能\"}]}"
)


def _build_curate_prompt(payload: dict) -> str:
    return "\n".join([
        f"区域 area: {payload.get('area')}  类型 type: {payload.get('type')}",
        f"房间边界 boundary: {json.dumps(payload.get('boundary', {}), ensure_ascii=False)}",
        f"门/开口 doors: {json.dumps(payload.get('doors', []), ensure_ascii=False)}",
        "=== 物体清单(只能从这里选 id；abs_pose/size 仅供你判断，绝不回填或修改) ===",
        json.dumps(payload.get("objects", []), ensure_ascii=False),
        "=== 代码检出的离谱候选 sanity_candidates(只能对这里的 id 做 merge/drop) ===",
        json.dumps(payload.get("sanity_candidates", []), ensure_ascii=False),
        "请把物体聚成功能子区(A)、纠正误标(B)、给关系(C)、对上面候选给合并/丢弃裁决(D)。"
        "绝不输出坐标。只输出一个 JSON。",
    ])


SYSTEM_ROUTE = (
    "你是室内机器人的【语义路线规划官】。你【看不到图像】——只收到你之前整理好的词典式地图："
    "功能子区(label/type/摘要/中心)、地标物体(名字+世界坐标)、物体关系、门/开口(带 label)，以及机器人起点与一个目标描述。\n"
    "机器人底层用 VFH 反应式避障(会自己绕开障碍/贴墙滑行)，但它【只会朝一个个语义地标推进】——所以你要"
    "把'从起点到目标'拆成一串【途经地标】，像给人指路：先到哪个地标附近、再到哪个、最后以什么方位靠近目标。\n"
    "输出有序 legs，每段：\n"
    "  · via：途经的【地标名(用地图里的物体名、子区 label、或门 label)】；\n"
    "  · manner：到该地标的方位/方式，只用 near(附近)/at(到该处)/behind(后面)/left(左侧)/right(右侧)/front(前面)；\n"
    "  · note：≤20字这段为什么这么走。\n"
    "规则：\n"
    "1. via 只能用地图里【真实出现的】子区 label、物体名、或【门 label】；不得编造。\n"
    "2. 按空间从起点【由近及远】串起来。**当起点与目标分处隔断/子区两侧时，必须先以 `via=门label,"
    " manner=at` 穿过那个开口（从哪个入口进哪个区要说清），再到门另一侧的地标**——别指望机器人自己"
    "找缝，明确给出该走哪个门。最后一段以目标描述要求的 manner 收尾。\n"
    "3. 【绝不输出任何坐标或距离】——via/manner 会由代码解析成坐标。\n"
    "4. 只输出一个合法 JSON(不要代码块、不要解释)。\n"
    "格式：{\"legs\":[{\"via\":\"沙发\",\"manner\":\"near\",\"note\":\"先到中央休息区\"},"
    "{\"via\":\"红柜隔断北口\",\"manner\":\"at\",\"note\":\"从北口穿隔断进办公区\"},"
    "{\"via\":\"东侧工位区\",\"manner\":\"at\",\"note\":\"进入工位区\"},"
    "{\"via\":\"绿植\",\"manner\":\"left\",\"note\":\"到最东绿植左侧\"}],\"goal_note\":\"≤30字总体思路\"}"
)


def _build_route_prompt(payload: dict) -> str:
    return "\n".join([
        f"区域『{payload.get('area')}』类型 {payload.get('type')}。{payload.get('summary','')}",
        f"机器人起点: {json.dumps(payload.get('start_pose', {}), ensure_ascii=False)}（朝东 +x）",
        f"【目标】: {payload.get('target')}",
        "=== 功能子区(label|type|中心x,y|摘要) ===",
        json.dumps(payload.get("sub_areas", []), ensure_ascii=False),
        "=== 地标物体(name|世界坐标 x,y) ===",
        json.dumps(payload.get("landmarks", []), ensure_ascii=False),
        "=== 门/开口(label 可直接作 via 目标；跨隔断/子区时用它穿过) ===",
        json.dumps(payload.get("doors", []), ensure_ascii=False),
        f"=== 物体关系 ===\n{json.dumps(payload.get('relations', []), ensure_ascii=False)}",
        "规划一条【途经地标 via + 方位 manner】的语义路线到目标，跨隔断先经门 label 穿开口，"
        "末段用目标要求的方位收尾。只用地图里真实的地标名/子区名/门 label，绝不输出坐标。只输出 JSON。",
    ])


def _build_plan_prompt(payload: dict) -> str:
    return "\n".join([
        "=== 房间覆盖现状(数字单位米，已 round 到 0.1) ===",
        f"bbox: {json.dumps(payload.get('bbox', {}), ensure_ascii=False)}",
        f"occupancy 覆盖计数: {json.dumps(payload.get('coverage', {}), ensure_ascii=False)}",
        f"四象限统计: {json.dumps(payload.get('quadrants', {}), ensure_ascii=False)}",
        f"已记录物体: {json.dumps(payload.get('objects', []), ensure_ascii=False)}",
        f"候选 frontier 目标点(只能从这里选 id): {json.dumps(payload.get('candidates', []), ensure_ascii=False)}",
        f"轮次: {payload.get('round')}  剩余轮次: {payload.get('rounds_left')}",
        "请选出下一个要去补扫的候选 id（frontier 耗尽/都覆盖够了就 done:true）。只输出 JSON。",
    ])


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
