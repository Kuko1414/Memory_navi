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


class AnthropicProvider:
    """Claude 记忆作者（主）。"""

    def __init__(self, model: str = None, api_key: str = None, max_tokens: int = None):
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
