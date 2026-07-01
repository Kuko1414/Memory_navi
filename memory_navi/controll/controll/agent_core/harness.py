"""route-B 任务编排 harness：Qwen 规划子目标 → 代码逐步执行 + 真值核验 → 杜绝幻觉。

为什么这样设计（见 Report/harness_eval.md）：
- 单段对话让 Qwen 自跑整任务会幻觉（叙述代替指令 / 没做到却说做到 / 看图说谎）。
- 本 harness 把"控制流 + 动作执行 + 验证"交给确定性代码，Qwen 只负责①规划分解、②看图描述：
  * 规划器：把任务拆成原语步骤清单（结构化 JSON）。
  * 执行器：move/turn 由代码 ros.call 闭环下发（agent_actions 已闭环到真值），动作后用 get_pose 核验；
    绝不采纳模型"我已完成"的叙述。
  * 看图/报告：用一次全新短上下文 run（auto），代码核验本轮确实调了 look，再用返回的新帧描述。
vLLM(hermes) 不支持 tool_choice="required"，故不依赖强制工具，靠"代码执行动作 + 代码核验"达到等效确定性。
"""
import json
import math
import re

from agent_core.executor import Executor
from agent_core.image_utils import to_openai_image_url

PRIMITIVES = ("look", "move", "turn_left", "turn_right", "report")

PLANNER_SYS = (
    "你是室内机器人任务规划器。把用户任务拆成有序的原子步骤，只能用这些原语：\n"
    "- look：观察当前画面\n"
    "- move：前进，args {\"distance_m\": 数值}\n"
    "- turn_left：左转，args {\"degrees\": 数值}\n"
    "- turn_right：右转，args {\"degrees\": 数值}\n"
    "- report：描述当前所见\n"
    "规则：\n"
    "1) 忠实保留并落实任务里的方向词（左/右/前/后）。例如\"去某物左侧的区域\"="
    "先 turn_left 朝向左侧，再 move 前进过去；不要简化成只 move 前进。\n"
    "2) 动作数值不必精确（代码会闭环执行并核验真值）。\n"
    "3) 需要观察先插 look；任务要求报告就以 report 结尾。\n"
    "只输出 JSON，不要任何其它文字：\n"
    "{\"steps\":[{\"action\":\"...\",\"args\":{...},\"why\":\"...\"}]}"
)

DESCRIBE_BRIEF = "用 look 看一眼，简要描述你现在看到的主要物体和它们的方位（前/左/右）。"


def _extract_obj(text: str) -> dict:
    """从模型输出里抠出第一个 JSON 对象。"""
    if not text:
        return {}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def plan_task(ex: Executor, task: str, max_tokens: int = 600) -> list:
    """让 Qwen 规划子目标清单（纯 chat，无工具）。返回合法 steps 列表。"""
    resp = ex.client.chat.completions.create(
        model=ex.model,
        messages=[
            {"role": "system", "content": PLANNER_SYS},
            {"role": "user", "content": f"任务：{task}"},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        stream=False,
    )
    obj = _extract_obj(resp.choices[0].message.content or "")
    steps = obj.get("steps", []) if isinstance(obj, dict) else []
    return [s for s in steps if isinstance(s, dict) and s.get("action") in PRIMITIVES]


def _pose(ex: Executor) -> dict:
    return json.loads(ex.ros.call("get_pose", {}).text)


def _yaw_diff(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def execute_step(ex: Executor, step: dict) -> dict:
    """执行单个子目标并代码核验真值。返回结构化结果（含 verified 标记）。"""
    action = step.get("action")
    args = step.get("args") or {}

    if action in ("look", "report"):
        res = ex.run(DESCRIBE_BRIEF, max_iters=3)
        called = [s["name"] for s in res.tool_trace]
        return {
            "action": action,
            "called": called,
            "looked": "look" in called,        # 代码核验：报告前确实抓了新帧
            "report": res.text,
            "verified": "look" in called,
        }

    p0 = _pose(ex)
    if action == "move":
        d = float(args.get("distance_m", 0.5))
        out = ex.ros.call("move", {"distance_m": d}).text
        target, kind = d, "disp"
    elif action == "turn_left":
        deg = float(args.get("degrees", 90))
        out = ex.ros.call("turn_left_deg", {"degrees": deg}).text
        target, kind = deg, "yaw"
    elif action == "turn_right":
        deg = float(args.get("degrees", 90))
        out = ex.ros.call("turn_right_deg", {"degrees": deg}).text
        target, kind = -deg, "yaw"
    else:
        return {"action": action, "error": "unknown action"}

    p1 = _pose(ex)
    disp = math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"])
    dyaw = _yaw_diff(p1["yaw_deg"], p0["yaw_deg"])
    # 代码核验：实际达成是否接近目标（move 看位移、turn 看航向；safety 提前停也算合理完成）
    measured = disp if kind == "disp" else dyaw
    tol = max(0.1, abs(target) * 0.25)
    verified = abs(measured - target) <= tol or "safety_stop" in out
    return {
        "action": action,
        "args": args,
        "tool_result": _extract_obj(out),
        "measured": round(measured, 3),
        "target": round(target, 3),
        "verified": verified,
        "pose": p1,
    }


# ---------------------------------------------------------------------------
# 感知反馈环（supervisor 真形态）：记忆注入 + 看一步→决策→执行→核验，去补全地图缺口
# ---------------------------------------------------------------------------
EXPLORE_SYS = (
    "你是室内机器人探索器。给你：本区域已知语义记忆、当前位姿、激光各扇区最近障碍距离、"
    "以及当前相机画面。你的目标是把自己移动到能看清【信息缺失区域】的视角，到达后描述看到了什么。\n"
    "每次只决定【下一个】动作，可选：\n"
    "- move：前进，args {\"distance_m\": 数值}（前方 front 扇区距离不足 0.4m 时别前进，先转向）\n"
    "- turn_left / turn_right：转向，args {\"degrees\": 数值}\n"
    "- look：原地不动再看一眼（基本不需要，每步本来就给你画面）\n"
    "- arrived：已经能看到目标区域 → 在 report 里详细描述你现在看到了什么\n"
    "规则：参考已知记忆里物体的方位绕开它们、朝缺失区域走；前方被挡就转向找通路；"
    "动作数值不必精确（代码会闭环执行并核验）。只输出 JSON，不要其它文字：\n"
    "{\"action\":\"...\",\"args\":{...},\"report\":\"\",\"reason\":\"...\"}"
)


def _decide_next(ex, goal, mem_ctx, scan_text, pose_text, history, image):
    """给 Qwen 当前感知 + 记忆，让它只决定下一个原语（结构化 JSON）。"""
    text = (
        f"目标：{goal}\n\n已知记忆：\n{mem_ctx}\n\n"
        f"当前位姿：{pose_text}\n障碍扇区：{scan_text}\n"
        f"最近几步：{history if history else '（无）'}\n\n"
        "看这张当前相机画面，决定下一个动作（JSON）。"
    )
    content = [{"type": "text", "text": text}]
    if image is not None:
        content.append(to_openai_image_url(image))
    resp = ex.client.chat.completions.create(
        model=ex.model,
        messages=[{"role": "system", "content": EXPLORE_SYS},
                  {"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=400,
        stream=False,
    )
    return _extract_obj(resp.choices[0].message.content or "")


def explore_and_report(goal: str, mem_ctx: str, max_steps: int = 12):
    """感知反馈环：循环 看→决策→执行→核验，直到 Qwen 报 arrived 或步数用尽。

    返回 (result_dict, executor)；调用方负责 ex.close() 并查看最终相机帧。
    """
    ex = Executor()
    steps = []
    arrived = False
    final_report = ""
    for i in range(1, max_steps + 1):
        look_out = ex.ros.call("look", {})
        img = look_out.images[0] if look_out.images else None
        scan = ex.ros.call("scan_summary", {}).text
        pose = json.loads(ex.ros.call("get_pose", {}).text)
        hist = "; ".join(steps[-5:])
        d = _decide_next(ex, goal, mem_ctx, scan, json.dumps(pose, ensure_ascii=False), hist, img)
        action = d.get("action")
        args = d.get("args") or {}

        if action == "arrived":
            arrived = True
            final_report = d.get("report") or d.get("reason") or ""
            steps.append(f"{i}:arrived")
            break
        if action in ("turn_left", "turn_right", "move", "look"):
            rec = execute_step(ex, {"action": action, "args": args})
            tag = rec.get("measured") if action != "look" else "looked"
            steps.append(f"{i}:{action}{args}->{tag} pose={pose['x']},{pose['y']},{pose['yaw_deg']}")
        else:
            steps.append(f"{i}:invalid({action})")

    if not arrived and not final_report:
        # 超时兜底：至少看一眼并描述当前所见，保证有输出
        res = execute_step(ex, {"action": "report"})
        final_report = res.get("report", "")
    ex.ros.call("stop", {})
    final_pose = json.loads(ex.ros.call("get_pose", {}).text)
    return {"goal": goal, "arrived": arrived, "final_report": final_report,
            "steps": steps, "final_pose": final_pose, "model": ex.model}, ex


# ---------------------------------------------------------------------------
# 语义补全巡检（分层 harness 的 "VLM 只做语义" 那一层）
# 代码已用 navigator.geo_goto/geo_face_point 把车确定性地导到观察点并朝向目标区；
# 这里 Qwen 只看图 + 引用代码给的 scan/depth 距离，如实列物体，不做任何导航决策。
# ---------------------------------------------------------------------------
# 注意：本提示【不透题】——不告诉 Qwen 这里有什么/应该看到什么/目标区域在哪，
# 只让它如实汇报当前画面里真正看到的物体和可通行方向。任何"期望物清单/区域提示"都会 priming 幻觉。
INSPECT_SYS = (
    "你是室内机器人的语义记录员，只做语义、不做导航。代码已把你移动并停在某处。\n"
    "给你：当前相机画面、激光障碍摘要(scan)、深度每列距离(depth，左→右，米，-1=该列无效)。\n"
    "任务：\n"
    "1) 如实列出你在画面里确实看到的【独立家具/设备/可移动物品】。\n"
    "2) 描述画面里哪些方向的地面看起来开阔、可安全通行。\n"
    "命名规则（重要）：\n"
    "- 一律用【规范的中文通用单数名】：沙发/显示器/办公桌/办公椅/柜子/绿植/厨台/水槽/键盘/吊灯 等。\n"
    "- 同一类物体每次都用【同一个名字】，不要换着叫（如别一会儿『柜子』一会儿『橱柜/抽屉柜/红色柜门』）；\n"
    "  颜色/位置写进 spatial 描述，不要塞进名字。看不准具体类别时用最接近的通用类名。\n"
    "【绝对不要记】：\n"
    "- 墙面/地板/天花板/踢脚线/梁/隔断矮墙 等【建筑表面】；阴影/反光/光斑等视觉假象；\n"
    "- 机器人【自身】可见的轮子/机身/底盘（名字含 机器人/机器人本体/robot/wheel/self 的一律不列）；\n"
    "- 门/通道/开口【不要作为 objects 列出】（方向另在 passable 报）。\n"
    "字段规则：\n"
    "- 每个物体给 name、bearing(left/center/right)、bbox_center([x,y] Qwen 归一化 0-1000，物体中心)、\n"
    "  roi({\"x\":,\"y\":,\"w\":,\"h\":} 物体外接框：左上角 x,y + 宽高 w,h，均 0-1000；要贴合物体本身，"
    "不要把背景/远墙框进来——框大了尺寸会算错)、confidence(0-1)。\n"
    "- 【不要自己报距离】——距离由系统按 bearing 从深度图读取，你只需把方位和框判断准。\n"
    "- 只报确实看清的；看不清/没把握/被遮挡就不要列，宁缺勿编；绝不列『猜应该有』的东西。\n"
    "- passable: [{direction:left/center/right, free:true/false, note:\"\"}] —— 哪侧地面开阔、哪侧被挡。\n"
    "- 若想看的东西明显偏在画面一侧，用 recenter_deg 给建议转向(+左/-右，度；不需要就 0)。\n"
    "只输出 JSON，不要其它文字：\n"
    "{\"objects\":[{\"name\":\"沙发\",\"bearing\":\"center\",\"bbox_center\":[500,500],"
    "\"roi\":{\"x\":420,\"y\":430,\"w\":160,\"h\":140},\"confidence\":0.8}],"
    "\"passable\":[{\"direction\":\"left\",\"free\":true,\"note\":\"\"}],"
    "\"recenter_deg\":0,\"note\":\"\"}"
)


def _band_distance(depths: list, bearing: str):
    """按 bearing 取 depth 列段(左/中/右 三等分)的中位有效距离。代码接地，不让模型自报数字。"""
    valid_all = [d for d in depths if isinstance(d, (int, float)) and d >= 0]
    if not depths:
        return None
    n = len(depths)
    third = max(1, n // 3)
    b = (bearing or "center").lower()
    if "left" in b or "左" in b:
        seg = depths[:third]
    elif "right" in b or "右" in b:
        seg = depths[-third:]
    else:
        seg = depths[third:n - third] or depths
    seg = sorted(d for d in seg if isinstance(d, (int, float)) and d >= 0)
    if seg:
        return round(seg[len(seg) // 2], 2)
    return round(min(valid_all), 2) if valid_all else None


def inspect_and_report(ex: Executor, *, area_hint: str = "", max_tokens: int = 1000,
                       name_hints: list = None) -> dict:
    """到位后的一次语义巡检：look + scan + depth 喂给 Qwen，让它报【物体+方位+bbox+可通行方向】，
    距离由代码按 bearing 从 depth 列接地（不采纳模型自报的数字，杜绝距离幻觉）。

    name_hints: 本区域【已知物体名】列表（如 ["红色柜子","沙发","绿植"]）。仅作【命名锚点】——
    如画面里看到这些已知物体，请沿用相同名称（减少"红柜→工作台"这类叫错）；不泄露未探索区内容。

    返回 {objects, passable, recenter_deg, note, scan, depth, pose, image, raw}。
    image 供调用方核验图文一致；导航与控制流都不在这里。
    """
    look = ex.ros.call("look", {})
    img = look.images[0] if look.images else None
    scan_text = ex.ros.call("scan_summary", {}).text
    depth_text = ex.ros.call("depth_summary", {}).text
    pose = _pose(ex)
    depths = []
    try:
        depths = (json.loads(depth_text) or {}).get("depths_m", []) or []
    except (ValueError, TypeError):
        depths = []

    # 命名锚点（不透题）：只给【当前区域已知物体名】帮助一致命名，不告诉它未探索区有什么
    name_line = ""
    if name_hints:
        name_line = ("本区域已知物体（若在画面里看到它们，请沿用这些名称，不要另起名）："
                     + "、".join(str(n) for n in name_hints) + "。\n")
    user = (
        f"{name_line}"
        f"scan(激光最近障碍/扇区)：{scan_text}\n"
        f"depth(每列中位距离, 左→右 米)：{depth_text}\n\n"
        "看这张当前相机画面，按规则列出物体+可通行方向（JSON）。"
    )
    content = [{"type": "text", "text": user}]
    if img is not None:
        content.append(to_openai_image_url(img))
    raw = ""
    for attempt in range(2):
        resp = ex.client.chat.completions.create(
            model=ex.model,
            messages=[{"role": "system", "content": INSPECT_SYS},
                      {"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=max_tokens,
            stream=False,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.endswith("}"): break
    obj = _extract_obj(raw)
    objects = obj.get("objects", []) if isinstance(obj, dict) else []
    grounded = []
    for o in objects:
        if not (isinstance(o, dict) and o.get("name")):
            continue
        o = dict(o)
        o["distance_m"] = _band_distance(depths, o.get("bearing"))    # 代码接地
        o["distance_src"] = "depth_band"
        # roi 规范化为 dict {x,y,w,h}（Qwen 可能给数组）；供代码算 size/abs_pose（贴物体的框）
        roi = o.get("roi")
        if isinstance(roi, (list, tuple)) and len(roi) == 4:
            roi = {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]}
        if isinstance(roi, dict) and all(k in roi for k in ("x", "y", "w", "h")):
            o["roi"] = {k: float(roi[k]) for k in ("x", "y", "w", "h")}
        else:
            o.pop("roi", None)
        # bbox_center: Qwen 归一化[0-1000]；缺失则用 roi 中心补（保持 completion 兼容）
        bc = o.get("bbox_center")
        if isinstance(bc, (list, tuple)) and len(bc) == 2:
            o["bbox_center"] = [round(float(bc[0])), round(float(bc[1]))]
        elif isinstance(o.get("roi"), dict):
            r = o["roi"]
            o["bbox_center"] = [round(r["x"] + r["w"] / 2.0), round(r["y"] + r["h"] / 2.0)]
        grounded.append(o)
    passable = []
    for p in (obj.get("passable", []) or []):
        if isinstance(p, dict) and p.get("direction"):
            passable.append({
                "direction": p["direction"],
                "free": bool(p.get("free", True)),
                "note": p.get("note", ""),
            })
    return {
        "objects": grounded,
        "passable": passable,
        "recenter_deg": float(obj.get("recenter_deg", 0) or 0) if isinstance(obj, dict) else 0.0,
        "note": obj.get("note", "") if isinstance(obj, dict) else "",
        "scan": scan_text,
        "depth": depth_text,
        "pose": pose,
        "image": img,
        "raw": raw,
    }


def verify_passable(ex: Executor, direction: str, clear_thresh_m: float = 0.6) -> dict:
    """scan 交叉验证 Qwen 说的"X 侧可通行"是否为真。

    direction: left/center/right → 映射到 scan_summary 8 扇区标签（front/front_left/front_right）。
    返回 {verified, reported_free, scan_dist_m, sector_label}。
    """
    label = {"left": "left", "center": "front", "right": "right"}.get(
        (direction or "").lower(), "front")
    scan = json.loads(ex.ros.call("scan_summary", {}).text)
    sectors = scan.get("sectors", {})
    d = sectors.get(label)
    ok = d is not None and d >= clear_thresh_m
    return {"verified": ok, "reported_free": True, "scan_dist_m": d,
            "sector_label": label, "threshold_m": clear_thresh_m}


# ---------------------------------------------------------------------------
# 语义记忆驱动探索（Qwen 读缺口记忆 → 出定性方向 → 代码 scan 门控执行）
# ---------------------------------------------------------------------------
INSPECT_MEM_SYS = (
    "你是室内机器人的探索指引员。给你：\n"
    "1) 本区域的已知语义记忆（文本，含『信息缺失』标记）\n"
    "2) 当前相机画面 + 激光障碍(scan) + 深度(depth)\n"
    "3) 【位置变化】上一轮 vs 本轮参照物的 bearing/depth 变化（如有）\n"
    "任务：根据记忆缺失 + 画面 + 位置变化，决定下一步。\n"
    "你可以用这些工具（在 JSON 的 action 字段指定）：\n"
    "- **go_look_behind**: 记忆里某物体「后方信息缺失」→ 选这个！"
    "  例: {\"action\":\"go_look_behind\",\"object_name\":\"红色柜子\",\"u\":300,\"v\":380}\n"
    "  u/v=bbox_center。代码全自动算观察点+导航。这是最优先的探索方式。\n"
    "- go_look_at: 到参照物正面去看。"
    "  例: {\"action\":\"go_look_at\",\"object_name\":\"办公桌\",\"u\":400,\"v\":200}\n"
    "- 以上都用不了时，用 direction_hint: left/center/right（系统小步探索）\n"
    "- 如果暂时不需要精确移动，用 direction_hint: left/center/right（定性方向，系统会小步移动）\n"
    "规则：\n"
    "- dist_m = 你想到达的位置离参照物多远（standoff，通常 2-3m，不小于 1.2m）。"
    "不要用当前 depth 读数作为 dist_m——那是你离物体现在的距离，不是你想到达的距离\n"
    "- direction 只选 left/right/front（物体可见侧，不放 behind）\n"
    "- spatial_intent: 你计划对参照物做什么？behind(绕到背面)/left_of(绕到左侧)/right_of(绕到右侧)/front(靠近正面)。只选一个词。\n"
    "- 选 arrived 的条件（满足任一即可）：\n"
    "  a) 画面出现记忆里没有的新家具/物体\n"
    "  b) 看到柜子背面/侧面而非正面红门，且能看到后方空间\n"
    "  c) 已明显绕过柜子，前方/侧方出现新开阔区域\n"
    "- 满足条件就果断 direction_hint: arrived\n"
    "只输出 JSON：{\"direction_hint\":\"left|center|right|arrived\","
    "\"spatial_intent\":\"behind|left_of|right_of|front\",\"reference_object\":\"\","
    "\"action\":\"nav_distance|nav_object|none\",...}"
)


def inspect_with_memory(ex: Executor, gapped_memory: dict, max_tokens: int = 400) -> dict:
    """读缺口记忆 + 看图 → 出定性方向建议。返回 {direction_hint, reason, arrival_condition,
    objects, passable, scan, depth, pose, image, raw}。
    """
    look = ex.ros.call("look", {})
    img = look.images[0] if look.images else None
    scan_text = ex.ros.call("scan_summary", {}).text
    depth_text = ex.ros.call("depth_summary", {}).text
    pose = _pose(ex)

    # 把缺口记忆转成精简文本（不透题——不注入坐标/答案）
    objs = [f"- {o.get('name')}: {o.get('spatial','')}" for o in gapped_memory.get("objects", [])]
    summary = gapped_memory.get("summary", "")
    mem_text = f"区域 {gapped_memory.get('area','?')}。{summary}\n" + "\n".join(objs)

    user = (
        f"已知记忆：\n{mem_text}\n\n"
        f"scan: {scan_text}\n"
        f"depth(左→右,米): {depth_text}\n\n"
        "看这张画面。记忆里标记了信息缺失区域。往哪走？（JSON）"
    )
    content = [{"type": "text", "text": user}]
    if img is not None:
        content.append(to_openai_image_url(img))
    resp = ex.client.chat.completions.create(
        model=ex.model,
        messages=[{"role": "system", "content": INSPECT_MEM_SYS},
                  {"role": "user", "content": content}],
        temperature=0.2, max_tokens=max_tokens, stream=False,
    )
    raw = resp.choices[0].message.content or ""
    obj = _extract_obj(raw)
    return {
        "direction_hint": (obj.get("direction_hint") or "").lower() if isinstance(obj, dict) else "",
        "reason": obj.get("reason", "") if isinstance(obj, dict) else "",
        "arrival_condition": obj.get("arrival_condition", "") if isinstance(obj, dict) else "",
        "objects": [], "passable": [],    # explore mode: 不要求物体清单
        "scan": scan_text, "depth": depth_text, "pose": pose, "image": img, "raw": raw,
    }


def explore_with_memory(ex: Executor, gapped_memory: dict, *,
                        max_steps: int = 8, step_m: float = 1.0,
                        max_no_progress: int = 3) -> dict:
    """语义记忆驱动探索回路：每步 Qwen 看记忆+画面 → 出方向 → scan门控 → 执行。

    返回 {arrived, steps, pose, status}。
    若 Qwen 连续 N 步无进展或 scan 全堵 → 返回 arrived=False，调用方回退 geo_route。
    """
    steps = []
    prev_pose = None
    no_progress = 0
    arrived = False
    status = "max_steps"

    for i in range(max_steps):
        rep = inspect_with_memory(ex, gapped_memory)
        direction = rep["direction_hint"]
        steps.append(f"{i}:hint={direction} reason={rep['reason'][:80]}")

        if direction == "arrived":
            arrived = True
            status = "arrived"
            steps.append(f"{i}:Qwen says arrived: {rep['arrival_condition']}")
            break

        # scan 门控
        v = verify_passable(ex, direction)
        if not v["verified"]:
            # 尝试备选方向
            alt = {"left": "center", "center": "left", "right": "center"}.get(direction)
            if alt:
                v2 = verify_passable(ex, alt)
                if v2["verified"]:
                    direction = alt
                    v = v2
                    steps.append(f"{i}:redirect {direction} (orig blocked scan={v2['scan_dist_m']}m)")
                else:
                    steps.append(f"{i}:blocked {direction}({v['scan_dist_m']}m) alt {alt}({v2['scan_dist_m']}m)")
                    status = "blocked"
                    break
            else:
                steps.append(f"{i}:blocked {direction} scan={v['scan_dist_m']}m")
                status = "blocked"
                break

        # 执行一步
        if direction == "left":
            ex.ros.call("turn_left_deg", {"degrees": 30})
        elif direction == "right":
            ex.ros.call("turn_right_deg", {"degrees": 30})
        mv = json.loads(ex.ros.call("move", {"distance_m": step_m}).text)
        traveled = mv.get("traveled_m", 0)
        steps.append(f"{i}:move {direction} traveled={traveled}/{step_m} {mv.get('status')}")

        # 卡死检测
        cur = _pose(ex)
        if prev_pose:
            disp = math.hypot(cur["x"] - prev_pose["x"], cur["y"] - prev_pose["y"])
            no_progress = 0 if disp > 0.15 else no_progress + 1
            if no_progress >= max_no_progress:
                status = "no_progress"
                break
        prev_pose = cur

    ex.ros.call("stop", {})
    final_pose = _pose(ex)
    return {"arrived": arrived, "steps": steps, "pose": final_pose, "status": status}


def sweep_observe(ex: Executor, max_sweeps: int = 11, deg_per_sweep: float = 30.0) -> dict:
    """环顾扫视：到位后每 deg_per_sweep° look 一帧，收集 objects，≥3 条或扫满即停，回正。

    返回 {objects, passable, sweeps, pose}。objects 已去重（同名合并保留高置信度）。
    """
    all_objects = []
    all_passable = []
    sweeps_done = 0
    for i in range(max_sweeps):
        rep = inspect_and_report(ex)
        all_objects.extend(rep.get("objects", []))
        all_passable.extend(rep.get("passable", []))
        sweeps_done = i + 1
        if len(set(o.get("name") for o in all_objects)) >= 3:
            break
        ex.ros.call("turn_left_deg", {"degrees": deg_per_sweep})
    # 回正
    ex.ros.call("turn_right_deg", {"degrees": deg_per_sweep * sweeps_done})
    # 去重：同名取高置信度
    by_name = {}
    for o in all_objects:
        n = o.get("name", "")
        if n not in by_name or o.get("confidence", 0) > by_name[n].get("confidence", 0):
            by_name[n] = o
    pose = _pose(ex)
    return {"objects": list(by_name.values()), "passable": all_passable[:5],
            "sweeps": sweeps_done, "pose": pose}


def run_demo(task: str, max_steps: int = 10):
    """规划 + 逐步执行 + 核验。返回 (result_dict, executor)；调用方负责 ex.close()。"""
    ex = Executor()
    plan = plan_task(ex, task)
    records = []
    for step in plan[:max_steps]:
        records.append(execute_step(ex, step))
    ex.ros.call("stop", {})
    result = {
        "model": ex.model,
        "task": task,
        "plan": plan,
        "records": records,
        "all_verified": all(r.get("verified") for r in records) if records else False,
    }
    return result, ex
