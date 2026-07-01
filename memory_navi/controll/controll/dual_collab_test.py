#!/usr/bin/env python3
"""Qwen ↔ Claude 共享记忆【协作测试】（1 帧同视角，全走 MCP upsert_object）。

目的：在加几何回填之前，先钉死两模型在同一份本地记忆上的协作链路——
  · 格式一致（同一 C-format schema，读回不报错）
  · 互相读得懂（Claude 读 Qwen 写的 objects，沿用同名合并而非重复新增）
  · 不互擦（同一物体最终 verified_by 同时含 qwen + claude；Qwen 的 id 仍在）

流程：
  1) 抓 1 帧，存 /tmp/collab_frame.jpeg（供人工核验幻觉/命名）。
  2) Qwen(8B) 标注 → 逐物体 upsert_object(id=q{i}, verified_by=[qwen])。
  3) Claude 读 read_area_memory → 拿已有名单 → 看【同一帧】+ 补充 prompt →
     沿用同名补 aliases/roi、并补新物体 → 逐物体 upsert_object(verified_by=[claude])。
  4) 读回，打印每个对象 {name,id,verified_by,aliases,roi,confidence}，跑断言。

运行：conda run -n vllm python dual_collab_test.py
需：vLLM 8B(:8000) + MCP(:9000) + rosbridge(:9090) + ANTHROPIC_AUTH_TOKEN/BASE_URL 网关。
"""
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from agent_core import config
from agent_core.executor import Executor
from agent_core.cloud.memory_author import MemoryAuthor
from agent_core.cloud.providers import (
    AnthropicProvider, LocalVLMProvider, _extract_json,
)
from agent_core.image_utils import to_anthropic_image_block

AREA = "collab_test"          # 独立 area，不碰 explore_room / break_room
FRAME_OUT = "/tmp/collab_frame.jpeg"


def _hr(t):
    print("\n" + "=" * 70 + f"\n  {t}\n" + "=" * 70)


def _call_json(ros, name, args):
    """调 MCP 工具并把返回文本解析为 dict（FastMCP 把 dict 结果序列化成 JSON 文本）。"""
    out = ros.call(name, args)
    txt = (out.text or "").strip()
    try:
        return json.loads(txt)
    except (ValueError, TypeError):
        return {"_raw": txt, "_is_error": out.is_error}


def _reset_area():
    """清掉本 area 旧记录，保证干净测试。"""
    p = os.path.join(config.MEMORY_ROOT, config.ENV_NAME, AREA, "area.json")
    if os.path.exists(p):
        os.remove(p)
        print(f"  已清空旧记录 {p}")


# ---------- Claude 补充模式 prompt（测试内，不污染 provider 默认路径）----------
SUPPLEMENT_SYS = (
    "你是室内机器人的『记忆作者』，正在【接力补充】另一台模型已写好的房间记忆。\n"
    "下面给你：(1) 记忆里已登记的物体名单；(2) 当前相机的同一帧 RGB 图。\n"
    "你的任务：\n"
    " a) 对【已登记】的物体：沿用【完全相同的 name】（以便系统按名字合并，不要改名/翻译成别的词），"
    "为它补 aliases（中英文别名数组）、roi（归一化 bbox x,y,w,h，0~1）、并按需修正 confidence。\n"
    " b) 对图里【有但名单里没有】的显著物体：作为新物体补一条（至少 name + roi + confidence）。\n"
    "硬性要求：只输出一个合法 JSON（不要代码块/解释）：\n"
    "{\"objects\":[{\"name\":\"\",\"aliases\":[\"\"],\"roi\":{\"x\":0,\"y\":0,\"w\":0,\"h\":0},"
    "\"confidence\":0.8}]}\n"
    "abs_pose 一律不填（由几何管线回填）。name 必须与已登记物体逐字一致才会合并。"
)


def _claude_supplement(prov: AnthropicProvider, image, existing_names):
    """Claude 读已有名单 + 同帧 → 输出补充 objects（沿用同名 + aliases/roi + 新物体）。"""
    user_txt = (
        "记忆里已登记的物体 name 列表（请对这些沿用完全相同的 name）：\n"
        + json.dumps(existing_names, ensure_ascii=False)
        + "\n\n这是当前相机的同一帧图，请按系统指令补充/新增。"
    )
    msg = [{
        "role": "user",
        "content": [to_anthropic_image_block(image), {"type": "text", "text": user_txt}],
    }]
    resp = prov._client.messages.create(
        model=prov._model, max_tokens=prov._max_tokens,
        system=SUPPLEMENT_SYS, messages=msg,
    )
    txt = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    obj = _extract_json(txt)
    return obj.get("objects", []) if isinstance(obj, dict) else []


def main():
    _hr("Qwen ↔ Claude 共享记忆协作测试")
    _reset_area()

    ex = Executor()
    ros = ex.ros

    # 用 MemoryAuthor 复用抓帧逻辑（subscribe_once → 盘上降采样帧）
    qwen = LocalVLMProvider(base_url=config.VLLM_8B_BASE_URL)
    author = MemoryAuthor(ros, qwen, None)  # memory=None：本测试不走 author.record，只借 _grab_image
    image = author._grab_image()
    with open(FRAME_OUT, "wb") as f:
        f.write(base64.b64decode(image.b64))
    print(f"  抓帧 OK，存 {FRAME_OUT}（{len(image.b64)} b64 chars）")

    # ---------- ① Qwen 标注 → 逐物体 MCP upsert ----------
    _hr("① Qwen(8B) 标注并写入")
    raw_q = qwen.annotate(image, {"area_hint": AREA, "trigger": "collab_test"})
    objs_q = raw_q.get("objects", []) or []
    print(f"  Qwen 返回 {len(objs_q)} 个物体")
    for i, o in enumerate(objs_q):
        if not isinstance(o, dict) or not o.get("name"):
            continue
        payload = {
            "id": f"q{i}",
            "name": o["name"],
            "confidence": float(o.get("confidence", 0.6) or 0.6),
            "verified_by": ["qwen3-vl-8b"],
        }
        if isinstance(o.get("roi"), dict):
            payload["roi"] = o["roi"]
        if o.get("spatial"):
            payload["spatial"] = o["spatial"]
        r = _call_json(ros, "upsert_object",
                       {"area": AREA, "object_json": json.dumps(payload, ensure_ascii=False)})
        print(f"    [{payload['id']}] {payload['name']:<16} -> {r.get('action')} (n={r.get('n_objects')})")

    # ---------- ② Claude 读已有 + 补充 → 逐物体 MCP upsert ----------
    _hr("② Claude 读 read_area_memory → 补充并写回")
    mem = _call_json(ros, "read_area_memory", {"area": AREA})
    existing = [o.get("name") for o in mem.get("objects", []) if o.get("name")]
    print(f"  Claude 读到已有 {len(existing)} 物体：{existing}")

    claude = AnthropicProvider()
    objs_c = _claude_supplement(claude, image, existing)
    print(f"  Claude 补充返回 {len(objs_c)} 个物体")
    for o in objs_c:
        if not isinstance(o, dict) or not o.get("name"):
            continue
        payload = {
            "name": o["name"],
            "confidence": float(o.get("confidence", 0.7) or 0.7),
            "verified_by": ["claude-sonnet-4-6"],
        }
        if isinstance(o.get("aliases"), list):
            payload["aliases"] = o["aliases"]
        if isinstance(o.get("roi"), dict):
            payload["roi"] = o["roi"]
        r = _call_json(ros, "upsert_object",
                       {"area": AREA, "object_json": json.dumps(payload, ensure_ascii=False)})
        print(f"    {payload['name']:<16} -> {r.get('action')} (n={r.get('n_objects')})")

    # ---------- ③ 读回核验 ----------
    _hr("③ 读回最终记忆 + 断言")
    final = _call_json(ros, "read_area_memory", {"area": AREA})
    fobjs = final.get("objects", [])
    print(f"  最终 {len(fobjs)} 个对象：\n")
    both = same_name_dups = 0
    for o in fobjs:
        vb = o.get("verified_by", [])
        has_both = ("qwen3-vl-8b" in vb) and ("claude-sonnet-4-6" in vb)
        both += int(has_both)
        flag = "★双模型" if has_both else ("·" + ("/".join(s.split("-")[0] for s in vb)) if vb else "·无")
        print(f"   - {o.get('name','?'):<16} id={o.get('id',''):<4} "
              f"vby={vb} aliases={o.get('aliases',[])} roi={'有' if o.get('roi') else '无'} "
              f"conf={o.get('confidence')} [{flag}]")

    # 同名重复检测（合并失败 → 命名分歧）
    names = [(o.get("name") or "").strip().lower() for o in fobjs]
    same_name_dups = len(names) - len(set(names))

    _hr("结论")
    print(f"  · 总对象数：{len(fobjs)}")
    print(f"  · 被两模型共同标注(verified_by 含 qwen+claude)：{both}")
    print(f"  · 重名重复条目数(应为 0)：{same_name_dups}")
    print(f"  · 不互擦判定：{'PASS' if both > 0 else 'FAIL（无任何对象保留双方 verified_by → upsert 覆盖了数组）'}")
    print(f"  · 格式一致判定：{'PASS' if final.get('exists') and isinstance(fobjs, list) else 'FAIL'}")
    print(f"\n  人工核验：读 {FRAME_OUT} 对照上面 objects 是否真实、Qwen/Claude 命名是否对得上。")
    # 落一份机读结果
    with open("/tmp/collab_result.json", "w", encoding="utf-8") as f:
        json.dump({"qwen_objs": objs_q, "claude_objs": objs_c, "final": final}, f,
                  ensure_ascii=False, indent=2)
    print("  机读结果存 /tmp/collab_result.json")


if __name__ == "__main__":
    main()
