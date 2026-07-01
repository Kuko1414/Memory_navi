"""统一记忆 MCP 工具：Qwen 和 Claude 共用、读写同一份本地记忆。

记忆文件布局与 agent_core.memory.fs_memory.FsMemory 完全一致：
  <MEMORY_ROOT>/<ENV_NAME>/<area>/area.json   （C-format 记录）
本模块用自包含的原子写 + 时间戳，复刻 FsMemory 行为（MCP server 是独立 py3.10 包，
不便跨包 import controll 侧的 FsMemory），从而两侧读写互通。

工具（area 一律必填，不再硬编码 break_room）：
  - read_area_memory(area)         读完整 C-format 记录
  - upsert_object(area, object)    写/更新单个对象（按 id 或 name 合并）
  - get_room_boundary(area)        读房间粗边界
  - list_areas()                   列出所有已知区域
"""
import datetime
import json
import os
import tempfile

from fastmcp import FastMCP

from ros_mcp.utils.websocket import WebSocketManager

MEMORY_ROOT = os.environ.get("MEMORY_ROOT", "/home/kuko/Kuko1414/memory_navi/memory")
ENV_NAME = os.environ.get("ENV_NAME", "sim")


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _area_path(area):
    return os.path.join(MEMORY_ROOT, ENV_NAME, area, "area.json")


def _load(area):
    p = _area_path(area)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError):
        return None


def _atomic_save(area, rec):
    """原子写：写 .tmp → os.replace（复刻 FsMemory._atomic_write）。设置 observed_at + 各对象 last_seen。"""
    rec = dict(rec)
    rec["observed_at"] = _now_iso()
    for o in rec.get("objects", []) or []:
        if isinstance(o, dict) and not o.get("last_seen"):
            o["last_seen"] = rec["observed_at"]
    p = _area_path(area)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return p


# 协作合并：这些字段是【多模型共同贡献】的集合，必须并集而非覆盖，
# 否则后写的模型(如 Claude)会擦掉先写模型(如 Qwen)的贡献——"一个写一个擦"。
_UNION_KEYS = ("verified_by", "aliases", "affordance")


def _merge_object(existing: dict, incoming: dict) -> dict:
    """把 incoming 合并进 existing：
      · 数组贡献字段(_UNION_KEYS)做去重并集(保序)；
      · 其余标量/对象字段用 incoming 覆盖(仅当 incoming 显式提供)。
    """
    for k, v in incoming.items():
        if k in _UNION_KEYS:
            old = existing.get(k) or []
            if not isinstance(old, list):
                old = [old]
            add = v if isinstance(v, list) else ([v] if v is not None else [])
            seen = list(old)
            for item in add:
                if item not in seen:
                    seen.append(item)
            existing[k] = seen
        else:
            existing[k] = v
    return existing


def register_memory_tools(mcp: FastMCP, ws_manager: WebSocketManager) -> None:
    """注册统一记忆工具（Qwen/Claude 共用）。"""

    @mcp.tool(
        description=(
            "Read the full semantic memory record (C-format) for a room/area. "
            "Returns {area, type, summary, view_pose, boundary, objects[], hazards[]}, "
            "or {exists:false} if the area has no memory yet. "
            "Use this to check what is already known about a room before exploring or navigating."
        ),
    )
    def read_area_memory(area: str) -> dict:
        rec = _load(area)
        if rec is None:
            return {"exists": False, "area": area, "objects": []}
        rec["exists"] = True
        return rec

    @mcp.tool(
        description=(
            "List all known areas/rooms that have a memory record on disk. "
            "Returns {areas: [name, ...]}."
        ),
    )
    def list_areas() -> dict:
        base = os.path.join(MEMORY_ROOT, ENV_NAME)
        out = []
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                if os.path.exists(os.path.join(base, name, "area.json")):
                    out.append(name)
        return {"areas": out}

    @mcp.tool(
        description=(
            "Get the coarse room boundary (axis-aligned bbox) recorded for an area. "
            "Returns {boundary:{xmin,xmax,ymin,ymax}} or {boundary:null} if not recorded."
        ),
    )
    def get_room_boundary(area: str) -> dict:
        rec = _load(area)
        return {"area": area, "boundary": (rec or {}).get("boundary")}

    @mcp.tool(
        description=(
            "Insert or update ONE object in a room's semantic memory. "
            "object_json is a JSON object with fields: name (required), id, aliases[], "
            "abs_pose{x,y,z}, size{width_m,height_m,depth_m}, roi{x,y,w,h}, confidence, "
            "state, affordance[], spatial. Merges by id if given, else by name (case-insensitive). "
            "Provided fields overwrite; omitted fields are kept from the existing record. "
            "Creates the area record if it does not exist yet."
        ),
    )
    def upsert_object(area: str, object_json: str) -> dict:
        try:
            obj = json.loads(object_json) if isinstance(object_json, str) else object_json
        except (ValueError, TypeError):
            return {"ok": False, "error": "object_json must be valid JSON"}
        if not isinstance(obj, dict) or not obj.get("name"):
            return {"ok": False, "error": "object must be a dict with at least a name"}

        rec = _load(area) or {"area": area, "type": "unknown", "summary": "", "objects": []}
        if not isinstance(rec.get("objects"), list):
            rec["objects"] = []

        oid = (obj.get("id") or "").strip()
        oname = (obj.get("name") or "").strip().lower()
        # 找已有对象：优先 id，否则名字不区分大小写
        idx = None
        for i, e in enumerate(rec["objects"]):
            if oid and (e.get("id") or "") == oid:
                idx = i
                break
            if not oid and (e.get("name") or "").strip().lower() == oname:
                idx = i
                break
        obj["last_seen"] = _now_iso()
        if idx is None:
            rec["objects"].append(obj)
            action = "inserted"
        else:
            _merge_object(rec["objects"][idx], obj)   # 标量覆盖；数组(verified_by/aliases/affordance)并集
            action = "updated"
        path = _atomic_save(area, rec)
        return {"ok": True, "action": action, "area": area, "name": obj["name"],
                "n_objects": len(rec["objects"]), "path": path}
