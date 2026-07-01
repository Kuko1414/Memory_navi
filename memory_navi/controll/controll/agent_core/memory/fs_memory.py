"""文件系统式 Spatial Token Graph。

布局（落实"文件夹=拓扑词典、JSON=语义"）：
  <root>/<env>/
    topology.json        # 连通图边表 {"edges":[{from,to,via,direction,width_m,confidence}]}（BFS 用）
    roomA/area.json      # roomA 的 C 格式语义词条
    roomB/area.json

- 去 roomA 只读 roomA/area.json（按需查条目，稀疏低 token）。
- 连通关系单列根 topology.json 边表（纯文件夹嵌套只能表达树，房间连通是图）。
- 单写者（记忆作者）+ 多读者（agent），原子写（tmp + os.replace）。
"""
import json
import math
import os
import re
import tempfile
from collections import deque
from datetime import datetime, timezone

# 协作合并：这些字段是【多模型共同贡献】的集合，必须并集而非覆盖（否则后写者擦先写者）。
# 与 MCP 侧 memory_tools._merge_object 语义保持一致——全系统单一写盘语义。
_UNION_KEYS = ("verified_by", "aliases", "affordance")
INSTANCE_TOL_M = 0.8   # 同名物体 abs_pose 距离 ≤ 此值视作同一实例；更远=不同实例各自成条


class FsMemory:
    def __init__(self, root: str, env_name: str):
        self._dir = os.path.join(root, env_name)
        os.makedirs(self._dir, exist_ok=True)
        self._topology_path = os.path.join(self._dir, "topology.json")

    # ---- 区域语义词条 ----
    def _area_path(self, area: str) -> str:
        return os.path.join(self._dir, area, "area.json")

    def list_areas(self) -> list:
        if not os.path.isdir(self._dir):
            return []
        return sorted(
            d for d in os.listdir(self._dir)
            if os.path.exists(os.path.join(self._dir, d, "area.json"))
        )

    def load_area(self, area: str):
        p = self._area_path(area)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def upsert_area(self, area: str, record: dict) -> str:
        """原子写区域词条；补盖时间戳（observed_at / 每物体 last_seen）。"""
        record = dict(record)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record["observed_at"] = now  # 由存储层盖章真实记录时间（覆盖模型自填值）
        for obj in record.get("objects", []) or []:
            obj.setdefault("last_seen", now)
        p = self._area_path(area)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        _atomic_write(p, json.dumps(record, ensure_ascii=False, indent=2))
        return p

    def upsert_object(self, area: str, obj: dict, instance_tol_m: float = INSTANCE_TOL_M) -> str:
        """逐物体并集写入区域记录（数组字段并集、标量覆盖；同名远位=不同实例）。原子写。

        匹配键：id 优先；否则 name(忽略大小写) 且 abs_pose 距离 ≤ instance_tol_m。
        同名但坐标相距更远 → 视作不同实例各自成条（不被并掉，修复"两张 sofa 合一"）。
        缺 abs_pose 无法判实例时退化为按名合并最早一条（保守）。
        """
        rec = self.load_area(area) or {"area": area, "type": "unknown", "summary": "", "objects": []}
        rec.setdefault("objects", [])
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        oid = (obj.get("id") or "").strip()
        oname = (obj.get("name") or "").strip().lower()
        o_abs = obj.get("abs_pose")

        idx = None
        if oid:
            for i, e in enumerate(rec["objects"]):
                if isinstance(e, dict) and (e.get("id") or "").strip() == oid:
                    idx = i
                    break
        if idx is None and not oid and oname:
            pos_idx, pos_d, name_idx = None, None, None
            for i, e in enumerate(rec["objects"]):
                if not isinstance(e, dict) or (e.get("name") or "").strip().lower() != oname:
                    continue
                d = _abs_dist(o_abs, e.get("abs_pose"))
                if d is None:
                    if name_idx is None:
                        name_idx = i                       # 同名但无法定位 → 备选
                elif d <= instance_tol_m and (pos_d is None or d < pos_d):
                    pos_d, pos_idx = d, i                   # 同名且近 → 同实例（取最近）
            idx = pos_idx if pos_idx is not None else name_idx

        obj = dict(obj)
        obj["last_seen"] = now
        if idx is None:
            rec["objects"].append(obj)                     # 新实例
        else:
            _merge_object(rec["objects"][idx], obj)         # 并集合并
        rec["observed_at"] = now
        p = self._area_path(area)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        _atomic_write(p, json.dumps(rec, ensure_ascii=False, indent=2))
        return p

    # ---- 拓扑 ----
    def load_topology(self) -> dict:
        if not os.path.exists(self._topology_path):
            return {"edges": []}
        with open(self._topology_path, encoding="utf-8") as f:
            return json.load(f)

    def add_edge(self, frm: str, to: str, **attrs) -> None:
        topo = self.load_topology()
        edge = {"from": frm, "to": to}
        edge.update(attrs)
        topo.setdefault("edges", []).append(edge)
        _atomic_write(self._topology_path, json.dumps(topo, ensure_ascii=False, indent=2))

    def topology_path(self, frm: str, to: str):
        """BFS 求区域路径（边按无向看待）；不可达返回 None。"""
        if frm == to:
            return [frm]
        adj: dict = {}
        for e in self.load_topology().get("edges", []):
            adj.setdefault(e["from"], set()).add(e["to"])
            adj.setdefault(e["to"], set()).add(e["from"])
        q = deque([[frm]])
        seen = {frm}
        while q:
            path = q.popleft()
            for nxt in adj.get(path[-1], ()):
                if nxt in seen:
                    continue
                if nxt == to:
                    return path + [nxt]
                seen.add(nxt)
                q.append(path + [nxt])
        return None


def _abs_dist(a, b):
    """两 abs_pose 的 xy 平面距离；任一缺坐标返回 None（无法判同实例）。"""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    if a.get("x") is None or b.get("x") is None:
        return None
    return math.hypot(a["x"] - b["x"], (a.get("y") or 0.0) - (b.get("y") or 0.0))


def _norm_name(n: str) -> str:
    """归一化物体名用于去重：小写、空白/下划线统一。"""
    return re.sub(r"[\s_]+", " ", (n or "").strip().lower())


def _dedup_aliases(aliases, main_name) -> list:
    """别名保序去重（按归一化名），剔除等于主名的项（修复重复 red_cabinet / 别名==主名）。"""
    out, seen = [], {_norm_name(main_name)}
    for a in aliases or []:
        na = _norm_name(a)
        if not na or na in seen:
            continue
        seen.add(na)
        out.append(a)
    return out


def _merge_object(existing: dict, incoming: dict) -> dict:
    """把 incoming 合并进 existing：数组贡献字段(_UNION_KEYS)去重并集(保序)，其余标量覆盖(仅当提供)。"""
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
    # aliases 最终按归一化去重并剔除主名（避免重复别名 / 别名==主名）
    if existing.get("aliases"):
        existing["aliases"] = _dedup_aliases(existing["aliases"], existing.get("name"))
        if not existing["aliases"]:
            existing.pop("aliases", None)
    return existing


def _atomic_write(path: str, text: str) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
