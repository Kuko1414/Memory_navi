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
        idx = match_index(rec["objects"], obj_id=obj.get("id"), name=obj.get("name"),
                          abs_pose=obj.get("abs_pose"), tol=instance_tol_m)

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

    # ---- 整理层（Claude 整理官）写回 ----
    @staticmethod
    def _sub_area_range(members: list):
        """从成员 abs_pose 算 3D 包围盒（几何归代码）；无可用坐标返回 None。"""
        xs, ys, zs = [], [], []
        for o in members:
            ap = o.get("abs_pose") if isinstance(o, dict) else None
            if not isinstance(ap, dict):
                continue
            if ap.get("x") is not None:
                xs.append(ap["x"])
            if ap.get("y") is not None:
                ys.append(ap["y"])
            if ap.get("z") is not None:
                zs.append(ap["z"])
        if not xs or not ys:
            return None
        r = {"xmin": min(xs), "xmax": max(xs), "ymin": min(ys), "ymax": max(ys)}
        if zs:
            r["zmin"] = min(zs)
            r["zmax"] = max(zs)
        return r

    def apply_curation(self, area: str, verdicts: dict, *, record=None,
                       author: str = "claude_curate", validator=None) -> str:
        """把 Claude 整理官裁决应用到区域记录，单次原子写。

        verdicts = {
            "sub_areas":   [{label,type,member_ids[],summary}],   # A 功能子区
            "corrections": [{id,new_name,why}],                   # B 上下文纠错（改名）
            "relations":   [{subject_id,predicate,object_id,why}],# C 关系
            "merges":      [{keep_id,drop_ids[],name,why}],       # D 共坐标/重复合并
            "drops":       [{id,why}],                            # D 离谱剔除
        }
        record（可选）：调用方已 load+补 id 的记录；给了就用它（避免 reload 丢失内存中新分配的 id），
        否则从磁盘读。铁律：绝不改动任何物体的 abs_pose；sub_area.range 由代码从成员坐标算。
        调用方须已用 gate 把 merges/drops 限制在代码检出候选内。
        validator（可选，如 schema.validate_curated_record）在写盘前校验，避免耦合 cloud 层。
        返回文件路径。
        """
        rec = record if record is not None else self.load_area(area)
        if rec is None:
            raise FileNotFoundError(f"area 记录不存在: {area}")
        rec.setdefault("objects", [])
        by_id = {}
        for o in rec["objects"]:
            oid = (o.get("id") or "").strip() if isinstance(o, dict) else ""
            if oid:
                by_id[oid] = o
        corrections = list(rec.get("corrections") or [])
        dropped_ids = set()

        # 1) merges（先）：贡献集并入 keep、被并名转别名、可选改名；被并项待移除
        for m in verdicts.get("merges") or []:
            keep_id = (m.get("keep_id") or "").strip()
            keep = by_id.get(keep_id)
            if keep is None:
                continue
            merged = []
            for did in m.get("drop_ids") or []:
                did = (did or "").strip()
                src = by_id.get(did)
                if src is None or did == keep_id or did in dropped_ids:
                    continue
                _union_contrib(keep, src)
                if src.get("name"):
                    keep.setdefault("aliases", []).append(src["name"])
                dropped_ids.add(did)
                merged.append(did)
            if not merged:
                continue
            new_name = (m.get("name") or "").strip()
            if new_name and new_name != (keep.get("name") or ""):
                keep["old_name"] = keep.get("name")
                keep.setdefault("aliases", []).append(keep.get("name"))
                keep["name"] = new_name
            _clean_aliases(keep)
            _append_unique(keep.setdefault("verified_by", []), author)
            corrections.append({"id": keep_id, "op": "merge", "merged_ids": merged,
                                "new_name": new_name or None, "why": m.get("why", "")})

        # 2) drops：仅登记；随后统一移除
        for d in verdicts.get("drops") or []:
            did = (d.get("id") or "").strip()
            if did and did in by_id and did not in dropped_ids:
                dropped_ids.add(did)
                corrections.append({"id": did, "op": "drop", "why": d.get("why", "")})

        if dropped_ids:
            rec["objects"] = [
                o for o in rec["objects"]
                if (o.get("id") or "").strip() not in dropped_ids
            ]
            by_id = {k: v for k, v in by_id.items() if k not in dropped_ids}

        # 3) renames(B)：原名转 old_name+别名、改名；幂等跳过 no-op；abs_pose 不动
        for c in verdicts.get("corrections") or []:
            oid = (c.get("id") or "").strip()
            o = by_id.get(oid)
            if o is None:
                continue
            new_name = (c.get("new_name") or "").strip()
            if not new_name or new_name == (o.get("name") or ""):
                continue
            old_name = o.get("name")
            o["old_name"] = old_name
            o.setdefault("aliases", []).append(old_name)
            o["name"] = new_name
            _clean_aliases(o)
            _append_unique(o.setdefault("verified_by", []), author)
            corrections.append({"id": oid, "op": "rename", "old_name": old_name,
                                "new_name": new_name, "why": c.get("why", "")})

        # 4) relations：过滤到 subject/object 都存活
        survivors = set(by_id.keys())
        relations = []
        for r in verdicts.get("relations") or []:
            s = (r.get("subject_id") or "").strip()
            t = (r.get("object_id") or "").strip()
            if s in survivors and t in survivors:
                rel = {"subject_id": s, "predicate": r.get("predicate"), "object_id": t}
                if r.get("why"):
                    rel["why"] = r["why"]
                relations.append(rel)
        rec["relations"] = relations

        # 5) sub_areas：过滤成员到存活 id；range 由代码算
        sub_areas = []
        for i, sa in enumerate(verdicts.get("sub_areas") or []):
            member_ids = [
                mid for mid in (sa.get("member_ids") or [])
                if (mid or "").strip() in survivors
            ]
            if not member_ids:
                continue
            members = [by_id[mid] for mid in member_ids]
            sa_type = (sa.get("type") or "area").strip() or "area"
            entry = {
                "id": f"sa_{sa_type}_{i}",
                "label": sa.get("label", ""),
                "type": sa_type,
                "range": self._sub_area_range(members),
                "member_ids": member_ids,
            }
            if sa.get("summary"):
                entry["summary"] = sa["summary"]
            sub_areas.append(entry)
        rec["sub_areas"] = sub_areas

        rec["corrections"] = corrections
        rec["curated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if validator is not None:
            validator(rec)  # 写盘前校验（不合法抛异常，文件不动）

        p = self._area_path(area)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        _atomic_write(p, json.dumps(rec, ensure_ascii=False, indent=2))
        return p


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


def match_index(objects, *, obj_id=None, name=None, abs_pose=None, tol=INSTANCE_TOL_M):
    """在 objects 里找与 (id / name+abs_pose) 匹配的记忆物体【下标】（无匹配 None）。

    id 优先精确匹配（给了 id 但没命中即视为新实例，不再退化按名）；否则同名(归一化)中
    abs_pose 平面距离 ≤ tol 取【最近】的一条；缺 abs_pose 无法判实例时退化为同名最早一条。
    这是 upsert_object 写路径与只读 grounding(match_object) 共用的单一匹配语义。
    """
    oid = (obj_id or "").strip()
    if oid:
        for i, e in enumerate(objects):
            if isinstance(e, dict) and (e.get("id") or "").strip() == oid:
                return i
        return None
    nn = _norm_name(name)
    if not nn:
        return None
    pos_idx, pos_d, name_idx = None, None, None
    for i, e in enumerate(objects):
        if not isinstance(e, dict) or _norm_name(e.get("name")) != nn:
            continue
        d = _abs_dist(abs_pose, e.get("abs_pose"))
        if d is None:
            if name_idx is None:
                name_idx = i                        # 同名但无法定位 → 备选
        elif d <= tol and (pos_d is None or d < pos_d):
            pos_d, pos_idx = d, i                     # 同名且近 → 同实例（取最近）
    return pos_idx if pos_idx is not None else name_idx


def match_object(objects, abs_pose=None, name=None, *, obj_id=None, tol=INSTANCE_TOL_M):
    """只读接地：给 abs_pose(+name / id) 返回记忆里匹配的物体 dict（含其 id），无匹配 None。

    供 bbox→深度→世界坐标后"这落点是记忆里哪个物体"的身份确认（不写盘）。语义同 match_index。
    """
    idx = match_index(objects or [], obj_id=obj_id, name=name, abs_pose=abs_pose, tol=tol)
    return objects[idx] if idx is not None else None


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


def _append_unique(lst: list, item) -> None:
    """把 item 追加进 lst（保序去重、忽略空值）。"""
    if item and item not in lst:
        lst.append(item)


def _union_contrib(keep: dict, src: dict) -> None:
    """把 src 的贡献集字段(_UNION_KEYS)并入 keep（保序去重，不动其余标量/几何）。"""
    for k in _UNION_KEYS:
        sv = src.get(k)
        if not sv:
            continue
        add = sv if isinstance(sv, list) else [sv]
        old = keep.get(k) or []
        if not isinstance(old, list):
            old = [old]
        for it in add:
            if it not in old:
                old.append(it)
        keep[k] = old


def _clean_aliases(o: dict) -> None:
    """就地按归一化去重 aliases 并剔除等于主名的项；空则删键。"""
    if o.get("aliases"):
        o["aliases"] = _dedup_aliases(o["aliases"], o.get("name"))
        if not o["aliases"]:
            o.pop("aliases", None)


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
