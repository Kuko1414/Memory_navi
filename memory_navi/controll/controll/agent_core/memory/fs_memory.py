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
import os
import tempfile
from collections import deque
from datetime import datetime, timezone


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
