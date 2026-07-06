"""occupancy/coverage 栅格 + 轻量 A* 路由（几何核心，纯函数，可脱 ROS/LLM 单测）。

用途（Q1 覆盖修复，见 Process.md §10.3）：用 lidar 射线把房间**逐步**标成
unknown/free/occupied/visited，`frontier_cells` = 已知自由格邻接 unknown 的格 → 供覆盖候选生成，
破"只从已扫 bbox 派生候选"的自我强化偏置（被隔断遮挡区一旦从开口扫到，就作为 free 进 grid、
其邻接的 unknown 成 frontier，代码会被驱动去补）。

坐标约定同 navigator / potential_field / depth_projection：世界系 +X 东、+Y 北，yaw=0 朝 +X，+yaw CCW。
格 = (ix, iy) 整数索引；格中心世界坐标 = (ix*res, iy*res)。

状态优先级 visited > occupied > free > unknown（不降级写入）：
- 射线经过的格标 free；射线终点(命中)标 occupied（occupied 不被后续 free 覆盖=墙不被穿过它的射线擦掉）；
- 车所在格标 visited（visited 覆盖 occupied：真开进去过=那里不是墙，纠正误标）；
- unknown = 不在 grid 里（未观测）。
"""
import heapq
import math

FREE = "free"
OCCUPIED = "occupied"
VISITED = "visited"
UNKNOWN = "unknown"

_PRIORITY = {FREE: 1, OCCUPIED: 2, VISITED: 3}
_KNOWN_FREE = (FREE, VISITED)          # frontier/低覆盖的"已知可站/可过"基底
_NEIGH4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGH8 = _NEIGH4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


class OccGrid:
    """稀疏 occupancy 栅格：dict{(ix,iy): state}，不在表里 = unknown。"""

    def __init__(self, res_m=0.4):
        self.res = float(res_m)
        self.cells = {}

    # —— 坐标 <-> 格 ——
    def cell_of(self, x, y):
        return (int(round(float(x) / self.res)), int(round(float(y) / self.res)))

    def center(self, c):
        return (c[0] * self.res, c[1] * self.res)

    def state(self, c):
        return self.cells.get(c, UNKNOWN)

    def mark(self, c, st):
        """按优先级写入（不降级）：仅当新状态优先级 ≥ 现状态才覆盖。"""
        cur = self.cells.get(c)
        if cur is None or _PRIORITY[st] >= _PRIORITY[cur]:
            self.cells[c] = st

    def cells_by_state(self, *states):
        s = set(states)
        return [c for c, st in self.cells.items() if st in s]


def update_from_beams(grid, pose, beams, *, range_max_m=8.0, max_mark_m=6.0):
    """把一帧 lidar 射线投进栅格：沿每条射线标 free、命中点标 occupied。

    beams: [[bearing_deg(机体系,0=前,+左), dist_m], ...]；dist_m <=0 或 None = 无命中(自由到 max)。
    range_max_m: 传感器最大量程；max_mark_m: 单帧最远标记距离(限制远处不确定射线过度铺 free)。
    命中距离 > max_mark_m 视作"该向本帧不确定"，只标 free 到 max_mark_m、不标 occupied（避免远墙抖动）。
    纯几何、原地改 grid。返回标记的 (n_free_cells 估, n_occ) 供日志（近似）。
    """
    px, py = float(pose["x"]), float(pose["y"])
    yaw = float(pose.get("yaw_deg", pose.get("yaw", 0.0)) or 0.0)
    step = grid.res * 0.5                       # 沿射线采样步长（<格距，保证不漏格）
    n_occ = 0
    for b in beams:
        if not (isinstance(b, (list, tuple)) and len(b) >= 2):
            continue
        bearing, d = b[0], b[1]
        hit = isinstance(d, (int, float)) and d > 0
        reach = min(float(d), max_mark_m) if hit else min(range_max_m, max_mark_m)
        wa = math.radians(yaw + float(bearing))
        cos_a, sin_a = math.cos(wa), math.sin(wa)
        # 沿射线标 free（到 reach 之前；命中格留给下面标 occupied）
        n = max(1, int(reach / step))
        for i in range(1, n):
            t = i * step
            grid.mark(grid.cell_of(px + t * cos_a, py + t * sin_a), FREE)
        # 命中且在可信标记距离内 → 终点标 occupied（墙/障碍）
        if hit and float(d) <= max_mark_m:
            grid.mark(grid.cell_of(px + float(d) * cos_a, py + float(d) * sin_a), OCCUPIED)
            n_occ += 1
    return n_occ


def mark_visited(grid, pose):
    """车当前格标 visited（visited 覆盖 occupied：真到过=非墙）。返回该格。"""
    c = grid.cell_of(float(pose["x"]), float(pose["y"]))
    grid.mark(c, VISITED)
    return c


def frontier_cells(grid, *, diagonal=False):
    """frontier = 已知自由格(free/visited)且【邻接至少一个 unknown 格】的格集合。

    这是破 Q1 偏置的核心：只要从开口扫到遮挡区一小片 free，其与 unknown 的交界即成 frontier，
    覆盖回路据此被驱动去补——不再受"已扫 bbox 凸包"约束。
    """
    deltas = _NEIGH8 if diagonal else _NEIGH4
    out = set()
    for c, st in grid.cells.items():
        if st not in _KNOWN_FREE:
            continue
        for dx, dy in deltas:
            if (c[0] + dx, c[1] + dy) not in grid.cells:      # 邻居 unknown
                out.add(c)
                break
    return out


def low_coverage_cells(grid, *, radius_m=1.2):
    """低覆盖格：已知自由格但距任一 visited 格 > radius_m（远处观测到、没走近细看）。

    供候选生成"补细看"：这些格观测过但站得远，走近能提升召回/几何精度。无 visited 时返回空集。
    """
    visited = grid.cells_by_state(VISITED)
    if not visited:
        return set()
    rcells = radius_m / grid.res
    out = set()
    for c, st in grid.cells.items():
        if st not in _KNOWN_FREE:
            continue
        if min(math.hypot(c[0] - v[0], c[1] - v[1]) for v in visited) > rcells:
            out.add(c)
    return out


def observed_bbox(grid, *, margin_cells=0):
    """所有非 unknown 格的世界坐标 bbox {xmin,xmax,ymin,ymax}；空则 None。仅供边界展示/打分。"""
    if not grid.cells:
        return None
    xs = [c[0] for c in grid.cells]
    ys = [c[1] for c in grid.cells]
    r = grid.res
    return {"xmin": (min(xs) - margin_cells) * r, "xmax": (max(xs) + margin_cells) * r,
            "ymin": (min(ys) - margin_cells) * r, "ymax": (max(ys) + margin_cells) * r}


def coverage_summary(grid):
    """计数摘要（日志/payload）：{free, occupied, visited, frontier, unknown_adj}。"""
    n = {FREE: 0, OCCUPIED: 0, VISITED: 0}
    for st in grid.cells.values():
        if st in n:
            n[st] += 1
    fr = frontier_cells(grid)
    return {"free": n[FREE], "occupied": n[OCCUPIED], "visited": n[VISITED],
            "frontier": len(fr), "cells": len(grid.cells), "res_m": grid.res}


def astar(grid, start, goal, *, diagonal=True, max_expand=20000):
    """在【已知自由格(free/visited)】上做 A* 找路（格坐标）。返回格序列(含起终点)或 None。

    这是"门导向 waypoint 桥接"的路由核心：occupancy 把穿过开口的自由空间连通起来后，A* 自动绕过
    occupied 墙、经开口连到远侧候选——无需手标走廊、无需 Nav2 全栈。unknown 格不可走（未知≠自由，
    保守）；goal 必须已是已知自由格（frontier 候选本就是 free）。不可达→None（该候选暂时到不了，正确跳过）。
    """
    if start == goal:
        return [start]
    free = set(grid.cells_by_state(FREE, VISITED))
    free.add(start)                              # 起点(车所在)必在自由集
    if goal not in free:
        return None
    deltas = _NEIGH8 if diagonal else _NEIGH4
    def _h(c):
        return math.hypot(c[0] - goal[0], c[1] - goal[1])
    openh = [(_h(start), 0.0, start)]
    came = {}
    gscore = {start: 0.0}
    expanded = 0
    while openh:
        _, gc, cur = heapq.heappop(openh)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if gc > gscore.get(cur, 1e18):
            continue                             # 过期堆项
        expanded += 1
        if expanded > max_expand:
            return None
        for dx, dy in deltas:
            nb = (cur[0] + dx, cur[1] + dy)
            if nb not in free:
                continue
            ng = gc + math.hypot(dx, dy)
            if ng < gscore.get(nb, 1e18):
                gscore[nb] = ng
                came[nb] = cur
                heapq.heappush(openh, (ng + _h(nb), ng, nb))
    return None


def path_waypoints(grid, cell_path):
    """把 A* 格路径简化成世界坐标 waypoints（贪心 line_free 直线捷径合并共线段）。

    返回 [(x,y), ...]【不含起点】——直接喂 navigator.geo_route_around 逐段绕行。空/单点 → []。
    """
    if not cell_path or len(cell_path) < 2:
        return []
    pts = [grid.center(c) for c in cell_path]
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1:
            x0, y0 = out[-1]
            x1, y1 = pts[j]
            if line_free(grid, x0, y0, x1, y1):
                break
            j -= 1
        out.append(pts[j])
        i = j
    return [(round(x, 2), round(y, 2)) for x, y in out[1:]]


def line_free(grid, x0, y0, x1, y1, *, block_states=(OCCUPIED,)):
    """(x0,y0)→(x1,y1) 直线沿途是否不穿 occupied 格（供桥接/直达判定）。

    沿线按 res/2 采样查格状态；遇 block_states 返回 False。unknown 视作可疑但不阻断（返回 True 时
    调用方仍应保守）。纯查表、无 ROS。
    """
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(dist / (grid.res * 0.5)))
    for i in range(1, n):
        t = i / n
        if grid.state(grid.cell_of(x0 + t * (x1 - x0), y0 + t * (y1 - y0))) in block_states:
            return False
    return True
