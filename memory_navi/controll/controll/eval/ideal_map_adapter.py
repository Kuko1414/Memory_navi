#!/usr/bin/env python3
"""理想全局语义地图适配器：break_room_ground_truth.json → memory/sim/break_room_ideal/area.json。

为什么要它：磁盘上 break_room/area.json 是【缺口图】(删了红柜后办公区)，且任何 area.json 都没有
sub_areas。执行模式初期实验要在【完整理想地图】上验证链路(绕过探索幻觉)，故用 GT 答案 key 造一份
完整记忆：objects(含三株绿植，尤其目标 6.23,1.42 与 -5.78,-2.17) + doors(隔断开口) +
三个功能子区(起始休息区/红柜后办公区/西侧办公簇)。

GT schema(flat x,y / alias / match_names / route) → area.json schema(abs_pose{x,y,z} / aliases)
的转换在此完成。sub_areas 走整理层通道 apply_curation(range 由代码从成员坐标算)。

幂等：写独立 area 名 break_room_ideal，不碰既有缺口图 break_room。
运行：conda run -n vllm python eval/ideal_map_adapter.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                      # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core import config
from agent_core.memory.fs_memory import FsMemory

DEFAULT_GT = os.path.join(HERE, "break_room_ground_truth.json")
OUT_AREA = "break_room_ideal"
AREA_TYPE = "lounge"

# 区域键 → 子区标签/类型/摘要
_SUBAREAS = {
    "start": ("起始休息区", "lounge",
              "小车起点所在的休息区：沙发/厨台/矮柜/红柜(任务锚点 cabinet1)。东有 wall1 隔断挡直行。"),
    "office": ("红柜后办公区", "office",
               "红柜/浅紫隔断墙后(东侧)的办公区：办公桌/显示器/办公椅/键盘/橙色高柜/东北绿植。"),
    "west": ("西侧办公簇", "office",
             "起点背后(西侧 -x)隔断墙后的另一片办公区：显示器群/西南绿植。"),
}


def _obj(region, name, x, y, z=0.3, aliases=None, state=None):
    o = {"name": name, "abs_pose": {"x": float(x), "y": float(y), "z": float(z)},
         "aliases": aliases or [], "confidence": 1.0, "verified_by": ["ground_truth"]}
    if state:
        o["state"] = state
    return region, o


def _entries():
    """(region, object) 清单——手工从 GT 展开(坐标即 GT 真值)。"""
    e = []
    # ---- 起始休息区 ----
    e += [
        _obj("start", "沙发", 1.23, -2.0, 0.3, ["sofa", "米色沙发"]),
        _obj("start", "绿植", 2.38, -2.15, 0.3, ["盆栽", "potted plant"], state="起点右前方(SE)"),
        _obj("start", "厨台", -1.79, -2.25, 0.3, ["kitchen_counter", "水槽", "sink"]),
        _obj("start", "矮柜", -0.01, -2.49, 0.3, ["cabinet", "红门矮柜"]),
        _obj("start", "矮柜", -0.66, -2.49, 0.3, ["cabinet", "红门矮柜"]),
        _obj("start", "矮柜", -1.52, -2.50, 0.3, ["cabinet", "米色矮柜"]),
        _obj("start", "矮柜", -2.31, -2.50, 0.3, ["cabinet", "米色矮柜"]),
        _obj("start", "红柜", 2.62, 0.22, 0.0, ["红色柜子", "红柜子", "cabinet", "red_cabinet"],
             state="任务锚点 cabinet(1)，红门朝 -x，紧贴 wall1 西侧"),
        _obj("start", "红柜(北)", 1.40, 2.50, 0.3, ["北侧红柜", "cabinet4"],
             state="北侧另一红柜，勿与任务锚点混淆"),
    ]
    # ---- 红柜后办公区 ----
    e += [
        _obj("office", "办公桌", 3.18, 0.05, 0.3, ["桌子", "desk", "table"]),
        _obj("office", "办公桌", 3.18, -1.66, 0.3, ["桌子", "desk", "table"]),
        _obj("office", "显示器", 2.877, 0.36, 0.76, ["屏幕", "monitor", "screen"]),
        _obj("office", "显示器", 2.877, -0.18, 0.76, ["屏幕", "monitor", "screen"]),
        _obj("office", "显示器", 2.877, -1.29, 0.76, ["屏幕", "monitor", "screen"]),
        _obj("office", "显示器", 2.879, -1.79, 0.76, ["屏幕", "monitor", "screen"]),
        _obj("office", "键盘", 3.07, 0.10, 0.74, ["keyboard"]),
        _obj("office", "键盘", 2.98, -1.53, 0.74, ["keyboard"]),
        _obj("office", "办公椅", 4.02, -0.04, 0.3, ["椅子", "chair", "office chair"]),
        _obj("office", "办公椅", 3.94, -1.53, 0.3, ["椅子", "chair"]),
        _obj("office", "办公椅", 5.36, -1.48, 0.3, ["椅子", "chair"]),
        _obj("office", "办公桌", 6.14, -1.66, 0.3, ["桌子", "desk"]),
        _obj("office", "办公桌", 6.14, 0.05, 0.3, ["桌子", "desk"]),
        _obj("office", "显示器", 6.479, -1.79, 0.76, ["屏幕", "monitor"]),
        _obj("office", "显示器", 6.477, -1.29, 0.76, ["屏幕", "monitor"]),
        _obj("office", "橙色高柜", 6.65, -0.70, 1.6, ["cabinet2", "橙色柜"]),
        # ★目标1★ 东北绿植
        _obj("office", "绿植", 6.23, 1.42, 0.3, ["盆栽", "potted plant"], state="远东北绿植(执行目标1)"),
    ]
    # ---- 西侧办公簇 ----
    e += [
        _obj("west", "显示器", -2.84, 0.0, 0.76, ["屏幕", "monitor", "西区显示器群"],
             state="西隔断墙后 8 台显示器的代表地标"),
        # ★目标2★ 西南绿植
        _obj("west", "绿植", -5.78, -2.17, 0.3, ["盆栽", "potted plant"], state="远西南绿植(执行目标2)"),
    ]
    return e


def _doors():
    """隔断开口提示(几何从 GT interior_partitions 推导；带 id/label 供 Claude 门到门路由落点)。"""
    return [
        {"id": "door_0", "label": "红柜隔断北口", "pose": {"x": 2.65, "y": 1.30},
         "dir": "center", "count": 2,
         "reason": "红柜后隔断墙 wall1@x=2.65 北端开口(y>1.0)，从此绕到红柜后办公区"},
        {"id": "door_1", "label": "西隔断北口", "pose": {"x": -2.65, "y": 2.70},
         "dir": "left", "count": 1,
         "reason": "西隔断墙 wall@x=-2.65 北端开口，通往西侧办公簇"},
    ]


def build(gt: dict):
    """→ (base_record, curation_verdicts)。base_record 无 sub_areas(schema-clean)；子区走整理层。"""
    entries = _entries()
    objects, region_ids = [], {k: [] for k in _SUBAREAS}
    for i, (region, o) in enumerate(entries):
        oid = f"gt_{i:02d}"
        o = dict(o, id=oid)
        objects.append(o)
        region_ids[region].append(oid)

    env = (gt.get("room_envelope") or {}).get("outer_walls") or {}
    boundary = {"xmin": env.get("west_x", -6.17), "xmax": env.get("east_x", 6.68),
                "ymin": env.get("south_y", -2.9), "ymax": env.get("north_y", 4.8)}
    start = gt.get("robot_start") or {"x": -0.5, "y": -1.0, "yaw_deg": 0.0}

    base = {
        "area": OUT_AREA, "type": AREA_TYPE,
        "summary": "理想全局语义地图(由 break_room_ground_truth 适配)：3 子区、"
                   f"{len(objects)} 物体、含三株绿植。执行模式初期实验用，绕过探索幻觉。",
        "view_pose": {"x": start["x"], "y": start["y"], "yaw": start.get("yaw_deg", 0.0)},
        "objects": objects,
        "hazards": [],
        "doors": _doors(),
        "boundary": boundary,
    }
    verdicts = {
        "sub_areas": [
            {"label": _SUBAREAS[k][0], "type": _SUBAREAS[k][1],
             "member_ids": region_ids[k], "summary": _SUBAREAS[k][2]}
            for k in ("start", "office", "west") if region_ids[k]
        ],
        "relations": [], "corrections": [], "merges": [], "drops": [],
    }
    return base, verdicts


def seed(gt_path=DEFAULT_GT, out_area=OUT_AREA, *, mem=None, validate=True) -> str:
    """造理想图并落盘：upsert_area(base) → apply_curation(sub_areas)。返回 area.json 路径。"""
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    base, verdicts = build(gt)
    mem = mem or FsMemory(config.MEMORY_ROOT, config.ENV_NAME)

    mem_validator = curate_validator = None
    if validate:
        from agent_core.cloud.schema import validate_curated_record, validate_memory_record
        mem_validator = validate_memory_record
        curate_validator = validate_curated_record
        mem_validator(base)                          # base 记录 schema 校验(无 sub_areas)

    rec = mem.load_area(out_area)
    if rec is not None:
        # 幂等：清掉旧的再重建，避免跨轮 objects merge
        os.remove(mem._area_path(out_area))
    mem.upsert_area(out_area, base)
    path = mem.apply_curation(out_area, verdicts, author="ideal_map_adapter",
                              validator=curate_validator)
    return path


def main() -> int:
    path = seed()
    mem = FsMemory(config.MEMORY_ROOT, config.ENV_NAME)
    rec = mem.load_area(OUT_AREA)
    objs = rec.get("objects", [])
    plants = [o for o in objs if o.get("name") == "绿植"]
    print(f"[理想图] 写入 {path}")
    print(f"  物体 {len(objs)}、子区 {len(rec.get('sub_areas', []))}、门 {len(rec.get('doors', []))}")
    for sa in rec.get("sub_areas", []):
        print(f"    · {sa['label']}({sa['type']}) 成员{len(sa['member_ids'])} range={sa.get('range')}")
    print(f"  绿植 {len(plants)}：{[(o['abs_pose']['x'], o['abs_pose']['y']) for o in plants]}")
    tgt = {(6.23, 1.42), (-5.78, -2.17)}
    have = {(o['abs_pose']['x'], o['abs_pose']['y']) for o in plants}
    ok = tgt <= have
    print(f"  两执行目标绿植齐备：{ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
