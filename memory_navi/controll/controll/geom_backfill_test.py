#!/usr/bin/env python3
"""几何回填测试（方法B size + abs_pose）。

两段：
  A) 直接对 collab_test 记忆里已有 ROI 调 MCP depth_roi → 打印每个 ROI 的深度统计 +
     roi_to_size + back_project 出的 size/abs_pose（隔离几何，不依赖标注）。
  B) 端到端：MemoryAuthor(Qwen).record("geom_test", view_pose=pose) 跑完整管线，
     读回 area.json 确认 size/abs_pose 被代码回填（不是模型瞎填）。

运行：conda run -n vllm python geom_backfill_test.py
需：vLLM 8B + MCP(含 depth_roi) + rosbridge + 深度相机在发 /<ns>/camera/depth/image。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from agent_core import config, harness
from agent_core.executor import Executor
from agent_core.geometry import depth_projection as dp
from agent_core.cloud.memory_author import MemoryAuthor
from agent_core.cloud.providers import LocalVLMProvider
from agent_core.memory.fs_memory import FsMemory


def _hr(t):
    print("\n" + "=" * 70 + f"\n  {t}\n" + "=" * 70)


def _call_json(ros, name, args):
    out = ros.call(name, args)
    try:
        return json.loads((out.text or "").strip())
    except (ValueError, TypeError):
        return {"_raw": out.text, "_is_error": out.is_error}


def _pose(ex, retries=4):
    import time
    for _ in range(retries):
        p = harness._pose(ex)
        if isinstance(p, dict) and "x" in p and "yaw_deg" in p:
            return p
        time.sleep(0.3)
    return None


def main():
    _hr("几何回填测试（方法B size + abs_pose）")
    ex = Executor()
    ros = ex.ros
    pose = _pose(ex)
    print(f"  当前位姿：{pose}")

    # ---------- A) 隔离几何：对 collab_test 已有 ROI 跑 depth_roi ----------
    _hr("A) collab_test 已有 ROI → depth_roi → size/abs_pose")
    mem = _call_json(ros, "read_area_memory", {"area": "collab_test"})
    objs = [o for o in mem.get("objects", []) if isinstance(o.get("roi"), dict)]
    print(f"  collab_test 带 ROI 的物体：{len(objs)}")
    if objs:
        rois = [o["roi"] for o in objs]
        dr = _call_json(ros, "depth_roi", {"rois_json": json.dumps(rois, ensure_ascii=False)})
        if not dr.get("ok"):
            print(f"  depth_roi 失败：{dr}")
        else:
            stats = dr.get("stats", [])
            print(f"  depth_roi OK（帧 {dr.get('width')}x{dr.get('height')}）\n")
            for o, st in zip(objs, stats):
                if not st:
                    print(f"   - {o['name']:<14} 无有效深度（太远/天空/桌面高处）")
                    continue
                size = dp.roi_to_size(o["roi"], st)
                ap = None
                if pose:
                    uc, vc = dp.roi_center_pixel(o["roi"])
                    try:
                        ap = dp.back_project(uc, vc, st["median_m"], robot_pose=pose)["abs_pose"]
                    except Exception as e:  # noqa: BLE001
                        ap = f"err:{e}"
                print(f"   - {o['name']:<14} d_med={st['median_m']}m near=[{st['near_min_m']},{st['near_max_m']}] "
                      f"n={st['n_valid']}")
                print(f"       size(W×H×厚)= {size['width_m']} × {size['height_m']} × {size['depth_m']} m   abs_pose={ap}")

    # ---------- B) 端到端 MemoryAuthor.record ----------
    _hr("B) 端到端：MemoryAuthor(Qwen).record('geom_test')")
    p = os.path.join(config.MEMORY_ROOT, config.ENV_NAME, "geom_test", "area.json")
    if os.path.exists(p):
        os.remove(p)
    qwen = LocalVLMProvider(base_url=config.VLLM_8B_BASE_URL)
    fsmem = FsMemory(config.MEMORY_ROOT, config.ENV_NAME)
    author = MemoryAuthor(ros, qwen, fsmem)
    res = author.record("geom_test", trigger="geom_test", view_pose=pose)
    print(f"  record ok={res.ok} error={res.error}")
    if res.ok:
        rec = res.record
        print(f"  写入 {len(rec.get('objects', []))} 物体，逐个看 size/abs_pose 是否被回填：\n")
        n_size = n_pose = 0
        for o in rec.get("objects", []):
            sz = o.get("size") or {}
            ap = o.get("abs_pose")
            has_sz = sz.get("width_m") is not None
            has_ap = bool(ap and ap.get("x") is not None)
            n_size += int(has_sz); n_pose += int(has_ap)
            print(f"   - {o.get('name','?'):<14} roi={'有' if o.get('roi') else '无'} "
                  f"size={'✓' if has_sz else '·'}({sz.get('width_m')}×{sz.get('height_m')}×{sz.get('depth_m')}) "
                  f"abs_pose={'✓' if has_ap else '·'}{ap}")
        print(f"\n  回填统计：size {n_size}/{len(rec.get('objects', []))}，abs_pose {n_pose}/{len(rec.get('objects', []))}")
        print(f"  area.json: {p}")


if __name__ == "__main__":
    main()
