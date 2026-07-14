"""补全模式（Mode 封装）：用整理后语义地图导航到缺失/误标目标物体，近观确认后补写记忆。

由 complete_target.py 的 main 体搬移而来（几何归代码、语义归 Qwen，绝不把答案坐标喂 Qwen）。
角色：Qwen-only，不调云；memory_access='write'（只 upsert_object 补一个物体，结构性禁止重写/整理）。
"""
import math

from agent_core import config, harness
from agent_core import navigator as nav
from agent_core.modes.base import Mode, ModeResult, RunContext

_DEFAULTS = {
    "target": "红柜",                                  # 地图里缺失/误标的目标物体
    "target_aliases": ["红色柜子", "red_cabinet", "cabinet", "柜子"],
    "cabinet_keys": ["柜", "cabinet", "红"],           # 到位观测里判定"找到柜子"的关键词
    "arrival_tol_m": 1.2,                              # 到位判定（补全用宽松阈值）
}


def _map_brief(rec):
    """地图子区结构摘要（供 Qwen 理解工作区，不含答案坐标）。"""
    lines = [f"区域『{rec.get('area')}』类型 {rec.get('type')}。{rec.get('summary','')}", "功能子区："]
    for sa in rec.get("sub_areas", []):
        lines.append(f"  · {sa.get('label')}({sa.get('type')})：{sa.get('summary','')}")
    return "\n".join(lines)


def _frontal_candidates(rec, pose, k=6):
    """从地图取【机器人前方(x≥当前)】的已知物体作为候选前往点(按距离近→远)。"""
    cands = []
    for o in rec.get("objects", []):
        ap = o.get("abs_pose") or {}
        if ap.get("x") is None:
            continue
        if ap["x"] < pose["x"] - 0.3:       # 只要前方/侧前方，排除身后
            continue
        d = math.hypot(ap["x"] - pose["x"], (ap.get("y") or 0) - pose["y"])
        cands.append((d, o))
    cands.sort(key=lambda t: t[0])
    out, seen = [], []
    for d, o in cands:
        ap = o["abs_pose"]
        if any(math.hypot(ap["x"] - s[0], ap["y"] - s[1]) < 0.6 for s in seen):
            continue
        seen.append((ap["x"], ap["y"]))
        out.append(o)
        if len(out) >= k:
            break
    return out


def _hr(t):
    print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


class CompletionMode(Mode):
    name = "completion"
    skill = "读整理后地图，导航到缺失/误标目标物体，近观确认后补写记忆（Qwen-only，不调云）"
    allowlist = config.ACTION_ALLOWLIST
    memory_access = "write"              # 只 upsert_object 补一个物体
    uses_cloud = False

    def do_run(self, ctx: RunContext) -> ModeResult:
        p = {**_DEFAULTS, **(ctx.params or {})}
        target = p["target"]
        target_aliases = p["target_aliases"]
        cabinet_keys = p["cabinet_keys"]
        arrival_tol = p["arrival_tol_m"]
        ex, mem, area = ctx.ex, ctx.mem, ctx.area

        _hr(f"补全模式：用整理后地图找缺失目标『{target}』")
        rec = mem.load_area(area)
        if rec is None:
            return ModeResult(self.name, "NO_MAP", False,
                              summary=f"找不到整理后地图 {area}", exit_code=2)
        names = {o.get("name") for o in rec.get("objects", [])}
        print(f"[地图] {len(rec.get('objects',[]))} 物体、{len(rec.get('sub_areas',[]))} 子区；"
              f"是否已含『{target}』：{'是' if target in names else '否（缺失，需补全）'}")

        pose0 = harness._pose(ex)
        print(f"[起点] pose={pose0}")
        cands = _frontal_candidates(rec, pose0)
        if not cands:
            return ModeResult(self.name, "NO_CANDIDATE", False,
                              summary="地图里没有前方候选物体可供导航", exit_code=1)

        # ---- 阶段1：Qwen 看地图+相机，从候选里选一个前往点去找目标 ----
        _hr("阶段1 Qwen 读地图选前往点")
        cand_lines = []
        for i, o in enumerate(cands):
            ap = o["abs_pose"]
            cand_lines.append(f"  [{i}] {o.get('name')} @世界坐标({ap['x']:.2f},{ap['y']:.2f})"
                              f" 别名{o.get('aliases') or []}")
        look = ex.ros.call("look", {})
        img = look.images[0] if look.images else None
        scan_t = ex.ros.call("scan_summary", {}).text

        user = (f"{_map_brief(rec)}\n\n"
                f"你在休息/办公混合区，起点朝东(+x)。\n"
                f"任务：地图里【缺少『{target}』】。据记忆它应在你正前方偏左一带。\n"
                f"下面是地图里【你前方已知物体】及其世界坐标，选一个【最可能就是目标、或离目标最近】的"
                f"作为前往点(代码会导航过去让你近看确认)：\n" + "\n".join(cand_lines) + "\n\n"
                f"scan(最近障碍): {scan_t}\n"
                f"只输出 JSON：{{\"goto_id\": 候选编号整数, \"reason\": \"≤40字\"}}")
        content = [{"type": "text", "text": user}]
        if img:
            content.append(harness.to_openai_image_url(img))
        resp = ex.client.chat.completions.create(
            model=ex.model,
            messages=[{"role": "system", "content": harness.INSPECT_MEM_SYS},
                      {"role": "user", "content": content}],
            temperature=0.2, max_tokens=200, stream=False)
        dec = harness._extract_obj(resp.choices[0].message.content or "")
        gid = dec.get("goto_id") if isinstance(dec, dict) else None
        try:
            gid = int(gid)
        except (TypeError, ValueError):
            gid = None
        if gid is None or not (0 <= gid < len(cands)):
            gid = 0
            print("[Qwen] 未给有效 goto_id，兜底选最近候选 [0]")
        tgt = cands[gid]
        tx, ty = tgt["abs_pose"]["x"], tgt["abs_pose"]["y"]
        print(f"[Qwen] 选 [{gid}] {tgt.get('name')} @({tx:.2f},{ty:.2f})  理由: {dec.get('reason','')[:60]}")

        # ---- 阶段2：代码 geo_goto 导航到地图坐标（几何归代码，闭环 get_pose 判停）----
        _hr("阶段2 geo_goto 导航到地图坐标")
        g = nav.geo_goto(ex, tx, ty)
        posef = harness._pose(ex)
        print(f"[geo_goto] status={g['status']} dist_m={g['dist_m']} 到位pose=({posef['x']:.2f},{posef['y']:.2f})")

        # ---- 阶段3：到位面向目标 + 环顾观测 ----
        _hr("阶段3 到位环顾观测")
        nav.geo_face_point(ex, tx, ty)
        rep = harness.sweep_observe(ex)
        seen = [o.get("name") for o in rep.get("objects", [])]
        print(f"[sweep] {rep.get('sweeps')} 次扫视，看到: {seen}")
        cabinet_obj = next((o for o in rep.get("objects", [])
                            if any(k in (o.get("name") or "") for k in cabinet_keys)), None)
        arrived = g.get("dist_m", 99) is not None and g["dist_m"] <= arrival_tol

        # ---- 阶段4：补全回写 ----
        _hr("阶段4 补全回写")
        if cabinet_obj:
            ap = {"x": round(tx, 3), "y": round(ty, 3), "z": tgt["abs_pose"].get("z", 0.78)}
            mem.upsert_object(area, {
                "name": target, "aliases": target_aliases, "abs_pose": ap,
                "spatial": "中央区正前方偏左", "confidence": cabinet_obj.get("confidence", 0.6),
                "verified_by": ["qwen3-vl-8b", "completion(近观确认)"],
                "state": f"近观命名={cabinet_obj.get('name')}",
            })
            print(f"✅ 补全：写入『{target}』@{ap}（Qwen 近观报『{cabinet_obj.get('name')}』）")
            status, finish, code = "SUCCESS", True, 0
        else:
            print(f"⚠️ 到位但未在视野里辨认出目标（看到 {seen}）—— 未补全")
            status = "ARRIVED_NO_CABINET" if arrived else "NOT_ARRIVED"
            finish, code = False, 1

        ex.ros.call("stop", {})

        _hr("结果")
        saw = bool(cabinet_obj)
        print(f"到位(dist≤{arrival_tol}m): {arrived}  |  近观见目标: {saw}")
        print(f"判定: {status}")
        return ModeResult(
            self.name, status, finish,
            summary="",                                # base 用 Qwen 补齐
            exit_code=code,
            metrics={"dist_m": g.get("dist_m"), "arrived": arrived,
                     "sweeps": rep.get("sweeps"), "saw_target": saw,
                     "goto_id": gid, "chosen": tgt.get("name")},
            artifacts={"area_path": mem._area_path(area)},
        )
