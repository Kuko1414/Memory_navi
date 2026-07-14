"""理想地图适配器 + 执行评分器 单测（纯函数、无仿真、无 LLM）。

- ideal_map_adapter.build/seed：产出含两执行目标绿植、三子区；schema 校验(jsonschema 在则校，否则跳)。
- score_execution.score：到达/幻觉/总结/回报 硬门判定正确。
可在 base pytest 直接跑（validate 自动按 jsonschema 可用性开关）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core.memory.fs_memory import FsMemory  # noqa: E402
from eval import ideal_map_adapter as ida  # noqa: E402
from eval import score_execution as se  # noqa: E402

try:
    import jsonschema  # noqa: F401
    _HAS_JSONSCHEMA = True
except Exception:  # noqa: BLE001
    _HAS_JSONSCHEMA = False


# ---- 适配器 ----
def test_build_has_two_target_plants_and_three_subareas():
    import json
    with open(ida.DEFAULT_GT, encoding="utf-8") as f:
        gt = json.load(f)
    base, verdicts = ida.build(gt)
    plants = [o for o in base["objects"] if o["name"] == "绿植"]
    xy = {(o["abs_pose"]["x"], o["abs_pose"]["y"]) for o in plants}
    assert (6.23, 1.42) in xy and (-5.78, -2.17) in xy       # 两执行目标齐备
    assert len(verdicts["sub_areas"]) == 3                    # 三功能子区
    # 每个物体都有 id（apply_curation 的 member_ids 依赖）
    assert all(o.get("id") for o in base["objects"])
    labels = {sa["label"] for sa in verdicts["sub_areas"]}
    assert {"起始休息区", "红柜后办公区", "西侧办公簇"} == labels


def test_seed_writes_valid_ideal_map(tmp_path):
    mem = FsMemory(str(tmp_path), "sim")
    path = ida.seed(mem=mem, out_area="break_room_ideal", validate=_HAS_JSONSCHEMA)
    assert os.path.exists(path)
    rec = mem.load_area("break_room_ideal")
    assert rec["type"] and len(rec["objects"]) >= 20
    assert len(rec.get("sub_areas", [])) == 3                 # sub_areas 经整理层落盘
    plants = [o for o in rec["objects"] if o["name"] == "绿植"]
    assert len(plants) == 3
    # sub_area.range 由代码从成员坐标算（非模型）
    for sa in rec["sub_areas"]:
        assert sa.get("range") and sa["range"].get("xmin") is not None


def test_doors_carry_id_label_and_validate():
    import json
    with open(ida.DEFAULT_GT, encoding="utf-8") as f:
        gt = json.load(f)
    base, _ = ida.build(gt)
    doors = base["doors"]
    assert len(doors) == 2
    assert all(d.get("id") and d.get("label") for d in doors)   # 门带 id/label（门到门路由）
    labels = {d["label"] for d in doors}
    assert "西隔断北口" in labels
    if _HAS_JSONSCHEMA:                                          # 加了 id/label 后 schema 仍校验通过
        from agent_core.cloud.schema import validate_memory_record
        validate_memory_record(base)


def test_seed_idempotent(tmp_path):
    mem = FsMemory(str(tmp_path), "sim")
    ida.seed(mem=mem, validate=_HAS_JSONSCHEMA)
    ida.seed(mem=mem, validate=_HAS_JSONSCHEMA)               # 二次不叠加
    rec = mem.load_area("break_room_ideal")
    assert len([o for o in rec["objects"] if o["name"] == "绿植"]) == 3


# ---- 执行评分器 ----
def _good_result():
    return {
        "targets": [
            {"name": "绿植", "target_xy": [6.23, 1.42], "min_surface_dist_m": 0.30,
             "saw_target": True, "objects_seen": ["绿植", "办公桌"], "arrival_pose": {"x": 6.0, "y": 1.3}},
            {"name": "绿植", "target_xy": [-5.78, -2.17], "min_surface_dist_m": 0.42,
             "saw_target": True, "objects_seen": ["绿植"], "arrival_pose": {"x": -5.5, "y": -2.0}},
        ],
        "collisions": 1, "duration_s": 120.0, "qwen_summary": "到达两处绿植并回报",
        "supervisor_report": {"phase": "execution", "mode": "execution", "finish": True,
                              "duration_s": 120.0, "summary": "到达两处绿植并回报"},
    }


def test_score_pass_when_all_arrived_no_halluc():
    s = se.score(_good_result(), se.load_gt())
    assert s["pass"] is True and s["arrival"]["arrived"] == 2


def test_score_fail_when_one_target_too_far():
    r = _good_result()
    r["targets"][1]["min_surface_dist_m"] = 1.2               # 超阀值
    s = se.score(r, se.load_gt())
    assert s["pass"] is False and s["arrival"]["arrived"] == 1


def test_score_fail_on_hallucination():
    r = _good_result()
    r["targets"][0]["objects_seen"] = ["绿植", "电视"]         # 幻觉黑名单命中
    s = se.score(r, se.load_gt())
    assert s["pass"] is False and "电视" in s["hallucination"]["offending"]


def test_score_fail_without_supervisor_report():
    r = _good_result()
    r["supervisor_report"] = {}                               # 缺回报
    s = se.score(r, se.load_gt())
    assert s["supervisor_report"]["ok"] is False and s["pass"] is False


def test_surface_distance_computed_from_pose_when_missing():
    r = _good_result()
    # 去掉预算的表面距离，让 scorer 从 arrival_pose 现算
    r["targets"][0].pop("min_surface_dist_m")
    r["targets"][0]["arrival_pose"] = {"x": 6.0, "y": 1.42}   # 中心距≈0.23 → 表面≈-0.13 ≤0.45
    s = se.score(r, se.load_gt())
    assert s["targets"][0]["arrived"] is True


def test_single_target_smoke_not_false_fail():
    """单目标 smoke（跑 1/2）：该目标达标应记 arrived，但整体因未完整不算总 PASS（非假 1/2 FAIL）。"""
    r = _good_result()
    r["targets"] = r["targets"][:1]                          # 只跑第 1 个目标
    s = se.score(r, se.load_gt())
    a = s["arrival"]
    assert a["attempted"] == 1 and a["gt_total"] == 2 and a["complete"] is False
    assert a["arrived"] == 1                                  # 跑的那个达标
    assert s["targets"][0]["arrived"] is True                # 逐目标行绿
    assert s["pass"] is False                                # 但未完整 → 非总 PASS


def test_full_two_targets_complete_pass():
    s = se.score(_good_result(), se.load_gt())
    assert s["arrival"]["complete"] is True and s["arrival"]["attempted"] == 2
    assert s["pass"] is True
