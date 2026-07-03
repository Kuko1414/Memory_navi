"""探索模式稳定化纯函数单测（不需仿真）。

跑：conda run -n vllm python -m pytest memory_navi/controll/controll/test/test_explore_stable.py
或： conda run -n vllm python memory_navi/controll/controll/test/test_explore_stable.py
覆盖：A3 数组并集去重（fs_memory + explore_probe）、召回打分器匹配逻辑。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROLL = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CONTROLL)))
sys.path.insert(0, CONTROLL)
sys.path.insert(0, os.path.join(CONTROLL, "eval"))


# ---------- A3: fs_memory 并集去重 ----------
def test_fs_merge_object_alias_dedup():
    from agent_core.memory.fs_memory import _merge_object
    e = {"name": "cabinet", "aliases": ["red_cabinet", "bookshelf", "red_cabinet"]}
    _merge_object(e, {"name": "cabinet", "aliases": ["red_cabinet", "cabinet", "storage_cabinet"]})
    assert e["aliases"].count("red_cabinet") == 1, e["aliases"]
    assert "cabinet" not in e["aliases"]           # 别名不含主名
    assert "storage_cabinet" in e["aliases"]


def test_fs_dedup_aliases_normalized():
    from agent_core.memory.fs_memory import _dedup_aliases
    # 'red cabinet' ≡ 'red_cabinet'（归一化），主名剔除
    assert _dedup_aliases(["red_cabinet", "red cabinet", "desk"], "cabinet") == ["red_cabinet", "desk"]
    assert _dedup_aliases(["桌子", "桌子"], "显示器") == ["桌子"]


# ---------- A3: explore_probe 合并（含置信度切主名）----------
def test_explore_merge_obj_pair_no_dup():
    import explore_probe as ep
    keep = {"name": "cabinet", "confidence": 0.6, "aliases": ["red_cabinet"]}
    ep._merge_obj_pair(keep, {"name": "red_cabinet", "confidence": 0.5})
    ep._merge_obj_pair(keep, {"name": "red_cabinet", "confidence": 0.5})
    assert keep["aliases"].count("red_cabinet") == 1, keep["aliases"]
    ep._merge_obj_pair(keep, {"name": "storage_cabinet", "confidence": 0.9})  # 高置信→换主名
    assert keep["name"] == "storage_cabinet"
    assert "cabinet" in keep["aliases"] and keep["aliases"].count("red_cabinet") == 1
    assert "storage_cabinet" not in keep["aliases"]


# ---------- A4: 名感知几何去重（桌椅不并坨、同名两视角合一）----------
def test_dedup_name_aware():
    import explore_probe as ep
    recs = [{"objects": [
        {"name": "办公桌", "abs_pose": {"x": 3.0, "y": 0.0, "z": 0.5}, "confidence": 0.8},
        {"name": "办公椅", "abs_pose": {"x": 3.3, "y": 0.0, "z": 0.4}, "confidence": 0.8},
        {"name": "办公桌", "abs_pose": {"x": 3.1, "y": 0.05, "z": 0.5}, "confidence": 0.7},
    ]}]
    out = ep._dedup_objects(recs, {"xmin": -1, "xmax": 6, "ymin": -3, "ymax": 5})
    assert sorted(o["name"] for o in out) == ["办公桌", "办公椅"]   # 桌椅分开；两张桌合一


def test_dedup_keeps_distinct_instances():
    import explore_probe as ep
    # 南墙两矮柜 0.8m 间距(>DEDUP_M) → 各自成条
    recs = [{"objects": [
        {"name": "柜子", "abs_pose": {"x": -0.0, "y": -2.0, "z": 0.3}, "confidence": 0.8},
        {"name": "柜子", "abs_pose": {"x": -0.8, "y": -2.0, "z": 0.3}, "confidence": 0.8},
    ]}]
    out = ep._dedup_objects(recs, {"xmin": -3, "xmax": 3, "ymin": -3, "ymax": 3})
    assert len(out) == 2, [o["abs_pose"] for o in out]


# ---------- B1: 多视角融合取更近帧几何 ----------
def test_merge_prefers_closer_view():
    import explore_probe as ep
    keep = {"name": "显示器", "confidence": 0.9, "distance_m": 3.0,
            "abs_pose": {"x": 5.0, "y": 0.0}, "size_unreliable": True}
    # 更近帧(距离1.2)几何更准 → abs_pose/distance 换成近帧，清掉不可信标记
    ep._merge_obj_pair(keep, {"name": "显示器", "confidence": 0.7, "distance_m": 1.2,
                              "abs_pose": {"x": 2.9, "y": 0.1}, "size": {"width_m": 0.5}})
    assert keep["abs_pose"]["x"] == 2.9 and keep["distance_m"] == 1.2
    assert "size_unreliable" not in keep
    # 更远帧不覆盖近帧几何
    ep._merge_obj_pair(keep, {"name": "显示器", "confidence": 0.95, "distance_m": 4.0,
                              "abs_pose": {"x": 9.9, "y": 9.9}})
    assert keep["abs_pose"]["x"] == 2.9


# ---------- A7: 门校验 ----------
def test_validate_doors():
    import explore_probe as ep
    bbox = {"xmin": -2, "xmax": 6, "ymin": -2, "ymax": 4}
    doors = [{"pose": {"x": 0.0, "y": 0.0}, "count": 10, "dir": "left", "reason": ""},   # 正中→丢
             {"pose": {"x": 5.8, "y": 1.0}, "count": 3, "dir": "right", "reason": ""},    # 近边界→留
             {"pose": {"x": 5.9, "y": -1.0}, "count": 1, "dir": "left", "reason": ""}]    # count1→丢
    v = ep._validate_doors(doors, bbox)
    assert len(v) == 1 and v[0]["count"] == 3


# ---------- 召回打分器匹配逻辑 ----------
def test_scorer_cat_and_pos_match():
    import score_explore as se
    gt_point = {"name": "红柜", "x": 2.6, "y": 0.2, "match_names": ["红柜", "cabinet", "red_cabinet"]}
    # 类别：记录名/别名任一命中
    assert se._cat_match({"name": "desk", "aliases": ["cabinet"]}, gt_point)
    assert not se._cat_match({"name": "door", "aliases": []}, gt_point)
    # 位置：点物体距离阈值
    assert se._pos_match((2.7, 0.1), gt_point, 1.0)
    assert not se._pos_match((5.0, 0.1), gt_point, 1.0)
    # region 物体：落在范围±margin
    gt_region = {"name": "显示器群", "type": "region", "x_range": [-4.2, -2.84],
                 "y_range": [-2.0, 2.15], "region_margin_m": 0.6,
                 "match_names": ["显示器", "monitor"]}
    assert se._pos_match((-2.4, -1.0), gt_region, 1.0)     # x 在 -2.84+0.6 余量内
    assert not se._pos_match((0.0, 0.0), gt_region, 1.0)


def test_scorer_recall_end_to_end(tmp_path=None):
    import json
    import score_explore as se
    area = {"objects": [
        {"name": "sofa", "abs_pose": {"x": 1.2, "y": -2.0}},          # 命中 沙发
        {"name": "door", "abs_pose": {"x": -1.8, "y": -2.2}},          # 厨台位置但类别错 → 误标, 厨台漏
        {"name": "ufo", "abs_pose": {"x": 0.0, "y": 0.0}},             # 幻觉
    ]}
    gt = {"pos_tol_m": 1.0, "objects": [
        {"name": "沙发", "x": 1.23, "y": -2.0, "observable": True, "match_names": ["沙发", "sofa"]},
        {"name": "厨台", "x": -1.79, "y": -2.25, "observable": True, "match_names": ["厨台", "counter"]},
    ]}
    ap = os.path.join(os.path.dirname(__file__), "_t_area.json")
    gp = os.path.join(os.path.dirname(__file__), "_t_gt.json")
    json.dump(area, open(ap, "w")); json.dump(gt, open(gp, "w"))
    try:
        r = se.score(ap, gp)
        assert abs(r - 0.5) < 1e-6, r          # 2 observable, 命中 1 → 50%
    finally:
        for p in (ap, gp):
            if os.path.exists(p):
                os.remove(p)


# ---------- 象限调度官（Claude Role-1）纯函数 ----------
def test_recon_corners():
    import explore_probe as ep
    c = ep._recon_corners({"xmin": 0, "xmax": 5, "ymin": 0, "ymax": 4}, inset=0.8)
    assert c == [(0.8, 0.8), (4.2, 0.8), (4.2, 3.2), (0.8, 3.2)], c
    assert ep._recon_corners({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 4}, 0.8) == []  # 太窄→退化
    assert ep._recon_corners(None) == []


def test_quadrant_stats_undercover_and_allblocked():
    import explore_probe as ep
    bbox = {"xmin": -2.8, "xmax": 2.8, "ymin": -2.8, "ymax": 2.8}
    cells = ep._bbox_cells(bbox)
    ne = {c for c in cells if c[0] >= 0 and c[1] >= 0}
    se = {c for c in cells if c[0] >= 0 and c[1] < 0}
    visited = set(ne)                       # NE 全访问
    blocked = set(se) | {(-2, -2)}          # SE 全 blocked；SW 只 blocked 一格(仍有开洞)
    st = ep._quadrant_stats(bbox, visited, blocked, [])
    assert st["NE"]["coverage"] == 1.0 and not st["NE"]["under_covered"]   # 全访问→不欠
    assert st["SE"]["coverage"] == 1.0 and not st["SE"]["under_covered"]   # 全 blocked→不欠(不可达)
    assert st["SW"]["under_covered"] and st["SW"]["coverage"] < 0.6        # 低覆盖+有开洞→欠
    assert st["NW"]["under_covered"]                                       # 全开→欠


def test_candidate_cells_reopen_flag():
    import explore_probe as ep
    bbox = {"xmin": -1.4, "xmax": 1.4, "ymin": -1.4, "ymax": 1.4}   # 3x3=9 格
    ring = [c for c in ep._bbox_cells(bbox) if c != (0, 0)]         # (0,0) 的 8 邻居全访问
    cands = ep._candidate_cells(bbox, set(ring), {(0, 0)})
    assert len(cands) == 1 and cands[0]["cell"] == (0, 0) and cands[0]["reopen_blocked"]
    # 全开：每象限 top-N 覆盖候选，无 reopen，id 唯一
    cands2 = ep._candidate_cells(bbox, set(), set())
    ids = [c["id"] for c in cands2]
    assert len(ids) == len(set(ids)) and not any(c["reopen_blocked"] for c in cands2)


def test_validate_plan_choice():
    import explore_probe as ep
    cands = [{"id": "c0", "cell": (1, 0)}, {"id": "c1", "cell": (2, 0)}]
    assert ep._validate_plan_choice({"target_id": "c1"}, cands)["cell"] == (2, 0)
    assert ep._validate_plan_choice({"target_id": "c9"}, cands) is None   # 幻觉 id
    assert ep._validate_plan_choice({}, cands) is None
    assert ep._validate_plan_choice({"target_id": None}, cands) is None
    assert ep._validate_plan_choice("nope", cands) is None


def test_is_forward_open():
    import explore_probe as ep
    # 前向锥(±30)内最近障碍在目标格之外(+margin) → 开
    assert ep._is_forward_open({0: 3.0, 45: 0.2, -45: 0.3}, dist_to_cell=1.5)   # 45°在锥外不算
    assert not ep._is_forward_open({0: 1.0}, dist_to_cell=1.5)                  # 障碍近于格→墙
    assert not ep._is_forward_open({90: 5.0}, dist_to_cell=1.5)                 # 无前向读数→保守当墙
    assert not ep._is_forward_open({}, dist_to_cell=1.0)


def test_stop_rule_object_standoff():
    import explore_probe as ep
    cell = (2, 0)                       # 中心 (2.8, 0)
    fb, tol = ep._stop_rule(cell, [("柜子", 2.8, 0.0)])   # 格上有物体
    assert fb == ep.OBJECT_STANDOFF_M and tol > ep.OBJECT_STANDOFF_M
    fb2, _ = ep._stop_rule(cell, [("柜子", 5.0, 5.0)])    # 远处物体→默认
    assert fb2 == 0.4


def test_min_vantage_spacing_ok():
    import explore_probe as ep
    assert ep._min_vantage_spacing_ok({"x": 0, "y": 0}, None)          # 首个总 OK
    assert ep._min_vantage_spacing_ok({"x": 2, "y": 0}, (0.0, 0.0))    # 够远
    assert not ep._min_vantage_spacing_ok({"x": 0.5, "y": 0}, (0.0, 0.0))  # 太近


# ---------- Role-2：代码聚簇 → Claude 命名 → 代码合并 ----------
def test_position_clusters():
    import explore_probe as ep
    recs = [{"name": "桌子", "abs_pose": {"x": 3.0, "y": 0.0}},
            {"name": "办公桌", "abs_pose": {"x": 3.1, "y": 0.0}},   # 与 0 同簇(<0.6m)
            {"name": "显示器", "abs_pose": {"x": 5.0, "y": 0.0}}]   # 远，独簇
    cl = ep._position_clusters(recs)
    idsets = sorted(sorted(ids) for ids, _cen in cl)
    assert idsets == [[0, 1], [2]], idsets


def test_consolidate_merge_synonyms():
    import explore_probe as ep
    recs = [{"name": "桌子", "abs_pose": {"x": 3.0, "y": 0.0}, "confidence": 0.7},
            {"name": "办公桌", "abs_pose": {"x": 3.1, "y": 0.0}, "confidence": 0.8}]
    cl = ep._position_clusters(recs)                              # 一个簇 {0,1}
    out = ep._apply_consolidation(recs, cl, {"clusters": [{"id": 0, "name": "办公桌"}], "drop": []})
    assert len(out) == 1 and out[0]["name"] == "办公桌", out       # 同物不同名→合一
    assert "桌子" in (out[0].get("aliases") or [])


def test_consolidate_keep_distinct_clusters():
    import explore_probe as ep
    recs = [{"name": "桌子", "abs_pose": {"x": 3.0, "y": 0.0}},
            {"name": "椅子", "abs_pose": {"x": 5.0, "y": 0.0}}]    # 远→两簇
    cl = ep._position_clusters(recs)
    out = ep._apply_consolidation(recs, cl, {"clusters": [{"id": 0, "name": "办公桌"},
                                                          {"id": 1, "name": "办公椅"}]})
    assert sorted(o["name"] for o in out) == ["办公桌", "办公椅"]


def test_consolidate_colocated_forces_one():
    import explore_probe as ep
    # 名字矛盾的记录挤在一簇(<0.6m) → 恒合并成一个物体（消除同位重名，不因名字矛盾拆分）
    recs = [{"name": "办公桌", "abs_pose": {"x": 3.0, "y": 0.0}},
            {"name": "柜子", "abs_pose": {"x": 3.2, "y": 0.0}}]
    cl = ep._position_clusters(recs)
    assert len(cl) == 1                                           # 确实同簇
    out = ep._apply_consolidation(recs, cl, {"clusters": [{"id": 0, "name": "办公桌"}]})
    assert len(out) == 1 and out[0]["name"] == "办公桌"


def test_consolidate_drop_and_fallback():
    import explore_probe as ep
    recs = [{"name": "沙发", "abs_pose": {"x": 1.0, "y": 0.0}},
            {"name": "ufo", "abs_pose": {"x": 5.0, "y": 0.0}},     # 独簇→drop
            {"name": "绿植", "abs_pose": {"x": 9.0, "y": 0.0}}]    # 独簇, Claude 没判定→多数名兜底
    cl = ep._position_clusters(recs)                              # 3 独簇 (id 0,1,2)
    out = ep._apply_consolidation(recs, cl, {"clusters": [{"id": 0, "name": "沙发"}], "drop": [1]})
    assert sorted(o["name"] for o in out) == ["沙发", "绿植"]      # ufo 删、绿植兜底保留


def test_consolidate_empty_plan_noop():
    import explore_probe as ep
    recs = [{"name": "沙发", "abs_pose": {"x": 1.0, "y": 0.0}}]
    cl = ep._position_clusters(recs)
    assert ep._apply_consolidation(recs, cl, {}) == recs          # 空计划=安全降级
    assert ep._apply_consolidation(recs, cl, "bad") == recs


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
