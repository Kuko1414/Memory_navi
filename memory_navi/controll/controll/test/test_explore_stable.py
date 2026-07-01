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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
