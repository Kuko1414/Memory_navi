"""name_boxes 输出解析纯函数测试：容错 + 漏报默认弃权。无 vLLM/ROS。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core import harness  # noqa: E402


def test_parse_well_formed():
    raw = ('{"boxes":[{"idx":1,"completeness":"完整","name":"沙发","keep":true},'
           '{"idx":2,"completeness":"部分","name":null,"keep":false}]}')
    out = harness._parse_box_judgments(raw, [1, 2])
    assert out[1] == {"name": "沙发", "completeness": "完整", "keep": True}
    assert out[2] == {"name": None, "completeness": "部分", "keep": False}


def test_tolerates_code_fence_and_trailing_text():
    raw = ('```json\n{"boxes":[{"idx":1,"completeness":"完整","name":"显示器","keep":true}]}'
           '\n```\n以上是判定。')
    out = harness._parse_box_judgments(raw, [1])
    assert out[1]["name"] == "显示器" and out[1]["keep"] is True


def test_omitted_index_defaults_to_abstain():
    raw = '{"boxes":[{"idx":1,"completeness":"完整","name":"绿植","keep":true}]}'
    out = harness._parse_box_judgments(raw, [1, 2, 3])   # 2/3 未上报 → 默认弃权
    assert out[1]["keep"] is True
    assert out[2] == {"name": None, "completeness": "部分", "keep": False}
    assert out[3] == {"name": None, "completeness": "部分", "keep": False}


def test_keep_forced_false_when_incomplete_or_unnamed():
    # 模型自相矛盾：keep=true 但完整度非完整 / 无名 → 一律纠正为弃权
    raw = ('{"boxes":[{"idx":1,"completeness":"部分","name":"柜子","keep":true},'
           '{"idx":2,"completeness":"完整","name":null,"keep":true}]}')
    out = harness._parse_box_judgments(raw, [1, 2])
    assert out[1]["keep"] is False and out[1]["name"] is None
    assert out[2]["keep"] is False and out[2]["name"] is None


def test_garbage_input_all_abstain():
    out = harness._parse_box_judgments("not json at all", [1, 2])
    assert all(v == {"name": None, "completeness": "部分", "keep": False}
               for v in out.values())


if __name__ == "__main__":
    test_parse_well_formed()
    test_tolerates_code_fence_and_trailing_text()
    test_omitted_index_defaults_to_abstain()
    test_keep_forced_false_when_incomplete_or_unnamed()
    test_garbage_input_all_abstain()
    print("test_name_boxes_parse: OK")
