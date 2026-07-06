"""geo_goto_around 余量封顶单测（确定性，不连真实 ROS）：注入假 ros，断言前进步长被前向 clearance 封顶。

整机跑受 Qwen/Claude 随机性主导、单轮碰撞代理噪声大，无法隔离这个 3 行改动；本测用假 ros 直接证明：
墙在 0.65m 时直冲不再盲冲 1.0m 而是 ≤ front-clear_margin；转向滑行也按转后实测 front 封顶。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core import navigator as nav  # noqa: E402


class _Res:
    def __init__(self, text):
        self.text = text


class _FakeRos:
    """假 ros：get_pose 沿 +x 前进(正对目标不需转向)、scan 前向固定、记录 move 下发距离。"""

    def __init__(self, front_min_m, sectors=None):
        self.x = 0.0
        self.front = front_min_m
        self.sectors = sectors or {}
        self.moves = []            # 记录所有 move 下发的 distance_m（含正负）

    def call(self, name, args):
        if name == "get_pose":
            return _Res(json.dumps({"x": self.x, "y": 0.0, "yaw_deg": 0.0}))
        if name == "scan_summary":
            return _Res(json.dumps({"front_min_m": self.front, "sectors": self.sectors}))
        if name in ("turn_left_deg", "turn_right_deg"):
            return _Res(json.dumps({"status": "ok"}))
        if name == "move":
            d = float(args.get("distance_m", 0.0))
            self.moves.append(d)
            self.x += d if d > 0 else 0.0      # 只有正向前进推进位置（后退解钉不推进，便于触发 stuck 收尾）
            return _Res(json.dumps({"traveled_m": d, "status": "ok"}))
        return _Res(json.dumps({}))


class _FakeEx:
    def __init__(self, ros):
        self.ros = ros


def test_direct_move_capped_by_clearance():
    """墙在 0.65m、目标远(2m)：直冲步长应被 front-clear_margin(0.40) 封顶，而非盲冲 max_step 1.0。"""
    ros = _FakeRos(front_min_m=0.65)
    nav.geo_goto_around(_FakeEx(ros), 2.0, 0.0, clear_margin_m=0.25)
    fwd = [d for d in ros.moves if d > 0]
    assert fwd, "应有前进 move"
    assert max(fwd) <= 0.45, f"直冲未按 clearance 封顶：{fwd}"   # 0.65-0.25=0.40 (+容差)
    assert max(fwd) < 1.0, "仍在盲冲 max_step"


def test_slide_step_capped_after_turn():
    """墙在 0.5m(<front_block 0.6→绕行)：转向后滑行步长应按转后 front 封顶(≤0.30)，而非盲滑 0.7。"""
    sectors = {"front_left": 1.5, "left": 1.5, "front_right": 0.4, "right": 0.4}
    ros = _FakeRos(front_min_m=0.5, sectors=sectors)
    nav.geo_goto_around(_FakeEx(ros), 2.0, 0.0, clear_margin_m=0.25, max_stuck=3)
    fwd = [d for d in ros.moves if d > 0]
    assert fwd, "应有前进(滑行) move"
    assert max(fwd) <= 0.30, f"滑行未按转后 clearance 封顶：{fwd}"   # 0.5-0.25=0.25 (+容差)


if __name__ == "__main__":
    test_direct_move_capped_by_clearance()
    test_slide_step_capped_after_turn()
    print("test_navigator_clearance: all PASS")
