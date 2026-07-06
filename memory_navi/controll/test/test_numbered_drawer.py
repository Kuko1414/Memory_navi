"""编号框绘制器测试：整图画 YOLO 框 + 编号。纯 PIL，无 vLLM/ROS。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from PIL import Image  # noqa: E402

import draw_dual_boxes  # noqa: E402


def test_draw_returns_same_size_image():
    im = Image.new("RGB", (640, 480), (128, 128, 128))
    boxes = [
        {"idx": 1, "roi": {"x": 100, "y": 100, "w": 200, "h": 150}},
        {"idx": 2, "roi": {"x": 400, "y": 200, "w": 150, "h": 150}},
        {"idx": 3, "roi": {"x": 50, "y": 300, "w": 120, "h": 120}},
    ]
    out = draw_dual_boxes.draw_numbered_boxes(im, boxes)
    assert out.size == (640, 480)
    assert out.mode == "RGB"
    # 至少某些像素被画成蓝框颜色（图不再是纯灰）
    assert out.getcolors(maxcolors=1 << 20) is None or len(out.getcolors(1 << 20)) > 1


def test_draw_tolerates_bad_and_out_of_range_roi():
    im = Image.new("RGB", (320, 240), (0, 0, 0))
    boxes = [
        {"idx": 1, "roi": {"x": 10, "y": 10, "w": 50, "h": 50}},
        {"idx": 2, "roi": {"x": 9999, "y": 9999, "w": 10, "h": 10}},   # 越界
        {"idx": 3, "roi": {"x": 0, "y": 0, "w": 0, "h": 0}},           # 退化
        {"idx": 4, "roi": None},                                       # 非法
    ]
    out = draw_dual_boxes.draw_numbered_boxes(im, boxes)               # 不应抛异常
    assert out.size == (320, 240)


def test_draw_empty_boxes_noop():
    im = Image.new("RGB", (100, 100), (10, 20, 30))
    out = draw_dual_boxes.draw_numbered_boxes(im, [])
    assert out.size == (100, 100)


if __name__ == "__main__":
    test_draw_returns_same_size_image()
    test_draw_tolerates_bad_and_out_of_range_roi()
    test_draw_empty_boxes_noop()
    print("test_numbered_drawer: OK")
