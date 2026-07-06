#!/usr/bin/env python3
"""常驻 YOLO 检测服务（在 yolo conda env 里跑）：模型【只加载一次】，之后从 stdin 逐行读图片路径，
每行回一行 JSON {ok, image, width, height, raw_detections}。读到 __QUIT__ 或 EOF 退出。

为什么：explore 混合/YOLOE 模式原本每帧起一个 subprocess 且重载 YOLO 模型（CPU 冷加载 ~3.6s/帧），
72 帧≈4.5 分钟纯加载。改常驻后模型只加载一次，每帧仅推理（~200ms）。见 explore_probe._yolo_detect。

协议：父进程 Popen 本脚本，先读一行 "READY"（模型加载完）；随后每写一行【图片绝对路径】就读回一行
JSON 结果；结束写 "__QUIT__\\n"。stderr 留给加载/推理报错（父进程可重定向到日志）。
"""
import argparse
import json
import sys

from agent_core.perception import yoloe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=yoloe.DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()

    model = yoloe.load_model(args.model)            # 只此一次加载（含 mobileclip 文本编码器）
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        if path == "__QUIT__":
            break
        try:
            res = yoloe.predict_image(model, path, device=args.device, conf=args.conf)
            out = {"ok": True, "image": path, "width": res.get("width"),
                   "height": res.get("height"), "raw_detections": res.get("raw_detections") or []}
        except Exception as e:  # noqa: BLE001
            out = {"ok": False, "image": path, "error": str(e)[:200]}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
