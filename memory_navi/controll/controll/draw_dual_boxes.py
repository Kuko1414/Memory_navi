#!/usr/bin/env python3
"""把双标注实验(manifest.jsonl)里 Qwen / YOLO 各自标注的框 + 名字/置信度画到原图上。

保留 imgs_raw/ 原图不动，额外生成两套：
  imgs_qwen/NNNN.jpg  —— 该帧 Qwen 上报的所有物体框（绿=进记忆/留存，橙=被过滤丢弃）
  imgs_yolo/NNNN.jpg  —— 该帧 YOLO 上报的所有物体框
中文标签用系统 Noto Sans CJK 字体（PIL 默认位图字体不支持中文；truetype+CJK 可正常渲染）。

用法： python draw_dual_boxes.py [dual_review_dir]   （默认 Report/dual_review）
纯离线：只读 manifest.jsonl + imgs_raw/*.jpg，不连 ROS/LLM，可随时重跑。
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

# ROI 约定：{x,y,w,h} 归一化到 0..1000（x,y=左上角；见 depth_projection._roi_to_frac）
ROI_SCALE = 1000.0
_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts-droid-fallback/truetype/DroidSansFallback.ttf",
]
KEPT_COLOR = (60, 200, 60)      # 绿：进记忆/留存
DROP_COLOR = (255, 150, 30)     # 橙：上报了但被过滤丢弃


def _load_font(size=16):
    for p in _FONT_CANDIDATES:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def _roi_px(roi, w, h):
    """归一化 0..1000 ROI → 像素 (x0,y0,x1,y1)。非法返回 None。"""
    if not isinstance(roi, dict):
        return None
    try:
        x = float(roi.get("x", 0)) / ROI_SCALE * w
        y = float(roi.get("y", 0)) / ROI_SCALE * h
        bw = float(roi.get("w", 0)) / ROI_SCALE * w
        bh = float(roi.get("h", 0)) / ROI_SCALE * h
    except (TypeError, ValueError):
        return None
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w - 1, x + bw), min(h - 1, y + bh)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _kept_names(kept):
    """留存物体名多重集合（用于给上报框着色：留存=绿、被丢=橙）。"""
    from collections import Counter
    return Counter(o.get("name") for o in (kept or []) if o.get("name"))


NUMBER_COLOR = (30, 144, 255)   # 蓝：混合模式待 Qwen 判定的编号框
YOLO_COLOR = (255, 150, 30)     # 橙：YOLO 提议框
QWEN_COLOR = (60, 200, 60)      # 绿：Qwen 留存（命名/未弃权）框


def draw_labeled_boxes(raw_img, items, *, color=NUMBER_COLOR, font=None):
    """通用画框：在整图上画每个 item 的框 + 文本标签。返回同尺寸 PIL.Image。

    raw_img: PIL.Image（原始帧）。items: [{"roi":{x,y,w,h 0..1000}, "label": str}]。
    复用 _roi_px / _load_font；非法 roi 跳过、不抛异常。
    """
    im = raw_img.convert("RGB")
    w, h = im.size
    dr = ImageDraw.Draw(im)
    if font is None:
        font = _load_font(max(16, int(h / 24)))
    for it in (items or []):
        px = _roi_px(it.get("roi"), w, h)
        if px is None:
            continue
        dr.rectangle(px, outline=color, width=3)
        label = str(it.get("label", ""))
        if not label:
            continue
        tb = dr.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        lx, ly = px[0], max(0, px[1] - th - 6)
        dr.rectangle([lx, ly, lx + tw + 6, ly + th + 6], fill=color)
        dr.text((lx + 3, ly + 3), label, fill=(0, 0, 0) if color == QWEN_COLOR
                else (255, 255, 255), font=font)
    return im


def draw_numbered_boxes(raw_img, boxes, *, font=None):
    """把 YOLO 出的框 + 大号【编号】画在整图上（混合模式方案一：整图批量喂 Qwen）。

    boxes: [{"idx": int, "roi": {...}}]。返回带框图，供 name_boxes 喂 Qwen（Qwen 只看编号，
    不看 YOLO 的 OOD 类名，避免 priming）。复用 draw_labeled_boxes（label=编号）。
    """
    items = [{"roi": b.get("roi"), "label": str(b.get("idx", "?"))} for b in (boxes or [])]
    return draw_labeled_boxes(raw_img, items, color=NUMBER_COLOR, font=font)


def _draw(raw_path, reported, kept, out_path, font):
    im = Image.open(raw_path).convert("RGB")
    w, h = im.size
    dr = ImageDraw.Draw(im)
    kept_ct = _kept_names(kept)
    for o in (reported or []):
        px = _roi_px(o.get("roi"), w, h)
        if px is None:
            continue
        name = str(o.get("name", "?"))
        conf = o.get("confidence")
        keep = kept_ct.get(name, 0) > 0
        if keep:
            kept_ct[name] -= 1
        color = KEPT_COLOR if keep else DROP_COLOR
        dr.rectangle(px, outline=color, width=3)
        label = f"{name} {conf:.2f}" if isinstance(conf, (int, float)) else name
        tb = dr.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        lx, ly = px[0], max(0, px[1] - th - 4)
        dr.rectangle([lx, ly, lx + tw + 4, ly + th + 4], fill=color)
        dr.text((lx + 2, ly + 2), label, fill=(0, 0, 0), font=font)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path, quality=90)


def draw_dual(dual_dir):
    """对 dual_dir 逐帧生成 imgs_qwen/ 与 imgs_yolo/。返回 (n_frames, n_qwen_img, n_yolo_img)。"""
    mpath = os.path.join(dual_dir, "manifest.jsonl")
    if not os.path.isfile(mpath):
        print(f"⚠️ 找不到 {mpath}")
        return (0, 0, 0)
    font = _load_font(16)
    nq = ny = nf = 0
    with open(mpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            raw_rel = r.get("raw_image")
            if not raw_rel:
                continue
            raw_path = os.path.join(dual_dir, raw_rel)
            if not os.path.isfile(raw_path):
                continue
            nf += 1
            seq = r.get("seq", nf)
            try:
                _draw(raw_path, r.get("qwen_reported"), r.get("qwen_kept"),
                      os.path.join(dual_dir, "imgs_qwen", f"{seq:04d}.jpg"), font)
                nq += 1
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ qwen 帧{seq} 画框失败: {e}")
            try:
                _draw(raw_path, r.get("yolo_reported"), r.get("yolo_kept"),
                      os.path.join(dual_dir, "imgs_yolo", f"{seq:04d}.jpg"), font)
                ny += 1
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ yolo 帧{seq} 画框失败: {e}")
    print(f"[画框] {dual_dir}  帧={nf}  imgs_qwen={nq} 张  imgs_yolo={ny} 张  "
          f"(绿=留存/进记忆, 橙=上报但被过滤)")
    return (nf, nq, ny)


def _recall_line(dual_dir, tag):
    import re
    p = os.path.join(dual_dir, f"score_{tag}.txt")
    if os.path.isfile(p):
        m = re.search(r"召回率 = \d+/\d+ = [\d.]+%", open(p, encoding="utf-8").read())
        if m:
            return m.group(0)
    return "n/a"


def write_index(dual_dir):
    """重写 index.md：逐帧并列 原图 / Qwen 画框 / YOLO 画框 + 各自上报清单。"""
    mpath = os.path.join(dual_dir, "manifest.jsonl")
    if not os.path.isfile(mpath):
        return
    rows = [json.loads(x) for x in open(mpath, encoding="utf-8") if x.strip()]

    def _names(items):
        return "、".join(f"{o.get('name')}({o.get('confidence')})" for o in (items or [])) or "无"
    lines = ["# Qwen vs YOLO 双标注对比（同一轨迹/同一帧）", "",
             f"- 帧数：{len(rows)}",
             f"- **Qwen 记忆召回：{_recall_line(dual_dir, 'qwen')}**（`score_qwen.txt`）",
             f"- **YOLO 记忆召回：{_recall_line(dual_dir, 'yolo')}**（`score_yolo.txt`）",
             "- 每帧三图：原图 / Qwen 画框 / YOLO 画框（框内 绿=进记忆留存，橙=上报但被过滤）。", ""]
    for r in rows:
        p = r.get("observer_pose") or {}
        seq = r.get("seq")
        lines.append(f"## 帧{seq:04d} · v{r.get('vantage_idx')} · heading={r.get('heading')}° "
                     f"· pose=({p.get('x')},{p.get('y')})")
        trio = []
        if r.get("raw_image"):
            trio.append(f"![raw]({r['raw_image']})")
        trio.append(f"![qwen](imgs_qwen/{seq:04d}.jpg)")
        trio.append(f"![yolo](imgs_yolo/{seq:04d}.jpg)")
        lines.append(" ".join(trio))
        lines.append(f"- **Qwen 上报**：{_names(r.get('qwen_reported'))}")
        lines.append(f"- **YOLO 上报**：{_names(r.get('yolo_reported'))}")
        lines.append("")
    with open(os.path.join(dual_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[index] {os.path.join(dual_dir, 'index.md')} 已更新（原图/Qwen/YOLO 三图并列）")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "Report/dual_review"
    draw_dual(d)
    write_index(d)
