#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_synthetic_data —— 生成**合成**测试图（不含任何真实人脸）
=============================================================
用途：CI 冒烟测试与本地「缺图不报错」验证。

这些图像是用 OpenCV 基本图形**程序化绘制**的卡通人脸，
不含任何真实人物、不来自任何数据集、**不可用于任何身份识别用途**。

真实检测模型（YuNet / SCRFD）通常不会把卡通脸当作人脸检出，
因此本脚本主要验证：
  * 流水线在缺图 / 图中无人脸时是否**优雅报错**而非崩溃
  * 报告 JSON 结构是否正确、是否含敏感字段
  * 图表生成环节是否可跑通

用法
----
    python scripts/make_synthetic_data.py --out-dir data
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def _draw_face(canvas, cx, cy, w, h, skin=(190, 175, 160)):
    """在画布上绘制一个卡通人脸（椭圆脸 + 五官）。"""
    # 脸
    cv2.ellipse(canvas, (cx, cy), (w // 2, h // 2), 0, 0, 360, skin, -1)
    # 头发
    cv2.ellipse(canvas, (cx, cy - int(h * 0.32)), (w // 2, int(h * 0.26)),
                0, 180, 360, (40, 40, 45), -1)
    # 眼睛
    eye_dx = max(2, w // 6)
    for sx in (-1, 1):
        ex = cx + sx * eye_dx
        ey = cy - int(h * 0.10)
        cv2.ellipse(canvas, (ex, ey), (max(2, w // 14), max(2, h // 16)),
                    0, 0, 360, (250, 250, 250), -1)
        cv2.circle(canvas, (ex, ey), max(1, w // 18), (30, 30, 30), -1)
    # 眉毛
    for sx in (-1, 1):
        ex = cx + sx * eye_dx
        ey = cy - int(h * 0.26)
        cv2.line(canvas, (ex - w // 8, ey), (ex + w // 8, ey), (60, 60, 65), 2)
    # 鼻
    cv2.line(canvas, (cx, cy - int(h * 0.02)), (cx, cy + int(h * 0.12)),
             (140, 130, 120), 2)
    # 嘴
    cv2.ellipse(canvas, (cx, cy + int(h * 0.26)), (w // 4, max(2, h // 14)),
                0, 0, 180, (90, 90, 160), 2)
    # 脖子 + 肩（白色上衣，满足「白衣」启发式）
    cv2.rectangle(canvas, (cx - w // 6, cy + h // 2 - 2),
                  (cx + w // 6, cy + h), (skin[0], skin[1], skin[2]), -1)
    cv2.ellipse(canvas, (cx, cy + int(h * 1.15)), (int(w * 0.95), int(h * 0.55)),
                0, 180, 360, (245, 245, 245), -1)


def make_group(path, width=1684, height=1080, rows=3, per_row=6):
    """生成一张「大合照」：多排卡通人脸。"""
    img = np.full((height, width, 3), (225, 228, 232), dtype=np.uint8)
    # 背景色块，模拟真实场景的层次
    cv2.rectangle(img, (0, 0), (width, height // 3), (200, 205, 212), -1)
    rng = np.random.default_rng(20261118)
    for r in range(rows):
        y = int(height * (0.42 + 0.24 * r))
        base_w = int(width / (per_row + 1.2))
        base_h = int(base_w * 1.3)
        for c in range(per_row):
            x = int(width * (c + 1) / (per_row + 1))
            jitter = int(rng.integers(-base_w // 8, base_w // 8 + 1))
            _draw_face(img, x + jitter, y, base_w, base_h)
    cv2.imwrite(path, img)
    return img.shape


def make_single(path, width=1267, height=951, cx=None, cy=None):
    """生成一张「特写」：单张大脸 + 少量背景人物。"""
    img = np.full((height, width, 3), (215, 218, 224), dtype=np.uint8)
    if cx is None:
        cx, cy = int(width * 0.52), int(height * 0.45)
    # 背景小脸
    for i in range(3):
        _draw_face(img, int(width * (0.08 + 0.08 * i)), int(height * 0.88),
                   int(width * 0.14), int(height * 0.18))
    # 主体大脸
    _draw_face(img, cx, cy, int(width * 0.28), int(width * 0.36))
    cv2.imwrite(path, img)
    return img.shape


def main():
    ap = argparse.ArgumentParser(description="生成合成测试图（不含真实人脸）")
    ap.add_argument("--out-dir", default="data", help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p_group = os.path.join(args.out_dir, "group.jpg")
    p_host = os.path.join(args.out_dir, "host.png")
    p_pano = os.path.join(args.out_dir, "classroom.png")

    print(f"[synthetic] 合照      -> {p_group}  {make_group(p_group)}")
    print(f"[synthetic] 讲台特写  -> {p_host}   {make_single(p_host)}")
    print(f"[synthetic] 教室全景  -> {p_pano}   "
          f"{make_single(p_pano, cx=int(1266 * 0.18), cy=int(950 * 0.35))}")
    print("\n[synthetic] 全部为程序绘制的卡通人脸，不含任何真实人物。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
