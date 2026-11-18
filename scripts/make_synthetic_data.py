#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_synthetic_data —— 生成测试图（**不含任何真实人脸**）
==========================================================
用途：CI 冒烟测试与本地「缺图不报错」验证。

两种模式：

``--mode synthetic``（默认）
    用 OpenCV 基本图形**程序化绘制**的卡通人脸。
    完全由代码生成，不来自任何数据集，**不含任何真实人物**。
    注意：卡通脸通常无法被 YuNet / SCRFD 检出，因此本模式主要用于
    验证「缺图 / 无人脸时优雅报错」这条路径。

``--mode sample``
    使用 matplotlib 自带的**公有领域**示例图
    （Grace Hopper 肖像，美国海军部公有领域照片），
    拼贴成多脸合照 / 特写 / 全景三张图。
    该图片随 matplotlib 安装包分发，**不下载、不入库**，
    仅在运行本脚本时临时生成到 ``data/``（该目录已 gitignore）。

真实检测模型对公有领域示例图检出稳定，因此 ``--mode sample``
可以真正跑通「检测 → 特征 → 比对 → 报告 → 图表」全链路。

用法
----
    python scripts/make_synthetic_data.py --out-dir data --mode sample
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


def _load_public_sample():
    """取 matplotlib 自带的公有领域示例图，返回 BGR ndarray。"""
    try:
        import matplotlib.cbook as cbook
        with cbook.get_sample_data("grace_hopper.jpg") as fh:
            path = fh.name
    except Exception as exc:
        raise RuntimeError(
            f"无法获取 matplotlib 示例图: {exc}\n"
            "请改用 --mode synthetic（卡通脸，不保证可被检出）。"
        ) from exc
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"无法读取示例图: {path}")
    return img


def make_synthetic_group(path, width=1684, height=1080, rows=3, per_row=6):
    """程序绘制的「合照」：多排卡通人脸。"""
    img = np.full((height, width, 3), (225, 228, 232), dtype=np.uint8)
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


def make_synthetic_single(path, width=1267, height=951):
    """程序绘制的「特写」：单张大脸 + 背景小脸。"""
    img = np.full((height, width, 3), (215, 218, 224), dtype=np.uint8)
    for i in range(3):
        _draw_face(img, int(width * (0.08 + 0.08 * i)), int(height * 0.88),
                   int(width * 0.14), int(height * 0.18))
    _draw_face(img, int(width * 0.52), int(height * 0.45),
               int(width * 0.28), int(width * 0.36))
    cv2.imwrite(path, img)
    return img.shape


def make_sample_group(path, base):
    """用公有领域示例图拼贴成多脸「合照」（2×3 网格）。"""
    h, w = base.shape[:2]
    img = np.full((h * 2, w * 3, 3), 225, dtype=np.uint8)
    for r in range(2):
        for c in range(3):
            img[r * h:(r + 1) * h, c * w:(c + 1) * w] = base
    cv2.imwrite(path, img)
    return img.shape


def make_sample_single(path, base):
    """用公有领域示例图拼贴成「特写」（主体居中偏右 + 背景小脸）。"""
    h, w = base.shape[:2]
    img = np.full((h, w, 3), 218, dtype=np.uint8)
    small = cv2.resize(base, (w // 3, h // 3))
    for i in range(3):                                   # 背景人物
        x = int(w * 0.06 + i * w * 0.30)
        y = int(h * 0.66)
        img[y:y + small.shape[0], x:x + small.shape[1]] = small
    big = cv2.resize(base, (int(w * 0.55), int(h * 0.75)))
    bx, by = int(w * 0.30), int(h * 0.06)
    img[by:by + big.shape[0], bx:bx + big.shape[1]] = big
    cv2.imwrite(path, img)
    return img.shape


def make_sample_panorama(path, base):
    """用公有领域示例图拼贴成「全景」（主持人在左上，满足 x/W<0.25）。"""
    h, w = base.shape[:2]
    img = np.full((h * 2, w * 2, 3), 225, dtype=np.uint8)
    small = cv2.resize(base, (w // 2, h // 2))
    img[0:h // 2, 0:w // 2] = small                      # 左上：主持人
    quarter = cv2.resize(base, (w // 4, h // 4))
    for i in range(3):                                   # 下方观众
        x = w // 2 + i * (w // 4)
        y = h // 2 + i * (h // 4)
        img[y:y + quarter.shape[0], x:x + quarter.shape[1]] = quarter
    cv2.imwrite(path, img)
    return img.shape


def main():
    ap = argparse.ArgumentParser(description="生成测试图（不含真实人脸）")
    ap.add_argument("--out-dir", default="data", help="输出目录")
    ap.add_argument("--mode", choices=("synthetic", "sample"), default="sample",
                    help="synthetic=程序绘制的卡通脸；sample=matplotlib 自带公有领域示例图")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p_group = os.path.join(args.out_dir, "group.jpg")
    p_host = os.path.join(args.out_dir, "host.png")
    p_pano = os.path.join(args.out_dir, "classroom.png")

    if args.mode == "sample":
        base = _load_public_sample()
        print(f"[sample] 公有领域示例图 {base.shape}")
        print(f"[sample] 合照      -> {p_group}  {make_sample_group(p_group, base)}")
        print(f"[sample] 讲台特写  -> {p_host}   {make_sample_single(p_host, base)}")
        print(f"[sample] 教室全景  -> {p_pano}   {make_sample_panorama(p_pano, base)}")
        print("\n[sample] 素材为 matplotlib 自带的公有领域照片（Grace Hopper，"
              "美国海军部公有领域），仅临时生成，不入库。")
    else:
        print(f"[synthetic] 合照      -> {p_group}  {make_synthetic_group(p_group)}")
        print(f"[synthetic] 讲台特写  -> {p_host}   {make_synthetic_single(p_host)}")
        print(f"[synthetic] 教室全景  -> {p_pano}   {make_synthetic_single(p_pano)}")
        print("\n[synthetic] 全部为程序绘制的卡通人脸，不含任何真实人物；"
              "卡通脸通常无法被检测模型检出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
