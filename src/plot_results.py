#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results —— 由 JSON 报告生成可视化图表
==========================================
1. similarity_chart.png  合照所有人脸对目标人物的相似度分布（阈值分界线 + Top-N 标注）
2. confidence_bars.png   三组三角验证的余弦相似度对比条形图

依赖: matplotlib
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import font_manager      # noqa: E402

# ---------------- 配色（深色科技风） ----------------
BG = "#0f1420"
FG = "#e8edf7"
ACCENT = "#3ddc97"
ACCENT2 = "#4aa8ff"
WARN = "#ffb020"
MUTED = "#6b7896"
GRID = "#232b3d"


def pick_font():
    """挑一个能显示中文的字体，避免方块乱码。"""
    prefer = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
              "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "SimHei",
              "Microsoft YaHei", "PingFang SC", "Arial Unicode MS"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in prefer:
        if name in have:
            return name
    return "DejaVu Sans"


def style_axes(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def similarity_chart(report, out_path):
    ranking = report["ranking"]
    thr = report["threshold"]["cosine"]
    values = [r["cosine"] for r in ranking]
    labels = [f"#{i+1} ({int(r['center'][0])},{int(r['center'][1])})" for i, r in enumerate(ranking)]

    fig, ax = plt.subplots(figsize=(11, 6.4), facecolor=BG)
    style_axes(ax)

    colors = []
    for r in ranking:
        if r["is_target"]:
            colors.append(ACCENT)
        elif r["same_person"]:
            colors.append(ACCENT2)
        else:
            colors.append("#3a4661")

    y = range(len(values))
    ax.barh(list(y), values, color=colors, height=0.68, zorder=3)
    ax.axvline(thr, color=WARN, linestyle="--", linewidth=1.8, zorder=4)
    ax.text(thr, len(values) - 0.2, f" 判定阈值 {thr}", color=WARN, fontsize=10,
            va="top", ha="left", fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.12, max(max(values) * 1.18, thr * 1.4))
    ax.set_xlabel("Cosine Similarity", color=FG, fontsize=11)
    ax.set_title("合照全部人脸 vs 目标人物 · 相似度分布", color=FG, fontsize=15,
                 fontweight="bold", pad=16)

    for i, v in enumerate(values):
        if i < 8 or ranking[i]["is_target"]:
            ax.text(v + 0.012, i, f"{v:.4f}", va="center", fontsize=8.5,
                    color=ACCENT if ranking[i]["is_target"] else MUTED,
                    fontweight="bold" if ranking[i]["is_target"] else "normal")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=ACCENT, label="目标本人"),
        Patch(color=ACCENT2, label="阈值内(同一人)"),
        Patch(color="#3a4661", label="阈值外(非同一人)"),
    ], loc="lower right", frameon=False, labelcolor=FG, fontsize=9)

    fig.text(0.012, 0.015,
             f"backend={report['backend']}  ·  dim={report['embedding_dim']}  ·  "
             f"faces={report['group_image']['faces_detected']}  ·  "
             f"top1 cosine={values[0]:.4f}  ·  结论："
             f"{'同一人' if report['comparison']['same_person'] else '非同一人'}",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"[save] {out_path}")


def confidence_bars(tri, out_path):
    pairs = tri["pairs"]
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=BG)
    style_axes(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)

    names = [p["pair"] for p in pairs]
    vals = [p["cosine"] for p in pairs]
    thr = tri["threshold"]
    colors = [ACCENT if p["same_person"] else "#ff6b6b" for p in pairs]

    bars = ax.barh(names, vals, color=colors, height=0.52, zorder=3)
    ax.axvline(thr, color=WARN, linestyle="--", linewidth=1.8, zorder=4)
    ax.text(thr, -0.62, f"阈值 {thr}", color=WARN, fontsize=10,
            va="bottom", ha="center", fontweight="bold")
    ax.set_xlim(0, max(max(vals) * 1.25, thr * 1.5))
    ax.set_ylim(len(names) - 0.35, -0.85)
    ax.set_xlabel("Cosine Similarity", color=FG, fontsize=11)
    ax.set_title("三角验证 · 三组两两比对", color=FG, fontsize=14, fontweight="bold", pad=16)

    for bar, p in zip(bars, pairs):
        ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{p['cosine']:.4f}  ({p['confidence_pct']:.1f}%)",
                va="center", fontsize=10, color=FG, fontweight="bold")

    fig.text(0.012, 0.02,
             f"三对全部一致：{'同一人' if tri['all_consistent'] else '需人工复核'}"
             f"   ·   backend={tri['backend']}",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"[save] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="生成结果可视化图表")
    ap.add_argument("--report", default="results/report.json")
    ap.add_argument("--triangulation", default="results/triangulation.json")
    ap.add_argument("--out-dir", default="assets/charts")
    args = ap.parse_args()

    font = pick_font()
    plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    print(f"[font] 使用字体: {font}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.report, encoding="utf-8") as fh:
        report = json.load(fh)
    similarity_chart(report, os.path.join(args.out_dir, "similarity_chart.png"))

    if os.path.exists(args.triangulation):
        with open(args.triangulation, encoding="utf-8") as fh:
            tri = json.load(fh)
        confidence_bars(tri, os.path.join(args.out_dir, "confidence_bars.png"))


if __name__ == "__main__":
    main()
