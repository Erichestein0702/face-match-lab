#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
triangulate —— 三图三角验证
================================
用三张互相独立的照片构成证据链，交叉验证「同一人」结论，避免单张误判：

    A  大合照中的目标（第一排最右 + 白衣）
    B  讲台特写中的主持人
    C  教室全景中的主持人（讲台左侧站立者）

A-B、A-C、B-C 三对两两比对，若均判定「同一人」，则结论高度可信。
同时输出合照内全部人脸对 C 的排名，确认不存在得分更高的其他人。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_compare import (                                          # noqa: E402
    COS_THR_ARCFACE, BACKEND, detect_faces, cosine, euclidean,
    resolve_source, white_cloth_check,
)

THR = COS_THR_ARCFACE if BACKEND == "ArcFace" else 0.363


def confidence_pct(cos_sim: float, thr: float, k: float = 12.0) -> float:
    return 100.0 / (1.0 + math.exp(-k * (cos_sim - thr)))


def pair(a: dict, b: dict, label: str) -> dict:
    c = cosine(a["emb"], b["emb"])
    e = euclidean(a["emb"], b["emb"])
    row = {
        "pair": label,
        "cosine": round(c, 4),
        "euclidean": round(e, 4),
        "confidence_pct": round(confidence_pct(c, THR), 2),
        "same_person": c >= THR,
    }
    print(f"{label:<32} 余弦={c:.4f} ({c*100:5.2f}%) | 欧氏={e:.4f} | "
          f"置信度={row['confidence_pct']:6.2f}% | {'同一人' if row['same_person'] else '非同一人'}")
    return row


def main():
    ap = argparse.ArgumentParser(description="三图三角验证")
    ap.add_argument("--group", default="samples/group.jpg", help="大合照")
    ap.add_argument("--host", default="samples/host.png", help="讲台特写")
    ap.add_argument("--panorama", default="samples/classroom.png", help="教室全景")
    ap.add_argument("--out-dir", default="results", help="输出目录")
    ap.add_argument("--report", default="results/triangulation.json", help="JSON 输出")
    args = ap.parse_args()

    img_g = cv2.imread(resolve_source(args.group))
    img_c = cv2.imread(resolve_source(args.host))
    img_p = cv2.imread(resolve_source(args.panorama))
    if img_g is None or img_c is None or img_p is None:
        sys.exit("[error] 图片读取失败")

    hg, _ = img_g.shape[:2]
    hc, _ = img_c.shape[:2]
    hp, wp = img_p.shape[:2]

    dg, _ = detect_faces(img_g)
    dc, _ = detect_faces(img_c)
    dp, _ = detect_faces(img_p)
    print(f"检测数: 合照={len(dg)}  讲台特写={len(dc)}  教室全景={len(dp)}\n")

    # --- A: 合照目标（第一排最右 + 白衣） ---
    front = [f for f in dg if f.center[1] > 0.52 * hg]
    front.sort(key=lambda f: f.center[0], reverse=True)
    tgt = None
    for f in front:
        ok, sat, val = white_cloth_check(img_g, f.bbox)
        if ok:
            tgt = f
            print(f"A 合照目标: bbox={f.bbox} 肩部HSV S={sat:.0f} V={val:.0f} (白衣 ✓)")
            break
    if tgt is None:
        tgt = front[0] if front else max(dg, key=lambda f: f.center[1])

    # --- B: 讲台特写主持人（排除前景观众头顶后取最大脸） ---
    cands = [f for f in dc if f.center[1] < 0.85 * hc] or dc
    host = max(cands, key=lambda f: f.area)
    print(f"B 讲台主持人: bbox={host.bbox} 中心={host.center}")

    # --- C: 教室全景中的主持人（讲台左侧、画面偏上） ---
    pcands = [f for f in dp if f.center[0] < 0.25 * wp and f.center[1] < 0.62 * hp]
    pcands.sort(key=lambda f: f.center[1])
    if not pcands:
        sys.exit("[error] 教室全景中未定位到主持人")
    pan = pcands[0]
    print(f"C 全景主持人: bbox={pan.bbox} 中心={pan.center}\n")

    A = {"emb": tgt.embedding}
    B = {"emb": host.embedding}
    C = {"emb": pan.embedding}

    print("========== 三角验证 ==========")
    pairs = [
        pair(A, B, "A 合照目标  vs  B 讲台特写"),
        pair(A, C, "A 合照目标  vs  C 教室全景"),
        pair(B, C, "B 讲台特写  vs  C 教室全景"),
    ]
    all_same = all(p["same_person"] for p in pairs)
    print(f"\n三对全部一致: {'确认为同一人' if all_same else '存在不一致，需人工复核'}")

    # --- 合照全部人脸 vs 全景主持人，验证唯一性 ---
    ranking = []
    for f in dg:
        c = cosine(f.embedding, C["emb"])
        ranking.append({"bbox": f.bbox, "center": [round(v, 1) for v in f.center],
                        "cosine": round(c, 4),
                        "euclidean": round(euclidean(f.embedding, C["emb"]), 4),
                        "is_target": f is tgt, "same_person": c >= THR})
    ranking.sort(key=lambda r: -r["cosine"])
    print(f"\n========== 合照 {len(dg)} 张人脸 vs 全景主持人（C）==========")
    print(f"{'排名':<6}{'中心(x,y)':<14}{'余弦':<10}{'欧氏':<9}{'判定'}")
    for rank, r in enumerate(ranking, 1):
        mark = "  <== 目标" if r["is_target"] else ""
        print(f"{rank:<6}({int(r['center'][0])},{int(r['center'][1])})".ljust(20)
              + f"{r['cosine']:<10.4f}{r['euclidean']:<9.4f}"
              f"{'是' if r['same_person'] else '否'}{mark}")

    os.makedirs(args.out_dir, exist_ok=True)
    vis = img_p.copy()
    x1, y1, x2, y2 = pan.bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 60), 4)
    cv2.putText(vis, "HOST", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 60), 2)
    out = os.path.join(args.out_dir, "result_panorama_annotated.jpg")
    cv2.imwrite(out, vis)
    print(f"\n[save] {out}")

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({"backend": BACKEND, "threshold": THR, "pairs": pairs,
                       "all_consistent": all_same, "ranking_vs_panorama": ranking},
                      fh, ensure_ascii=False, indent=2)
        print(f"[save] {args.report}")


if __name__ == "__main__":
    main()
