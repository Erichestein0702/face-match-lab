#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
triangulate —— 三图三角验证
================================
用三张互相独立的照片构成证据链，交叉验证「同一人」结论，避免单张误判：

    A  群像图中的目标人物
    B  参考像（单人特写）
    C  第三视角（另一张照片中的同一人）

A-B、A-C、B-C 三对两两比对，若均判定「同一人」，则结论高度可信。
同时输出群像图内全部人脸对 C 的排名，确认不存在得分更高的其他人。

第三视角 C 中的人物同样通过**与 B 比对后取最高分**的方式定位，
因此不依赖任何关于画面构图的位置假设。
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
    resolve_source, select_target, anonymize_ranking,
)
import config                                                      # noqa: E402

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
    print(f"{label:<34} 余弦={c:.4f} ({c*100:5.2f}%) | 欧氏={e:.4f} | "
          f"置信度={row['confidence_pct']:6.2f}% | {'同一人' if row['same_person'] else '非同一人'}")
    return row


def main():
    ap = argparse.ArgumentParser(
        description="三图三角验证",
        epilog="路径优先级：命令行 > 环境变量 > src/config.py 默认值。",
    )
    ap.add_argument("--group", default=None, help="群像图（多人）")
    ap.add_argument("--reference", default=None, help="参考像（单人）")
    ap.add_argument("--third", default=None, help="第三视角（同一人另一张照片）")
    ap.add_argument("--host", dest="host_opt", help="参考像路径的旧别名")
    ap.add_argument("--panorama", dest="panorama_opt", help="第三视角路径的旧别名")
    ap.add_argument("--data-dir", default=None, help=f"数据目录（默认 {config.DATA_DIR}）")
    ap.add_argument("--out-dir", default=None, help=f"输出目录（默认 {config.OUT_DIR}）")
    ap.add_argument("--report", default=None, help=f"JSON 输出（默认 {config.TRI_REPORT}）")
    ap.add_argument("--samples", action="store_true",
                    help="使用仓库自带的公开示例（NASA 公有领域照片）")
    ap.add_argument("--target-strategy", default="retrieve",
                    choices=("retrieve", "largest", "front-right", "white-shirt"),
                    help="群像图中目标人物的定位策略（默认 retrieve）")
    args = ap.parse_args()

    data_dir = config.resolve_dir(args.data_dir, config.DATA_DIR)
    # --samples 只填补「用户未显式指定」的路径，显式指定的优先级更高
    if args.samples:
        args.group = args.group or config.sample(config.SAMPLE_GROUP_NAME)
        args.reference = (args.reference or args.host_opt
                          or config.sample(config.SAMPLE_REFERENCE_NAME))
        args.third = (args.third or args.panorama_opt
                      or config.sample(config.SAMPLE_THIRD_NAME))
    else:
        args.group = args.group or config.default_input(data_dir)
        args.reference = (args.reference or args.host_opt
                          or config.default_input(data_dir, config.DEFAULT_REFERENCE_NAME))
        args.third = (args.third or args.panorama_opt
                      or config.default_input(data_dir, config.DEFAULT_THIRD_NAME))
    args.out_dir = config.resolve_dir(args.out_dir, config.OUT_DIR)
    args.report = config.resolve_dir(args.report, config.TRI_REPORT)

    print(f"[path] 群像图  : {args.group}")
    print(f"[path] 参考像  : {args.reference}")
    print(f"[path] 第三视角: {args.third}")

    img_g = cv2.imread(resolve_source(args.group))
    img_c = cv2.imread(resolve_source(args.reference))
    img_p = cv2.imread(resolve_source(args.third))
    missing = [p for p, im in ((args.group, img_g), (args.reference, img_c),
                               (args.third, img_p)) if im is None]
    if missing:
        print("[error] 以下图片无法读取：")
        for p in missing:
            print(f"          - {p}")
        print("\n提示：把照片放进数据目录后重试，结构应为：")
        print(f"          data/{config.DEFAULT_GROUP_NAME}      （群像，多人）")
        print(f"          data/{config.DEFAULT_REFERENCE_NAME}      （参考像，单人）")
        print(f"          data/{config.DEFAULT_THIRD_NAME}      （第三视角，同一人）")
        print("        也可加 --samples 直接使用仓库自带的公开示例，")
        print("        或用 --group/--reference/--third 直接指定路径。")
        sys.exit(1)

    hg, _ = img_g.shape[:2]

    dg, _ = detect_faces(img_g)
    dc, _ = detect_faces(img_c)
    dp, _ = detect_faces(img_p)
    print(f"检测数: 群像={len(dg)}  参考像={len(dc)}  第三视角={len(dp)}\n")

    # --- B: 参考像中的主体（最大人脸） ---
    if not dc:
        sys.exit("[error] 参考像中未检测到人脸")
    subject = max(dc, key=lambda f: f.area)
    print(f"B 参考像主体: bbox={subject.bbox} 中心={subject.center}")

    # --- A: 群像图中的目标 ---
    tgt, target_info = select_target(dg, img_g, args.target_strategy,
                                     subject_emb=subject.embedding)
    if tgt is None:
        sys.exit("[error] 群像图中未检测到人脸")
    print(f"A 群像目标: bbox={tgt.bbox} 定位方式={target_info['matched_by']}")

    # --- C: 第三视角中与 B 最相似的人 ---
    if not dp:
        sys.exit("[error] 第三视角中未检测到人脸")
    c_best = max(dp, key=lambda f: cosine(f.embedding, subject.embedding))
    print(f"C 第三视角人物: bbox={c_best.bbox} 中心={c_best.center}\n")

    A = {"emb": tgt.embedding}
    B = {"emb": subject.embedding}
    C = {"emb": c_best.embedding}

    print("========== 三角验证 ==========")
    pairs = [
        pair(A, B, "A 群像目标  vs  B 参考像"),
        pair(A, C, "A 群像目标  vs  C 第三视角"),
        pair(B, C, "B 参考像    vs  C 第三视角"),
    ]
    all_same = all(p["same_person"] for p in pairs)
    print(f"\n三对全部一致: {'确认为同一人' if all_same else '存在不一致，需人工复核'}")

    # --- 群像图全部人脸 vs 第三视角人物，验证唯一性 ---
    # 与 face_compare 一致：保留未舍入余弦用于排序与 margin 计算
    ranking = []
    for f in dg:
        c = cosine(f.embedding, C["emb"])
        ranking.append({"bbox": f.bbox, "center": [round(v, 1) for v in f.center],
                        "position": "", "cosine": round(c, 4), "_cos_raw": c,
                        "euclidean": round(euclidean(f.embedding, C["emb"]), 4),
                        "is_target": f is tgt, "same_person": c >= THR})
    ranking.sort(key=lambda r: -r["_cos_raw"])
    print(f"\n========== 群像图 {len(dg)} 张人脸 vs 第三视角人物（C）==========")
    print(f"{'排名':<6}{'余弦':<10}{'欧氏':<9}{'判定'}")
    for rank, r in enumerate(ranking, 1):
        mark = "  <== 目标" if r["is_target"] else ""
        print(f"#{rank:<5}{r['cosine']:<10.4f}{r['euclidean']:<9.4f}"
              f"{'是' if r['same_person'] else '否'}{mark}")
    margin = ((ranking[0]["_cos_raw"] - ranking[1]["_cos_raw"])
              if len(ranking) > 1 else None)

    os.makedirs(args.out_dir, exist_ok=True)
    vis = img_p.copy()
    x1, y1, x2, y2 = c_best.bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 60), 4)
    cv2.putText(vis, "SUBJECT", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 60), 2)
    out = os.path.join(args.out_dir, "result_third_annotated.jpg")
    cv2.imwrite(out, vis)
    print(f"\n[save] {out}")

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        payload = {"backend": BACKEND, "threshold": THR, "pairs": pairs,
                   "all_consistent": all_same,
                   "target_strategy": target_info.get("strategy"),
                   "ranking_vs_third": anonymize_ranking(ranking),
                   "privacy_note": "本报告已移除人脸框坐标与原始图片路径；"
                                   "index 为按相似度排序后的匿名序号。"}
        if margin is not None:
            payload["ranking_margin"] = {
                "top1_minus_top2": round(margin, 4),
                "note": "Top-1 与 Top-2 的余弦差，差值越大说明匹配越唯一",
            }
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"[save] {args.report}")


if __name__ == "__main__":
    main()
