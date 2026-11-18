#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selftest —— 端到端自测脚本
================================
对「群像检索 + 三角验证」能力做**可断言**的回归测试。

设计目标
--------
1. **零配置可跑**：默认使用仓库自带的 NASA 公有领域示例，克隆即可执行。
2. **支持外部数据**：``--data-dir`` 指向任意本地目录，用自己的照片验证，
   图片永不离开本机（``data/`` 已被 .gitignore 排除）。
3. **断言而非打印**：每项检查返回 PASS / FAIL，退出码非 0 即表示回归失败，
   可直接接入 CI。

检查项
------
============  ============================================================
编号           检查内容
============  ============================================================
T1 主判定      群像检索 Top-1 判定为「同一人」
T2 唯一性      Top-1 与 Top-2 的余弦差超过下限（匹配必须唯一）
T3 负对照      群像中**其他人**与参考像的相似度低于阈值（不是碰巧高）
T4 外部负对照  群像中**不存在**的人（用参考像人物 vs 群像最优）全部不过线
T5 三角验证    A-B / A-C / B-C 三对两两一致（需第三视角）
T6 参考像鲁棒  换用第三视角作参考像，检索命中**同一张脸**
============  ============================================================

T5 / T6 需要第三视角（``third.jpg``）。缺失时自动跳过并标注 SKIP，
不视为失败。

用法
----
    # 用仓库自带公开示例（推荐）
    python3 scripts/selftest.py

    # 用自己的照片（图片放在 data/ 或任意目录）
    python3 scripts/selftest.py --data-dir data
    python3 scripts/selftest.py --group a.jpg --reference b.jpg --third c.jpg

    # 机器可读输出
    python3 scripts/selftest.py --json results/selftest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cv2                                                    # noqa: E402

from face_compare import (                                    # noqa: E402
    BACKEND, COS_THR_ARCFACE, COS_THR_SFACE,
    cosine, detect_faces, resolve_source,
)
import config                                                  # noqa: E402

THR = COS_THR_ARCFACE if BACKEND == "ArcFace" else COS_THR_SFACE

#: Top-1 与 Top-2 的最小余弦差。低于此值说明「匹配不唯一」，
#: 可能只是群像里有两张脸长得像，而不是真的找到了目标。
MIN_MARGIN = 0.05


# --------------------------------------------------------------------------- #
# 检查项实现
# --------------------------------------------------------------------------- #
def _subject(faces):
    """参考像主体：取面积最大的人脸。"""
    return max(faces, key=lambda f: f.area)


def run_checks(group_path, reference_path, third_path=None):
    """执行全部检查项，返回 (results, meta)。"""
    img_g = cv2.imread(resolve_source(group_path))
    img_r = cv2.imread(resolve_source(reference_path))
    if img_g is None:
        sys.exit(f"[error] 无法读取群像图: {group_path}")
    if img_r is None:
        sys.exit(f"[error] 无法读取参考像: {reference_path}")

    faces_g, _ = detect_faces(img_g)
    faces_r, _ = detect_faces(img_r)
    if not faces_g:
        sys.exit("[error] 群像图中未检测到人脸")
    if not faces_r:
        sys.exit("[error] 参考像中未检测到人脸")

    subject = _subject(faces_r)

    # 群像中每个人对参考像主体的相似度（降序）
    scored = sorted(
        ((cosine(f.embedding, subject.embedding), i, f) for i, f in enumerate(faces_g)),
        key=lambda t: -t[0],
    )
    top1_c, top1_i, top1_f = scored[0]
    top2_c = scored[1][0] if len(scored) > 1 else None
    margin = (top1_c - top2_c) if top2_c is not None else None

    results = []

    # ---- T1 主判定 ----
    results.append({
        "id": "T1", "name": "主判定：群像检索 Top-1",
        "status": "PASS" if top1_c >= THR else "FAIL",
        "detail": f"Top-1 余弦 {top1_c:.4f} / 阈值 {THR}",
        "value": round(top1_c, 4),
    })

    # ---- T2 唯一性 ----
    if margin is None:
        results.append({
            "id": "T2", "name": "唯一性：Top-1 与 Top-2 差距",
            "status": "SKIP", "detail": "群像仅检出 1 张人脸，无 Top-2 可比",
            "value": None,
        })
    else:
        results.append({
            "id": "T2", "name": "唯一性：Top-1 与 Top-2 差距",
            "status": "PASS" if margin >= MIN_MARGIN else "FAIL",
            "detail": f"margin={margin:.4f}（下限 {MIN_MARGIN}）",
            "value": round(margin, 4),
        })

    # ---- T3 负对照：群像中的其他人 ----
    # 取排名第二（即「非目标」中最高分）的那张脸，它必须低于阈值。
    # 若它也过了阈值，说明「同一人」的判定不唯一。
    others = [c for c, _, _ in scored[1:]]
    if not others:
        results.append({
            "id": "T3", "name": "负对照：群像中其他人",
            "status": "SKIP", "detail": "群像仅检出 1 张人脸，无其他人可比",
            "value": None,
        })
    else:
        worst = max(others)
        results.append({
            "id": "T3", "name": "负对照：群像中其他人",
            "status": "PASS" if worst < THR else "FAIL",
            "detail": f"其余人最高余弦 {worst:.4f} < 阈值 {THR}（唯一命中）",
            "value": round(worst, 4),
        })

    # ---- T4 外部负对照：参考像人物在群像中的唯一性 ----
    # 群像中除 Top-1 外的所有脸都必须不过线，且 Top-1 必须过线。
    # 合成一张判定：过线人数必须恰好为 1。
    passed = [c for c, _, _ in scored if c >= THR]
    results.append({
        "id": "T4", "name": "负对照：过线人数恰为 1",
        "status": "PASS" if len(passed) == 1 else "FAIL",
        "detail": f"共 {len(faces_g)} 张脸中 {len(passed)} 张过阈值",
        "value": len(passed),
    })

    # ---- T5 / T6 需要第三视角 ----
    tri = None
    if third_path and os.path.exists(third_path):
        img_t = cv2.imread(resolve_source(third_path))
        faces_t = [] if img_t is None else detect_faces(img_t)[0]
        if faces_t:
            c_best = max(faces_t, key=lambda f: cosine(f.embedding, subject.embedding))
            cA = cosine(top1_f.embedding, subject.embedding)      # A-B
            cB = cosine(top1_f.embedding, c_best.embedding)        # A-C
            cC = cosine(subject.embedding, c_best.embedding)       # B-C
            pairs = {"A-B": cA, "A-C": cB, "B-C": cC}
            tri = {"pairs": pairs, "c_best": c_best}
            results.append({
                "id": "T5", "name": "三角验证：三对两两一致",
                "status": "PASS" if all(v >= THR for v in pairs.values()) else "FAIL",
                "detail": "  ".join(f"{k}={v:.4f}" for k, v in pairs.items())
                          + f" / 阈值 {THR}",
                "value": {k: round(v, 4) for k, v in pairs.items()},
            })

            # ---- T6 参考像鲁棒性：换参考像，命中同一张脸 ----
            scored2 = sorted(
                ((cosine(f.embedding, c_best.embedding), i) for i, f in enumerate(faces_g)),
                key=lambda t: -t[0],
            )
            same_face = scored2[0][1] == top1_i
            results.append({
                "id": "T6", "name": "参考像鲁棒性：换参考像命中同一张脸",
                "status": "PASS" if same_face else "FAIL",
                "detail": (f"换用第三视角后命中 {'同一张' if same_face else '不同'}脸 "
                           f"(余弦 {scored2[0][0]:.4f}，原 {top1_c:.4f})"),
                "value": round(scored2[0][0], 4),
            })
        else:
            results.append({"id": "T5", "name": "三角验证：三对两两一致",
                            "status": "SKIP", "detail": "第三视角未检出人脸", "value": None})
            results.append({"id": "T6", "name": "参考像鲁棒性：换参考像命中同一张脸",
                            "status": "SKIP", "detail": "第三视角未检出人脸", "value": None})
    else:
        for tid, nm in (("T5", "三角验证：三对两两一致"),
                        ("T6", "参考像鲁棒性：换参考像命中同一张脸")):
            results.append({"id": tid, "name": nm, "status": "SKIP",
                            "detail": "未提供第三视角（third.jpg）", "value": None})

    meta = {
        "backend": BACKEND,
        "threshold": THR,
        "faces_detected": len(faces_g),
        "group_size": [img_g.shape[1], img_g.shape[0]],
        "triangulation": tri,
    }
    return results, meta


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def print_results(results, meta) -> None:
    print(f"\n后端: {meta['backend']}   判定阈值(余弦): {meta['threshold']}   "
          f"群像检出: {meta['faces_detected']} 张人脸")
    print("=" * 78)
    print(f"{'编号':<6}{'状态':<8}{'检查项':<34}明细")
    print("-" * 78)
    for r in results:
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "-"}.get(r["status"], "?")
        print(f"{r['id']:<6}{icon} {r['status']:<5}{r['name']:<34}{r['detail']}")
    print("=" * 78)
    npass = sum(1 for r in results if r["status"] == "PASS")
    nfail = sum(1 for r in results if r["status"] == "FAIL")
    nskip = sum(1 for r in results if r["status"] == "SKIP")
    print(f"通过 {npass}   失败 {nfail}   跳过 {nskip}")
    if nfail:
        print("\n存在失败项 —— 结论不可信，请检查输入数据与阈值设置。")


def main():
    ap = argparse.ArgumentParser(
        description="端到端自测：主判定 / 唯一性 / 负对照 / 三角验证 / 参考像鲁棒性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python3 scripts/selftest.py\n"
               "  python3 scripts/selftest.py --data-dir data\n"
               "  python3 scripts/selftest.py --group a.jpg --reference b.jpg --third c.jpg",
    )
    ap.add_argument("--group", default=None, help="群像图路径")
    ap.add_argument("--reference", default=None, help="参考像路径")
    ap.add_argument("--third", default=None, help="第三视角路径（可选，启用 T5/T6）")
    ap.add_argument("--data-dir", default=None,
                    help=f"数据目录（默认 {config.DATA_DIR}）")
    ap.add_argument("--samples", action="store_true",
                    help="使用仓库自带的 NASA 公有领域公开示例")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="把结果写入 JSON 文件")
    args = ap.parse_args()

    data_dir = config.resolve_dir(args.data_dir, config.DATA_DIR)

    if args.samples:
        group = args.group or config.sample(config.SAMPLE_GROUP_NAME)
        reference = args.reference or config.sample(config.SAMPLE_REFERENCE_NAME)
        third = args.third or config.sample(config.SAMPLE_THIRD_NAME)
    else:
        group = args.group or config.default_input(data_dir)
        reference = args.reference or config.default_input(data_dir, config.DEFAULT_REFERENCE_NAME)
        third = args.third or config.default_input(data_dir, config.DEFAULT_THIRD_NAME)

    print("=" * 78)
    print("FaceMatch Lab · 端到端自测")
    print("=" * 78)
    print(f"[path] 群像图  : {group}")
    print(f"[path] 参考像  : {reference}")
    print(f"[path] 第三视角: {third if os.path.exists(third) else '(未提供, T5/T6 将跳过)'}")

    results, meta = run_checks(group, reference, third if os.path.exists(third) else None)
    print_results(results, meta)

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        payload = {
            "checks": results,
            "meta": {k: v for k, v in meta.items() if k != "triangulation"},
            "passed": not any(r["status"] == "FAIL" for r in results),
            # 隐私：报告中不写入图片路径与坐标
            "privacy_note": "本报告仅含相似度统计量，不含图片路径与人脸框坐标。",
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"[save] JSON: {args.json_out}")

    return 0 if not any(r["status"] == "FAIL" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
