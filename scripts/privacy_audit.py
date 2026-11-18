#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
privacy_audit —— face-match-lab 隐私终审体检
=============================================
对仓库做四层扫描，任何一层命中即判定为**未通过**：

  L1  git 历史对象    全历史是否仍存在人脸图像 blob
  L2  工作树文件      是否存在未纳入版本控制的图像文件
  L3  文本内容扫描    源码 / 文档 / 配置中是否残留坐标、路径、真实数据目录引用
  L4  产物结构        JSON 报告是否含 bbox / center / path 等敏感字段

用法
----
    python scripts/privacy_audit.py            # 对当前仓库做完整审计
    python scripts/privacy_audit.py --json     # 机器可读输出
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff")
#: 允许入库的图像（纯数据图表 + NASA 公有领域公开示例）
ALLOWED_IMAGES = (
    "assets/charts/similarity_chart.png",
    "assets/charts/confidence_bars.png",
    "samples/mercury7_group.jpg",
    "samples/glenn_portrait.jpg",
    "samples/glenn_spacesuit.jpg",
    "samples/cooper_portrait.jpg",
)

#: 公开示例的 SHA-256 指纹 —— 防止有人把 NASA 素材偷换成私人照片
#: 重新计算：sha256sum samples/*.jpg
SAMPLE_SHA256 = {
    "samples/mercury7_group.jpg":
        "110fbc442dcce4ae0aa4a6fff510513ae4c6f82b8506eed678b7722c74cc02b8",
    "samples/glenn_portrait.jpg":
        "cdd906af1375b8189245046cec481e42f827c081ad22c3cc336fc110cae330ea",
    "samples/glenn_spacesuit.jpg":
        "7df27271b632b3ad4044f16e3679dde7857dd7b07dfba6e4350c2d9442e127b3",
    "samples/cooper_portrait.jpg":
        "3327e1a5f9396647ef184ef65a85ad55d1b9297c6d3ca5fbafad098a56f7c462",
}

#: 报告中禁止出现的字段
FORBIDDEN_JSON_KEYS = ('"bbox"', '"center"', '"path"', '"face_path"', '"image_path"')

#: 文本中禁止出现的模式
#: 注：``samples/`` 现在是**合法的公开示例目录**（NASA 公有领域照片，已入库），
#: 因此不再禁止引用；改为禁止引用历史上已删除的**私人**样例目录名。
FORBIDDEN_TEXT_PATTERNS = [
    (r"(?<![\w/])samples/group\.jpg", "已删除的私人样例文件引用"),
    (r"(?<![\w/])samples/host\.", "已删除的私人样例文件引用"),
    (r"(?<![\w/])samples/classroom\.", "已删除的私人样例文件引用"),
    (r"face-match-lab/samples/(?!mercury7|glenn|cooper)", "已删除的私人样例文件引用"),
]

#: 审计脚本自身：其中必然包含模式定义（FORBIDDEN_TEXT_PATTERNS），
#: 扫描时需排除这些「定义行」，否则会自我误报。
SELF_REL = os.path.relpath(__file__, ROOT).replace(os.sep, "/")


def _matches_forbidden(text: str, rel: str) -> list[str]:
    """返回命中的描述列表；对审计脚本自身跳过模式定义行。"""
    descs = []
    for pattern, desc in FORBIDDEN_TEXT_PATTERNS:
        for line in text.splitlines():
            if rel == SELF_REL and ("FORBIDDEN_TEXT_PATTERNS" in line
                                    or re.search(r'\(r?"', line)):
                continue          # 定义行本身，非实际引用
            if re.search(pattern, line):
                descs.append(desc)
                break
    return descs


def git(*args) -> str:
    return subprocess.run(["git", "-C", ROOT, *args],
                          capture_output=True, text=True).stdout


def l1_history_images() -> list[str]:
    """L1：全历史对象中的图像文件路径（允许清单内的纯数据图表除外）。"""
    out = git("rev-list", "--objects", "--all")
    hits = []
    for line in out.splitlines():
        path = line.split(" ", 1)[1] if " " in line else ""
        if not path.lower().endswith(IMAGE_EXT):
            continue
        if path in ALLOWED_IMAGES:
            continue
        hits.append(path)
    return sorted(set(hits))


def l2_worktree_images() -> list[str]:
    """L2：**会被提交**的文件中是否存在图像文件（允许清单除外）。

    注意：不能直接扫工作树 —— ``data/`` 与 ``results/`` 本就被
    ``.gitignore`` 排除，里面的图像是正常的运行时产物。
    这里以 ``git ls-files``（已跟踪文件）+ ``git status``（未跟踪且
    未被忽略的文件）为准，只检查真正会进入版本库的内容。
    """
    hits = []
    tracked = git("ls-files").splitlines()
    for rel in tracked:
        if rel.lower().endswith(IMAGE_EXT) and rel not in ALLOWED_IMAGES:
            hits.append(rel)

    # 未跟踪但未被忽略的文件（git status --porcelain 输出以 ?? 开头）
    for line in git("status", "--porcelain", "--untracked-files=all").splitlines():
        if not line.startswith("?? "):
            continue
        rel = line[3:].strip().strip('"')
        if rel.lower().endswith(IMAGE_EXT) and rel not in ALLOWED_IMAGES:
            hits.append(rel)
    return sorted(set(hits))


def l3_text_scan() -> list[tuple[str, str]]:
    """L3：会被提交的文本文件中的禁止模式。

    只检查已跟踪文件与未忽略的新增文件 —— ``data/`` 下的说明文件
    属于使用者本地目录，不构成仓库泄露。
    """
    hits = []
    exts = (".py", ".md", ".yml", ".yaml", ".txt", ".json", ".cfg", ".toml", ".in")
    candidates = set(git("ls-files").splitlines())
    for line in git("status", "--porcelain", "--untracked-files=all").splitlines():
        if line.startswith("?? "):
            candidates.add(line[3:].strip().strip('"'))

    for rel in sorted(candidates):
        if not rel.endswith(exts):
            continue
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            continue
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for desc in _matches_forbidden(text, rel):
            hits.append((rel, desc))
    return sorted(set(hits))


def l4_report_fields() -> list[tuple[str, str]]:
    """L4：会被提交的 JSON 报告中的敏感字段。"""
    hits = []
    candidates = set(git("ls-files").splitlines())
    for line in git("status", "--porcelain", "--untracked-files=all").splitlines():
        if line.startswith("?? "):
            candidates.add(line[3:].strip().strip('"'))
    for rel in sorted(candidates):
        if not rel.endswith(".json"):
            continue
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            continue
        text = open(path, encoding="utf-8", errors="ignore").read()
        for key in FORBIDDEN_JSON_KEYS:
            if key in text:
                hits.append((rel, f"含敏感字段 {key}"))
    return sorted(set(hits))


def l5_gitignore_guard() -> list[str]:
    """L5：.gitignore 是否真正拦住了数据目录与图像文件。"""
    problems = []
    # 用 --no-index 检查尚不存在于索引中的路径是否会被忽略
    out = git("check-ignore", "-v", "--no-index",
              "data/group.jpg", "data/person_01/01.jpg",
              "results/result_group_annotated.jpg", "results/crops/x.jpg")
    ignored = set()
    for line in out.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            ignored.add(parts[1].strip())
    for probe in ("data/group.jpg", "data/person_01/01.jpg",
                  "results/result_group_annotated.jpg", "results/crops/x.jpg"):
        if probe not in ignored:
            problems.append(f"{probe} 未被 .gitignore 拦截")
    return problems


def l6_sample_integrity() -> list[str]:
    """L6：公开示例文件的 SHA-256 是否与登记的指纹一致。

    ``samples/`` 里的图像是仓库内**唯一**的真实人脸素材，
    必须确保它们始终是登记在册的 NASA 公有领域原图，
    而不会在某次提交中被悄悄替换成私人照片。
    """
    problems = []
    for rel, expected in SAMPLE_SHA256.items():
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            problems.append(f"{rel} 缺失")
            continue
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if digest != expected:
            problems.append(f"{rel} 指纹不匹配（期望 {expected[:12]}…，实际 {digest[:12]}…）")
    # 反向检查：samples/ 下不应存在白名单之外的图像
    samples_dir = os.path.join(ROOT, "samples")
    if os.path.isdir(samples_dir):
        for name in sorted(os.listdir(samples_dir)):
            rel = f"samples/{name}"
            if name.lower().endswith(IMAGE_EXT) and rel not in SAMPLE_SHA256:
                problems.append(f"{rel} 未登记指纹")
    return problems


def run_audit() -> dict:
    result = {
        "L1_history_images": l1_history_images(),
        "L2_worktree_images": l2_worktree_images(),
        "L3_text_patterns": l3_text_scan(),
        "L4_report_fields": l4_report_fields(),
        "L5_ignore_guard": l5_gitignore_guard(),
        "L6_sample_integrity": l6_sample_integrity(),
    }
    result["passed"] = not any(
        result[k] for k in ("L1_history_images", "L2_worktree_images",
                            "L3_text_patterns", "L4_report_fields",
                            "L5_ignore_guard", "L6_sample_integrity")
    )
    return result


def main():
    ap = argparse.ArgumentParser(description="仓库隐私终审体检")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    res = run_audit()
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res["passed"] else 1

    print("=" * 66)
    print("  face-match-lab 隐私终审体检")
    print("=" * 66)

    def section(title, items, fmt=None):
        print(f"\n[{title}]")
        if not items:
            print("  ✓ 未发现任何命中项")
            return 0
        for it in items:
            print(f"  ✗ {fmt(it) if fmt else it}")
        return len(items)

    n = 0
    n += section("L1 全历史图像 blob", res["L1_history_images"])
    n += section("L2 工作树图像文件", res["L2_worktree_images"])
    n += section("L3 文本禁止模式", res["L3_text_patterns"],
                 lambda x: f"{x[0]} — {x[1]}")
    n += section("L4 报告敏感字段", res["L4_report_fields"],
                 lambda x: f"{x[0]} — {x[1]}")
    n += section("L5 .gitignore 兜底", res["L5_ignore_guard"])
    n += section("L6 公开示例指纹", res["L6_sample_integrity"])

    print("\n" + "=" * 66)
    if res["passed"]:
        print("  结论：✓ 隐私体检全部通过，未发现泄露")
    else:
        print(f"  结论：✗ 发现 {n} 处问题，需处理")
    print("=" * 66)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
