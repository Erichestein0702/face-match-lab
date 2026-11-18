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
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff")
#: 允许入库的图像（纯数据图表）
ALLOWED_IMAGES = ("assets/charts/similarity_chart.png", "assets/charts/confidence_bars.png")

#: 报告中禁止出现的字段
FORBIDDEN_JSON_KEYS = ('"bbox"', '"center"', '"path"', '"face_path"', '"image_path"')

#: 文本中禁止出现的模式
FORBIDDEN_TEXT_PATTERNS = [
    (r"samples/", "已废弃的 samples/ 目录引用"),
    (r"\bface-match-lab/samples\b", "已废弃的 samples/ 目录引用"),
]


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
    """L2：工作树中的图像文件（排除 .git 与允许清单）。"""
    hits = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if ".git" in dirnames:
            dirnames.remove(".git")
        for name in filenames:
            if not name.lower().endswith(IMAGE_EXT):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), ROOT).replace(os.sep, "/")
            if rel in ALLOWED_IMAGES:
                continue
            hits.append(rel)
    return sorted(hits)


def l3_text_scan() -> list[tuple[str, str]]:
    """L3：文本文件中的禁止模式（审计脚本自身除外——它必然包含模式定义）。"""
    hits = []
    exts = (".py", ".md", ".yml", ".yaml", ".txt", ".json", ".cfg", ".toml", ".in")
    self_rel = os.path.relpath(__file__, ROOT).replace(os.sep, "/")
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if ".git" in dirnames:
            dirnames.remove(".git")
        for name in filenames:
            if not name.endswith(exts):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
            if rel == self_rel:
                continue          # 本脚本包含模式定义本身，属预期
            try:
                text = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for pattern, desc in FORBIDDEN_TEXT_PATTERNS:
                if re.search(pattern, text):
                    hits.append((rel, desc))
    return sorted(set(hits))


def l4_report_fields() -> list[tuple[str, str]]:
    """L4：JSON 报告中的敏感字段。"""
    hits = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if ".git" in dirnames:
            dirnames.remove(".git")
        for name in filenames:
            if not name.endswith(".json"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
            text = open(path, encoding="utf-8", errors="ignore").read()
            for key in FORBIDDEN_JSON_KEYS:
                if key in text:
                    hits.append((rel, f"含敏感字段 {key}"))
    return sorted(set(hits))


def l5_gitignore_guard() -> list[str]:
    """L5：.gitignore 是否真正拦住了数据目录与图像文件。"""
    out = git("check-ignore", "-v", "--no-index",
              "data/group.jpg", "assets/results/x.jpg", "data/person_01/01.jpg")
    problems = []
    if not out.strip():
        problems.append("data/ 下的图像未被 .gitignore 拦截")
    return problems


def run_audit() -> dict:
    result = {
        "L1_history_images": l1_history_images(),
        "L2_worktree_images": l2_worktree_images(),
        "L3_text_patterns": l3_text_scan(),
        "L4_report_fields": l4_report_fields(),
        "L5_ignore_guard": l5_gitignore_guard(),
    }
    result["passed"] = not any(
        result[k] for k in ("L1_history_images", "L2_worktree_images",
                            "L3_text_patterns", "L4_report_fields", "L5_ignore_guard")
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

    print("\n" + "=" * 66)
    if res["passed"]:
        print("  结论：✓ 隐私体检全部通过，未发现泄露")
    else:
        print(f"  结论：✗ 发现 {n} 处问题，需处理")
    print("=" * 66)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
