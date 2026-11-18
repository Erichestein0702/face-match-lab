#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data —— 数据准备与结构校验
====================================
本脚本**不包含、也不产生**任何真实人脸数据。它只做三件事：

1. ``--check``        检查 ``data/`` 目录结构是否符合约定，给出可操作的修复建议
2. ``--template``     在 ``data/`` 下生成目录骨架与说明文件（全部为文本，无图像）
3. ``--use-samples``  把仓库自带的公开示例（NASA 公有领域照片）复制到 ``data/``
4. ``--dataset``      从公开人脸数据集下载样本到 ``data/``（需用户明确指定）

隐私约定
--------
* ``data/`` 目录已被 ``.gitignore`` 排除，**永远不会进入 git 历史**
* 本仓库**只**提供 NASA 公有领域示例照片，不提供任何使用者的真实人脸样本
* 公开数据集下载仅在用户显式指定 ``--dataset`` 时发生

用法
----
    python scripts/prepare_data.py --check
    python scripts/prepare_data.py --template
    python scripts/prepare_data.py --use-samples
    python scripts/prepare_data.py --dataset lfw --num-identities 10
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import config  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

#: 顶层三张场景图的期望文件名 -> 用途说明
SCENE_FILES = {
    config.DEFAULT_GROUP_NAME: "群像图（多人，用于遍历检索）",
    config.DEFAULT_REFERENCE_NAME: "参考像（单人，用于比对基准）",
    config.DEFAULT_THIRD_NAME: "第三视角（可选，用于三角验证）",
}

#: 仓库自带公开示例 -> data/ 下的目标文件名
SAMPLE_FILES = {
    config.SAMPLE_GROUP_NAME: config.DEFAULT_GROUP_NAME,
    config.SAMPLE_REFERENCE_NAME: config.DEFAULT_REFERENCE_NAME,
    config.SAMPLE_THIRD_NAME: config.DEFAULT_THIRD_NAME,
}

README_TEMPLATE = """\
# data/ —— 本地数据目录

**本目录已被 .gitignore 排除，其中的任何文件都不会进入 git 历史。**

## 期望结构

```
data/
├── group.jpg          群像图（多人，用于遍历检索）
├── reference.jpg      参考像（单人，用于比对基准）
├── third.jpg          第三视角（可选，用于三角验证）
└── <人名>/            单人人脸样本库（可选，供检索/复现）
    ├── 01.jpg
    └── 02.jpg
```

## 放入照片前的检查清单

- [ ] 你是照片中人物的肖像权人，或已取得**明确书面授权**
- [ ] 照片仅用于本地算法验证，不会被提交到任何远程仓库
- [ ] 不涉及未成年人、证件照、监控抓拍等敏感场景
- [ ] 需要分享结果时，请只分享**统计量与图表**，不要分享原图

## 目录名约定

`<人名>/` 下的子目录名会作为匿名标识出现在报告里。
若不希望暴露真实姓名，可改用 `person_01`、`person_02` 之类的代号。
"""


def _iter_images(directory: str):
    """列出目录下的图像文件。"""
    if not os.path.isdir(directory):
        return
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXT:
            yield path


def check(data_dir: str) -> int:
    """检查数据目录结构，返回问题计数。"""
    print(f"[check] 数据目录: {data_dir}")
    problems = 0

    if not os.path.isdir(data_dir):
        print(f"  ✗ 目录不存在。可用 `--template` 生成骨架，或 `--use-samples` 填入公开示例。")
        return 1

    print("\n  【场景图】(顶层)")
    for name, desc in SCENE_FILES.items():
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✓ {name:<16} {size/1024:8.1f} KB   {desc}")
        else:
            legacy = [old for old, new in config.LEGACY_ALIASES.items()
                      if new == name and os.path.exists(os.path.join(data_dir, old))]
            if legacy:
                print(f"  ! {name:<16} 使用旧名 {legacy[0]}，建议重命名为 {name}")
                problems += 1
            else:
                print(f"  ✗ {name:<16} {'缺失':>10}   {desc}")
                problems += 1

    print("\n  【单人样本库】(data/<人名>/*.jpg)")
    subs = [d for d in sorted(os.listdir(data_dir))
            if os.path.isdir(os.path.join(data_dir, d))
            and not d.startswith((".", "__"))]
    if not subs:
        print("  - 未发现任何子目录（可选，不影响主流程）")
    for sub in subs:
        subpath = os.path.join(data_dir, sub)
        imgs = list(_iter_images(subpath))
        if imgs:
            print(f"  ✓ {sub:<20} {len(imgs)} 张图片")
        else:
            # 空目录（仅 .gitkeep 等占位文件）属于「模板骨架」，
            # 说明用户还没往里放照片，不是结构错误，不计入问题数。
            placeholder = [n for n in os.listdir(subpath)
                           if not n.startswith(".")]
            print(f"  - {sub:<20} 空目录（模板骨架，可选）"
                  + (f"，含占位文件 {placeholder}" if placeholder else ""))
    print("\n  【检查结果】")
    if problems == 0:
        print("  ✓ 结构完整，可以运行主流程：")
        print("      python src/face_compare.py")
    else:
        print(f"  发现 {problems} 个问题：")
        print("      · 补齐缺失的场景图，或用 --group/--reference/--third 直接指定路径")
        print("      · 目录名建议用代号，避免在报告中暴露真实姓名")
        print("      · 结构确认无误后可忽略（非阻塞项）")
    return problems


def make_template(data_dir: str) -> int:
    """生成目录骨架与说明文件（纯文本，无图像）。"""
    os.makedirs(data_dir, exist_ok=True)
    readme = os.path.join(data_dir, "README.md")
    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as fh:
            fh.write(README_TEMPLATE)
        print(f"[template] 已创建 {readme}")
    else:
        print(f"[template] 已存在，跳过 {readme}")

    for person in ("person_01", "person_02"):
        pdir = os.path.join(data_dir, person)
        os.makedirs(pdir, exist_ok=True)
        keep = os.path.join(pdir, ".gitkeep")
        if not os.path.exists(keep):
            open(keep, "w").close()
        print(f"[template] 已创建 {pdir}/")
    print("\n下一步：把你的照片按上述结构放入 data/，再运行 --check 复核。")
    return 0


def use_samples(data_dir: str) -> int:
    """把仓库自带的公开示例复制到 data/，作为零配置上手素材。"""
    print(f"[samples] 公开示例目录: {config.SAMPLES_DIR}")
    os.makedirs(data_dir, exist_ok=True)
    copied = 0
    for src_name, dst_name in SAMPLE_FILES.items():
        src = config.sample(src_name)
        dst = os.path.join(data_dir, dst_name)
        if not os.path.exists(src):
            print(f"  ✗ 缺少示例文件: {src}")
            continue
        shutil.copyfile(src, dst)
        print(f"  ✓ {src_name:<26} -> {dst_name:<14} "
              f"{os.path.getsize(dst)/1024:7.1f} KB")
        copied += 1
    print(f"\n[done] 已复制 {copied} 张公开示例（NASA 公有领域照片）-> {data_dir}")
    print("       现在可以直接运行：")
    print("           python src/face_compare.py --data-dir data")
    return 0 if copied else 1


# --------------------------------------------------------------------------- #
# 公开数据集
# --------------------------------------------------------------------------- #
DATASETS = {
    "lfw": {
        "desc": "Labeled Faces in the Wild（宾夕法尼亚大学，非商业研究用）",
        "loader": "sklearn.datasets.fetch_lfw_people",
        "url": "http://vis-www.cs.umass.edu/lfw/",
    },
    "olivetti": {
        "desc": "AT&T ORL  Olivetti 人脸库（学术免费使用）",
        "loader": "sklearn.datasets.fetch_olivetti_faces",
        "url": "https://cam-orl.co.uk/facedatabase.html",
    },
}


def download(dataset: str, data_dir: str, num_identities: int) -> int:
    """从公开数据集下载样本，落地为 data/<人名>/*.jpg 结构。"""
    if dataset not in DATASETS:
        print(f"[error] 未知数据集: {dataset}")
        print(f"        可选: {', '.join(DATASETS)}")
        return 1

    meta = DATASETS[dataset]
    print(f"[download] 数据集: {dataset} — {meta['desc']}")
    print(f"[download] 主页: {meta['url']}")
    print("[download] 首次运行需要联网下载，可能耗时较长\n")

    try:
        if dataset == "lfw":
            from sklearn.datasets import fetch_lfw_people
            data = fetch_lfw_people(min_faces_per_person=1, resize=None,
                                    color=True, download_if_missing=True)
            names = list(data.target_names)
            images, labels = data.images, data.target
        else:
            from sklearn.datasets import fetch_olivetti_faces
            data = fetch_olivetti_faces(download_if_missing=True)
            names = [f"person_{i+1:02d}" for i in range(data.target.shape[1])]
            images = data.images.reshape(-1, 64, 64)
            labels = data.target.argmax(axis=1)
    except Exception as exc:
        print(f"[error] 数据集下载失败: {exc}")
        print("        可手动下载后按 data/<人名>/*.jpg 结构放入，")
        print(f"        结构说明见 {meta['url']}")
        return 1

    try:
        import cv2
    except ImportError:
        print("[error] 需要 opencv 才能写出图片：pip install opencv-python-headless")
        return 1

    out_root = os.path.join(data_dir, dataset)
    os.makedirs(out_root, exist_ok=True)
    counts: dict[str, int] = {}
    for img, label in zip(images, labels):
        person = names[label]
        pdir = os.path.join(out_root, person)
        os.makedirs(pdir, exist_ok=True)
        idx = counts.get(person, 0) + 1
        if idx > num_identities and num_identities > 0:
            continue
        counts[person] = idx
        arr = img
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[-1] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        path = os.path.join(pdir, f"{idx:03d}.jpg")
        cv2.imwrite(path, arr)

    total = sum(counts.values())
    print(f"\n[done] 已写入 {total} 张图片，{len(counts)} 个身份目录 -> {out_root}")
    print(f"       目录结构: {out_root}/<人名>/001.jpg ...")
    print("\n注意：公开数据集的图像同样受各自许可约束，商用前请自行确认授权范围。")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="数据准备与结构校验（不含任何真实人脸数据）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python scripts/prepare_data.py --check\n"
               "  python scripts/prepare_data.py --template\n"
               "  python scripts/prepare_data.py --use-samples\n"
               "  python scripts/prepare_data.py --dataset lfw --num-identities 10\n",
    )
    ap.add_argument("--data-dir", default=None,
                    help=f"数据目录（默认 {config.DATA_DIR}）")
    ap.add_argument("--check", action="store_true", help="检查目录结构")
    ap.add_argument("--template", action="store_true", help="生成目录骨架与说明")
    ap.add_argument("--use-samples", action="store_true",
                    help="把仓库自带的公开示例复制到数据目录")
    ap.add_argument("--dataset", default=None,
                    choices=sorted(DATASETS), help="从公开数据集下载样本")
    ap.add_argument("--num-identities", type=int, default=10,
                    help="每个身份最多保留的图片数（0 = 不限，默认 10）")
    args = ap.parse_args()

    data_dir = config.resolve_dir(args.data_dir, config.DATA_DIR)

    if args.dataset:
        return download(args.dataset, data_dir, args.num_identities)
    if args.template:
        return make_template(data_dir)
    if args.use_samples:
        return use_samples(data_dir)
    # 默认行为就是 check
    return check(data_dir)


if __name__ == "__main__":
    sys.exit(main())
