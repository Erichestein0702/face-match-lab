#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config —— 集中管理路径与运行参数
=================================
设计原则：**代码里不写死任何用户数据路径**。

优先级（从高到低）：

    1. 命令行显式参数      --data-dir / --out-dir / --report ...
    2. 环境变量            FACE_DATA_DIR / FACE_OUT_DIR / FACE_REPORT ...
    3. 本模块内置默认值

数据目录约定
------------
``data/`` 为使用者自己的数据根目录，已在 ``.gitignore`` 中排除，**永不入库**：

    data/
    ├── group.jpg          群像图（画面中有多个人，用于遍历检索）
    ├── reference.jpg      参考像（单个人，用于比对基准）
    ├── third.jpg          第三视角（可选，用于三角验证）
    └── <人名>/            单人人脸样本库，供检索/复现使用
        ├── 01.jpg
        └── 02.jpg

``samples/`` 为仓库自带的**公开示例**（NASA 公有领域照片），
可直接运行，也可复制到 ``data/`` 作为上手素材。

其中前三个文件是 ``face_compare.py`` / ``triangulate.py`` 的默认输入；
``<人名>/`` 子目录由 ``scripts/prepare_data.py`` 负责校验。
"""
from __future__ import annotations

import os

# 仓库根目录（本文件位于 <repo>/src/config.py）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def env(name: str, default: str) -> str:
    """读取环境变量，空串视为未设置。"""
    value = os.environ.get(name, "")
    return value if value else default


def _env_path(name: str, default_rel: str) -> str:
    """环境变量路径统一解析为绝对路径（相对仓库根目录）。"""
    raw = env(name, "")
    if not raw:
        return os.path.join(ROOT, default_rel)
    return os.path.abspath(os.path.expanduser(raw))


# --------------------------------------------------------------------------- #
# 数据与输出目录
# --------------------------------------------------------------------------- #
#: 数据根目录：使用者自己的照片放在这里，已被 .gitignore 排除
DATA_DIR = _env_path("FACE_DATA_DIR", "data")

#: 仓库自带的公开示例目录（NASA 公有领域照片，已入库）
SAMPLES_DIR = _env_path("FACE_SAMPLES_DIR", "samples")

#: 标注图 / 裁剪图 / JSON 报告的输出根目录
OUT_DIR = _env_path("FACE_OUT_DIR", "results")

#: 人脸裁剪图输出目录
CROP_DIR = _env_path("FACE_CROP_DIR", os.path.join(OUT_DIR, "crops"))

#: 主 JSON 报告路径
REPORT = _env_path("FACE_REPORT", os.path.join(OUT_DIR, "report.json"))

#: 三角验证 JSON 报告路径
TRI_REPORT = _env_path("FACE_TRI_REPORT", os.path.join(OUT_DIR, "triangulation.json"))

#: 可视化图表输出目录
CHART_DIR = _env_path("FACE_CHART_DIR", os.path.join("assets", "charts"))

#: 模型权重缓存目录（首次运行自动下载）
MODEL_ROOT = _env_path("FACE_MODEL_ROOT", os.path.join("~", ".cache", "face_models"))
if MODEL_ROOT.startswith("~"):
    MODEL_ROOT = os.path.expanduser(MODEL_ROOT)

#: 设为 1 时强制跳过 insightface，降级到 OpenCV Zoo (YuNet + SFace)
NO_ARCFACE = bool(os.environ.get("FACE_NO_ARCFACE"))


# --------------------------------------------------------------------------- #
# 默认输入文件（相对 data/ 目录）
# --------------------------------------------------------------------------- #
DEFAULT_GROUP_NAME = "group.jpg"
DEFAULT_REFERENCE_NAME = "reference.jpg"
DEFAULT_THIRD_NAME = "third.jpg"

#: 旧版命名 -> 新版命名（保持向后兼容，--check 时给出迁移提示）
LEGACY_ALIASES = {
    "host.png": DEFAULT_REFERENCE_NAME,
    "host.jpg": DEFAULT_REFERENCE_NAME,
    "classroom.png": DEFAULT_THIRD_NAME,
    "classroom.jpg": DEFAULT_THIRD_NAME,
}

# --------------------------------------------------------------------------- #
# 仓库自带公开示例（NASA 公有领域照片）
# --------------------------------------------------------------------------- #
SAMPLE_GROUP_NAME = "mercury7_group.jpg"        # 7 人群像
SAMPLE_REFERENCE_NAME = "glenn_portrait.jpg"    # 单人官方肖像
SAMPLE_THIRD_NAME = "glenn_spacesuit.jpg"      # 另一视角（戴头盔）


def sample(name: str) -> str:
    """返回仓库自带公开示例的绝对路径。"""
    return os.path.join(SAMPLES_DIR, name)


def default_input(data_dir: str | None = None, name: str = DEFAULT_GROUP_NAME) -> str:
    """给定数据目录，返回默认输入图片的绝对路径。"""
    return os.path.join(data_dir or DATA_DIR, name)


def resolve_dir(cli_value: str | None, env_value: str) -> str:
    """命令行 > 环境变量 > 默认值，统一解析目录。"""
    if cli_value:
        return os.path.abspath(os.path.expanduser(cli_value))
    return env_value


def describe() -> str:
    """打印当前生效的全部路径配置，便于排查。"""
    return "\n".join([
        f"  仓库根目录      : {ROOT}",
        f"  数据目录        : {DATA_DIR}",
        f"  公开示例目录    : {SAMPLES_DIR}",
        f"  输出目录        : {OUT_DIR}",
        f"  裁剪图目录      : {CROP_DIR}",
        f"  主报告          : {REPORT}",
        f"  三角验证报告    : {TRI_REPORT}",
        f"  图表目录        : {CHART_DIR}",
        f"  模型缓存        : {MODEL_ROOT}",
        f"  强制降级 SFace  : {'是' if NO_ARCFACE else '否'}",
    ])


if __name__ == "__main__":
    print("当前生效配置：")
    print(describe())
