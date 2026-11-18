#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face-compare —— 人脸检测 / 特征提取 / 相似度比对一体化流水线
================================================================

功能
----
1. 在**群像图**（画面中有多个人）中定位目标人物
2. 提取 512 维 ArcFace（或 128 维 SFace）深度特征
3. 计算欧氏距离 / 余弦相似度 / 置信度百分比，给出是否同一人判定
4. 遍历群像图中全部人脸，输出匹配排名表（用于「如果不是，请遍历查询其他人」）
5. 产出标注可视化图与结构化 JSON 报告

目标定位策略
------------
``--target-strategy`` 决定「群像图里要拿谁去和参考像比对」：

==============  ============================================================
策略             说明
==============  ============================================================
``retrieve``     群像图中与参考像最相似的人（默认，检索型）
``largest``      画面中面积最大的人脸（几何型）
``front-right``  画面下半区中最靠右的人脸（几何型，竖排合影场景）
``white-shirt``  下半区中靠右且肩部为白色衣物的人脸（几何型）
``index:N``      直接指定群像图中第 N 张人脸（按检测顺序，从 1 开始）
==============  ============================================================

**检索型 vs 几何型**：

* ``retrieve`` 回答「这群人里有没有这个人」——本工具的核心能力。
* 其余策略回答「我指认的这个位置的人，是不是这个人」。由于选人时
  **不看参考像**，判定结果具有独立意义，可用于交叉验证。
* 所有策略都带兜底：命中失败时退化为「最大人脸 → 唯一人脸」，
  不会因场景差异而崩溃。

后端
----
* 主后端  insightface `buffalo_l`  = SCRFD 检测器 + ArcFace(w600k_r50) 512 维特征
* 降级后端  OpenCV Zoo              = YuNet 检测器 + SFace 128 维特征
  （insightface 在部分平台无预编译 wheel 时自动切换，保证「零配置可跑」）

阈值标准
--------
* ArcFace（归一化 512 维）: 余弦 >= 0.40 判为同一人，等价于欧氏距离 <= 1.095
* SFace（128 维）:         余弦 >= 0.363 判为同一人（OpenCV Zoo 官方阈值）

用法
----
    python3 face_compare.py --group data/group.jpg --reference data/reference.jpg
    python3 face_compare.py --samples            # 直接用仓库自带的公开示例

路径配置
--------
所有路径均不写死，取值优先级：命令行参数 > 环境变量 > src/config.py 默认值。
详见 ``python3 config.py`` 或 README「数据准备」章节。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import urllib.request

# --------------------------------------------------------------------------- #
# 依赖与模型自动准备
# --------------------------------------------------------------------------- #
BANNER = r"""
  ___ _         ___                      __
 |   \| |__ ___ | _ )___ _ _ _____ __ __ | _ \ ___ _ _
 | |) | '  \(_-< | _ / _ \ '_/ _ \ V  V /  _/ _ \ '_|
 |__/|_|_|_|/__/|___/\___/_| \___/\_/\_/ \___\___/_|
"""


def _ensure(pkg: str, mod: str) -> None:
    """缺依赖就自动 pip install，保证脚本零配置可跑。"""
    if importlib.util.find_spec(mod) is None:
        print(f"[setup] pip install {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


FORCE_NO_ARCFACE = bool(os.environ.get("FACE_NO_ARCFACE"))
ARCFACE_OK = not FORCE_NO_ARCFACE

if ARCFACE_OK:
    try:
        _ensure("insightface", "insightface")
    except Exception as exc:                       # 无预编译 wheel 的平台
        print(f"[warn] insightface 安装失败 ({exc})，降级为 OpenCV Zoo (YuNet+SFace)")
        ARCFACE_OK = False

try:
    _ensure("onnxruntime", "onnxruntime")
except Exception as exc:
    print(f"[error] onnxruntime 安装失败: {exc}")
    sys.exit(1)

if importlib.util.find_spec("cv2") is None:
    try:
        _ensure("opencv-python-headless", "cv2")
    except Exception:
        _ensure("opencv-python", "cv2")
_ensure("numpy", "numpy")

import cv2                                                  # noqa: E402
import numpy as np                                         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config                                              # noqa: E402

DEFAULT_MODEL_ROOT = config.MODEL_ROOT
MODEL_ROOT = os.environ.get("FACE_MODEL_ROOT") or DEFAULT_MODEL_ROOT

# --------------------------------------------------------------------------- #
# 常量：判定阈值与几何经验值
# --------------------------------------------------------------------------- #
COS_THR_ARCFACE = 0.40          # insightface / ArcFace 常用验证阈值（余弦）
COS_THR_SFACE = 0.363           # OpenCV Zoo SFace 官方余弦判定阈值
FRONT_ROW_Y_RATIO = 0.52        # 竖排合影中「下半区」的人脸中心 y/H 下界
WHITE_MAX_SAT = 70              # 白衣判据：肩部区域平均饱和度上限
WHITE_MIN_VAL = 140             # 白衣判据：肩部区域平均明度下限


# --------------------------------------------------------------------------- #
# 模型初始化
# --------------------------------------------------------------------------- #
def init_models():
    """初始化检测 + 识别模型，返回 (backend_name, app, yunet, sface)。"""
    global ARCFACE_OK
    app = yunet = sface = None

    if ARCFACE_OK:
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name="buffalo_l",
                root=MODEL_ROOT,
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_size=(640, 640))
            app.det_model.det_thresh = 0.45
            return "ArcFace", app, None, None
        except Exception as exc:
            print(f"[warn] insightface 初始化失败 ({exc})，降级为 OpenCV Zoo")
            app, ARCFACE_OK = None, False

    # ---- 降级路径：OpenCV Zoo (YuNet + SFace) ----
    for name, url in {
        "yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/"
                      "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sface.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/"
                      "face_recognition_sface/face_recognition_sface_2021dec.onnx",
    }.items():
        dst = os.path.join(MODEL_ROOT, name)
        if not os.path.exists(dst):
            print(f"[setup] 下载 {name} ...")
            urllib.request.urlretrieve(url, dst)

    yunet = cv2.FaceDetectorYN_create(f"{MODEL_ROOT}/yunet.onnx", "", (320, 320), 0.6, 0.3, 5000)
    sface = cv2.FaceRecognizerSF_create(f"{MODEL_ROOT}/sface.onnx", "")
    return "SFace", None, yunet, sface


BACKEND, APP, YUNET, SFACE = init_models()


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def l2_norm(v):
    return v / np.linalg.norm(v)


def euclidean(a, b) -> float:
    return float(np.linalg.norm(a - b))


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def confidence_pct(cos_sim: float, thr: float, k: float = 12.0) -> float:
    """以阈值为中点的 logistic 映射：cos == thr -> 50%，越高越接近 100%。"""
    return 100.0 / (1.0 + math.exp(-k * (cos_sim - thr)))


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def det_sizes(width: int, height: int):
    """多尺度输入：保持长宽比，边长取 32 的倍数。

    小脸需要大输入才检得出来，特写在大输入下反而会漏检，故两档都跑；
    边长必须为 32 倍数，否则 SCRFD 输出锚点数与网络不匹配会直接报错。
    """
    sizes = []
    for long_side in (640, 1600):
        scale = long_side / float(max(width, height))
        w = max(32, int(round(width * scale / 32)) * 32)
        h = max(32, int(round(height * scale / 32)) * 32)
        sizes.append((w, h))
    return sizes


def resolve_source(src: str) -> str:
    """支持本地路径，也支持 http(s) 直链（手机上传图床后直接用链接）。"""
    if src.startswith("http://") or src.startswith("https://"):
        ext = os.path.splitext(src.split("?")[0])[1] or ".jpg"
        dst = os.path.join("/tmp", f"dl_{hashlib.md5(src.encode()).hexdigest()[:10]}{ext}")
        if not os.path.exists(dst):
            print(f"[download] {src}")
            urllib.request.urlretrieve(src, dst)
        return dst
    return src


class Face:
    """统一的人脸容器：bbox + 归一化 embedding。"""

    __slots__ = ("bbox", "embedding")

    def __init__(self, bbox, embedding=None):
        self.bbox = [int(v) for v in bbox]
        self.embedding = embedding

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


# --------------------------------------------------------------------------- #
# 检测 + 特征提取
# --------------------------------------------------------------------------- #
def detect_faces(img):
    """检测图片中所有人脸并提取特征，返回 (faces, backend)。"""
    height, width = img.shape[:2]
    faces = []

    if BACKEND == "ArcFace":
        for w, h in det_sizes(width, height):
            APP.det_model.input_size = (w, h)
            for f in APP.get(img):
                bbox = [int(v) for v in f.bbox]
                if all(iou(bbox, d.bbox) < 0.4 for d in faces):
                    faces.append(Face(bbox, l2_norm(f.embedding)))
        return faces, BACKEND

    # 降级：YuNet 检测 + SFace 特征
    YUNET.setInputSize((width, height))
    ok, rows = YUNET.detect(img)
    for r in [] if not ok or rows is None else rows:
        bbox = [r[0], r[1], r[0] + r[2], r[1] + r[3]]
        aligned = SFACE.alignCrop(img, r)
        faces.append(Face(bbox, l2_norm(SFACE.feature(aligned).flatten())))
    return faces, BACKEND


def row_label(cy: float, height: int) -> str:
    """按人脸中心的相对高度给出粗略的上下位置标签（仅用于报告可读性）。"""
    r = cy / height
    if r > FRONT_ROW_Y_RATIO:
        return "下半区"
    if r > 0.30:
        return "中部"
    return "上半区"


def white_cloth_check(img, bbox):
    """检测人脸下方肩部区域是否为白色衣服（低饱和度 + 高亮度）。"""
    height, width = img.shape[:2]
    x1, y1, x2, y2 = bbox
    fh = y2 - y1
    ys, ye = max(0, int(y2 + 0.10 * fh)), min(height, int(y2 + 0.90 * fh))
    xs, xe = max(0, int(x1 + 0.10 * (x2 - x1))), min(width, int(x2 - 0.10 * (x2 - x1)))
    if ye <= ys or xe <= xs:
        return False, -1.0, -1.0
    hsv = cv2.cvtColor(img[ys:ye, xs:xe], cv2.COLOR_BGR2HSV)
    sat = float(hsv[..., 1].mean())
    val = float(hsv[..., 2].mean())
    return (sat < WHITE_MAX_SAT and val > WHITE_MIN_VAL), sat, val


def save_crop(img, bbox, path, pad: float = 0.35) -> None:
    height, width = img.shape[:2]
    x1, y1, x2, y2 = bbox
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    cv2.imwrite(path, img[max(0, int(y1 - ph)):min(height, int(y2 + ph)),
                          max(0, int(x1 - pw)):min(width, int(x2 + pw))])


# --------------------------------------------------------------------------- #
# 目标定位（通用策略）
# --------------------------------------------------------------------------- #
def select_target(faces, img, strategy: str = "retrieve", subject_emb=None):
    """在群像图中选出要拿去和参考像比对的那张人脸。

    返回 ``(target, info)``；``info`` 记录命中策略与关键判据，便于复盘。

    策略语义分两类：

    * **检索型**（``retrieve``）：目标就是「群像图中与参考像最相似的人」。
      这回答的是「这群人里有没有这个人」——本工具的核心能力。
      必须提供 ``subject_emb``（参考像主体的 embedding）。
    * **几何型**（``largest`` / ``front-right`` / ``white-shirt`` / ``index:N``）：
      先按**与身份无关**的图像几何特征指定某张脸，再问「这张脸是不是参考像那个人」。
      这回答的是「我指认的这个位置的人，是不是这个人」。
      由于选人时不看参考像，判定结果具有独立意义。

    任何策略命中失败都会退化为「最大人脸」，保证流水线不中断。
    """
    if not faces:
        return None, {"strategy": strategy, "matched_by": "none"}

    height = img.shape[0]

    # ---- 检索型：与参考像最相似者 ----
    if strategy == "retrieve":
        if subject_emb is None:
            print("[warn] retrieve 策略需要参考像特征，退化为最大人脸")
            strategy = "largest"
        else:
            target = max(faces, key=lambda f: cosine(f.embedding, subject_emb))
            return target, {"strategy": "retrieve", "matched_by": "best_match_to_reference"}

    # ---- 显式指定序号 ----
    if strategy.startswith("index:"):
        try:
            idx = int(strategy.split(":", 1)[1])
        except ValueError:
            idx = 1
        if 1 <= idx <= len(faces):
            return faces[idx - 1], {"strategy": strategy, "matched_by": "explicit_index"}
        print(f"[warn] index={idx} 超出检测范围（共 {len(faces)} 张），退化为最大人脸")
        strategy = "largest"

    # ---- 下半区候选（竖排合影场景） ----
    lower = [f for f in faces if f.center[1] > FRONT_ROW_Y_RATIO * height]

    if strategy == "front-right":
        pool = lower or faces
        target = max(pool, key=lambda f: f.center[0])
        return target, {"strategy": strategy, "matched_by": "rightmost_in_lower_half"}

    if strategy == "white-shirt":
        pool = sorted(lower or faces, key=lambda f: -f.center[0])
        for f in pool:
            ok, sat, val = white_cloth_check(img, f.bbox)
            if ok:
                return f, {"strategy": strategy, "matched_by": "white_shoulder",
                           "shoulder_sat": round(sat, 1), "shoulder_val": round(val, 1)}
        target = max(pool, key=lambda f: f.center[0])
        return target, {"strategy": strategy, "matched_by": "fallback_rightmost"}

    # ---- 兜底：最大人脸 ----
    target = max(faces, key=lambda f: f.area)
    return target, {"strategy": "largest", "matched_by": "largest_face"}


# --------------------------------------------------------------------------- #
# 报告脱敏
# --------------------------------------------------------------------------- #
def anonymize_ranking(ranking):
    """剥离人脸框坐标，只保留匿名序号与统计量。

    隐私考量：``bbox`` / ``center`` 是像素坐标，一旦与原图同时泄露，
    即可反推群像图中每个人的位置。报告只保留排序后的匿名序号、位置标签
    与相似度指标，足以完整表达结论且不可还原个体位置。
    """
    out = []
    for i, r in enumerate(ranking, 1):
        out.append({
            "index": i,
            "position": r.get("position", ""),
            "cosine": r.get("cosine"),
            "euclidean": r.get("euclidean"),
            "is_target": bool(r.get("is_target")),
            "same_person": bool(r.get("same_person")),
        })
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(args):
    group_path = resolve_source(args.group)
    reference_path = resolve_source(args.reference)

    img_g = cv2.imread(group_path)
    if img_g is None:
        print(f"[error] 无法读取群像图: {group_path}")
        print("\n提示：把照片放进数据目录后重试，结构应为：")
        print(f"          data/{config.DEFAULT_GROUP_NAME}   （群像，多人）")
        print(f"          data/{config.DEFAULT_REFERENCE_NAME}   （参考像，单人）")
        print("        也可加 --samples 直接使用仓库自带的公开示例，")
        print("        或运行 `python scripts/prepare_data.py --check` 检查数据准备情况。")
        sys.exit(1)
    img_r = cv2.imread(reference_path)
    if img_r is None:
        print(f"[error] 无法读取参考像: {reference_path}")
        print("\n提示：把照片放进数据目录后重试，或用 --reference 直接指定路径。")
        sys.exit(1)

    hg, wg = img_g.shape[:2]
    hr, wr = img_r.shape[:2]

    thr = COS_THR_ARCFACE if BACKEND == "ArcFace" else COS_THR_SFACE
    dim = 512 if BACKEND == "ArcFace" else 128

    faces_g, _ = detect_faces(img_g)
    faces_r, _ = detect_faces(img_r)
    print(f"[group]    群像图 {wg}x{hg} 检测到 {len(faces_g)} 张人脸")
    print(f"[reference] 参考像 {wr}x{hr} 检测到 {len(faces_r)} 张人脸")

    # ---- 1) 参考像中的主体（取最大人脸） ----
    if not faces_r:
        sys.exit("[error] 参考像中未检测到人脸")
    subject = max(faces_r, key=lambda f: f.area)
    print(f"[reference] 主体 bbox={subject.bbox} 中心={subject.center}")

    # ---- 2) 群像图中的目标定位 ----
    target, target_info = select_target(faces_g, img_g, args.target_strategy,
                                        subject_emb=subject.embedding)
    if target is None:
        sys.exit("[error] 群像图中未检测到人脸")
    _, sat, val = white_cloth_check(img_g, target.bbox)
    print(f"[group]    目标定位 [{target_info['matched_by']}] "
          f"bbox={target.bbox} 肩部HSV S={sat:.0f} V={val:.0f}")

    os.makedirs(args.crop_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    save_crop(img_g, target.bbox, os.path.join(args.crop_dir, "target_group.jpg"))
    save_crop(img_r, subject.bbox, os.path.join(args.crop_dir, "subject_reference.jpg"))

    # ---- 3) 特征比对 ----
    dist = euclidean(target.embedding, subject.embedding)
    cos = cosine(target.embedding, subject.embedding)
    same = cos >= thr
    title = "insightface ArcFace (512 维)" if BACKEND == "ArcFace" else "OpenCV SFace (128 维)"

    print(f"\n========== {title} ==========")
    print(f"欧氏距离 Euclidean Distance : {dist:.4f}  (同一人参考: <= {math.sqrt(2 * (1 - thr)):.3f})")
    print(f"余弦相似度 Cosine Similarity: {cos:.4f}  ({cos * 100:.2f}%)")
    print(f"判定阈值(余弦)              : {thr}")
    print(f"相似度置信度                : {confidence_pct(cos, thr):.2f}%")
    print(f"判定结论                    : {'是同一人' if same else '不是同一人'}")

    # ---- 4) 遍历群像图全部人脸 ----
    # 内部保留 bbox 供可视化使用；写入 JSON 前会剥离坐标（见 anonymize_ranking）。
    # 注意：_cos_raw 保留未舍入的余弦值。排序与 margin 差值必须用原始值计算，
    # 否则「先舍入再相减」会引入 1e-4 量级的误差，
    # 导致同一份数据在 face_compare 与 selftest 中给出不同的 margin。
    ranking = []
    for f in faces_g:
        c = cosine(f.embedding, subject.embedding)
        ranking.append({
            "bbox": f.bbox,
            "center": [round(v, 1) for v in f.center],
            "position": row_label(f.center[1], hg),
            "cosine": round(c, 4),
            "_cos_raw": c,
            "euclidean": round(euclidean(f.embedding, subject.embedding), 4),
            "is_target": f is target,
            "same_person": c >= thr,
        })
    ranking.sort(key=lambda r: -r["_cos_raw"])
    print(f"\n========== 遍历群像图 {len(faces_g)} 张人脸 vs 参考像主体 ==========")
    print(f"{'排名':<6}{'位置':<10}{'余弦':<10}{'欧氏':<9}{'判定'}")
    for rank, r in enumerate(ranking, 1):
        mark = "  <== 目标" if r["is_target"] else ""
        print(f"#{rank:<5}{r['position']:<10}{r['cosine']:<10.4f}{r['euclidean']:<9.4f}"
              f"{'是' if r['same_person'] else '否'}{mark}")

    best = ranking[0]
    save_crop(img_g, best["bbox"], os.path.join(args.crop_dir, "best_match_group.jpg"))
    # margin 用未舍入值相减，最后再舍入 —— 与 selftest.py 的算法保持一致
    margin = ((ranking[0]["_cos_raw"] - ranking[1]["_cos_raw"])
              if len(ranking) > 1 else None)
    if margin is not None:
        print(f"\n最佳匹配: 排名1 ({best['position']}), 余弦={best['cosine']:.4f}, "
              f"置信度={confidence_pct(best['cosine'], thr):.2f}%, "
              f"与排名2 相差 {margin:.4f}")

    # ---- 5) 可视化 ----
    vis_g = img_g.copy()
    for rank, r in enumerate(ranking, 1):
        x1, y1, x2, y2 = r["bbox"]
        color = (0, 220, 60) if r["same_person"] else (0, 165, 255)
        cv2.rectangle(vis_g, (x1, y1), (x2, y2), color, 3)
        cv2.putText(vis_g, f"#{rank} {r['cosine']:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    out_g = os.path.join(args.out_dir, "result_group_annotated.jpg")
    cv2.imwrite(out_g, vis_g)

    vis_r = img_r.copy()
    x1, y1, x2, y2 = subject.bbox
    cv2.rectangle(vis_r, (x1, y1), (x2, y2), (0, 220, 60), 3)
    cv2.putText(vis_r, "SUBJECT", (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 60), 2)
    out_r = os.path.join(args.out_dir, "result_reference_annotated.jpg")
    cv2.imwrite(out_r, vis_r)
    print(f"\n[save] 标注图: {out_g} , {out_r}")

    # ---- 6) JSON 报告 ----
    # 隐私约定：报告中**不写入**人脸框坐标与原始图片路径。
    # 坐标若与原图同时泄露，可反推出群像图中每个人的位置；
    # 保留相似度/判定等统计量即可完整表达结论。
    report = {
        "backend": BACKEND,
        "embedding_dim": dim,
        "threshold": {"cosine": thr, "euclidean_max": round(math.sqrt(2 * (1 - thr)), 4)},
        "source": {
            "group": {"faces_detected": len(faces_g), "size": [wg, hg]},
            "reference": {"faces_detected": len(faces_r), "size": [wr, hr]},
        },
        "target": {
            "strategy": target_info.get("strategy"),
            "matched_by": target_info.get("matched_by"),
            "position": row_label(target.center[1], hg),
            "shoulder_hsv": {"saturation": round(sat, 1), "value": round(val, 1)},
        },
        "comparison": {
            "euclidean_distance": round(dist, 4),
            "cosine_similarity": round(cos, 4),
            "confidence_pct": round(confidence_pct(cos, thr), 2),
            "same_person": same,
        },
        "ranking": anonymize_ranking(ranking),
        "privacy_note": "本报告已移除人脸框坐标与原始图片路径；"
                        "ranking.index 为按相似度排序后的匿名序号。",
    }
    if margin is not None:
        report["ranking_margin"] = {
            "top1_minus_top2": round(margin, 4),
            "note": "Top-1 与 Top-2 的余弦差，差值越大说明匹配越唯一",
        }
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"[save] JSON 报告: {args.report}")
    return report


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(
        description="人脸检测 / 特征提取 / 相似度比对",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="路径优先级：命令行 > 环境变量 > src/config.py 默认值。"
               "运行 `python src/config.py` 查看当前生效配置。",
    )
    parser.add_argument("group", nargs="?", default=None, help="群像图路径（多人）")
    parser.add_argument("reference", nargs="?", default=None, help="参考像路径（单人）")
    parser.add_argument("--group", dest="group_opt", help="群像图路径（覆盖位置参数）")
    parser.add_argument("--reference", dest="reference_opt",
                        help="参考像路径（覆盖位置参数）")
    parser.add_argument("--host", dest="host_opt",
                        help="参考像路径的旧别名（等价于 --reference）")
    parser.add_argument("--data-dir", default=None,
                        help=f"数据目录（默认 {config.DATA_DIR}）")
    parser.add_argument("--out-dir", default=None,
                        help=f"标注图输出目录（默认 {config.OUT_DIR}）")
    parser.add_argument("--crop-dir", default=None,
                        help=f"人脸裁剪图输出目录（默认 {config.CROP_DIR}）")
    parser.add_argument("--report", default=None,
                        help=f"JSON 报告输出路径（默认 {config.REPORT}）")
    parser.add_argument("--samples", action="store_true",
                        help="使用仓库自带的公开示例（NASA 公有领域照片）")
    parser.add_argument("--target-strategy", default="retrieve",
                        choices=("retrieve", "largest", "front-right", "white-shirt"),
                        help="群像图中目标人物的定位策略："
                             "retrieve=群像中与参考像最相似者（默认）；"
                             "largest=最大人脸；front-right=下半区最右；"
                             "white-shirt=下半区最右且肩部为白色衣物")
    args = parser.parse_args()

    # --- 解析生效路径（命令行 > 环境变量 > 默认值） ---
    data_dir = config.resolve_dir(args.data_dir, config.DATA_DIR)
    # --samples 只填补「用户未显式指定」的路径，显式指定的优先级更高
    if args.samples:
        args.group = args.group_opt or args.group or config.sample(config.SAMPLE_GROUP_NAME)
        args.reference = (args.reference_opt or args.host_opt or args.reference
                          or config.sample(config.SAMPLE_REFERENCE_NAME))
    else:
        args.group = args.group_opt or args.group or config.default_input(data_dir)
        args.reference = (args.reference_opt or args.host_opt or args.reference
                          or config.default_input(data_dir, config.DEFAULT_REFERENCE_NAME))
    args.out_dir = config.resolve_dir(args.out_dir, config.OUT_DIR)
    args.crop_dir = config.resolve_dir(args.crop_dir, config.CROP_DIR)
    args.report = config.resolve_dir(args.report, config.REPORT)

    print(f"[init] 主后端: {'insightface ArcFace (512维)' if BACKEND == 'ArcFace' else 'OpenCV SFace (128维, 降级模式)'}")
    print(f"[path] 群像图: {args.group}")
    print(f"[path] 参考像: {args.reference}")
    run(args)


if __name__ == "__main__":
    main()
