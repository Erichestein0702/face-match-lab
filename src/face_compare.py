#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face-compare —— 人脸检测 / 特征提取 / 相似度比对一体化流水线
================================================================

功能
----
1. 在大合照中定位指定人员（"第一排最右 + 穿白色衣服 + 手持横幅"）
2. 在讲台照片中定位主持人
3. 提取 512 维 ArcFace 深度特征
4. 计算欧氏距离 / 余弦相似度 / 置信度百分比，给出是否同一人判定
5. 遍历合照中全部人脸，输出匹配排名表（用于"如果不是，请遍历查询其他人"）
6. 产出标注可视化图与结构化 JSON 报告

后端
----
* 主后端  insightface `buffalo_l`  = SCRFD 检测器 + ArcFace(w600k_r50) 512 维特征
* 降级后端  OpenCV Zoo              = YuNet 检测器 + SFace 128 维特征
  （insightface 在部分平台无预编译 wheel 时自动切换，保证"零配置可跑"）

阈值标准
--------
* ArcFace（归一化 512 维）: 余弦 >= 0.40 判为同一人，等价于欧氏距离 <= 1.095
* SFace（128 维）:         余弦 >= 0.363 判为同一人（OpenCV Zoo 官方阈值）

用法
----
    python3 face_compare.py <合照> <讲台照片>
    python3 face_compare.py --group samples/group.jpg --host samples/classroom.png \
                             --report results/report.json
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
  ___  _         ___                      __
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

DEFAULT_MODEL_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "face_models")
MODEL_ROOT = os.environ.get("FACE_MODEL_ROOT") or DEFAULT_MODEL_ROOT

# --------------------------------------------------------------------------- #
# 常量：判定阈值与几何经验值
# --------------------------------------------------------------------------- #
COS_THR_ARCFACE = 0.40          # insightface / ArcFace 常用验证阈值（余弦）
COS_THR_SFACE = 0.363           # OpenCV Zoo SFace 官方余弦判定阈值
FRONT_ROW_Y_RATIO = 0.52        # 合照中"第一排"人脸中心 y/H 的经验下界
HOST_MAX_Y_RATIO = 0.85         # 讲台照中排除前景观众头顶的 y/H 上界
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
    r = cy / height
    if r > FRONT_ROW_Y_RATIO:
        return "第一排"
    if r > 0.44:
        return "第二排"
    return "第三排"


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
# 主流程
# --------------------------------------------------------------------------- #
def run(args):
    group_path = resolve_source(args.group)
    host_path = resolve_source(args.host)

    img_g = cv2.imread(group_path)
    if img_g is None:
        sys.exit(f"[error] 无法读取合照: {group_path}")
    img_h = cv2.imread(host_path)
    if img_h is None:
        sys.exit(f"[error] 无法读取讲台照片: {host_path}")

    hg, wg = img_g.shape[:2]
    hh, wh = img_h.shape[:2]

    thr = COS_THR_ARCFACE if BACKEND == "ArcFace" else COS_THR_SFACE
    dim = 512 if BACKEND == "ArcFace" else 128

    faces_g, _ = detect_faces(img_g)
    faces_h, _ = detect_faces(img_h)
    print(f"[group] 合照 {wg}x{hg} 检测到 {len(faces_g)} 张人脸")
    print(f"[host]  讲台 {wh}x{hh} 检测到 {len(faces_h)} 张人脸")

    # ---- 1) 合照目标：第一排最右 + 白衣 ----
    front = [f for f in faces_g if f.center[1] > FRONT_ROW_Y_RATIO * hg]
    front.sort(key=lambda f: f.center[0], reverse=True)      # 从最右往左
    target, white_info = None, None
    for f in front:
        is_white, sat, val = white_cloth_check(img_g, f.bbox)
        if is_white:
            target, white_info = f, (sat, val)
            break
    if target is None and front:                             # 放宽：取第一排最右
        target = front[0]
        _, sat, val = white_cloth_check(img_g, target.bbox)
        white_info = (sat, val)
    if target is None:                                       # 兜底：全图最低人脸
        target = max(faces_g, key=lambda f: f.center[1]) if faces_g else None
    if target is None:
        sys.exit("[error] 合照中未检测到人脸")
    print(f"[group] 目标(第一排最右白衣): bbox={target.bbox} 中心={target.center} "
          f"肩部HSV S={white_info[0]:.0f} V={white_info[1]:.0f}")

    # ---- 2) 讲台主持人：排除前景观众头顶后取最大人脸 ----
    cands = [f for f in faces_h if f.center[1] < HOST_MAX_Y_RATIO * hh] or faces_h
    host = max(cands, key=lambda f: f.area)
    print(f"[host]  主持人 bbox={host.bbox} 中心={host.center}")

    os.makedirs(args.crop_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    save_crop(img_g, target.bbox, os.path.join(args.crop_dir, "target_group.jpg"))
    save_crop(img_h, host.bbox, os.path.join(args.crop_dir, "target_host.jpg"))

    # ---- 3) 特征比对 ----
    dist = euclidean(target.embedding, host.embedding)
    cos = cosine(target.embedding, host.embedding)
    same = cos >= thr
    title = "insightface ArcFace (512 维)" if BACKEND == "ArcFace" else "OpenCV SFace (128 维)"

    print(f"\n========== {title} ==========")
    print(f"欧氏距离 Euclidean Distance : {dist:.4f}  (同一人参考: <= {math.sqrt(2 * (1 - thr)):.3f})")
    print(f"余弦相似度 Cosine Similarity: {cos:.4f}  ({cos * 100:.2f}%)")
    print(f"判定阈值(余弦)              : {thr}")
    print(f"相似度置信度                : {confidence_pct(cos, thr):.2f}%")
    print(f"判定结论                    : {'是同一人' if same else '不是同一人'}")

    # ---- 4) 遍历合照全部人脸 ----
    ranking = []
    for f in faces_g:
        c = cosine(f.embedding, host.embedding)
        ranking.append({
            "bbox": f.bbox,
            "center": [round(v, 1) for v in f.center],
            "row": row_label(f.center[1], hg),
            "cosine": round(c, 4),
            "euclidean": round(euclidean(f.embedding, host.embedding), 4),
            "is_target": f is target,
            "same_person": c >= thr,
        })
    ranking.sort(key=lambda r: -r["cosine"])

    print(f"\n========== 遍历合照 {len(faces_g)} 张人脸 vs 讲台人物 ==========")
    print(f"{'排名':<6}{'中心(x,y)':<14}{'排':<8}{'余弦':<10}{'欧氏':<9}{'判定'}")
    for rank, r in enumerate(ranking, 1):
        mark = "  <== 目标(第一排最右白衣)" if r["is_target"] else ""
        print(f"{rank:<6}({int(r['center'][0])},{int(r['center'][1])})".ljust(20)
              + f"{r['row']:<8}{r['cosine']:<10.4f}{r['euclidean']:<9.4f}"
              f"{'是' if r['same_person'] else '否'}{mark}")

    best = ranking[0]
    save_crop(img_g, best["bbox"], os.path.join(args.crop_dir, "best_match_group.jpg"))
    print(f"\n最佳匹配: 排名1 中心=({int(best['center'][0])},{int(best['center'][1])}) "
          f"{best['row']}, 余弦={best['cosine']:.4f}, "
          f"置信度={confidence_pct(best['cosine'], thr):.2f}%")

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

    vis_h = img_h.copy()
    x1, y1, x2, y2 = host.bbox
    cv2.rectangle(vis_h, (x1, y1), (x2, y2), (0, 220, 60), 3)
    cv2.putText(vis_h, "HOST", (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 60), 2)
    out_h = os.path.join(args.out_dir, "result_host_annotated.jpg")
    cv2.imwrite(out_h, vis_h)
    print(f"\n[save] 标注图: {out_g} , {out_h}")

    # ---- 6) JSON 报告 ----
    report = {
        "backend": BACKEND,
        "embedding_dim": dim,
        "threshold": {"cosine": thr, "euclidean_max": round(math.sqrt(2 * (1 - thr)), 4)},
        "group_image": {"path": group_path, "size": [wg, hg], "faces_detected": len(faces_g)},
        "host_image": {"path": host_path, "size": [wh, hh], "faces_detected": len(faces_h)},
        "target": {"bbox": target.bbox, "center": [round(v, 1) for v in target.center],
                   "shoulder_hsv": {"saturation": round(white_info[0], 1),
                                    "value": round(white_info[1], 1)}},
        "host": {"bbox": host.bbox, "center": [round(v, 1) for v in host.center]},
        "comparison": {
            "euclidean_distance": round(dist, 4),
            "cosine_similarity": round(cos, 4),
            "confidence_pct": round(confidence_pct(cos, thr), 2),
            "same_person": same,
        },
        "ranking": ranking,
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
    )
    parser.add_argument("group", nargs="?", default="samples/group.jpg", help="大合照路径")
    parser.add_argument("host", nargs="?", default="samples/host.png", help="讲台特写照片路径")
    parser.add_argument("--group", dest="group_opt", help="大合照路径（覆盖位置参数）")
    parser.add_argument("--host", dest="host_opt", help="讲台照片路径（覆盖位置参数）")
    parser.add_argument("--out-dir", default="results", help="标注图输出目录")
    parser.add_argument("--crop-dir", default="results/crops", help="人脸裁剪图输出目录")
    parser.add_argument("--report", default="results/report.json", help="JSON 报告输出路径")
    args = parser.parse_args()
    args.group = args.group_opt or args.group
    args.host = args.host_opt or args.host

    print(f"[init] 主后端: {'insightface ArcFace (512维)' if BACKEND == 'ArcFace' else 'OpenCV SFace (128维, 降级模式)'}")
    run(args)


if __name__ == "__main__":
    main()
