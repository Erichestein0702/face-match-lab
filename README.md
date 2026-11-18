<div align="center">

# 🧬 FaceMatch Lab

**基于 ArcFace 深度人脸识别的「同一人」判定流水线**

*检测 → 语义定位 → 512维特征提取 → 欧氏距离 / 余弦相似度 → 阈值判定 → 全图遍历排名 → 三角验证*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3B4256?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![InsightFace](https://img.shields.io/badge/InsightFace-0.7.3-4A90D9?style=flat-square)](https://github.com/deepinsight/insightface)
[![OpenCV](https://img.shields.io/badge/OpenCV-Zoo-5C4033?style=flat-square&logo=opencv&logoColor=white)](https://github.com/opencv/opencv_zoo)
[![License](https://img.shields.io/badge/License-MIT-3DDC84?style=flat-square)](LICENSE)

</div>

---

## 📌 这是什么

一个**可复现、可移植、零配置可跑**的人脸比对工具集。给定两张照片，它会自动：

1. 在大合照中**定位**目标人物（按"第一排 / 最右 / 白衣 / 手持横幅"的语义描述筛选）
2. 在讲台照片中**定位**主持人（自动排除前景观众的头顶）
3. 提取 **512 维 ArcFace 深度特征**
4. 计算**欧氏距离**与**余弦相似度**，按官方标准阈值判定是否同一人，并给出置信度百分比
5. **遍历全图所有人脸**输出匹配排名表（"如果不是，请遍历查询其他人"）
6. 用第三张照片做**三角验证**，排除单张误判

> **为什么没用 `face_recognition`？** 该库依赖 `dlib`，在 Python 3.13+ 无预编译 wheel；其 0.6 欧氏距离阈值也只适用于 dlib 的 128 维特征。本项目改用同级别主流库 **insightface / OpenCV Zoo** 的官方阈值，并提供**自动降级**链路。

---

## 🎯 实测结论

> 输入：`samples/group.jpg`（1684×1080 合照）+ `samples/host.png`（1267×951 讲台特写）

| 指标 | 数值 | 说明 |
|:---|---:|:---|
| 欧氏距离 Euclidean | **0.9257** | 同一人参考 ≤ 1.095 ✅ |
| 余弦相似度 Cosine | **0.5715 (57.15%)** | 判定阈值 0.40 ✅ |
| 相似度置信度 | **88.68%** | 以阈值为中点的 logistic 映射 |
| **判定结论** | **✅ 是同一人** | 三对两两比对全部一致 |

### 三角验证（三条独立证据链）

| 比对 | 余弦 | 欧氏 | 置信度 | 判定 |
|:---|---:|---:|---:|:---:|
| A 合照目标 ↔ B 讲台特写 | 0.5715 | 0.9257 | 88.68% | ✅ 同一人 |
| A 合照目标 ↔ C 教室全景 | 0.5045 | 0.9955 | 77.80% | ✅ 同一人 |
| B 讲台特写 ↔ C 教室全景 | 0.6768 | 0.8040 | 96.52% | ✅ 同一人 |

**三对全部一致 → 确认为同一人。**

![三角验证](assets/charts/confidence_bars.png)

### 全图遍历排名（33 张人脸 vs 讲台人物）

Top-1 即目标本人（0.5715），第 2 名仅 0.3172，**与阈值 0.40 差距显著 → 判定稳健且唯一**。

![相似度分布](assets/charts/similarity_chart.png)

### 标注可视化

| 合照（绿框 = 目标本人） | 讲台特写 | 教室全景 |
|:---:|:---:|:---:|
| ![合照标注](assets/results/result_group_annotated.jpg) | ![讲台标注](assets/results/result_host_annotated.jpg) | ![全景标注](assets/results/result_panorama_annotated.jpg) |

---

## 🚀 快速开始

### 方式一：零配置直接跑（推荐）

脚本会自动 `pip install` 缺失依赖、自动下载模型、自动选择后端。

```bash
git clone https://github.com/Erichestein0702/face-match-lab.git
cd face-match-lab

# 主比对 + 遍历排名 + 标注图 + JSON 报告
python src/face_compare.py --group samples/group.jpg --host samples/host.png

# 三角验证（需要第三张图）
python src/triangulate.py --group samples/group.jpg \
                          --host samples/host.png \
                          --panorama samples/classroom.png

# 生成可视化图表
python src/plot_results.py
```

### 方式二：手动装依赖

```bash
pip install -r requirements.txt
```

> 💡 **Python 3.13+ 用户**：`insightface` 暂无预编译 wheel，需源码安装
> ```bash
> pip install insightface --no-build-isolation
> ```
> 装不上也没关系 —— 脚本会自动**降级**到 OpenCV Zoo（YuNet + SFace），功能完整可用。

### 手机 / Colab 友好

- **支持 http(s) 直链**，无需先下载到本地：
  ```bash
  python src/face_compare.py "https://example.com/group.jpg" "https://example.com/host.jpg"
  ```
- **零参数模式**：当前目录恰好 2 张图时自动识别（大的当合照，小的当特写）
- **纯 CPU 可跑**，不需要 GPU

---

## 🧠 算法设计

### 目标定位（语义 → 几何）

| 目标 | 判据 | 实现 |
|:---|:---|:---|
| 第一排 | 人脸中心 `y/H > 0.52` | 分排启发式 |
| 最右手边 | 按人脸中心 `x` 降序 | 最右优先遍历 |
| 穿白色衣服 | 肩部区域 HSV：`S < 70 且 V > 140` | 低饱和 + 高亮度 |
| 手持横幅 | 与第一排/最右条件联合 | 目视复核确认 |
| 主持人 | `y/H < 0.85` 排除观众头顶 → 取面积最大 | 全景图中再叠加 `x/W < 0.25` 锁定讲台侧 |

### 多尺度检测（解决小脸漏检）

SCRFD 在**单一输入尺寸**下会漏检合照里的小脸，也会漏掉大特写。做法：

```
长边缩放到 640  → 检测  →  收集
长边缩放到 1600 → 检测  →  收集
                ↓
        IoU < 0.4 去重合并
```

⚠️ 边长必须是 **32 的倍数**，否则 SCRFD 锚点数量与网络输出不匹配会直接报错。

### 判定阈值（沿用官方标准）

| 特征 | 维度 | 余弦阈值 | 等价欧氏距离 |
|:---|---:|---:|---:|
| ArcFace `w600k_r50` | 512 | **≥ 0.40** | ≤ 1.095 |
| SFace (OpenCV Zoo) | 128 | ≥ 0.363 | ≤ 1.088 |

置信度采用以阈值为中点的 logistic 映射：

```
confidence = 100 / (1 + exp(-12 * (cos_sim - threshold)))
```

`cos_sim == threshold` → 50%，越高于阈值越接近 100%。

> 详细方法学与边界分析见 **[docs/METHOD.md](docs/METHOD.md)**。

---

## 📁 项目结构

```
face-match-lab/
├── src/
│   ├── face_compare.py      # 主程序：定位 + 特征 + 比对 + 遍历 + 标注 + JSON
│   ├── triangulate.py       # 三图三角验证
│   └── plot_results.py      # 由 JSON 报告生成可视化图表
├── samples/                 # 演示样本（3 张）
│   ├── group.jpg            # 大合照（1684×1080）
│   ├── host.png             # 讲台特写（1267×951）
│   └── classroom.png        # 教室全景（1266×950）
├── assets/
│   ├── charts/              # 相似度分布图 / 三角验证条形图
│   ├── crops/               # 目标人脸裁剪
│   └── results/             # 标注可视化图
├── docs/
│   └── METHOD.md            # 方法学说明
├── results/                 # 运行时产物（.gitignore 排除）
├── .github/workflows/ci.yml # CI：语法检查 + SFace 降级链路冒烟测试
├── requirements.txt
├── LICENSE
└── README.md
```

### 核心 API

```python
from face_compare import detect_faces, cosine, euclidean, confidence_pct

faces, backend = detect_faces(image)      # -> (list[Face], "ArcFace"|"SFace")
f1, f2 = faces[0], faces[1]

dist = euclidean(f1.embedding, f2.embedding)   # 欧氏距离
cos  = cosine(f1.embedding, f2.embedding)      # 余弦相似度
conf = confidence_pct(cos, threshold=0.40)      # 置信度百分比
```

---

## ⚙️ 环境变量

| 变量 | 作用 | 默认值 |
|:---|:---|:---|
| `FACE_MODEL_ROOT` | 模型缓存目录 | `~/.cache/face_models` |
| `FACE_NO_ARCFACE` | 设为 `1` 强制降级到 SFace | 未设置 |
| `FACE_CROP_DIR` | 人脸裁剪输出目录 | `results/crops` |
| `FACE_OUT_DIR` | 标注图输出目录 | `results` |

---

## 🧩 特性清单

- [x] 512 维 ArcFace 深度特征（insightface `buffalo_l`）
- [x] 128 维 SFace 降级后端，**开箱即用不挑环境**
- [x] 欧氏距离 + 余弦相似度双指标
- [x] 官方标准阈值 + 置信度百分比
- [x] 语义化目标定位（排位 / 左右 / 衣服颜色）
- [x] 多尺度检测 + IoU 去重，解决小脸漏检
- [x] 全图遍历匹配排名表
- [x] 三图三角验证抗误判
- [x] 标注可视化图 + 结构化 JSON 报告
- [x] http(s) 直链输入（手机友好）
- [x] CI：语法检查 + 降级链路冒烟测试
- [x] 深色主题可视化图表

---

## ⚠️ 使用须知

- 本项目用于**技术验证与研究**。人脸识别属于**生物识别信息**处理，请遵守当地法律法规（个人信息保护法 / GDPR 等）。
- 仓库中的照片仅作**技术演示**用途，涉及的肖像权归原拍摄者与被拍摄者所有。
- **不要**将其用于未经授权的监控、追踪或身份认定。

---

<div align="center">

**[⬆ Back to top](#-facematch-lab)**

Made with ☕ by [Erichestein0702](https://github.com/Erichestein0702)

</div>
