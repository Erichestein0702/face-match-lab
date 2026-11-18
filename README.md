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

一个**可复现、可移植、零配置可跑**的人脸比对工具集。给定一张群像图和一张参考像，它会自动：

1. 在群像图中**定位**目标人物（可按"检索 / 位置 / 衣着"多种策略）
2. 提取 **512 维 ArcFace 深度特征**
3. 计算**欧氏距离**与**余弦相似度**，按官方标准阈值判定是否同一人，并给出置信度百分比
4. **遍历全图所有人脸**输出匹配排名表（"如果不是，请遍历查询其他人"）
5. 用第三张照片做**三角验证**，排除单张误判

仓库自带 **NASA 公有领域公开示例**，克隆即可运行，无需准备任何数据。

> **为什么没用 `face_recognition`？** 该库依赖 `dlib`，在 Python 3.13+ 无预编译 wheel；其 0.6 欧氏距离阈值也只适用于 dlib 的 128 维特征。本项目改用同级别主流库 **insightface / OpenCV Zoo** 的官方阈值，并提供**自动降级**链路。

---

## 💡 工程亮点

> 这一节回答「这个项目到底做对了什么」，而不是罗列功能。

### 1. 结论可复现，而不是「看起来能跑」

仓库内置一组**自带答案的公开对照组**（NASA 公有领域照片），
所有文档数值都是**跑出来的**，不是手写的：

| 验证项 | 期望 | 实测 | 结论 |
|:---|:---|:---|:---:|
| 群像检测 | 稳定检出 7 张人脸 | 7 | ✅ |
| 主判定 | Top-1 超阈值 | 0.5645 / 阈值 0.40 | ✅ |
| 唯一性 | Top-1 与 Top-2 拉开差距 | 0.5645 vs 0.1831（差 **0.3814**） | ✅ |
| 三角验证 | 三对两两一致 | 0.5645 / 0.7105 / 0.5650 | ✅ |
| **负对照** | 换参考像，Top-1 **必须切换** | Glenn 0.5645 → Cooper 0.4363 | ✅ |

最后一项是关键：**换成另一个人的肖像，Top-1 自动切换到群像图里的另一个人**。
这证明排名反映的是真实身份相似性，而不是"永远给某张固定的脸打高分"——
这是绝大多数人脸检索 Demo 做不出来的。

### 2. 双后端自动降级，不挑环境

主后端 `insightface`（ArcFace 512 维）在部分平台没有预编译 wheel，
此时**自动降级**到 OpenCV Zoo（YuNet + SFace 128 维），功能完整可用。
两条链路的阈值分别取各自官方标准值，不是拍脑袋抄的。

### 3. 多尺度检测，解决群像小脸漏检

群像图里后排的人脸只有几十像素，单一输入尺寸必然漏检。
本项目在 **640 / 1600** 两档长边分别检测后用 **IoU < 0.4** 贪心 NMS 合并，
特写与大合照同时兼顾。

> ⚠️ SCRFD 的 anchor stride 为 32，输入边长必须为 32 的倍数，
> 否则 `onnxruntime` 直接抛 `Invalid input shape`。`det_sizes()` 做了修正。

### 4. 定位策略与身份解耦，结论才有独立意义

目标定位分成两类：

- **检索型**（`retrieve`，默认）：群像中与参考像最相似的人 → 回答「这群人里有没有这个人」
- **几何型**（`largest` / `front-right` / `white-shirt` / `index:N`）：
  先按**与身份无关**的图像几何特征指认某张脸 → 回答「我指认的这个位置的人，是不是这个人」

几何型策略选人时**完全不看参考像**，因此判定结果具有独立意义，可与检索型交叉验证。

### 5. 隐私是架构约束，不是事后补救

生物识别信息的最小暴露被写进代码结构，而不是靠 README 提醒：

- JSON 报告**不写入** `bbox` / `center` / `path`（坐标一旦与原图同时泄露，可反推每个人在画面中的位置）
- `data/` 目录全程 gitignore，使用者的照片**永远不会**进入 git 历史
- 五层隐私体检（历史 blob / 工作树 / 文本模式 / 报告字段 / ignore 规则）**在 CI 中强制执行**
- CI 还跑**负对照断言**，防止算法退化后仍"看起来通过"

### 6. 全链路配置化，同一份代码换环境不用改

路径优先级统一为 **命令行 > 环境变量 > `src/config.py` 默认值**；
`python src/config.py` 可直接打印当前生效的全部配置。
手机 / Colab 场景还能直接喂 http(s) 直链，无需先下载到本地。

### 7. 结果不只是"是/否"，而是带置信度与唯一性度量

- 置信度采用**以阈值为中点的 logistic 映射**：`cos == 阈值 → 50%`
- 报告额外给出 **Top-1 与 Top-2 的差值**（`ranking_margin`），
  差值大说明匹配唯一、不存在边界抖动——这比单看一个分数稳健得多

---

## 🎯 实测结论

> **数据说明**：以下数值由仓库自带的 NASA 公有领域示例照片计算得出，
> **可一键复现**（`python src/face_compare.py --samples`）。
> 素材来源与授权说明见 [`samples/README.md`](samples/README.md)。

### 主判定

| 指标 | 数值 | 说明 |
|:---|---:|:---|
| 欧氏距离 Euclidean | **0.9333** | 同一人参考 ≤ 1.095 ✅ |
| 余弦相似度 Cosine | **0.5645 (56.45%)** | 判定阈值 0.40 ✅ |
| 相似度置信度 | **87.80%** | 以阈值为中点的 logistic 映射 |
| **判定结论** | **✅ 是同一人** | — |
| Top-1 与 Top-2 差值 | **0.3814** | 匹配唯一，无边界抖动 |

### 三角验证（三条独立证据链）

| 比对 | 余弦 | 欧氏 | 置信度 | 判定 |
|:---|---:|---:|---:|:---:|
| A 群像目标 ↔ B 参考像 | 0.5645 | 0.9333 | 87.80% | ✅ 同一人 |
| A 群像目标 ↔ C 第三视角 | 0.7105 | 0.7609 | 97.65% | ✅ 同一人 |
| B 参考像 ↔ C 第三视角 | 0.5650 | 0.9327 | 87.87% | ✅ 同一人 |

**三对全部一致 → 确认为同一人。**

![三角验证](assets/charts/confidence_bars.png)

### 全图遍历排名（7 张人脸 vs 参考像主体）

Top-1 即目标本人（0.5645），第 2 名仅 0.1831，**与阈值 0.40 差距显著 → 判定稳健且唯一**。

![相似度分布](assets/charts/similarity_chart.png)

### 负对照（证明检索真的在认人）

把参考像换成另一位宇航员 Cooper 的肖像，Top-1 **自动切换**到群像图中后排的 Cooper：

| 参考像 | Top-1 余弦 | Top-1 位置 |
|:---|---:|:---|
| Glenn 肖像 | 0.5645 | 中部 |
| Cooper 肖像 | 0.4363 | 上半区（后排另一人） |

---

## 📦 数据准备

### 方式一：直接用公开示例（零配置，推荐）

```bash
python src/face_compare.py --samples      # 主流程
python src/triangulate.py --samples        # 三角验证
python src/plot_results.py                 # 生成图表
```

`samples/` 内的照片全部来自 **NASA 影像库（公有领域）**，可自由使用与再分发，
来源与编号见 [`samples/README.md`](samples/README.md)。

### 方式二：用自己的照片

> 🔒 **隐私原则**：本仓库**不提供、不接受**任何真实人脸照片。
> 照片属于生物识别信息，必须由使用者自行准备、放在本地 `data/` 目录，
> 该目录已被 `.gitignore` 排除，**永远不会进入 git 历史**。

```
data/                          # 本地数据目录（已 gitignore，永不入库）
├── group.jpg                 # 群像图（多人）—— 遍历排名的输入
├── reference.jpg             # 参考像（单人）—— 比对基准
├── third.jpg                 # 第三视角（可选）—— 三角验证的第三张
└── <人名>/                   # 单人人脸样本库（可选，供检索 / 复现）
    ├── 01.jpg
    └── 02.jpg
```

三步准备：

```bash
# 1) 生成目录骨架与说明文件（纯文本，不含任何图像）
python scripts/prepare_data.py --template

# 2) 把你已获授权的照片按上述结构放进 data/
#    目录名建议用代号（如 person_01），避免在报告中暴露真实姓名

# 3) 检查结构是否符合约定
python scripts/prepare_data.py --check
```

也可以把公开示例复制到 `data/` 作为上手素材：

```bash
python scripts/prepare_data.py --use-samples
```

### 从公开数据集获取样本

若你没有合适的照片，可用公开数据集先跑通流程：

```bash
python scripts/prepare_data.py --dataset lfw --num-identities 10
```

| 数据集 | 说明 | 许可 |
|:---|:---|:---|
| `lfw` | Labeled Faces in the Wild | 非商业研究用途 |
| `olivetti` | AT&T ORL 人脸库 | 学术免费使用 |

下载结果落在 `data/<数据集名>/<人名>/*.jpg`，同样不入库。
> ⚠️ 公开数据集的图像同样受各自许可约束，商用前请自行确认授权范围。

---

## 🚀 快速开始

### 方式一：零配置直接跑（推荐）

脚本会自动 `pip install` 缺失依赖、自动下载模型、自动选择后端。

```bash
git clone https://github.com/Erichestein0702/face-match-lab.git
cd face-match-lab

# 用仓库自带的公开示例跑通全流程
python src/face_compare.py --samples
python src/triangulate.py --samples
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
  python src/face_compare.py "https://example.com/group.jpg" "https://example.com/reference.jpg"
  ```
- **纯 CPU 可跑**，不需要 GPU

---

## 🧠 算法设计

### 目标定位策略

| 策略 | 类型 | 说明 |
|:---|:---|:---|
| `retrieve` | 检索 | 群像中与参考像最相似的人（**默认**） |
| `largest` | 几何 | 画面中面积最大的人脸 |
| `front-right` | 几何 | 下半区中最靠右的人脸 |
| `white-shirt` | 几何 | 下半区中靠右且肩部为白色衣物 |
| `index:N` | 几何 | 直接指定第 N 张人脸 |

几何型策略的判据：

| 判据 | 实现 |
|:---|:---|
| 下半区 | 人脸中心 `y/H > 0.52` |
| 最右侧 | 按人脸中心 `x` 降序 |
| 白色衣物 | 肩部区域 HSV：`S < 70` 且 `V > 140` |

### 多尺度检测（解决小脸漏检）

SCRFD 在**单一输入尺寸**下会漏检群像里的小脸，也会漏掉大特写。做法：

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
│   ├── config.py             # 路径与环境变量集中管理（无第三方依赖）
│   ├── face_compare.py       # 主程序：定位 + 特征 + 比对 + 遍历 + 标注 + JSON
│   ├── triangulate.py        # 三图三角验证
│   └── plot_results.py       # 由 JSON 报告生成可视化图表
├── scripts/
│   ├── prepare_data.py       # 数据准备：结构检查 / 骨架生成 / 公开示例复制 / 数据集下载
│   ├── make_synthetic_data.py# 生成程序绘制的卡通测试图（不含真实人脸）
│   └── privacy_audit.py      # 隐私终审体检：五层扫描历史/工作树/文本/报告/忽略规则
├── samples/                  # 公开示例（NASA 公有领域照片，已入库）
│   ├── mercury7_group.jpg    # 群像：7 人
│   ├── glenn_portrait.jpg    # 参考像：单人官方肖像
│   ├── glenn_spacesuit.jpg   # 第三视角
│   └── cooper_portrait.jpg   # 负对照素材
├── data/                     # 用户本地数据目录（已 gitignore，永不入库）
│   ├── group.jpg
│   ├── reference.jpg
│   ├── third.jpg
│   └── <人名>/*.jpg          # 单人人脸样本库（可选）
├── assets/
│   ├── charts/               # 相似度分布图 / 三角验证条形图（纯数据图，不含人脸）
│   ├── crops/                # 目标人脸裁剪（运行期生成，已 gitignore）
│   └── results/              # 脱敏后的 JSON 报告
├── docs/
│   └── METHOD.md             # 方法学说明
├── results/                  # 运行时产物（已 gitignore）
├── .github/workflows/ci.yml  # CI：语法检查 + 公开样例冒烟 + 负对照 + 泄露校验 + 隐私体检
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

## ⚙️ 路径与环境变量

**所有路径都不写死在代码里**，优先级为：命令行参数 > 环境变量 > `src/config.py` 默认值。

运行 `python src/config.py` 可查看当前生效的完整配置。

| 环境变量 | 作用 | 默认值 |
|:---|:---|:---|
| `FACE_DATA_DIR` | 数据根目录 | `<repo>/data` |
| `FACE_SAMPLES_DIR` | 公开示例目录 | `<repo>/samples` |
| `FACE_OUT_DIR` | 标注图 / 报告输出根目录 | `<repo>/results` |
| `FACE_CROP_DIR` | 人脸裁剪图输出目录 | `<repo>/results/crops` |
| `FACE_REPORT` | 主 JSON 报告路径 | `<repo>/results/report.json` |
| `FACE_TRI_REPORT` | 三角验证报告路径 | `<repo>/results/triangulation.json` |
| `FACE_CHART_DIR` | 图表输出目录 | `<repo>/assets/charts` |
| `FACE_MODEL_ROOT` | 模型缓存目录 | `~/.cache/face_models` |
| `FACE_NO_ARCFACE` | 设为 `1` 强制降级到 SFace | 未设置 |

对应的命令行参数：`--data-dir` / `--out-dir` / `--crop-dir` / `--report` /
`--triangulation` / `--group` / `--reference` / `--third` / `--samples` /
`--target-strategy`。

---

## 🧩 特性清单

- [x] 512 维 ArcFace 深度特征（insightface `buffalo_l`）
- [x] 128 维 SFace 降级后端，**开箱即用不挑环境**
- [x] 欧氏距离 + 余弦相似度双指标
- [x] 官方标准阈值 + 置信度百分比
- [x] 检索型 / 几何型两族目标定位策略
- [x] 多尺度检测 + IoU 去重，解决小脸漏检
- [x] 全图遍历匹配排名表 + Top-1/Top-2 唯一性差值
- [x] 三图三角验证抗误判
- [x] 结构化 JSON 报告（坐标脱敏）
- [x] http(s) 直链输入（手机友好）
- [x] 路径全配置化：命令行 > 环境变量 > 默认值
- [x] **自带 NASA 公有领域公开示例**，零配置可跑、结论可复现
- [x] **隐私设计**：报告不含人脸坐标与原始路径，`data/` 全程不入库
- [x] CI：语法检查 + 公开样例冒烟 + **负对照断言** + 报告泄露校验 + 五层隐私体检
- [x] 深色主题可视化图表

---

## 🔒 隐私与合规

本仓库在设计上就把**生物识别信息**的最小暴露作为硬约束：

| 措施 | 说明 |
|:---|:---|
| **不含私人照片** | 使用者的照片一律要求放在 `data/`，该目录已 gitignore，永不入库 |
| **公开素材可追溯** | 仓库内人脸图像仅为 NASA 公有领域素材，来源与编号见 `samples/README.md` |
| **JSON 报告脱敏** | 不写出人脸框坐标（`bbox`/`center`）与原始图片路径，仅保留相似度等统计量 |
| **图表匿名化** | 纵轴标签使用匿名序号 + 位置标签，不含任何像素坐标 |
| **数据目录不入库** | `data/` 已在 `.gitignore` 中排除；`.gitignore` 另设兜底规则排除一切图像文件 |
| **CI 自动校验** | 流水线会检查生成的报告是否含 `bbox`/`center`/`path` 字段，命中即失败 |
| **可自检** | `python scripts/privacy_audit.py` 五层扫描历史/工作树/文本/报告/忽略规则 |

> 🔍 **自检命令**
> ```bash
> python scripts/privacy_audit.py          # 完整体检
> python scripts/privacy_audit.py --json   # 机器可读
> ```
> 扫描层级：L1 全历史图像 blob · L2 工作树图像文件 · L3 文本禁止模式 ·
> L4 报告敏感字段 · L5 `.gitignore` 兜底拦截。任一层命中即判定未通过。

> ℹ️ **坐标为何必须移除**：人脸框坐标属于「可定位的生物识别信息」。
> 若原图与坐标同时泄露，即可反推出群像中每个人的位置。
> 保留相似度、欧氏距离、置信度与判定结论，已足以完整表达算法行为。

### 使用须知

- 本项目用于**技术验证与研究**。人脸识别属于**生物识别信息**处理，请遵守当地法律法规（个人信息保护法 / GDPR 等）。
- 使用者放入 `data/` 的照片，须确保**已取得被拍摄者的明确授权**。
- **不要**将人脸识别技术用于未经授权的监控、追踪或身份认定。
- 若需展示结果，建议只分享**统计量与图表**（如本仓库 README 所示），不要分享原图。

---

<div align="center">

**[⬆ Back to top](#-facematch-lab)**

Made with ☕ by [Erichestein0702](https://github.com/Erichestein0702)

</div>
