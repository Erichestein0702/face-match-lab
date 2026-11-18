<div align="center">

# 🧬 FaceMatch Lab

**基于 ArcFace 深度人脸识别的「同一人」判定流水线**

*检测 → 语义定位 → 512维特征 → 欧氏距离/余弦相似度 → 阈值判定 → 全图遍历排名 → 三角验证*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3B4256?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![InsightFace](https://img.shields.io/badge/InsightFace-4A90D9?style=flat-square)](https://github.com/deepinsight/insightface)
[![License](https://img.shields.io/badge/License-MIT-3DDC84?style=flat-square)](LICENSE)

</div>

---

## 📌 这是什么

给定**一张群像图**和**一张参考像**，自动完成：定位目标人物 → 提取 **512 维 ArcFace 特征**
→ 按官方标准阈值判定「是否同一人」并给出置信度 → **遍历全图所有人脸**输出排名表
→ 用第三张照片做**三角验证**排除单张误判。

仓库自带 **NASA 公有领域公开示例**，克隆即可运行，无需准备任何数据。

> **为什么没用 `face_recognition`？** 它依赖 `dlib`，Python 3.13+ 无预编译 wheel；
> 其 0.6 欧氏阈值也只适用于 dlib 的 128 维特征。本项目改用同级别主流库的官方阈值，
> 并提供**自动降级**链路。

---

## 💡 工程亮点

### 1. 结论可复现，且有负对照兜底

文档里所有数值都是**跑出来的**，不是手写的。仓库内置一组自带答案的公开对照组，
并用**负对照**坐实算法有效性 —— 这是多数人脸检索 Demo 做不出来的。

**NASA 公开示例（7 人群像，ArcFace 512 维）**

| 指标 | 数值 | 说明 |
|:---|---:|:---|
| 群像检测 | 检出 7 张人脸 | — |
| 欧氏距离 | **0.9333** | 同一人参考 ≤ 1.095 ✅ |
| 余弦相似度 | **0.5645 (56.45%)** | 阈值 0.40 ✅ |
| 相似度置信度 | **87.80%** | logistic 映射，阈值处为 50% |
| **判定结论** | **✅ 是同一人** | — |
| Top-1 与 Top-2 差值 | **0.3814** | 匹配唯一，无边界抖动 |
| 三角验证 | 0.5645 / 0.7105 / 0.5650 | 三对两两一致 ✅ |
| **负对照** | 换 Cooper 肖像后 Top-1 **自动切换**（0.4363） | ✅ |

最后一项最关键：换成另一个人的肖像，Top-1 会自动跳到群像图里的另一个人。
这证明排名反映的是**真实身份相似性**，而不是「永远给某张固定的脸打高分」。

**真实场景实测（17 人大合照）**

用一张 17 人的黑白大合照 + 两张同一人的不同照片（室内半身 / 户外特写，
拍摄条件差异大）验证：

| 验证项 | 实测 | 结论 |
|:---|:---|:---:|
| 群像检测 | 检出 **17** 张人脸 | ✅ |
| 主判定 | Top-1 余弦 **0.5370**（命中目标） | ✅ 同一人 |
| 唯一性 | Top-1 0.5370 vs Top-2 0.0760，差 **0.4610** | ✅ 匹配唯一 |
| 负对照 | 群像中其余人最高仅 **0.0760**，全员不过线 | ✅ 不是碰巧高分 |
| 三角验证 | 0.5370 / 0.4020 / 0.6663 三对全过 | ✅ |
| **参考像鲁棒** | 换成户外特写当参考像，**仍命中同一张脸** | ✅ |

> 该组照片仅用于本地验证，**未入库**（含可识别水印，属生物识别信息）。
> 仓库内所有图像均为 NASA 公有领域素材。

![三角验证](assets/charts/confidence_bars.png)

![相似度分布](assets/charts/similarity_chart.png)

### 2. 双后端自动降级，不挑环境

主后端 `insightface`（ArcFace 512 维）在部分平台没有预编译 wheel，
此时**自动降级**到 OpenCV Zoo（YuNet + SFace 128 维），功能完整可用。
两条链路的阈值分别取各自官方标准值。

### 3. 多尺度检测，解决群像小脸漏检

群像后排人脸只有几十像素，单一输入尺寸必然漏检。本项目在 **640 / 1600**
两档长边分别检测后用 **IoU < 0.4** 贪心 NMS 合并，特写与大合照同时兼顾。

> ⚠️ SCRFD 的 anchor stride 为 32，输入边长必须为 32 的倍数，否则
> `onnxruntime` 直接抛 `Invalid input shape`。`det_sizes()` 已做修正。

### 4. 定位策略与身份解耦，结论才有独立意义

| 类型 | 策略 | 回答的问题 |
|:---|:---|:---|
| **检索型** | `retrieve`（默认） | 「这群人里**有没有**这个人」 |
| **几何型** | `largest` / `front-right` / `white-shirt` / `index:N` | 「我指认的**这个位置**的人，是不是这个人」 |

几何型策略选人时**完全不看参考像**，判定结果具独立意义，可与检索型交叉验证。
所有策略带兜底链，任何构图下不中断。

### 5. 隐私是架构约束，不是事后补救

- JSON 报告**不写入** `bbox` / `center` / `path`（坐标一旦与原图同时泄露，可反推每个人在画面中的位置）
- `data/` 目录全程 gitignore，使用者的照片**永远不会**进入 git 历史
- **六层隐私体检**（历史 blob / 工作树 / 文本模式 / 报告字段 / ignore 规则 / 公开样例指纹）在 CI 中强制执行
- CI 还跑**负对照断言**，防止算法退化后仍「看起来通过」

### 6. 全链路配置化 + 结果带唯一性度量

路径优先级统一为 **命令行 > 环境变量 > `src/config.py` 默认值**，
手机 / Colab 还能直接喂 http(s) 直链。置信度采用**以阈值为中点的 logistic 映射**
（`cos == 阈值 → 50%`），报告额外给出 **Top-1 与 Top-2 的差值**（`ranking_margin`）——
差值大说明匹配唯一、不存在边界抖动，比单看一个分数稳健得多。

---

## 🚀 快速开始

脚本会自动 `pip install` 缺失依赖、自动下载模型、自动选择后端。

```bash
git clone https://github.com/Erichestein0702/face-match-lab.git
cd face-match-lab

python src/face_compare.py --samples      # 主流程：定位 + 比对 + 遍历 + JSON
python src/triangulate.py  --samples       # 三角验证
python src/plot_results.py                 # 生成图表
python scripts/selftest.py --samples       # 端到端自测（6 项断言）
```

### 用自己的照片

> 🔒 照片属于生物识别信息，必须由使用者自行准备、放在本地 `data/`，
> 该目录已被 `.gitignore` 排除，**永远不会进入 git 历史**。

```
data/                          # 本地数据目录（已 gitignore，永不入库）
├── group.jpg                 # 群像图（多人）—— 遍历排名的输入
├── reference.jpg             # 参考像（单人）—— 比对基准
├── third.jpg                 # 第三视角（可选）—— 三角验证
└── <代号>/                   # 单人人脸样本库（可选，供检索/复现）
    └── 01.jpg
```

```bash
python scripts/prepare_data.py --template    # 生成目录骨架（纯文本）
# 把已获授权的照片按上述结构放进 data/
python scripts/prepare_data.py --check       # 检查结构是否符合约定

python src/face_compare.py --data-dir data
python scripts/selftest.py --data-dir data  # 用自己的数据跑 6 项断言
```

没有合适的照片？可从公开数据集获取（同样落在 `data/`，不入库）：

```bash
python scripts/prepare_data.py --dataset lfw --num-identities 10
```

> ⚠️ 公开数据集的图像同样受各自许可约束，商用前请自行确认授权范围。

**其他**：`pip install -r requirements.txt` 手动装依赖；
Python 3.13+ 需 `pip install insightface --no-build-isolation`，装不上会自动降级；
也支持 http(s) 直链输入（手机 / Colab 友好）。

---

## 🧪 端到端自测

`scripts/selftest.py` 把「跑得通」变成**可断言的回归测试**，每项检查返回 PASS / FAIL，
退出码非 0 即回归失败，可直接接入 CI。

| 编号 | 检查项 | 断言 |
|:---|:---|:---|
| T1 | 主判定 | 群像检索 Top-1 判定为「同一人」 |
| T2 | 唯一性 | Top-1 与 Top-2 的余弦差 > 0.05（匹配必须唯一） |
| T3 | 负对照 | 群像中**其他人**的相似度低于阈值 |
| T4 | 过线人数 | 群像中恰好 **1** 张脸过阈值 |
| T5 | 三角验证 | A-B / A-C / B-C 三对两两一致（需第三视角） |
| T6 | 参考像鲁棒 | 换用第三视角作参考像，**命中同一张脸** |

```bash
python scripts/selftest.py                          # NASA 公开示例
python scripts/selftest.py --data-dir data          # 自己的照片
python scripts/selftest.py --group a.jpg --reference b.jpg --third c.jpg
python scripts/selftest.py --json results/selftest.json
```

T5 / T6 需要第三视角，缺失时自动标记 SKIP（不视为失败）。

---

## 🧠 算法设计

| 特征 | 维度 | 余弦阈值 | 等价欧氏距离 |
|:---|---:|---:|---:|
| ArcFace `w600k_r50`（主） | 512 | **≥ 0.40** | ≤ 1.095 |
| SFace / OpenCV Zoo（降级） | 128 | ≥ 0.363 | ≤ 1.088 |

```
confidence = 100 / (1 + exp(-12 * (cos_sim - threshold)))
```

`cos_sim == threshold` → 50%，越高于阈值越接近 100%。
几何型策略判据：下半区 `y/H > 0.52`、最右侧按 `x` 降序、白衣为肩部 HSV `S<70 且 V>140`。

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
│   ├── selftest.py           # 端到端自测：6 项断言，支持外部数据目录
│   ├── prepare_data.py       # 数据准备：结构检查 / 骨架生成 / 公开示例复制 / 数据集下载
│   ├── make_synthetic_data.py# 程序绘制的卡通测试图（不含真实人脸）
│   └── privacy_audit.py      # 隐私终审：六层扫描历史/工作树/文本/报告/忽略规则/样例指纹
├── samples/                  # 公开示例（NASA 公有领域照片，已入库）
├── data/                     # 用户本地数据目录（已 gitignore，永不入库）
├── assets/charts/            # 相似度分布图 / 三角验证条形图（纯数据图，不含人脸）
├── assets/results/           # 脱敏后的 JSON 报告
├── docs/METHOD.md            # 方法学说明
├── results/                  # 运行时产物（已 gitignore）
└── .github/workflows/ci.yml  # CI：语法检查 + 公开样例冒烟 + 负对照 + 泄露校验 + 隐私体检
```

---

## 🔒 隐私与合规

**不含私人照片** —— 使用者照片一律放在 `data/`，该目录已 gitignore，永不入库；
仓库内人脸图像仅为 NASA 公有领域素材（来源见 `samples/README.md`）。

**报告脱敏** —— JSON 不写出 `bbox`/`center`/`path`；图表纵轴用匿名序号 + 位置标签，
不含任何像素坐标。坐标属于「可定位的生物识别信息」，原图与坐标同时泄露即可反推出
群像中每个人的位置；而相似度、欧氏距离、置信度已足以完整表达算法行为。

**ignore 兜底** —— `.gitignore` 排除一切图像文件，仅显式放行 `samples/` 与 `assets/charts/`。

**CI 自动校验** —— 流水线检查报告是否含敏感字段，命中即失败；
`python scripts/privacy_audit.py` 可本地跑六层扫描。

**使用须知** —— 本项目用于技术验证与研究，人脸识别属生物识别信息处理，请遵守当地法律法规；
放入 `data/` 的照片须确保已取得被拍摄者的明确授权；不要用于未经授权的监控、追踪或身份认定；
展示结果时只分享统计量与图表，不要分享原图。

---

<div align="center">

**[⬆ Back to top](#-facematch-lab)**

Made with ☕ by [Erichestein0702](https://github.com/Erichestein0702)

</div>
