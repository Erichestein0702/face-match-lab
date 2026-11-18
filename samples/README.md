# samples/ —— 公开示例素材

本目录存放**公开可再分发**的示例照片，用于让使用者零配置跑通全流程，
并为文档中的实测数值提供**可复现**的数据来源。

> English version: [README.en.md](README.en.md)

## 素材清单

| 文件 | 内容 | NASA 素材编号 | 来源 |
|:---|:---|:---|:---|
| `mercury7_group.jpg` | 水星七人宇航员合影（7 张人脸，用于遍历检索） | `s62-08774` | <https://images.nasa.gov/details/s62-08774> |
| `glenn_portrait.jpg` | John H. Glenn 官方肖像（单人，用于比对基准） | `S62-05540` | <https://images.nasa.gov/details/S62-05540> |
| `glenn_spacesuit.jpg` | John H. Glenn 戴水星宇航服（第三视角，用于三角验证） | `S87-41328` | <https://images.nasa.gov/details/S87-41328> |
| `cooper_portrait.jpg` | L. Gordon Cooper 官方肖像（**负对照**，非群像图中的人） | `s64-31847` | <https://images.nasa.gov/details/s64-31847> |

## 授权状态

这三张均为 **NASA（美国国家航空航天局）影像库**中的素材，由美国联邦政府雇员
在公务中拍摄，属**公有领域（Public Domain）**作品，可自由使用、修改与再分发，
无需额外授权。

> NASA 影像使用指南：<https://www.nasa.gov/nasa-brand-center/images-and-media/>

图像已按最长边缩放并重新压缩（质量 88）以控制仓库体积，未做内容修改。

## 为什么选这三张

它们构成了一个**自带答案的对照组**，可以完整验证算法的每一步：

| 验证点 | 期望行为 |
|:---|:---|
| 群像检测 | `mercury7_group.jpg` 应稳定检出 **7** 张人脸（含后排 3 张小脸） |
| 检索能力 | 以 `glenn_portrait.jpg` 为参考像，Glenn 本人应排 **Top-1** |
| 唯一性 | Top-1 与 Top-2 的余弦差应显著大于 0，说明匹配不靠"随便挑一个" |
| 三角验证 | 三张图中同属 Glenn 的三张脸应两两一致判定「同一人」 |
| 负对照 | 把参考像换成 Cooper 肖像，Top-1 应**切换到 Cooper**（后排另一人） |

负对照是这套示例最有价值的部分：它证明排名反映的是**真实的身份相似性**，
而不是"总是给相似度最高的那张脸一个高分"。

```bash
# 负对照：把参考像换成 Cooper，Top-1 应该切换到群像图中后排的 Cooper
python src/face_compare.py --samples --reference samples/cooper_portrait.jpg
```

## 使用方式

```bash
# 直接使用仓库自带示例（无需任何准备）
python src/face_compare.py --samples
python src/triangulate.py --samples

# 或者复制到 data/ 后按标准流程运行
python scripts/prepare_data.py --use-samples
python src/face_compare.py --data-dir data
```

## 隐私声明

本目录**不包含任何使用者或第三方的私人照片**。
仓库中的所有真实人脸图像，仅限于上表列出的 NASA 公有领域素材；
使用者的照片一律要求放在 `data/` 下，该目录已被 `.gitignore` 排除，永不入库。
