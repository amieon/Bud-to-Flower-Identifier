# 🌸 花卉识别器

> 基于多模型集成的100类花卉精细化识别系统

------

## 赛题背景

本项目为花卉识别图像分类竞赛的参赛方案。赛题要求构建高精度模型，对100种花卉（含稀有品种及形态相似品种）进行准确分类，服务于智慧农业与生态保护场景。

**数据集概览：**

- 训练集：100类 × 100~150张高质量图片（JPG/PNG，600×600）
- 测试集：100类 × 50张（服务器后台评测，不可下载）
- 标注格式：CSV文件，包含图片路径与类别标签

------

## 方案架构

```
花卉识别系统
├── 骨干网络（三路并行）
│   ├── Swin Transformer-S        # 全局语义建模
│   ├── ConvNeXt-Small            # 局部纹理提取
│   └── EfficientNet-B6           # 高效多尺度特征
├── 训练策略
│   ├── 混合数据增强（CutMix / Mixup / FMix）
│   ├── GridMask + RandomErasing
│   ├── CosineAnnealingWarmRestarts 学习率调度
│   └── 梯度累积（等效 batch_size=48）
├── 推理策略
│   ├── 测试时增强（TTA，9种变换）
│   └── 多模型平均集成
└── 可选：Stacking 元学习集成
    └── LR / RF / XGBoost 元模型
```

------

## 项目结构

```
.
├── code/
│   ├── train.py            # 主训练入口（三模型并行训练）
│   ├── retrain.py          # 单模型微调 / 断点续训
│   ├── predict.py          # 推理脚本（TTA + 多模型平均集成）
│   ├── stacking.py         # Stacking 元学习集成（可选）
│   ├── model.py            # 模型定义（Swin / ConvNeXt / EfficientNet）
│   ├── utils.py            # 数据集、预处理、数据增强
│   ├── CutMix.py           # CutMix / Mixup / FMix 实现
│   ├── label_mapping.json  # 类别索引双向映射表
│   └── train_log.txt       # 训练日志
├── model/
│   ├── config.json         # 标签映射配置（推理时使用）
│   ├── best_Swin.pth       # Swin Transformer 最优权重
│   └── best_ConvNeXt.pth   # ConvNeXt 最优权重
├── data/                   # 数据目录（需自行准备）
│   ├── train/              # 训练图片
│   └── train_labels.csv    # 训练标签
└── requirements.txt        # 依赖列表
```

------

## 环境配置

**Python 3.10+ 推荐，需要 CUDA 环境。**

```bash
pip install -r requirements.txt
```

核心依赖：

| 库            | 版本要求 |
| ------------- | -------- |
| torch         | ≥ 2.4.0  |
| torchvision   | ≥ 0.19.0 |
| numpy         | ≥ 1.22.0 |
| pandas        | ≥ 2.0.0  |
| opencv-python | ≥ 4.12.0 |

Stacking 集成额外依赖：

```bash
pip install scikit-learn xgboost
```

------

## 快速开始

### 1. 准备数据

将数据按以下结构放置：

```
data/
├── train/
│   ├── image_001.jpg
│   └── ...
└── train_labels.csv   # 列：filename, category_id
```

### 2. 训练模型

```bash
cd code
python train.py
```

训练过程将依次训练 Swin Transformer、ConvNeXt、EfficientNet 三个模型，每个模型训练50个 epoch，最优权重自动保存为 `best_{ModelName}.pth`，训练日志实时写入 `train_log.txt`。

**关键超参数（可在 `train.py` 中修改）：**

```python
batch_size = 16
accumulation_steps = 3   # 等效 batch_size = 48
nb_epoch = 50
```

### 3. 推理预测

```bash
cd code
python predict.py <测试图片目录> <输出CSV路径>

# 示例
python predict.py ../data/test/ ../submission.csv
```

推理默认开启 TTA（9种增强变换），使用 ConvNeXt + Swin 双模型平均集成，输出包含预测类别和置信度。

### 4. （可选）Stacking 元学习集成

```bash
cd code
python stacking.py
```

在验证集上训练 Logistic Regression、Random Forest、XGBoost 三种元模型，自动选取最优元模型保存为 `meta_model.pkl`。

### 5. 断点续训 / 微调

```bash
cd code
python retrain.py
```

从指定权重文件加载，继续训练单一模型，适用于针对难分类别的定向优化。

------

## 🔧 核心技术细节

### 模型设计

三个骨干网络均以 ImageNet 预训练权重初始化，分类头替换为自定义多层结构：

```
Dropout → Linear(in_features, 1024) → GELU/ReLU
→ BatchNorm1d → Dropout → Linear(1024, num_classes)
```

各模型学习率设置有差异：Swin 使用较小的 `3e-5`（Transformer 对学习率敏感），ConvNeXt 和 EfficientNet 使用 `1e-4`，均配合 `CosineAnnealingWarmRestarts` 调度。

### 数据增强

**训练时：**

- 随机旋转（±90°）、随机裁剪、颜色抖动
- 水平/垂直翻转、随机灰度化
- **GridMask**（概率0.75，遮挡网格区域）
- **RandomErasing**（概率0.75，随机擦除矩形块）
- **CutMix / Mixup / FMix 混合策略**（各占25%概率随机选取）

**推理时（TTA 9种变换）：** 中心裁剪、水平翻转、垂直翻转、角点裁剪、轻微旋转、中心放大、亮度调整、多尺度预测等。

### 图像预处理

输入图像统一 resize 到 336×336，使用 ImageNet 标准归一化：

```
mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```

------

## 文件说明

| 文件                      | 说明                                          |
| ------------------------- | --------------------------------------------- |
| `model/config.json`       | 类别ID与模型输出索引的双向映射，推理时必须    |
| `code/label_mapping.json` | 训练时生成的标签映射，与 config.json 内容一致 |
| `model/best_*.pth`        | 各模型在验证集最高准确率时的权重快照          |
| `code/train_log.txt`      | 每个 epoch 的 loss / acc / lr 记录            |

------

## 注意事项

- 训练需要 GPU，显存建议 ≥ 16GB（batch_size=16，图像尺寸336）
- 若显存不足，可降低 `batch_size` 并增大 `accumulation_steps` 保持等效批大小
- 推理使用 ConvNeXt + Swin 双模型（EfficientNet 可按需加入 `model_names` 列表）
- `stacking.py` 中数据拆分的 `random_state` 需与训练时保持一致，否则会造成数据泄漏

------

## 竞赛限制说明

本仓库 `main` 分支为**完整开发版本**，保留了全部实验性代码与最优配置。由于比赛方设有多项硬性限制，实际提交版本（见 `competition` 分支）做了相应裁剪，**并未能完全发挥出本方案的实际潜力**：

| 限制项             | 约束内容                           | 实际影响                                                     |
| ------------------ | ---------------------------------- | ------------------------------------------------------------ |
| 模型总体积 ≤ 500MB | EfficientNet-B6 体积超限，被迫舍弃 | 三模型集成降级为双模型（Swin + ConvNeXt）                    |
| 预测时间上限       | 9路 TTA 超时                       | TTA 缩减至 2 路，推理精度有所损失                            |
| 文件结构限制       | 不允许多余脚本文件                 | 删除 `stacking.py`、`retrain.py`，`CutMix.py` 并入 `utils.py` |

查看竞赛实际提交版本：

```bash
git checkout competition
```

---

##  竞赛成绩

**测试集准确率：95.46%**

![image-20260502005439069](.asset\image-20260502005439069.png)


## 方案亮点

- **异构集成**：Transformer（Swin）与 CNN（ConvNeXt、EfficientNet）互补，覆盖全局与局部特征
- **多样化混合增强**：CutMix / Mixup / FMix 三策略随机切换，有效缓解过拟合
- **9路 TTA**：大幅提升推理稳定性，对低置信度样本效果显著
- **梯度累积**：在有限显存下模拟大批量训练效果