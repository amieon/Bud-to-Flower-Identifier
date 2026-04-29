# ========== stacking_ensemble.py ==========
import sys
import cv2
import pandas as pd
import numpy as np
import os
import json
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle

from model import load_swin, load_efficientnet, load_convnext

# 设备检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"{'✅ GPU ready:' if torch.cuda.is_available() else '❌ No GPU detected.'} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

# 加载标签映射
with open('label_mapping.json', 'r') as f:
    mapping_dict = json.load(f)

label_to_idx = {int(k): int(v) for k, v in mapping_dict['label_to_idx'].items()}
idx_to_label = {int(k): int(v) for k, v in mapping_dict['idx_to_label'].items()}
num_classes = len(idx_to_label)

print(f"类别数量: {num_classes}")

# 数据预处理
size = 336
test_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


print("\n📂 加载训练/验证数据...")

csv_path = 'data/train_labels.csv'
image_dir = 'data/train'

df = pd.read_csv(csv_path)
filenames = [os.path.join(image_dir, fname) for fname in df['filename'].values]
y_original = df['category_id'].values.astype(np.int64)

# 转换为索引
y = np.array([label_to_idx[int(label)] for label in y_original])

print(f"总样本数: {len(y)}")

# 使用与训练时相同的拆分（重要！）
from sklearn.model_selection import train_test_split

_, val_filenames, _, y_val = train_test_split(
    filenames, y, random_state=5, test_size=0.2, stratify=y
)

print(f"验证集样本数: {len(y_val)}")

# ==================== 加载训练好的模型 ====================
print("\n📦 加载训练好的模型...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_names = ['ConvNeXt', 'Swin', 'EfficientNet']
models_list = []

for model_name in model_names:
    print(f"  加载 {model_name}...")

    # 创建模型（只取模型对象，忽略其他）
    if model_name == 'ConvNeXt':
        model, _, _, _ = load_convnext(num_classes=num_classes, device=device)
    elif model_name == 'Swin':
        model, _, _, _ = load_swin(num_classes=num_classes, device=device)
    elif model_name == 'EfficientNet':
        model, _, _, _ = load_efficientnet(num_classes=num_classes, device=device)

    # 加载训练好的权重
    model_path = f'best_{model_name}.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    models_list.append(model)  # ✅ 只存储模型对象
    print(f"    ✅ {model_name} 加载完成")

print(f"\n✅ 已加载 {len(models_list)} 个模型")

# ==================== 第一阶段：生成验证集的预测概率 ====================
print("\n🔮 第一阶段：生成验证集预测（用于训练 Meta Model）...")

val_predictions = []  # 存储每个模型的预测概率

for idx, model in enumerate(models_list):
    model_name = model_names[idx]
    print(f"\n[{idx + 1}/{len(models_list)}] 使用 {model_name} 预测验证集...")

    probabilities = []

    with torch.no_grad():
        for i, img_path in enumerate(val_filenames):
            # 加载并预处理图片
            img = Image.open(img_path).convert('RGB')
            img_tensor = test_transform(img).unsqueeze(0).to(device)

            # 预测
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)

            probabilities.append(probs.cpu().numpy()[0])

            if (i + 1) % 1000 == 0:
                print(f"    处理进度: {i + 1}/{len(val_filenames)}")

    val_predictions.append(np.array(probabilities))
    print(f"  ✅ {model_name} 验证集预测完成")

# 将预测概率拼接成特征矩阵
# Shape: (n_val_samples, n_models * n_classes)
val_features = np.concatenate(val_predictions, axis=1)
print(f"\n验证集特征矩阵形状: {val_features.shape}")
print(f"  - 样本数: {val_features.shape[0]}")
print(f"  - 特征数: {val_features.shape[1]} = {len(models_list)} 模型 × {num_classes} 类")

# ==================== 第二阶段：训练 Meta Model ====================
print("\n🎯 第二阶段：训练 Meta Model...")

# 尝试多种 Meta Model
meta_models = {}

print("\n训练 Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs',
    C=1.0,
    random_state=42,
    n_jobs=-1
)
lr_model.fit(val_features, y_val)
lr_score = lr_model.score(val_features, y_val)
meta_models['LogisticRegression'] = lr_model
print(f"  Logistic Regression 验证准确率: {lr_score * 100:.2f}%")

print("\n训练 Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(val_features, y_val)
rf_score = rf_model.score(val_features, y_val)
meta_models['RandomForest'] = rf_model
print(f"  Random Forest 验证准确率: {rf_score * 100:.2f}%")

print("\n训练 XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=30,
    max_depth=2,
    learning_rate=0.3,
    min_child_weight=10,
    subsample=0.6,
    colsample_bytree=0.6,
    reg_alpha=1.0,
    reg_lambda=2.0,
    gamma=1.0,
    objective='multi:softprob',
    num_class=num_classes,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(val_features, y_val)
xgb_score = xgb_model.score(val_features, y_val)
meta_models['XGBoost'] = xgb_model
print(f"  XGBoost 验证准确率: {xgb_score * 100:.2f}%")

# 选择最佳 Meta Model
best_meta_name = max(meta_models.items(), key=lambda x: x[1].score(val_features, y_val))[0]
best_meta_model = meta_models[best_meta_name]
best_score = best_meta_model.score(val_features, y_val)

print(f"\n🏆 最佳 Meta Model: {best_meta_name} (准确率: {best_score * 100:.2f}%)")

# 保存 Meta Model
with open('meta_model.pkl', 'wb') as f:
    pickle.dump(best_meta_model, f)
print(f"✅ Meta Model 已保存为: meta_model.pkl")

# 计算基础模型在验证集上的准确率（作为对比）
print("\n📊 基础模型在验证集上的准确率（对比）:")
for idx, probs in enumerate(val_predictions):
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y_val) * 100
    print(f"  {model_names[idx]}: {acc:.2f}%")

# 简单平均的准确率
avg_probs = np.mean(val_predictions, axis=0)
avg_preds = np.argmax(avg_probs, axis=1)
avg_acc = np.mean(avg_preds == y_val) * 100
print(f"  简单平均: {avg_acc:.2f}%")
print(f"  Stacking ({best_meta_name}): {best_score * 100:.2f}%")
print(f"  🎉 Stacking 提升: +{best_score * 100 - avg_acc:.2f}%")
