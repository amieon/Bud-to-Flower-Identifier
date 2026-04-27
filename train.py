import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F

from flower.CutMix import CutMix, mixup_criterion
import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

import xgboost as xgb
from model import load_convnext, load_efficientnet, load_swin
from utils import prep_image, FlowerDataset, LossHistory, train_transform, val_transform

# 设备检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'✅ GPU ready:' if torch.cuda.is_available() else '❌ No GPU detected.'} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

# 数据路径
csv_path = '../../data/train_labels.csv'
image_dir = '../../data/train'

# 加载CSV
df = pd.read_csv(csv_path)
filenames = [os.path.join(image_dir, fname) for fname in df['filename'].values]
y = df['category_id'].values.astype(np.int64)

print(f"CSV加载: {len(y)} 个样本，标签范围: {y.min()} 到 {y.max()}")
print("类分布:", np.unique(y, return_counts=True))




data, y_idx, idx_to_label, num_classes = prep_image(filenames, y)


# 数据拆分
x_train, x_valid, y_train, y_valid = train_test_split(data, y_idx, random_state=1, test_size=0.2, stratify=y_idx)


test_transform = val_transform


train_dataset = FlowerDataset(x_train, y_train, transform=train_transform)
val_dataset = FlowerDataset(x_valid, y_valid, transform=val_transform)

# Loader
batch_size = 48
nb_epoch = 40
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model_loaders = [load_efficientnet, load_convnext, load_swin]
model_names = ['EfficientNet', 'ConvNeXt', 'Swin']
ensemble_models = []
# 训练循环
print("All begin")
for idx, loader in enumerate(model_loaders):
    print(f"训练模型 {idx + 1}/{len(model_loaders)}")
    model, optimizer, criterion, scheduler = loader(num_classes=len(np.unique(y_idx)), device=device)

    history = LossHistory()
    early_stopping = 10
    best_val_loss = float('inf')
    early_stop_counter = 0

    # 训练循环
    print("begin")
    for epoch in range(nb_epoch):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels_batch in train_loader:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)

            inputs, labels_a, labels_b, lam = CutMix(inputs, labels_batch, device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()

            # 梯度裁剪（Transformer 训练的常见技巧）
            if idx == 2:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels_batch.size(0)
            correct_train += (predicted == labels_batch).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Validate
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels_batch in val_loader:
                inputs = inputs.to(device)
                labels_batch = labels_batch.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels_batch.size(0)
                correct_val += (predicted == labels_batch).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct_val / total_val

        history.on_epoch_end(train_loss, val_loss)

        print(f'模型 {idx + 1} Epoch {epoch + 1}/{nb_epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f"当前 lr: {optimizer.param_groups[0]['lr']}")
        # scheduler.step(val_acc / 100)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f'best_model_{idx}.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stopping:
                print(f'模型 {idx + 1} Early stopping at epoch {epoch + 1}')
                torch.save(model.state_dict(), f'best_model_{idx}.pth')
                break

    ensemble_models.append(model)



x_train, x_valid, y_train, y_valid = train_test_split(data, y_idx, random_state=2, test_size=0.2, stratify=y_idx)
val_dataset = FlowerDataset(x_valid, y_valid, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ==================== 在验证集上生成预测（训练 Meta Model） ====================
print("\n🔮 生成验证集预测（用于训练 Meta Model）...")

# 确保定义了 model_names

val_predictions = []  # 存储每个模型的预测概率

for idx, model in enumerate(ensemble_models):
    model_name = model_names[idx]
    print(f"\n[{idx + 1}/{len(ensemble_models)}] 使用 {model_name} 预测验证集...")

    model.eval()  # 确保是评估模式
    probabilities = []

    with torch.no_grad():
        for inputs, labels_batch in val_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            probabilities.append(probs.cpu().numpy())

    # 拼接所有batch
    probabilities = np.concatenate(probabilities, axis=0)
    val_predictions.append(probabilities)
    print(f"  {model_name} 验证集预测完成，shape: {probabilities.shape}")

# 将预测概率拼接成特征矩阵
# Shape: (n_val_samples, n_models * n_classes)
val_features = np.concatenate(val_predictions, axis=1)
print(f"\n验证集特征矩阵形状: {val_features.shape}")
print(f"  样本数: {val_features.shape[0]}, 特征数: {val_features.shape[1]}")



# ==================== 训练 Meta Model ====================
print("\n训练 Meta Model...")

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

# 获取验证集标签
y_val = []
for _, labels in val_loader:
    y_val.append(labels.numpy())
y_val = np.concatenate(y_val, axis=0)

print(f"验证集标签数量: {len(y_val)}")

# 尝试多种 Meta Model
meta_models = {}

# Logistic Regression
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
print(f"  验证准确率: {lr_score * 100:.2f}%")

# Random Forest
print("\n训练 Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,     # ✅ 从 200 降低
    max_depth=8,          # ✅ 从 20 大幅降低
    min_samples_split=20, # ✅ 从 5 增加到 20
    min_samples_leaf=10,  # ✅ 新增叶子节点最小样本
    max_features='sqrt',  # ✅ 限制特征采样
    random_state=42,
    n_jobs=-1
)
rf_model.fit(val_features, y_val)
rf_score = rf_model.score(val_features, y_val)
meta_models['RandomForest'] = rf_model
print(f"  验证准确率: {rf_score * 100:.2f}%")

# XGBoost
print("\n训练 XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=30,      # ✅ 从 200 降到 30
    max_depth=2,          # ✅ 从 6 降到 2（极浅的树）
    learning_rate=0.3,    # ✅ 提高学习率（快速收敛）
    min_child_weight=10,  # ✅ 从默认增加到 10
    subsample=0.6,        # ✅ 从 0.8 降到 0.6
    colsample_bytree=0.6, # ✅ 从 0.8 降到 0.6
    reg_alpha=1.0,        # ✅ 从 0 增加 L1 正则化
    reg_lambda=2.0,       # ✅ 增加 L2 正则化
    gamma=1.0,            # ✅ 新增，增加树分裂的最小损失减少
    objective='multi:softprob',
    num_class=num_classes,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(val_features, y_val)
xgb_score = xgb_model.score(val_features, y_val)
meta_models['XGBoost'] = xgb_model
print(f"  验证准确率: {xgb_score * 100:.2f}%")

# 选择最佳
best_meta_name = max(meta_models.items(), key=lambda x: x[1].score(val_features, y_val))[0]
best_meta_model = meta_models[best_meta_name]
best_score = best_meta_model.score(val_features, y_val)

print(f"\n🏆 最佳 Meta Model: {best_meta_name} (准确率: {best_score * 100:.2f}%)")

# 保存
import pickle

with open('meta_model.pkl', 'wb') as f:
    pickle.dump(best_meta_model, f)
print(f"✅ Meta Model 已保存")

