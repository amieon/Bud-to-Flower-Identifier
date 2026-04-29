import pickle
import sys

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from model import load_convnext, load_efficientnet, load_swin
from utils import prep_test_image, FlowerDataset

# 设备检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

folder_path = "../results"

os.makedirs(folder_path, exist_ok=True)

# 参数检查
if len(sys.argv) != 3:
    print("Usage: python predict.py <test_folder_path> <output_csv_path>")
    sys.exit(1)

test_folder_path = sys.argv[1]
output_csv_path = sys.argv[2]

# 加载标签映射
import json

with open('../model/config.json', 'r') as f:
    mapping = json.load(f)
idx_to_label = {int(k): int(v) for k, v in mapping['idx_to_label'].items()}
num_classes = len(idx_to_label)

# 测试集路径
test_folder_path = test_folder_path
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
test_filenames = [
    os.path.join(test_folder_path, f)
    for f in os.listdir(test_folder_path)
    if f.lower().endswith(valid_exts)
]
img_names = [os.path.basename(p) for p in test_filenames]

print(f"找到 {len(test_filenames)} 张测试图片")

# 预处理测试集
test_data = prep_test_image(test_filenames)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_dataset = FlowerDataset(test_data, y=None, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False)

# ==================== 加载模型 ====================
model_loaders = [load_efficientnet, load_convnext, load_swin]
model_names = ['EfficientNet.pth', 'ConvNeXt.pth', 'best_model.pth']

# ==================== 预测测试集 ====================
print("\n预测测试集...")

# 存储所有模型的测试集预测
# Shape: (n_models, n_test_samples, n_classes)
all_test_predictions = []

for model_idx, (loader_func, model_name) in enumerate(zip(model_loaders, model_names)):


    # 加载模型
    model, _, _, _ = loader_func(num_classes=num_classes, device=device)
    model.load_state_dict(torch.load("../model/" + model_names[model_idx], weights_only=True))
    model.eval()

    # 预测
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            predictions.append(probs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    all_test_predictions.append(predictions)

    print(f"  {model_name} 预测完成")

# 转换为 numpy 数组
all_test_predictions = np.array(all_test_predictions)

# ==================== 使用 Meta Model ====================
print("\n使用 Meta Model 进行最终预测...")

with open('../model/meta_model.pkl', 'rb') as f:
    meta_model = pickle.load(f)

# 拼接特征
# Shape: (n_test_samples, n_models * n_classes)
meta_test_features = all_test_predictions.transpose(1, 0, 2).reshape(len(test_filenames), -1)
print(f"测试集 Meta 特征形状: {meta_test_features.shape}")

# 预测
final_predictions = meta_model.predict(meta_test_features)

try:
    final_probabilities = meta_model.predict_proba(meta_test_features)
    final_confidences = np.max(final_probabilities, axis=1)
except:
    avg_probs = np.mean(all_test_predictions, axis=0)
    final_confidences = np.max(avg_probs, axis=1)

# 转换为原始标签
final_labels = [idx_to_label[int(idx)] for idx in final_predictions]

# ==================== 保存结果 ====================
print("\n保存结果...")

# Stacking 结果
submission_stacking = pd.DataFrame({
    "img_name": img_names,
    "predicted_class": final_labels,
    "confidence": [f"{c:.4f}" for c in final_confidences]
})
output_csv = output_csv_path
submission_stacking.to_csv(output_csv, index=False)
print(f"  Stacking 结果: {output_csv}")
