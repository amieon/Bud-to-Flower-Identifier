import pickle
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
import numpy
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from model import load_convnext, load_efficientnet, load_swin
from utils import load_TTA, load_TTA_1

# ==================== 配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
start_time = time.time()
# TTA 配置
USE_TTA = True  # 是否使用 TTA

# 加载标签映射
import json

with open('../model/config.json', 'r') as f:
    mapping = json.load(f)
idx_to_label = {int(k): int(v) for k, v in mapping['idx_to_label'].items()}
num_classes = len(idx_to_label)

# 测试集路径
test_folder_path = sys.argv[1]
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
test_filenames = [
    os.path.join(test_folder_path, f)
    for f in os.listdir(test_folder_path)
    if f.lower().endswith(valid_exts)
]
img_names = [os.path.basename(p) for p in test_filenames]
print(f"📸 找到 {len(test_filenames)} 张测试图片")

# ==================== TTA 变换定义 ====================
# 基础变换（必需）
base_transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# TTA 变换列表
tta_transforms, TTA_NUM_AUGMENTS = load_TTA_1()

# 根据配置选择使用的变换
if USE_TTA:
    selected_transforms = tta_transforms[:TTA_NUM_AUGMENTS]
    print(f"\n✨ TTA 已启用: 每张图片将进行 {len(selected_transforms)} 次增强")
else:
    selected_transforms = [tta_transforms[0]]  # 只用中心裁剪
    print(f"\n⚠️  TTA 未启用: 使用标准预测")

# ==================== 加载基模型 ====================
print("\n📦 加载基模型...")
model_names = ['ConvNeXt', 'Swin']
base_models = []

for model_name in model_names:
    print(f"  加载 {model_name}...")

    # 创建模型
    if model_name == 'ConvNeXt':
        model, _, _, _ = load_convnext(num_classes=num_classes, device=device)
    elif model_name == 'Swin':
        model, _, _, _ = load_swin(num_classes=num_classes, device=device)

    # 加载权重
    model_path = f'../model/best_{model_name}.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    base_models.append(model)
    print(f"    ✅ {model_name} 加载完成")

# ==================== 加载 Meta 模型 ====================
print("\n🔮 加载 Meta 模型...")
with open('../model/meta_model.pkl', 'rb') as f:
    meta_model = pickle.load(f)
print(f"  ✅ Meta 模型加载完成 (类型: {type(meta_model).__name__})")

# ==================== 第一阶段：基模型预测（带 TTA）====================
print(f"\n🎯 第一阶段：使用基模型生成预测概率...")

all_base_predictions = []  # 存储所有基模型的预测

for idx, model in enumerate(base_models):
    model_name = model_names[idx]
    print(f"\n[{idx + 1}/{len(base_models)}] {model_name} 预测中...")

    probabilities = []

    with torch.no_grad():
        for i, img_path in enumerate(test_filenames):
            # 加载图片
            img = Image.open(img_path).convert('RGB')

            # TTA: 对同一张图片进行多次增强预测
            tta_probs = []
            for transform in selected_transforms:
                img_tensor = transform(img).unsqueeze(0).to(device)
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                tta_probs.append(probs.cpu().numpy()[0])

            # 平均所有 TTA 预测
            avg_probs = np.mean(tta_probs, axis=0)
            probabilities.append(avg_probs)

            # 进度显示
            if (i + 1) % 500 == 0:
                print(f"    进度: {i + 1}/{len(test_filenames)}")

    all_base_predictions.append(np.array(probabilities))
    print(f"  ✅ {model_name} 预测完成")

# ==================== 拼接特征 ====================
print("\n🔧 拼接基模型预测为特征矩阵...")
# Shape: (n_test_samples, n_models * n_classes)
test_features = np.concatenate(all_base_predictions, axis=1)
print(f"  特征矩阵形状: {test_features.shape}")
print(f"  - 样本数: {test_features.shape[0]}")
print(f"  - 特征数: {test_features.shape[1]} = {len(base_models)} 模型 × {num_classes} 类")

# ==================== 第二阶段：Meta 模型预测 ====================
print(f"\n🚀 第二阶段：使用 Meta 模型进行最终预测...")

# Meta 模型预测
meta_predictions = meta_model.predict(test_features)  # 预测类别
meta_probabilities = meta_model.predict_proba(test_features)  # 预测概率

# 获取每个预测的置信度（最大概率）
pred_confidences = np.max(meta_probabilities, axis=1)

# 转换为原始标签
final_labels = [idx_to_label[int(pred)] for pred in meta_predictions]

print(f"  ✅ Meta 模型预测完成")
print(f"  平均置信度: {pred_confidences.mean():.4f}")
print(f"  最低置信度: {pred_confidences.min():.4f}")
print(f"  最高置信度: {pred_confidences.max():.4f}")

# ==================== 保存结果 ====================
print("\n💾 保存预测结果...")

submission = pd.DataFrame({
    "img_name": img_names,
    "predicted_class": final_labels,
    "confidence": [f"{c:.4f}" for c in pred_confidences]
})

output_csv = sys.argv[2]
submission.to_csv(output_csv, index=False)
print(f"  ✅ 预测结果已保存: {output_csv}")

# ==================== 统计信息 ====================
print("\n📊 预测统计:")
print(f"  总样本数: {len(final_labels)}")
print(f"  预测类别数: {len(set(final_labels))}")
print(f"  TTA 状态: {'✅ 已启用 (' + str(len(selected_transforms)) + ' 次增强)' if USE_TTA else '❌ 未启用'}")

print("\n🎉 Stacking + TTA 预测完成！")

elapsed_time = time.time() - start_time
print(f"\n⏱️  总用时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")