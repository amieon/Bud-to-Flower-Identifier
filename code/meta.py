# ==================== 训练 Meta Model ====================
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression


def meta(ensemble_models, model_names, val_loader, num_classes, device):
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
                probs = torch.nn.functional.softmax(outputs, dim=1)

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

    print("\n训练 Meta Model...")
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
    print(f"Meta Model 已保存")
    return best_meta_model, best_meta_name

