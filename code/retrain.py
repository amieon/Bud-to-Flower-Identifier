import datetime
import pickle
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from CutMix import CutMix, mixup_criterion
import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

import xgboost as xgb

from meta import meta
from model import load_convnext, load_efficientnet, load_swin
from utils import prep_image, FlowerDataset, LossHistory, GridMask, val_transform, train_transform


class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout  # 保存原来的标准输出
        self.log = open(filename, "a", encoding="utf-8")  # 打开日志文件（追加模式）

    def write(self, message):
        self.terminal.write(message)  # 打印到控制台
        self.log.write(message)       # 写入文件
        self.log.flush()

    def flush(self):
        # Python 3 需要定义 flush，否则 print(..., flush=True) 会报错
        self.terminal.flush()
        self.log.flush()

# 启用日志记录（把 print 输出同时写入文件）
sys.stdout = Logger("train_log.txt")

# 设备检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"{'✅ GPU ready:' if torch.cuda.is_available() else '❌ No GPU detected.'} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

# 数据路径
csv_path = 'data/train_labels.csv'
image_dir = 'data/train'

# 加载CSV
df = pd.read_csv(csv_path)
filenames = [os.path.join(image_dir, fname) for fname in df['filename'].values]
y = df['category_id'].values.astype(np.int64)

print(f"CSV加载: {len(y)} 个样本，标签范围: {y.min()} 到 {y.max()}")
print("类分布:", np.unique(y, return_counts=True))

data, y_idx, idx_to_label, num_classes = prep_image(filenames, y)

# 数据拆分
x_train, x_valid, y_train, y_valid = train_test_split(data, y_idx, random_state=1, test_size=0.2,
                                                      stratify=y_idx)


test_transform = val_transform

train_dataset = FlowerDataset(x_train, y_train, transform=train_transform)
val_dataset = FlowerDataset(x_valid, y_valid, transform=val_transform)

# Loader
batch_size = 12
accumulation_steps = 3  # 等效 batch_size = 48
nb_epoch = 40
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
)

# 训练循环
print(f"All begin  {datetime.datetime.now()}")


model, _,_,_ = load_swin(num_classes=num_classes, device=device)
model.load_state_dict(torch.load('7_4.pth', weights_only=True))
model.to(device)

optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=5e-2,
    betas=(0.9, 0.999)
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=nb_epoch,
    T_mult=2,
    eta_min=1e-7
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
# criterion = nn.CrossEntropyLoss()

# if isinstance(model, torch.nn.Module):  # 通用冻结逻辑
#     for name, param in model.named_parameters():
#         if 'features.0' in name or 'features.1' in name or 'features.3' in name:
#             param.requires_grad = False


# 4. 继续训练
history = LossHistory()
early_stopping = 20
best_val_acc = float(0)
early_stop_counter = 0

# 训练循环
print("begin")
for epoch in range(nb_epoch):


    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    train_idx = 0

    for inputs, labels_batch in train_loader:
        # if (train_idx+1) % 100 == 0:
        #     print(f"epoch : {epoch} with its train_idx : {train_idx+1}/{len(train_loader) } in {datetime.datetime.now()}")
        inputs = inputs.to(device)
        labels_batch = labels_batch.to(device)

        inputs, labels_a, labels_b, lam = CutMix(inputs, labels_batch, device)

        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss = loss / accumulation_steps

        if (train_idx + 1) % accumulation_steps == 0:
            optimizer.zero_grad()  # 先清零

        loss.backward()  # 再反向传播

        if (train_idx + 1) % accumulation_steps == 0:
            optimizer.step()  # 最后更新

        train_idx += 1

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels_batch.size(0)
        correct_train += (predicted == labels_batch).sum().item()

    if (train_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

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

    print(
        f'模型 1 Epoch {epoch + 1}/{nb_epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print(f"当前 lr: {optimizer.param_groups[0]['lr']}  当前时间: {datetime.datetime.now()}")
    # scheduler.step(val_acc / 100)

    scheduler.step()

    torch.save(model.state_dict(), f'7_{epoch}.pth')
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     early_stop_counter = 0
    #     torch.save(model.state_dict(), f'best_ConvNeXt.pth')
    # else:
    #     early_stop_counter += 1
    #     if early_stop_counter >= early_stopping:
    #         print(f'模型 ConvNeXt Early stopping at epoch {epoch + 1}')
    #         torch.save(model.state_dict(), f'best_model_ConvNeXt.pth')
    #         break
