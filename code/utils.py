import json
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.fft as fft


class GridMask:
    def __init__(self, d_range=(96, 224), ratio=0.6, prob=0.5):
        self.d_range = d_range
        self.ratio = ratio
        self.prob = prob

    def __call__(self, img):
        if np.random.random() > self.prob:
            return img

        h, w = img.shape[1], img.shape[2]
        d = np.random.randint(self.d_range[0], self.d_range[1])
        l = int(d * self.ratio)

        mask = np.ones((h, w), dtype=np.float32)
        for i in range(0, h, d):
            for j in range(0, w, d):
                mask[i:i + l, j:j + l] = 0

        mask = torch.from_numpy(mask).unsqueeze(0)
        return img * mask

class LossHistory:
    def __init__(self):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, train_loss, val_loss):
        self.losses.append(train_loss)
        self.val_losses.append(val_loss)

class FlowerDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        img_hwc = np.transpose(img, (1, 2, 0)) * 255
        img_hwc = img_hwc.astype(np.uint8)
        img_pil = Image.fromarray(img_hwc)

        if self.transform:
            img_pil = self.transform(img_pil)

        if self.y is not None:
            label = torch.tensor(self.y[idx]).long()
            return img_pil, label
        return img_pil


# 数据预处理
channels = 3
rows = 224
cols = 224

def prep_image(img_paths, y_raw):
    unique_labels = np.unique(y_raw)
    unique_labels_sorted = np.sort(unique_labels)
    num_classes = len(unique_labels_sorted)
    label_to_idx = {int(label): idx for idx, label in enumerate(unique_labels_sorted)}
    y_idx = np.array([label_to_idx[int(label)] for label in y_raw]).astype(np.int64)

    idx_to_label = {idx: int(label) for idx, label in enumerate(unique_labels_sorted)}
    mapping_dict = {'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label}
    with open('../model/config.json', 'w') as f:
        json.dump(mapping_dict, f, indent=4)
    print(f"映射保存完成，num_classes: {num_classes}")

    data = np.empty((len(img_paths), channels, rows, cols), dtype=np.uint8)
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"未加载 {img_path}")
            continue
        img = cv2.resize(img, (rows, cols), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data[i] = np.transpose(img, (2, 0, 1))
        if i % 1000 == 0:
            print(f'Processed {i} of {len(img_paths)}')

    data = data.astype(np.float32) / 255.0
    return data, y_idx, idx_to_label, num_classes

def prep_test_image(img_paths):
    data = np.empty((len(img_paths), channels, rows, cols), dtype=np.uint8)
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"未加载 {img_path}")
            continue
        img = cv2.resize(img, (rows, cols), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data[i] = np.transpose(img, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        if i % 1000 == 0:
            print(f'Processed {i}/{len(img_paths)} images...')
    data = data.astype(np.float32) / 255.0
    return data


train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=90),
    transforms.RandomResizedCrop(size=224, scale=(0.95, 1.05), ratio=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.1, contrast=0.05, saturation=0.05, hue=0.05),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ToTensor(),
    GridMask(prob=0.5),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])