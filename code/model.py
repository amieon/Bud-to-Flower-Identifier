import torch
from torch import nn
from torch.optim import AdamW


# def cool3(model):
#     if isinstance(model, torch.nn.Module):  # 通用冻结逻辑
#         for name, param in model.named_parameters():
#             if 'features.0' in name or 'features.1' in name or 'features.3' in name:
#                 param.requires_grad = False


def load_convnext(num_classes, device):
    """加载 ConvNeXt-Tiny"""
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

    print("  加载 ConvNeXt...")
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    cool3(model)

    num_ftrs = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(1),
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 1024),
        nn.GELU(),
        nn.Dropout(0.35),
        nn.Linear(1024, num_classes)
    )
    model = model.to(device)
    # model.load_state_dict(torch.load('convnext_small_best.pth', weights_only=True))
    model.eval()
    optimizer = AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=5e-3,
        betas=(0.9, 0.999)
    )
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    return model, optimizer, criterion, scheduler


def load_efficientnet(num_classes, device):
    """加载 EfficientNet-B3"""
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

    print("  加载 EfficientNet...")
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

    cool3(model)

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)
    # model.load_state_dict(torch.load('efficientnet_b4_best.pth', weights_only=True))
    model.eval()
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=5e-3,
        betas=(0.9, 0.999)
    )
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    return model, optimizer, criterion, scheduler


def load_swin(num_classes, device):

    from torchvision.models import swin_t, Swin_T_Weights

    print("  加载 Swin Transformer...")
    model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

    cool3(model)

    num_ftrs = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(0.35),
        nn.Linear(num_ftrs, 1024),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes)
    )
    model = model.to(device)
    # model.load_state_dict(torch.load('swin_t_best.pth', weights_only=True))
    model.eval()
    optimizer = AdamW(
        model.parameters(),
        lr=4e-5,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    return model, optimizer, criterion, scheduler

