import numpy as np
import torch
import numpy as np
import torch
import torch.fft as fft



# ==================== 使用示例 ====================

def fmix_data(x, y, alpha=1.0, device='cuda'):
    """
    FMix 数据增强函数（与 mixup_data 接口兼容）

    Args:
        x: 输入图像 [batch_size, 3, 224, 224]
        y: 标签 [batch_size]
        alpha: Beta分布参数
        device: 设备

    Returns:
        mixed_x: 混合后的图像
        y_a, y_b: 两个原始标签
        lam: 混合比例
    """
    fmix = FMix(alpha=alpha, decay_power=3)
    return fmix(x, y, device=device)




def mixup_data(x, y, alpha=0.2, device='cuda'):
    """
    混合两个样本
    Args:
        x: 输入图像 [batch_size, 3, 224, 224]
        y: 标签 [batch_size]
        alpha: Beta分布参数，控制混合程度
    Returns:
        mixed_x: 混合后的图像
        y_a, y_b: 两个原始标签
        lam: 混合比例
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam



def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """
    CutMix增强 (修复 inplace 操作问题)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    x_mixed = x.clone()
    x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # 调整lambda为实际裁剪区域的比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]
    return x_mixed, y_a, y_b, lam

def CutMix(inputs, labels_batch, device):
    r = np.random.random()

    if r < 0.25:
        # 使用Mixup
        inputs, labels_a, labels_b, lam = mixup_data(
            inputs, labels_batch, alpha=0.4, device=device
        )
    elif r < 0.5:
        # 使用CutMix
        inputs, labels_a, labels_b, lam = cutmix_data(
            inputs, labels_batch, alpha=1.0, device=device
        )
    elif r < 0.75:
        inputs, labels_a, labels_b, lam = fmix_data(
            inputs, labels_batch, alpha=1.0, device=device
        )
    else:
        # 不使用增强
        labels_a = labels_batch
        labels_b = labels_batch
        lam = 1.0

    return inputs, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    return lam * loss_a + (1 - lam) * loss_b
