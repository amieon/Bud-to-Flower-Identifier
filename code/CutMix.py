import numpy as np
import torch
import numpy as np
import torch
import torch.fft as fft


class FMix:
    """
    FMix: Frequency domain mixup
    在频域进行混合，比空域 Mixup 更平滑自然
    """

    def __init__(self, alpha=1.0, decay_power=3, max_soft=0.0, reformulate=False):
        """
        Args:
            alpha: Beta 分布参数，控制混合比例
            decay_power: 频率衰减指数，越大低频占比越高
            max_soft: 最大软化系数（通常设为 0）
            reformulate: 是否使用改进公式
        """
        self.alpha = alpha
        self.decay_power = decay_power
        self.max_soft = max_soft
        self.reformulate = reformulate

    def _get_spectrum(self, lam, h, w):
        """
        生成频谱遮罩

        Args:
            lam: 混合比例
            h, w: 图像高度和宽度

        Returns:
            遮罩矩阵 [h, w]
        """
        # 生成频率网格
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)

        # 计算到中心的距离（频率）
        d = np.sqrt(xx ** 2 + yy ** 2)

        # 频率衰减：距离越大（高频），权重越小
        decay = (1 + d) ** (-self.decay_power)

        # 归一化到 [0, 1]
        decay = decay / np.max(decay)

        # 根据 lam 生成二值遮罩
        # lam 越大，保留越多低频（中心区域）
        threshold = np.percentile(decay, (1 - lam) * 100)
        mask = (decay >= threshold).astype(np.float32)

        return mask

    def _get_mask(self, h, w):
        """
        生成随机遮罩（简化版，直接随机）

        Args:
            h, w: 图像高度和宽度

        Returns:
            遮罩矩阵 [h, w]
        """
        # 随机生成 lam
        lam = np.random.beta(self.alpha, self.alpha)

        # 生成频谱遮罩
        mask = self._get_spectrum(lam, h, w)

        return mask

    def __call__(self, x, y, device='cuda'):
        """
        对批量数据应用 FMix

        Args:
            x: 输入图像 [batch_size, C, H, W]
            y: 标签 [batch_size]
            device: 设备

        Returns:
            mixed_x: 混合后的图像
            y_a, y_b: 两个原始标签
            lam: 混合比例
        """
        batch_size = x.size(0)

        # 生成混合比例
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # 随机打乱索引
        index = torch.randperm(batch_size).to(device)

        # 获取第二组样本
        x2 = x[index]
        y_a, y_b = y, y[index]

        # 生成遮罩
        h, w = x.shape[-2:]
        mask = self._get_mask(h, w)
        mask = torch.from_numpy(mask).float().to(device)

        # 扩展遮罩维度以匹配 [B, C, H, W]
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # 转到频域
        x1_fft = fft.fftn(x, dim=(-2, -1))
        x2_fft = fft.fftn(x2, dim=(-2, -1))

        # 使用 fftshift 确保低频在中心
        x1_fft = fft.fftshift(fft.fftn(x, dim=(-2, -1)))
        x2_fft = fft.fftshift(fft.fftn(x2, dim=(-2, -1)))

        # 频域混合
        mixed_fft = mask * x1_fft + (1 - mask) * x2_fft

        # 反变换回空域
        mixed_x = fft.ifftn(fft.ifftshift(mixed_fft), dim=(-2, -1)).real

        # 调整 lam（根据实际遮罩比例）
        lam = mask.sum().item() / (mask.numel())

        return mixed_x, y_a, y_b, lam


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


# ========== CutMix 数据增强函数 ==========
def rand_bbox(size, lam):
    """
    生成随机裁剪框
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 随机中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


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
