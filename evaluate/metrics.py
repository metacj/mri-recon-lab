import math
import torch
import torch.nn.functional as F


def to_magnitude(x):
    """
    Convert real or complex tensor to magnitude image.
    """
    if torch.is_complex(x):
        return torch.abs(x)
    return x


def normalize_to_unit_range(x, eps=1e-8):
    """
    Normalize each image in a batch to [0, 1].
    Expected shape: (B, C, H, W) or similar.
    """
    x = to_magnitude(x)

    dims = tuple(range(1, x.ndim))
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)

    return (x - x_min) / (x_max - x_min + eps)


def mse(pred, target):
    """
    Mean squared error.
    """
    pred = to_magnitude(pred)
    target = to_magnitude(target)
    return torch.mean((pred - target) ** 2)


def mae(pred, target):
    """
    Mean absolute error.
    """
    pred = to_magnitude(pred)
    target = to_magnitude(target)
    return torch.mean(torch.abs(pred - target))


def nmse(pred, target, eps=1e-8):
    """
    Normalized mean squared error.

    NMSE = ||pred - target||_2^2 / ||target||_2^2
    """
    pred = to_magnitude(pred)
    target = to_magnitude(target)

    numerator = torch.sum((pred - target) ** 2)
    denominator = torch.sum(target ** 2) + eps

    return numerator / denominator


def psnr(pred, target, max_value=1.0, eps=1e-8):
    """
    Peak signal-to-noise ratio in dB.

    Assumes images are normalized to [0, 1].
    """
    pred = to_magnitude(pred)
    target = to_magnitude(target)

    mse_value = torch.mean((pred - target) ** 2)

    if mse_value.item() == 0:
        return torch.tensor(float("inf"), device=pred.device)

    return 20.0 * torch.log10(torch.tensor(max_value, device=pred.device)) - 10.0 * torch.log10(mse_value + eps)


def ssim(pred, target, max_value=1.0, window_size=11, eps=1e-8):
    """
    Simple differentiable SSIM implementation using average pooling.

    Expected input shape:
        (B, C, H, W)

    Returns mean SSIM over the batch.
    """
    pred = to_magnitude(pred)
    target = to_magnitude(target)

    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)

    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2

    padding = window_size // 2

    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=padding) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu_y_sq
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=padding) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    denominator = torch.clamp(denominator, min=eps)
    ssim_map = numerator / denominator
    ssim_map = torch.clamp(ssim_map, 0.0, 1.0)

    return ssim_map.mean()


def compute_reconstruction_metrics(pred, target, normalize=True):
    """
    Compute common MRI reconstruction metrics.

    Parameters
    ----------
    pred : torch.Tensor
        Reconstructed image.

    target : torch.Tensor
        Ground-truth image.

    normalize : bool
        If True, normalize pred and target to [0, 1] before metric computation.

    Returns
    -------
    dict
        Dictionary containing MSE, MAE, NMSE, PSNR, and SSIM.
    """
    pred = to_magnitude(pred)
    target = to_magnitude(target)

    if normalize:
        pred = normalize_to_unit_range(pred)
        target = normalize_to_unit_range(target)

    return {
        "mse": float(mse(pred, target).detach().cpu()),
        "mae": float(mae(pred, target).detach().cpu()),
        "nmse": float(nmse(pred, target).detach().cpu()),
        "psnr": float(psnr(pred, target).detach().cpu()),
        "ssim": float(ssim(pred, target).detach().cpu()),
    }
