import torch

from evaluate import compute_reconstruction_metrics, psnr, ssim, nmse


def main():
    # Ground-truth image
    target = torch.zeros(1, 1, 256, 256)
    target[:, :, 80:176, 80:176] = 1.0

    # Slightly noisy reconstruction
    pred = target + 0.05 * torch.randn_like(target)
    pred = torch.clamp(pred, 0.0, 1.0)

    metrics = compute_reconstruction_metrics(pred, target, normalize=True)

    print("Reconstruction metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    # Perfect reconstruction test
    perfect_metrics = compute_reconstruction_metrics(target, target, normalize=True)

    print("\nPerfect reconstruction metrics:")
    for key, value in perfect_metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
