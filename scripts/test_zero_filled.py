import torch

from sampling.cartesian import random_cartesian_mask
from operators import SingleCoilCartesianOperator
from models import ZeroFilledReconstructor, zero_filled_reconstruction
from evaluate import compute_reconstruction_metrics


def create_synthetic_image(image_size=256):
    """
    Create a simple synthetic image for testing.
    """
    x = torch.zeros(1, 1, image_size, image_size)

    # Large square
    x[:, :, 70:185, 70:185] = 0.7

    # Smaller bright square
    x[:, :, 110:150, 110:150] = 1.0

    return x


def main():
    image_size = 256

    # Ground-truth image
    x = create_synthetic_image(image_size=image_size)

    # Cartesian undersampling mask
    mask_np = random_cartesian_mask(
        shape=(image_size, image_size),
        acceleration=4,
        center_fraction=0.08,
        seed=42,
    )

    mask = torch.tensor(mask_np)

    # MRI operator
    A = SingleCoilCartesianOperator(mask)

    # Forward operation: image -> undersampled k-space
    y = A.forward(x)

    # Method 1: using module
    zf_model = ZeroFilledReconstructor(return_magnitude=True)
    x_zf_module = zf_model(y, A)

    # Method 2: using function
    x_zf_function = zero_filled_reconstruction(y, A, return_magnitude=True)

    # Compare both zero-filled outputs
    diff = torch.mean(torch.abs(x_zf_module - x_zf_function)).item()

    # Compute metrics
    metrics = compute_reconstruction_metrics(x_zf_module, x, normalize=True)

    print("Zero-filled reconstruction test")
    print("--------------------------------")
    print("Ground truth shape:", x.shape)
    print("Mask shape:", mask.shape)
    print("K-space shape:", y.shape)
    print("Zero-filled shape:", x_zf_module.shape)
    print("Module/function difference:", diff)
    print("Sampling percentage:", 100.0 * mask.mean().item())

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    # Full sampling sanity check
    full_mask = torch.ones(image_size, image_size)
    A_full = SingleCoilCartesianOperator(full_mask)

    y_full = A_full.forward(x)
    x_full_zf = zero_filled_reconstruction(y_full, A_full, return_magnitude=True)

    full_metrics = compute_reconstruction_metrics(x_full_zf, x, normalize=True)

    print("\nFull-sampling zero-filled sanity check:")
    for key, value in full_metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
