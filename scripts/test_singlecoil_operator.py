import numpy as np
import torch

from sampling.cartesian import random_cartesian_mask
from operators import fft2c, ifft2c, SingleCoilCartesianOperator


def main():
    image_size = 256

    # Create a simple synthetic image
    x = torch.zeros(1, 1, image_size, image_size)
    x[:, :, 80:176, 80:176] = 1.0

    # Create undersampling mask
    mask_np = random_cartesian_mask(
        shape=(image_size, image_size),
        acceleration=4,
        center_fraction=0.08,
        seed=42,
    )

    mask = torch.tensor(mask_np)

    # Create MRI operator
    A = SingleCoilCartesianOperator(mask)

    # Forward: image -> undersampled k-space
    y = A.forward(x)

    # Adjoint: undersampled k-space -> zero-filled image
    x_zf = A.adjoint(y)

    print("Input image shape:", x.shape)
    print("Mask shape:", mask.shape)
    print("K-space shape:", y.shape)
    print("Zero-filled reconstruction shape:", x_zf.shape)

    print("Input dtype:", x.dtype)
    print("K-space dtype:", y.dtype)
    print("Zero-filled dtype:", x_zf.dtype)

    print("Mask sampling percentage:", 100.0 * mask.mean().item())

    # Check FFT inverse consistency with full sampling
    full_mask = torch.ones(image_size, image_size)
    A_full = SingleCoilCartesianOperator(full_mask)

    y_full = A_full.forward(x)
    x_full = A_full.adjoint(y_full)

    reconstruction_error = torch.mean(torch.abs(x_full.real - x)).item()

    print("Full-sampling reconstruction error:", reconstruction_error)

    if reconstruction_error < 1e-6:
        print("FFT/operator test passed.")
    else:
        print("FFT/operator test may have an issue.")


if __name__ == "__main__":
    main()
