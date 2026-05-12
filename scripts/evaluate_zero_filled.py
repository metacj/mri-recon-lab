import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets import MRISliceImageFolderDataset
from evaluate import compute_reconstruction_metrics
from models import ZeroFilledReconstructor
from operators import SingleCoilCartesianOperator
from sampling.cartesian import random_cartesian_mask, get_sampling_percentage


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_demo_dataset(root, num_images=16, image_size=256):
    """
    Create a tiny synthetic MRI-like dataset for testing the full pipeline.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    yy, xx = np.meshgrid(
        np.linspace(-1, 1, image_size),
        np.linspace(-1, 1, image_size),
        indexing="ij",
    )

    for i in range(num_images):
        radius = 0.32 + 0.01 * i

        brain = np.exp(-((xx**2 + yy**2) / (2 * radius**2)))
        tissue_1 = 0.35 * np.exp(-(((xx - 0.18) ** 2 + (yy + 0.12) ** 2) / (2 * 0.12**2)))
        tissue_2 = 0.25 * np.exp(-(((xx + 0.18) ** 2 + (yy - 0.10) ** 2) / (2 * 0.09**2)))

        image = brain + tissue_1 + tissue_2
        image = image - image.min()
        image = image / (image.max() + 1e-8)

        image_uint8 = (255 * image).astype(np.uint8)
        out_path = root / f"demo_slice_{i:03d}.png"
        Image.fromarray(image_uint8).save(out_path)


def save_sample_image(target, recon, mask, save_path):
    """
    Save one visual comparison image: ground truth, zero-filled reconstruction, and mask.
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    target_np = target.detach().cpu().squeeze().numpy()
    recon_np = recon.detach().cpu().squeeze().numpy()
    mask_np = mask.detach().cpu().squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(target_np, cmap="gray")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(recon_np, cmap="gray")
    axes[1].set_title("Zero-filled")
    axes[1].axis("off")

    axes[2].imshow(mask_np, cmap="gray")
    axes[2].set_title("Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def evaluate_zero_filled(args):
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = Path(args.data_root)

    if args.create_demo_data:
        create_demo_dataset(
            root=data_root,
            num_images=args.demo_images,
            image_size=args.image_size,
        )

    dataset = MRISliceImageFolderDataset(
        root=data_root,
        image_size=args.image_size,
        normalization=args.normalization,
        recursive=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    mask_np = random_cartesian_mask(
        shape=(args.image_size, args.image_size),
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        seed=args.seed,
    )

    mask = torch.tensor(mask_np, dtype=torch.float32).to(device)

    operator = SingleCoilCartesianOperator(mask).to(device)
    model = ZeroFilledReconstructor(return_magnitude=True).to(device)
    model.eval()

    metric_sums = {
        "mse": 0.0,
        "mae": 0.0,
        "nmse": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }

    num_samples = 0
    saved_sample = False

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            target = batch["image"].to(device)

            y = operator.forward(target)
            recon = model(y, operator)

            metrics = compute_reconstruction_metrics(
                pred=recon,
                target=target,
                normalize=True,
            )

            batch_size = target.shape[0]

            for key in metric_sums:
                metric_sums[key] += metrics[key] * batch_size

            num_samples += batch_size

            if args.save_samples and not saved_sample:
                sample_path = Path(args.output_dir) / "sample_zero_filled.png"
                save_sample_image(
                    target=target[0],
                    recon=recon[0],
                    mask=mask,
                    save_path=sample_path,
                )
                saved_sample = True

    avg_metrics = {key: value / num_samples for key, value in metric_sums.items()}

    result = {
        "experiment": "zero_filled_cartesian",
        "data_root": str(data_root),
        "num_samples": num_samples,
        "image_size": args.image_size,
        "sampling": {
            "type": "cartesian_random",
            "acceleration": args.acceleration,
            "center_fraction": args.center_fraction,
            "actual_sampling_percentage": get_sampling_percentage(mask_np),
        },
        "metrics": avg_metrics,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=4)

    print("\nZero-filled evaluation completed.")
    print("--------------------------------")
    print(f"Number of samples: {num_samples}")
    print(f"Sampling percentage: {result['sampling']['actual_sampling_percentage']:.2f}%")

    print("\nAverage metrics:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.6f}")

    print(f"\nSaved metrics to: {metrics_path}")

    if args.save_samples:
        print(f"Saved sample image to: {output_dir / 'sample_zero_filled.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate zero-filled single-coil Cartesian MRI reconstruction."
    )

    parser.add_argument("--data_root", type=str, default="data/demo_brain256")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--normalization", type=str, default="minmax_per_slice")

    parser.add_argument("--acceleration", type=float, default=4)
    parser.add_argument("--center_fraction", type=float, default=0.08)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output_dir", type=str, default="results/zero_filled_demo")

    parser.add_argument("--create_demo_data", action="store_true")
    parser.add_argument("--demo_images", type=int, default=16)
    parser.add_argument("--save_samples", action="store_true")

    args = parser.parse_args()

    evaluate_zero_filled(args)


if __name__ == "__main__":
    main()
