from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets import MRISliceImageFolderDataset


def create_demo_dataset(root, num_images=8, image_size=256):
    """
    Create a tiny synthetic MRI-like dataset for testing the loader.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    yy, xx = np.meshgrid(
        np.linspace(-1, 1, image_size),
        np.linspace(-1, 1, image_size),
        indexing="ij",
    )

    for i in range(num_images):
        radius = 0.35 + 0.02 * i

        brain = np.exp(-((xx ** 2 + yy ** 2) / (2 * radius ** 2)))
        inner = 0.4 * np.exp(-(((xx - 0.15) ** 2 + (yy + 0.10) ** 2) / (2 * 0.10 ** 2)))
        image = brain + inner

        image = image - image.min()
        image = image / (image.max() + 1e-8)

        image_uint8 = (255 * image).astype(np.uint8)

        out_path = root / f"demo_slice_{i:03d}.png"
        Image.fromarray(image_uint8).save(out_path)


def main():
    demo_root = Path("data/demo_brain256")

    create_demo_dataset(demo_root, num_images=8, image_size=256)

    dataset = MRISliceImageFolderDataset(
        root=demo_root,
        image_size=256,
        normalization="minmax_per_slice",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    print("Dataset length:", len(dataset))

    batch = next(iter(dataloader))

    images = batch["image"]
    paths = batch["path"]

    print("Batch image shape:", images.shape)
    print("Batch dtype:", images.dtype)
    print("Min value:", images.min().item())
    print("Max value:", images.max().item())
    print("First path:", paths[0])

    assert images.shape == (4, 1, 256, 256)
    assert torch.isfinite(images).all()

    print("Dataset loader test passed.")


if __name__ == "__main__":
    main()
