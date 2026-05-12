from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MRISliceImageFolderDataset(Dataset):
    """
    Simple MRI slice dataset.

    This dataset loads 2D MRI slices from a folder.

    Supported formats:
        .png, .jpg, .jpeg, .bmp, .tif, .tiff, .npy

    Output:
        {
            "image": tensor of shape (1, H, W),
            "path": file path
        }
    """

    SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".npy",
    }

    def __init__(
        self,
        root,
        image_size=256,
        normalization="minmax_per_slice",
        recursive=True,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.normalization = normalization

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        pattern = "**/*" if recursive else "*"

        self.files = sorted(
            [
                p
                for p in self.root.glob(pattern)
                if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
        )

        if len(self.files) == 0:
            raise RuntimeError(f"No supported image files found in: {self.root}")

    def __len__(self):
        return len(self.files)

    def _load_file(self, path):
        suffix = path.suffix.lower()

        if suffix == ".npy":
            image = np.load(path)
        else:
            image = Image.open(path).convert("F")

            if self.image_size is not None:
                image = image.resize(
                    (self.image_size, self.image_size),
                    resample=Image.BILINEAR,
                )

            image = np.array(image, dtype=np.float32)

        if image.ndim == 3:
            image = image[..., 0]

        image = image.astype(np.float32)

        return image

    def _normalize(self, image, eps=1e-8):
        if self.normalization == "none":
            return image

        if self.normalization == "minmax_per_slice":
            image_min = image.min()
            image_max = image.max()
            image = (image - image_min) / (image_max - image_min + eps)
            return image

        if self.normalization == "zscore_per_slice":
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (std + eps)
            return image

        raise ValueError(f"Unknown normalization: {self.normalization}")

    def __getitem__(self, index):
        path = self.files[index]

        image = self._load_file(path)
        image = self._normalize(image)

        image = torch.from_numpy(image).float()

        # Convert from (H, W) to (1, H, W)
        image = image.unsqueeze(0)

        return {
            "image": image,
            "path": str(path),
        }
