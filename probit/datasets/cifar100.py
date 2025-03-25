"""CIFAR100 dataset with train/val/test splits."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from torch import Tensor
from torchvision.datasets import CIFAR100


class CIFAR100Split(CIFAR100):
    """CIFAR100 dataset with custom validation/test split.

    This class extends the CIFAR100 dataset to handle all data splits:
    - 'train': Original training set (50k samples)
    - 'val': First 4k samples from the original test set
    - 'test': Remaining 6k samples from the original test set
    - 'all': All original *test* samples (10k)

    It also adds support for out-of-distribution transforms similar to SoftDataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset.
        split (str): Which split to use: 'train', 'val', 'test', or 'all'. Defaults to
            'test'.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. For OOD transforms, it should accept
            a random number generator as a second argument.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        *,
        download: bool = False,
    ) -> None:
        # For train split, use the original training set
        if split == "train":
            super().__init__(
                root=root,
                train=True,
                transform=transform,
                target_transform=target_transform,
                download=download,
            )
        else:
            # For val/test/all splits, load the original test set
            super().__init__(
                root=root,
                train=False,
                transform=transform,
                target_transform=target_transform,
                download=download,
            )

            # Split the test data (total 10k samples)
            if split == "val":
                # Use first 4k samples for validation
                self.data = self.data[:4000]
                self.targets = self.targets[:4000]
            elif split == "test":
                # Use remaining 6k samples for test
                self.data = self.data[4000:]
                self.targets = self.targets[4000:]
            elif split == "all":
                # Use all test samples (no change needed)
                pass
            elif split != "train":
                msg = (
                    f"Split {split} not recognized. "
                    "Use 'train', 'val', 'test', or 'all'."
                )
                raise ValueError(msg)

        # Add OOD flag
        self.is_ood = False
        self.split = split

    def __getitem__(self, index: int) -> tuple[Image.Image, Tensor]:
        """Get an item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Returns:
            A tuple containing the image and its label.
        """
        img, target = self.data[index], self.targets[index]

        # Convert to PIL Image
        img = Image.fromarray(img)

        # Apply transform with OOD handling if necessary
        if self.transform is not None and self.is_ood:
            rng = np.random.default_rng(seed=index)
            img = self.transform(img, rng)
        elif self.transform is not None:
            img = self.transform(img)

        # Apply target transform
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def set_ood(self) -> None:
        """Sets the dataset to use out-of-distribution transform."""
        self.is_ood = True

    def extra_repr(self) -> str:
        """Return extra information about the dataset."""
        return f"Split: {self.split}"
