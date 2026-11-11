from pathlib import Path

from torchvision.datasets import Imagenette
from torchvision import transforms


def download_imagenette_dataset(root_dir: str = "data", split: str = "train", image_size: int = 224):
    """Load Imagenette 320px as fixed-size float RGB tensors in [0, 1].

    Use split='train' or 'val'. Deterministic transforms keep each example fixed
    for the memorization check; both splits reuse the same downloaded archive.
    """
    transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

    return Imagenette(
        root=root_dir,
        split=split,
        size="320px",
        download=not (Path(root_dir) / "imagenette2-320").exists(),
        transform=transform,
    )

