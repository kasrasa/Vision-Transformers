from pathlib import Path

from torchvision.datasets import Imagenette
from torchvision import transforms


def download_imagenette_dataset(root_dir: str = "data", split: str = "train", image_size: int = 224,
                                transform=None):
    """Load Imagenette 320px as fixed-size float RGB tensors in [0, 1].

    Use split='train' or 'val'. The default deterministic transforms keep each
    example fixed. Supply a training transform on a separate dataset instance
    when augmenting; both instances reuse the same downloaded archive.
    """
    if transform is None:
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

