import json
from pathlib import Path

import torch
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
        download=True,
        transform=transform,
    )


def save_checkpoint(model, checkpoint_path, epoch, val_loss, val_accuracy):
    """Save the validation-selected weights and the metadata needed to reload them."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": model.__class__.__name__,
        "model_config": getattr(model, "config", {}),
        "epoch": epoch,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
    }, checkpoint_path)


def load_checkpoint(model, checkpoint_path, device):
    """Restore saved weights, including buffers such as BatchNorm statistics."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint


def save_results(results, path):
    """Write the result table explicitly requested by a notebook experiment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
