"""The notebook's TorchVision augmentation policy, distinct from DeiT's timm policy."""

from torch import nn
from torchvision import transforms
from torchvision.transforms import v2


def build_image_transform(image_size=224):
    """Augment one PIL image, then convert to a tensor before random erasing."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9,
                               interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25),
    ])


class _SmoothTargets(nn.Module):
    """Smooth the dense targets produced by batch mixing once."""

    def __init__(self, num_classes, label_smoothing):
        super().__init__()
        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError("label_smoothing must be between 0 and 1")
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def forward(self, images, targets):
        targets = (1.0 - self.label_smoothing) * targets + self.label_smoothing / self.num_classes
        return images, targets

    def extra_repr(self):
        return f"num_classes={self.num_classes}, label_smoothing={self.label_smoothing}"


def build_batch_transform(num_classes, label_smoothing=0.0):
    """Choose MixUp or CutMix, then optionally smooth their dense targets once."""
    return v2.Compose([
        v2.RandomChoice([
            v2.MixUp(num_classes=num_classes, alpha=0.8),
            v2.CutMix(num_classes=num_classes, alpha=1.0),
        ]),
        _SmoothTargets(num_classes, label_smoothing),
    ])
