"""The notebook's TorchVision augmentation policy, distinct from DeiT's timm policy."""

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


def build_batch_transform(num_classes):
    """Choose MixUp or CutMix for a loaded batch and return dense class targets."""
    return v2.RandomChoice([
        v2.MixUp(num_classes=num_classes, alpha=0.8),
        v2.CutMix(num_classes=num_classes, alpha=1.0),
    ])
