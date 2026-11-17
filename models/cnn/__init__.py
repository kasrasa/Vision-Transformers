"""CNN baselines shared by the article experiments."""

from .resnet18 import ResNet18
from .resnet50 import ResNet50

__all__ = ["ResNet18", "ResNet50"]
