from torchvision.models import resnet50


def build_resnet50(num_classes=10):
    """Build a randomly initialized ResNet-50 baseline that returns class logits.

    This uses Torchvision's ResNet-50 v1.5 architecture and downloads no weights.
    Train it from scratch with the same data splits and input preprocessing as ViT.
    Training and evaluation loops are supplied separately by the experiment.
    """
    return resnet50(weights=None, num_classes=num_classes)
