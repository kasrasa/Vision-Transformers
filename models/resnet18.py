import torch.nn as nn


class BasicBlock(nn.Module):
    """Two 3 x 3 convolutions with an identity or projected shortcut."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18(nn.Module):
    """Custom ResNet-18 with half-width stages by default: 32, 64, 128, 256.

    It keeps the [2, 2, 2, 2] basic-block layout and returns class logits.
    Use base_channels=64 for the standard stage widths.
    """

    def __init__(self, num_classes=10, base_channels=32):
        super().__init__()
        self.in_channels = base_channels
        self.config = {"num_classes": num_classes, "base_channels": base_channels}

        self.conv1 = nn.Conv2d(
            3, base_channels, kernel_size=7, stride=2, padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(base_channels, 2)
        self.layer2 = self._make_layer(base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        layers.extend(BasicBlock(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x).flatten(1)
        return self.fc(x)

