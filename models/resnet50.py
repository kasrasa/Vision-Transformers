import torch.nn as nn


class BottleNeckResNet50(nn.Module):
    """ResNet v1.5 bottleneck: out_channels is the internal width.

    The block produces expansion * out_channels output channels. Its shortcut
    projects the input when channels or spatial resolution need to change.
    """

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        expanded_channels = out_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, expanded_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expanded_channels)

        if stride != 1 or in_channels != expanded_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expanded_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    """ResNet-50 v1.5 built from bottleneck blocks, returning class logits."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleNeckResNet50.expansion, num_classes)

        # Match the initialization used by the reference ResNet implementation.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(BottleNeckResNet50(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * BottleNeckResNet50.expansion
        for _ in range(1, blocks):
            layers.append(BottleNeckResNet50(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

