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
