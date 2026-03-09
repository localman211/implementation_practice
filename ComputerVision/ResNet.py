    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, ):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=3 if in_channels != out_channels else 1,
                                   stride=1 if in_channels != out_channels else 2,
                                   padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=1, stride=2, padding=0)
                )

        def forward(self, x):
            f = F.relu(self.bn1(self.conv1(x)))
            f = F.relu(self.bn2(self.conv2(f)) + self.shortcut(x))
            return f

    class ResNet(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                                   kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.block1 = nn.Sequential(
                ResidualBlock(in_channels=64, out_channels=64),
                ResidualBlock(in_channels=64, out_channels=64),
                ResidualBlock(in_channels=64, out_channels=64),
            )

            self.block2 = nn.Sequential(
                ResidualBlock(in_channels=64, out_channels=128),
                ResidualBlock(in_channels=128, out_channels=128),
                ResidualBlock(in_channels=128, out_channels=128),
                ResidualBlock(in_channels=128, out_channels=128),
            )

            self.block3 = nn.Sequential(
                ResidualBlock(in_channels=128, out_channels=256),
                ResidualBlock(in_channels=256, out_channels=256),
                ResidualBlock(in_channels=256, out_channels=256),
                ResidualBlock(in_channels=256, out_channels=256),
                ResidualBlock(in_channels=256, out_channels=256),
                ResidualBlock(in_channels=256, out_channels=256),
            )

            self.block4 = nn.Sequential(
                ResidualBlock(in_channels=256, out_channels=512),
                ResidualBlock(in_channels=512, out_channels=512),
                ResidualBlock(in_channels=512, out_channels=512),
            )

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(in_features=512, out_features=num_classes)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)

            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)

            x = self.avgpool(x)

            x = x.view(x.size(0), -1)

            x = self.fc(x)

            return x