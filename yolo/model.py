"""
the structure of the model
RGB image
[B, C, H, W]
|
CNN backbone: extract the visual features
|
feature map [B, C, S, S]
|
detection head: turn features into predictions
|
predictions [B, 25, S, S]
|
permute
|
[B, S, S, 25]
"""

import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False # 功能上和batchnorm的偏置重复， 所以不用
            ),
            nn.BatchNorm2d(out_channels),   
            nn.LeakyReLU(0.1)               
        )

    def forward(self, x):
        return self.block(x)

class SimpleYOLO(nn.Module):
    def __init__(self, S=7, num_classes=20):
        super().__init__()
        self.S = S
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 512, stride=2),
            ConvBlock(512, 512, stride=2),
            ConvBlock(512, 512, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((S, S))
        self.head = nn.Conv2d(
            in_channels=512,
            out_channels=5 + num_classes,
            kernel_size=1
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)
        return x

if __name__ == "__main__":
    model = SimpleYOLO(S=7, num_classes=20)
    X = torch.randn(4, 3, 448, 448)
    pred = model(X)
    print(pred.shape)
