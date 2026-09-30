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
        """
        output tensor shape:
        [N, S, S, 25]
        """
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)
        x[..., :4] = torch.sigmoid(x[..., :4])
        return x

class SimpleYoloLoss(nn.Module):
    """
    针对一个/一组结果进行损失计算
    [boxes(4), objectness(1), classes_pred(20)]
    """
    def __init__(self):
        super().__init__()
        self.box_loss_fn = nn.MSELoss()
        self.objectness_loss_fn = nn.BCEWithLogitsLoss()
        self.classes_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        """
        target: torch tensor, dim=[N, S, S, 25]
        pred: torch tensor, dim=[N, S, S, 25]
        """
        target_box = target[..., 0:4]
        target_objectness = target[..., 4]
        target_classes = target[..., 5:]

        pred_box = pred[..., 0:4]
        pred_objectness = pred[..., 4]
        pred_classes = pred[..., 5:]

        positive_mask = target_objectness.bool()

        target_box_pos = target_box[positive_mask]
        target_classes_pos = target_classes[positive_mask]
        pred_box_pos = pred_box[positive_mask]
        pred_classes_pos = pred_classes[positive_mask]

        objectness_loss = self.objectness_loss_fn(pred_objectness, target_objectness)
        box_loss = self.box_loss_fn(pred_box_pos, target_box_pos)
        class_loss = self.classes_loss_fn(pred_classes_pos, target_classes_pos)

        loss = box_loss + objectness_loss + class_loss;

        return loss


if __name__ == "__main__":
    model = SimpleYOLO(S=7, num_classes=20)
    X = torch.randn(4, 3, 448, 448)
    pred = model(X)
    print(pred.shape)
