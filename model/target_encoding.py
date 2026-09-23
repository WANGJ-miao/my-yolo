"""
input targets: list of target
target = {
    "bboxes": boxes,
    "labels": labels
}
boxes: torch.tensor
[
    [xmin, ymin, xmax, ymax],
    [...]
]
labels: torch.tensor
[class ids...]
"""
import torch

from utils.box import xyxy2cxcywh
from data.voc import VOCDataset

# test
from data.visualization import draw_boxes

def encode_target(target):
    boxes = xyxy2cxcywh(target["bboxes"])
    labels = target["labels"]
    num = len(labels)
    print("len of item in one target: ", num)
    encoded_target = []
    for box, label in zip(boxes, labels):
        objectness = torch.tensor([1])
        class_score = torch.tensor([0] * 20)
        class_score[len(objectness) + len(box) + label] = 1
        encoded_target.append(torch.cat([box, objectness, class_score], dim=0))
    return torch.stack(encoded_target)

if __name__ == "__main__":
    # test
    dataset = VOCDataset()
    image, target = dataset[4]
    target = encode_target(target)
    boxes = target[:, :4]
    draw_boxes(image, boxes, mode="xywh")
