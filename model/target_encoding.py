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
from data.voc import VOCDataset, VOC_CLASSES, DataLoader

# test
from data.visualization import draw_serveral_images

def encode_target(target):
    boxes = xyxy2cxcywh(target["bboxes"])
    labels = target["labels"]
    # num = len(labels)
    # print("len of item in one target: ", num)
    encoded_target = []
    for box, label in zip(boxes, labels):
        objectness = torch.tensor([1])
        class_score = torch.tensor([0] * 20)
        # print("label: ", label, VOC_CLASSES[label])
        # print("objectness length:", len(objectness))
        # print("box length:", len(box))
        # print("label:", label)
        class_score[label] = 1
        encoded_target.append(torch.cat([box, objectness, class_score], dim=0))
    return torch.stack(encoded_target)

def encode_targets(targets):
    """
    targets need to be iterable
    返回targets list
    """
    new_targets = []
    for target in targets:
        new_targets.append(encode_target(target))
    return new_targets

if __name__ == "__main__":
    # test
    dataset = VOCDataset()
    dataloader = DataLoader(dataset, batch_size=4)
    for i, (images, targets) in enumerate(dataloader):
        new_targets = encode_targets(targets)
        boxes_list = [target[:, :4] for target in new_targets]
        draw_serveral_images(images, boxes_list, mode="xywh")
        # for target in new_targets:
        #     print(target)
        if i >= 2:
            break
