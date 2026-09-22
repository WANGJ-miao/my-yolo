from utils.box import xyxy2cxcywh
from data.voc import VOCDataset
from pathlib import Path

voc_root = Path("data/VOCdevkit/VOC2007")
dataset = VOCDataset(voc_root)

print("num of data for training: ", len(dataset))

image, annotation = dataset[0]
bboxes = annotation["bboxes"]
new_bboxes = xyxy2cxcywh(bboxes)
print(image)
print(bboxes)
print(new_bboxes)
