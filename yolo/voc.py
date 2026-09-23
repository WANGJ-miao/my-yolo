# Person: person
# Animal: bird, cat, cow, dog, horse, sheep
# Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
# Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor

import torch
from pathlib import Path
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from torchvision.transforms import functional as F

from yolo.transform import resize_image_and_boxes
from yolo.visualization import draw_boxes
from utils.box import xyxy2cxcywh
# test
from yolo.visualization import draw_serveral_images

VOC_CLASSES = [
    "person",
    "bird", 
    "cat", 
    "cow", 
    "dog", 
    "horse", 
    "sheep",
    "aeroplane", 
    "bicycle", 
    "boat", 
    "bus", 
    "car", 
    "motorbike", 
    "train",
    "bottle", 
    "chair", 
    "diningtable", 
    "pottedplant", 
    "sofa", 
    "tvmonitor",
]

CLASS_TO_IDX = {
    name: idx
    for idx, name in enumerate(VOC_CLASSES)
}

input_image_size = (460, 460) # height width

class VOCDataset(Dataset):
    """
    an OK dataset class should has the following methods and attributes:
    __init__:
        1. success torch.Dataset
        2. read train.txt
        3. set up class mapping
    __len__:
        1. return the number of data
    __getitem__:
        1. can be use like dataset[0], return the n^th data, containing GTboxes and corresponding class names

    the data format of an annotation:
    target = {
        "labels": torch.tensor (1 dim)
        "bboxes": torch.tensor (2 dim)
    }
    """
    def __init__(self, voc_root=Path("data/VOCdevkit/VOC2007"), split="train"):
        self.VOC_ROOT = Path(voc_root) 
        self.images_dir = self.VOC_ROOT / "JPEGImages"
        self.annotation_dir = self.VOC_ROOT / "Annotations"
        # VOC2007
        #  ├── Annotations
        #  ├── ImageSets
        #  │   ├── Layout
        #  │   ├── Main
        #  │   └── Segmentation
        #  ├── JPEGImages
        #  ├── SegmentationClass
        #  └── SegmentationObject

        split_file = self.VOC_ROOT / "ImageSets" / "Main" / f"{split}.txt"
        with open(split_file, "r") as f:
            self.images_ids = [line.strip() for line in f]

    def __len__(self):
        return len(self.images_ids)

    def __getitem__(self, index):
        index = self.images_ids[index]
        annotation_path = self.annotation_dir / f"{index}.xml"
        annotation_root = ET.parse(annotation_path)
        image_path = self.images_dir / f"{index}.jpg"
        image = F.to_tensor(Image.open(image_path).convert("RGB"))
        boxes = []
        labels = []
        for object in annotation_root.findall("object"):
              # <object>
              #     <name>chair</name>
              #     <pose>Unspecified</pose>
              #     <truncated>0</truncated>
              #     <difficult>0</difficult>
              #     <bndbox>
              #         <xmin>165</xmin>
              #         <ymin>264</ymin>
              #         <xmax>253</xmax>
              #         <ymax>372</ymax>
              #     </bndbox>
              # </object>
            name = object.find("name").text
            class_id = CLASS_TO_IDX[name]
            bandbox = object.find("bndbox")
            xmin = int(bandbox.find("xmin").text)
            ymin = int(bandbox.find("ymin").text)
            xmax = int(bandbox.find("xmax").text)
            ymax = int(bandbox.find("ymax").text)

            # width = xmax - xmin
            # height = ymax - ymin

            boxes.append([
                xmin,
                ymin,
                xmax,
                ymax
            ])
            labels.append(class_id)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        # transform to fixed size
        image, boxes = resize_image_and_boxes(image, boxes, input_image_size)
        target = {
            "bboxes": boxes,
            "labels": labels
        }

        return image, target

# target encoder
def encode_target(target):
    """
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
    返回值是一个25维的向量组
    并且其中的坐标已经归一化
    """
    boxes = xyxy2cxcywh(target["bboxes"])
    labels = target["labels"]
    # num = len(labels)
    # print("len of item in one target: ", num)
    encoded_target = []
    height = input_image_size[0]
    width = input_image_size[1]
    for box, label in zip(boxes, labels):
        box = box.clone() # 原来的box只是boxes的一个视图， 所以避免直接修改原来的boxes, 这里先clone
        box[[0, 2]] /= width
        box[[1, 3]] /= height
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

def collate_fn(batch):
    """
    batch is a list of tuples:(image, target)
    return a tuple (images:torch.tensor [N, C, H, W], targets:list of target)
    """
    images , targets = tuple(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, targets

class DataLoader():
    """provide __iter__ method, returning data batch"""
    def __init__(self, dataset, batch_size, shuffle=True, collate_fn=collate_fn):
        # TODO implement shuffle
        self.batch_size = batch_size
        self.dataset = dataset
        self.collate_fn = collate_fn

    def __iter__(self):
        for batch_start in range(0, len(self.dataset), self.batch_size):
            batch = []
            for i in range(batch_start, min(batch_start + self.batch_size, len(self.dataset))):
                # 目前是如果最后一个batch不够分, 就有多少返回多少
                sample = self.dataset[i]
                batch.append(sample)
            images, targets = collate_fn(batch)
            targets=encode_targets(targets)
            yield images, targets

if __name__ == "__main__":
    # test
    dataset = VOCDataset()
    dataloader = DataLoader(dataset, batch_size=4)
    for i, (images, targets) in enumerate(dataloader):
        boxes_list = [target[:, :4] for target in targets]
        print(targets)
        draw_serveral_images(images, boxes_list, mode="xywh", normalized=True, size=input_image_size)
        if i >= 6:
            break
