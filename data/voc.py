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
from transform import resize_image_and_boxes
from visualization import draw_boxes

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
    def __init__(self, voc_root, split="train"):
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
        size = (500, 500)
        image, boxes = resize_image_and_boxes(image, boxes, size)
        target = {
            "bboxes": boxes,
            "labels": labels
        }

        return image, target

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
            yield images, targets

if __name__ == "__main__":
    # test
    VOC_ROOT = Path("data/VOCdevkit/VOC2007")
    dataset = VOCDataset(VOC_ROOT, split="train")
    # for i in range(10):
    #     image, target = dataset[i]
    #     image = F.to_pil_image(image)
    #     boxes = target["bboxes"]
    #     draw_boxes(image, boxes)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=False)
    for images, targets in dataloader:
        image = images[0]
        target = targets[0]
        image = F.to_pil_image(image)
        boxes = target["bboxes"]
        draw_boxes(image, boxes)
        # print("len of images: ", len(images))
        # print(image.shape)
        # print(target["bboxes"])
        # print(target["labels"])
        break
