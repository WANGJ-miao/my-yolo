from pathlib import Path
from typing import reveal_type
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET

# .
# └── data
#     └── VOCdevkit
#         └── VOC2007
#             ├── Annotations
#             ├── ImageSets
#             │   ├── Layout
#             │   ├── Main
#             │   └── Segmentation
#             └── JPEGImages

VOC_ROOT = Path("data/VOCdevkit/VOC2007")

image_dir = VOC_ROOT / "JPEGImages"
annotation_dir = VOC_ROOT / "Annotations"

# images = list(image_dir.glob("*.jpg"))
# annotations = list(annotation_dir.glob("*.xml"))
# print("Number of images:", len(images))
# print("Number of annotations:", len(annotations))
# print("image 1: ", images[1])
# print("annotations 0: ", annotations[0])

image_id = "000007"
image_path = image_dir / f"{image_id}.jpg"
annotation_path = annotation_dir / f"{image_id}.xml"

image = Image.open(image_path)
annotation_tree = ET.parse(annotation_path)
annotation_root = annotation_tree.getroot()

fig, ax = plt.subplots()

ax.imshow(image) # pyright: ignore

for i, object in enumerate(annotation_root.findall("object")):
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
    bandbox = object.find("bndbox")
    xmin = int(bandbox.find("xmin").text)
    ymin = int(bandbox.find("ymin").text)
    xmax = int(bandbox.find("xmax").text)
    ymax = int(bandbox.find("ymax").text)

    width = xmax - xmin
    height = ymax - ymin

    rect = patches.Rectangle((xmin, ymin), width, height, fill=False, linewidth=2, color="green")
    ax.add_patch(rect)
    ax.text(xmin, ymin, name, fontsize=10, color="black")
    # print(name, i)
    # print("xmin: ", xmin)
    # print("ymin: ", ymin)
    # print("xmax: ", xmax)
    # print("ymax: ", ymax)

plt.show()
