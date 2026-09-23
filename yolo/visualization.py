import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F

def draw_boxes(image, boxes, mode="xyxy", normalized=False, size=None):
    """ 
    image 是torch.tensor, boxes 是此张图片中的框坐标
    boxes([ xmin, ymin, xmax, ymax ])
    mode: "xyxy" / "xywh" 其中xywh中的xy是中心坐标
    """
    if normalized and size is None:
        raise ValueError("size should be provided when normalized is True")

    image = F.to_pil_image(image)
    fig, ax = plt.subplots()
    ax.imshow(image) # pyright: ignore
    for box in boxes:
        if normalized:
            image_width, image_height = size
            if mode == "xyxy":
                box[[0, 2]] *= image_width
                box[[1, 3]] *= image_height
            elif mode == "xywh":
                box[[0, 2]] *= image_width
                box[[1, 3]] *= image_height
        if mode == "xyxy":
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
        elif mode == "xywh":
            cx, cy, width, height = box
            xmin = cx - width / 2
            ymin = cy - height / 2
        else:
            raise ValueError("mode should be either 'xyxy' or 'xywh'")
        rect = patches.Rectangle((xmin, ymin), width, height, fill=False, linewidth=2, color="green")
        ax.add_patch(rect)
    plt.show()

def draw_serveral_images(images, boxes_list, mode="xyxy", normalized=False, size=None):
    """
    images is a list of torch tensor
    boxes_list is a list of bounding boxes :)

    mode: 
        xyxy -> [xmin, ymin, xmax, ymax]
        xywh -> [cx, cy, width, height]
    normalized:
        True -> boxes are normalized to [0, 1]
        False -> boxes are already in pixel coordinates
    size:
        (height, width) used when normalized = True
    """
    if normalized and size is None:
        raise ValueError("size should be provided when normalized is True")

    num = len(images)
    if num == 1:
        draw_boxes(images[0], boxes_list[0], mode=mode, normalized=False, size=None)
        return
    ncols = math.ceil(num / 2)
    fig, axes = plt.subplots(nrows=2, ncols=ncols)
    for ax, image, boxes in zip(axes.flatten(), images, boxes_list): 
        ax.imshow(F.to_pil_image(image))
        for box in boxes:
            if normalized:
                image_width, image_height = size
                if mode == "xyxy":
                    box[[0, 2]] *= image_width
                    box[[1, 3]] *= image_height
                elif mode == "xywh":
                    box[[0, 2]] *= image_width
                    box[[1, 3]] *= image_height
            if mode == "xyxy":
                xmin, ymin, xmax, ymax = box
                width = xmax - xmin
                height = ymax - ymin
            elif mode == "xywh":
                cx, cy, width, height = box
                xmin = cx - width / 2
                ymin = cy - height / 2
            else:
                print("mode should be either xyxy or xywh! fall to default xyxy")
                xmin, ymin, xmax, ymax = box
                width = xmax - xmin
                height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height, fill=False, linewidth=2, color="green")
            ax.add_patch(rect)
    plt.show()
