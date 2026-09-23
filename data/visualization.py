import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F

def draw_boxes(image, boxes, mode="xyxy"):
    """ 
    image 是torch.tensor, boxes 是此张图片中的框坐标
    boxes([ xmin, ymin, xmax, ymax ])
    mode: "xyxy" / "xywh" 其中xywh中的xy是中心坐标
    """
    image = F.to_pil_image(image)
    fig, ax = plt.subplots()
    ax.imshow(image) # pyright: ignore
    for box in boxes:
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
