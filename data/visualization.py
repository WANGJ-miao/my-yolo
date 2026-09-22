import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_boxes(image, boxes):
    """ 
    image 是PIL.Image, boxes 是此张图片中的框坐标
    boxes([ xmin, ymin, xmax, ymax ])
    """
    fig, ax = plt.subplots()
    ax.imshow(image) # pyright: ignore
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, fill=False, linewidth=2, color="green")
        ax.add_patch(rect)
    plt.show()
