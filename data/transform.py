from torchvision.transforms import functional as F

def resize_image_and_boxes(image, boxes, size=(460, 460)):
    """size = (height, width)"""
    # image is torch.tensor, it's shape is [C, H, W]
    old_height, old_width = image.shape[-2:]
    new_height, new_width = size
    scale_x = new_width / old_width
    scale_y = new_height / old_height
    image = F.resize(image, size)
    boxes = boxes.clone()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return image, boxes
