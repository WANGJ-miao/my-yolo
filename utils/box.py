""" 
useful functions for box operations
"""

import torch

def xyxy2cxcywh(boxes):
    """ 
    convert xyxy to cx, cy, w, h
    boxes: torch.tensor
    shape of boxes [
        [x1, y1, x2, y2],
        [x1, y1, x2, y2],
        ...
    ]
    """
    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]
    width = xmax - xmin
    height = ymax - ymin
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    return torch.stack([cx, cy, width, height], dim=1)

def normalize_boxes(boxes, size):
    """
    support [cx, cy, w, h] or [xmin, ymin, xmax, ymax]
    size: [width, height]
    """
    w, h = size
    normalized_boxes = boxes.clone().float()
    normalized_boxes[:, [0, 2]] /= w
    normalized_boxes[:, [1, 3]] /= h
    return normalized_boxes

def cxcywh2txtywh(boxes, S):
    """
    coordinate should be normalized
    tx, ty: normalized coordinate inside a grid
    """
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0] * S - torch.floor(boxes[:, 0] * S)
    boxes[:, 1] = boxes[:, 1] * S - torch.floor(boxes[:, 1] * S)
    return boxes

if __name__ == "__main__":
    # # test normalize
    # print("test normalize_boxes:")
    # X = torch.tensor([
    #     [1, 2, 3, 4],
    #     [2, 3, 4, 5]
    # ])

    # size = (1, 2)
    # Y = normalize_boxes(X, size)
    # print(Y)
    # test txty
    print("test cxcywh2txtywh:")
    X = torch.tensor([
        [0.1, 0.2, 3, 4],
        [0.2, 0.3, 4, 5]
    ])
    S = 7
    Y = cxcywh2txtywh(X, S)
    print(Y)
