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

if __name__ == "__main__":
    # test
    X = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5]
    ])

    Y = xyxy2cxcywh(X)
    print(Y)
