# This is a handwrite yolo model by myself
Just to go through the whole process of training a computer vision model

# Model: mini yolo3

# project structure
```plain_text
.
├── data
│   └── VOCdevkit                   # data is here
├── README.md
├── utils                           # some useful functions
└── yolo                            # the model
    ├── transform.py
    ├── visualization.py
    └── voc.py
```

# DataSet: PASCAL VOC 2007
20 kinds of items
9963 pieces of pictures
24640 marked items
ADDRESS: [https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
