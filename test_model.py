from torch.utils import data
from yolo.model import SimpleYOLO
from yolo.voc import VOCDataset, DataLoader

dataset = VOCDataset()
dataloader = DataLoader(dataset, batch_size=5)
model = SimpleYOLO(7, 20)

for i, (images, targets) in enumerate(dataloader):
    images /= 255
    pred = model(images)
    print(i, pred.shape)
    if i >= 100:
        break
