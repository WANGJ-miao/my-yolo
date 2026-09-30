import torch
from yolo.model import SimpleYOLO, SimpleYoloLoss
from yolo.voc import VOCDataset, DataLoader

dataset = VOCDataset()
dataloader = DataLoader(dataset, batch_size=5)
model = SimpleYOLO(7, 20)
loss = SimpleYoloLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
model.train()

for i, (images, targets) in enumerate(dataloader):
    optimizer.zero_grad()
    preds = model(images)
    loss_value = loss(preds, targets)
    loss_value.backward()
    optimizer.step()
    print(i, loss_value)
    if i >= 100:
        break
