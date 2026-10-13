from pathlib import Path
import torch

from yolo.voc import VOC_CLASSES, VOCDataset, DataLoader
from yolo.model import SimpleYOLO, SimpleYoloLoss

dataset = VOCDataset()
dataloader = DataLoader(dataset, batch_size = 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleYOLO().to(device)
criterion = SimpleYoloLoss()

lr = 0.01
momentum_factor = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr, momentum_factor)

save_dir = Path("checkpoints")
save_dir.mkdir(exist_ok=True)

epochs = 1

for epoch in range(epochs):
    model.train()
    loss_sum = 0
    sample_count = 0

    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size

        if step % 20 == 0:
            print(f"step = {step}, ", f"loss = {loss.item(): .4f}")

    save_path = save_dir / f"simple_yolo_epoch_{epoch + 1:03d}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch + 1,
        "S": model.S,
        "num_classes": model.num_classes,
        "class_names": VOC_CLASSES,
        "image_size": (448, 448),
    }, save_path)
    print(f"saved: {save_path}")


checkpoint = torch.load("checkpoints/overfiting.pt",
                        map_location=device,
                        weights_only=True)
model = SimpleYOLO(checkpoint["S"], checkpoint["num_classes"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# model.train()
# (images, targets) = next(iter(dataloader))
# images = images.to(device)
# targets = targets.to(device)
# print(targets[..., 4])
# for i in range(100):
#     optimizer.zero_grad()
#     preds = model(images)
#     loss = criterion(preds, targets)
#     loss.backward()
#     optimizer.step()
#     print(i, loss)

# save_path = save_dir / f"overfiting.pt"
# torch.save({
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     "S": model.S,
#     "num_classes": model.num_classes,
#     "class_names": VOC_CLASSES,
#     "image_size": (448, 448),
# }, save_path)

