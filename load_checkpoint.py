import torch

from yolo.model import SimpleYOLO
from yolo.voc import VOCDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("checkpoints/simple_yolo_epoch_001.pt",
                        map_location=device,
                        weights_only=True)
model = SimpleYOLO(checkpoint["S"], checkpoint["num_classes"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_dataset = VOCDataset(split="val")
train_dataset = VOCDataset(split="train")
dataloader = DataLoader(test_dataset, 1)

S = checkpoint["S"]

for images, targets in dataloader:
    with torch.no_grad():
        pred = model(images)
        pred_objectness = pred[..., 4].sigmoid()
        print(pred_objectness)
        target_objectness = targets[..., 4]
        print(target_objectness)
    break
        # for pred_target in pred:
        #     for x_index in range(S):
        #         for y_index in range(S):
        #             pred_item = pred_target[y_index][x_index]
        #             objectness = pred_item[4]
        #             print(objectness)
