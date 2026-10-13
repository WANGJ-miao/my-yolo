import torch

# mask = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).bool()
mask = torch.tensor([1, 0, 1]).bool()
X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(X[mask])
