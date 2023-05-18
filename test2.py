import torch

tsr = torch.tensor([[3, 2, 1], [4, 6, 2], [2, 5, 8], [1, 1, 1], [3, 5, 2], [2, 1, 5]])

idx = torch.tensor([1, 2, 2, 3, 4, 2])

print(tsr[idx])