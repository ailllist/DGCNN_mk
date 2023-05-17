import torch

device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
print(device)