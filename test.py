import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 2, 1]])
a = a.T
a = a.repeat(2, 1, 1)  # 2, 3, 3

inner = -2 * torch.matmul(a.transpose(2, 1), a)
aa = torch.sum(a ** 2, dim=1, keepdim=True)
pd = -aa - inner - aa.transpose(2, 1)

idx = pd.topk(k=2, dim=-1)[1]
print(idx)
idx_base = torch.arange(0, 2).view(-1, 1, 1) * 3
idx = idx + idx_base
idx = idx.view(-1)
print(idx)
print(idx.shape)
print(a.view(2 * 3, -1).shape)

feature = a.view(2 * 3, -1)[idx, :]
print(feature)
print(feature.shape)

feature = feature.view(2, 3, 2, -1)
print(feature)
print(feature.shape)

a = [[[[1, 4, 7],
       [2, 5, 2]],  # 1의 neighbor
      [[2, 5, 2],
       [1, 4, 7]],  # 2의 neighbor
      [[3, 6, 1],
       [1, 4, 7]]],  # 3의 neighbor
     [[[1, 4, 7],
       [2, 5, 2]],
      [[2, 5, 2],
       [1, 4, 7]],
      [[3, 6, 1],
       [1, 4, 7]]]]
