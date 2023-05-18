import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy
from data import ModelNet40

def knn(x, k):
    # x: 32, 3, 1024
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # 32, 1024, 1024
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # norm의 제곱 # 32, 1, 1024
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # 가까운 점 반환
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.shape[0]
    num_points = x.shape[2]
    x = x.view(batch_size, -1, num_points)  # 32, 3 ,1024
    
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    # batch unpacking
    idx = idx.to(device)
    idx = idx + idx_base
    
    idx = idx.view(-1) # flatting
    
    _, num_dims, _ = x.size()
    
    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    # 32768, 3을 가지고 [655360 (len of idx), 3]의 tensor를 만든다.
    
    feature = feature.view(batch_size, num_points, k, -1)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # 32, 1024, 20, 3을 만든다.
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # extract edge feature
    
    return feature

class DGCNN(nn.Module):

    def __init__(self, k=20, output=40, emb_dim=1024):
        # emb_dim>
        super(DGCNN, self).__init__()
        self.k = k
        self.output = 40
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dim)
        
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        # 예는 왜 conv1d임? 아마 어차피 flatten 시킬 예정이기에 미리 시키는 듯.
        self.linear1 = nn.Linear(emb_dim * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.7)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.7)
        self.linear3 = nn.Linear(256, output)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)  # 2, 64, 1024, 20
        x1 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        

        x = torch.cat((x1, x2, x3, x4), dim=1)  # 각 layer별 Feature를 모두 합친다.
        x = self.conv5(x)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        # Fully connected
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x

if __name__ == "__main__":
    dataset = DataLoader(ModelNet40("train", num_points=1024, num_of_object=8),
                         batch_size=2, shuffle=True)

    model = DGCNN()
    for data, lbl in dataset:
        data = torch.permute(data, (0, 2, 1))
        model(data)
