import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy
from data import ModelNet40

class DGCNN(nn.Module):

    def __init__(self):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 7, kernel_size=1, bias=False)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        return x

if __name__ == "__main__":
    dataset = DataLoader(ModelNet40("train", num_points=1024, num_of_object=3),
                         batch_size=1, shuffle=True)

    model = DGCNN()
    for data, lbl in dataset:
        model(data)
