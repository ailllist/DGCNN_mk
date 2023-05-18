import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import ToTensor
from data import ModelNet40, ModelNet40suffie
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime

class TNet(nn.Module):
    def __init__(self, k=64):
        super().__init__()

        self.k = k
        self.conv1 = torch.nn.Conv1d(self.k, 64, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = torch.nn.Conv1d(128, 1024, kernel_size=1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.k * self.k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        # iden??
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)


        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
             batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNet(nn.Module):

    def __init__(self, k=40, num_of_points=1024, is_training=True):
        super().__init__()

        self.stn3 = TNet(3)
        self.stn64 = TNet(64)
        self.is_training = is_training
        self.num_of_points = num_of_points
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)


        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        # x: num_of_objects * 3 * num_point

        xt3 = self.stn3(x)
        x = x.permute(0, 2, 1)
        x = torch.matmul(x, xt3)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        xt64 = self.stn64(x)
        x = x.permute(0, 2, 1)
        x = torch.matmul(x, xt64)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.dropout(x, 0.7, training=self.is_training)
        x = F.relu(self.bn5(self.fc2(x)))
        x = F.dropout(x, 0.7, training=self.is_training)
        x = self.fc3(x)

        return x

def train(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    device = next(model.parameters()).device

    for i, data in enumerate(dataloader):

        points, target = data

        batch, label = points.to(device), target.to(device)
        batch = batch.permute(0, 2, 1)

        pred = model(batch)
        label = label.squeeze(1)
        loss = F.nll_loss(F.log_softmax(pred), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            loss, current = loss.item(), i * len(points)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for data in dataloader:
            points, target = data

            batch, label = points.to(device), target.to(device)
            batch = batch.permute(0, 2, 1)
            pred = model(batch)

            label = label.squeeze(1)
            test_loss += F.nll_loss(F.log_softmax(pred), label)

            predicted = pred.argmax(1)

            correct += (predicted == label).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100 * correct, test_loss

num_points = 1024
batch_size = 128
test_batch_size = 128
epoch = 100
train_num_of_object = -1
test_num_of_object = -1

if __name__ == "__main__":

    torch.cuda.empty_cache()
    train_loader = DataLoader(
        ModelNet40(partition='train', num_points=num_points, num_of_object=train_num_of_object),
        batch_size=batch_size, shuffle=True, drop_last=True)
    
    test_loader = DataLoader(
        ModelNet40(partition='test', num_points=num_points, num_of_object=test_num_of_object),
        batch_size=test_batch_size, shuffle=False, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = PointNet(40, num_points, is_training=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))  # TODO Need to understand
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # TODO Need to understand
    model = model.to(device)

    acc_list = []
    loss_list = []

    n_time = datetime.now()

    with open(f"record_{n_time.month}_{n_time.day}_{n_time.hour}_{n_time.minute}.csv", "w") as f:
        for ep in range(epoch):
            print("epoch : ", ep)
            scheduler.step()
            train(train_loader, model, optimizer)
            acc, loss = test(test_loader, model, ep)
            f.write(f"{acc}, {loss}\n")

    torch.save(model.state_dict(), f"model_{n_time.month}_{n_time.day}_{n_time.hour}_{n_time.minute}.pth")