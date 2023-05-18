import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import ModelNet40
from model import DGCNN
from utils import cal_loss
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import sklearn.metrics as metrics
from tqdm import tqdm

def train(dataloader, model, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    for data, label in tqdm(dataloader):
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.shape[0]
        optimizer.zero_grad()
        pred = model(data)
        loss = F.cross_entropy(pred, label, reduction='mean')
        loss.backward()
        opt.step()
        
    print(f"loss: {loss:>7f}")

def test(dataloader, model: "DGCNN"):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0.0, 0.0

    for data, label in tqdm(dataloader):
        data, label = data.to(data), label.to(label).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.shape[0]
        pred = model(data)
        loss = F.cross_entropy(pred, label, reduction='mean')
        loss.backward()

        predicted = pred.argmax(1)
        correct += (predicted == label).type(torch.float).sum().item()

    test_loss /= batch_size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return 100 * correct, test_loss
    

num_points = 512
batch_size = 2
test_batch_size = 2
epoch = 100
train_num_of_object = -1
test_num_of_object = -1
k = 20

if __name__ == "__main__":
    train_loader = DataLoader(ModelNet40(partition="train", num_points=num_points, num_of_object=train_num_of_object),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition="test", num_points=num_points, num_of_object=test_num_of_object),
                             batch_size=test_batch_size, shuffle=True, drop_last=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = DGCNN(k)
    model = model.to(device)
    
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, epoch, eta_min=0.001)
    
    n_time = datetime.now()
    
    with open(f"record_{n_time.month}_{n_time.day}_{n_time.hour}_{n_time.minute}.csv", "w") as f:
        best_test_acc = 0
        for ep in range(epoch):
            print("epoch : " , ep)
            scheduler.step()
            train(train_loader, model, opt)
            acc, loss = test(test_loader, model)
            
            f.write(f"{acc}, {loss}\n")
            if acc >= best_test_acc:
                best_test_acc = acc
                torch.save(model.state_dict(), f"model_{n_time.month}_{n_time.day}_{n_time.hour}_{n_time.minute}.pth")
            
            