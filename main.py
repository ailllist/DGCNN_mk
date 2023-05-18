import torch
import torch.nn as nn
import torch.optim as optim
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
    train_loss = 0.0
    count = 0.0
    train_pred = []
    train_true = []
    model.train()
    for data, label in dataloader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.shape[0]
        optimizer.zero_grad()
        pred = model(data)
        loss = cal_loss(pred, label)
        loss.backward()
        opt.step()
        preds = pred.max(dim=1)[1]
        count += batch_size
        train_loss += loss.item() * batch_size
        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    loss = train_loss * 1.0 / count
    acc = metrics.accuracy_score(train_true, train_pred)
    avg_acc = metrics.balanced_accuracy_score(train_true, train_pred)
    return loss, acc, avg_acc

def test(dataloader, model: "DGCNN"):
    model.eval()
    test_loss, count = 0.0, 0.0
    test_pred = []
    test_true = []
    
    for data, label in dataloader:
        data, label = data.to(data), label.to(label).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.shape[0]
        pred = model(data)
        loss = cal_loss(pred, label)
        preds = pred.max(dim=1)[1]
        count += batch_size
        test_loss += loss.item() * batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append((preds.detach().cpu().numpy()))
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    loss = test_loss * 1.0 / count
    
    return loss, test_acc, avg_per_class_acc
    

num_points = 1024
batch_size = 128
test_batch_size = 128
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
        for ep in tqdm(range(epoch)):
            print("epoch : " , ep)
            scheduler.step()
            train(train_loader, model, opt)
            loss, acc, avg_acc = test(test_loader, model)
            print(f"loss: {loss}, acc: {acc}, avg_acc: {avg_acc}")
            
            if acc >= best_test_acc:
                best_test_acc = acc
                torch.save(model.state_dict(), f"model_{n_time.month}_{n_time.day}_{n_time.hour}_{n_time.minute}.pth")
            
            