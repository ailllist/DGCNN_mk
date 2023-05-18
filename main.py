import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ConstantLR

from data import ModelNet40
from torch.utils.data import DataLoader
import numpy as np



if __name__ == "__main__":
    