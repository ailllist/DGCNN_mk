import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    all_data = []
    all_label = []

    for h5_name in glob.glob(os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048", "ply_data_%s*.h5" % partition)):
        with h5py.File(h5_name) as f:
            data = f["data"][:]  # 1648개의 data
            label = f["label"][:]  # 거기에 대응되는 1648개의 label
            all_data.append(data)
            all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label

class ModelNet40(Dataset):
    def __init__(self, partition="train", num_points=1024, num_of_object=-1, batch_size=1):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.data = self.data[:num_of_object, :self.num_points, :]
        self.label = self.label[:num_of_object, :]

    def __getitem__(self, index):
        return self.data[index], self.label[index]
        # return self.data, self.label

    def __len__(self):
        return self.data.shape[0]  # object의 갯수

class ModelNet40suffie(Dataset):
    def __init__(self, partition="train", num_points=1024, num_of_object=-1):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.num_of_object = num_of_object

    def __getitem__(self, item):
        s = np.arange(self.data.shape[0])
        np.random.shuffle(s)
        self.data = self.data[s][:self.num_of_object, :self.num_points, :]
        self.label = self.label[s][:self.num_of_object, :]
        return self.data, self.label

    def __len__(self):
        return self.label[:self.num_of_object].shape[0]  # object의 갯수

if __name__ == "__main__":
    train_data = ModelNet40("train")
    test_data = ModelNet40("test")
    print(len(train_data))
    print(len(test_data))
    print(set(i[0] for i in test_data.label))