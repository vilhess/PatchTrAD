import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch 
from torch.utils.data import Dataset, DataLoader

machines = [m.split(".")[0]+"_" for m in os.listdir("data/smd/train")]

class SMD(Dataset):
    def __init__(self, root="data/smd/processed", machine="machine-1-1_", mode="train", window_size=10):
        super().__init__()

        self.mode = mode
        self.window_size = window_size

        scaler = StandardScaler()
        
        x_normal = np.load(os.path.join(root, machine+"train.npy"))
        x_normal_scaled = scaler.fit_transform(x_normal)

        normal = pd.DataFrame(x_normal_scaled)


        labels = np.load(os.path.join(root, machine+"test_label.npy"))
        x_attack = np.load(os.path.join(root, machine+"test.npy"))
        x_attack_scaled = scaler.transform(x_attack)
        attack = pd.DataFrame(x_attack_scaled)
        
        assert self.mode in ['train', 'test'], "mode must be 'train' or 'test'"

        if self.mode=="train": 
            self.data = normal
            self.labels = np.zeros(len(normal))

        elif self.mode=="test": 
            self.data = attack
            self.labels = labels

        self.data = self.data.values

    def __len__(self):
        return len(self.data) - self.window_size
    
    def __getitem__(self, index):
        start = index
        end = index+self.window_size

        anomaly = self.labels[end]

        features = self.data[start:end+1]
        return torch.tensor(features, dtype=torch.float32),  anomaly

    

def get_loaders(root_dir="data/smd/processed", machine="machine-1-1_", window_size=10, batch_size=32):

    trainset = SMD(mode="train", machine=machine, window_size=window_size, root=root_dir)
    testset = SMD(mode="test", machine=machine, window_size=window_size, root=root_dir)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=21)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=21)

    return trainloader, testloader


