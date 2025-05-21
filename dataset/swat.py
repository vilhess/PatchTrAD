import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch 
from torch.utils.data import Dataset, DataLoader

class Swat(Dataset):
    def __init__(self, root="data/Physical", mode="train", window_size=10):
        super().__init__()

        self.mode = mode
        self.window_size = window_size
        
        x_normal_scaled = np.load(os.path.join(root, 'normal.npy'))[-20000:] # Reduction here

        normal = pd.DataFrame(x_normal_scaled)

        labels = np.load(os.path.join(root, 'attack_label.npy'))
        x_attack_scaled = np.load(os.path.join(root, 'attack.npy'))
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

def get_loaders(root_dir="data", window_size=10, batch_size=32):

    trainset = Swat(mode="train", window_size=window_size, root=root_dir)
    testset = Swat(mode="test", window_size=window_size, root=root_dir)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=21)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=21)

    return trainloader, testloader