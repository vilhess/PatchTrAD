import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch 
from torch.utils.data import Dataset, DataLoader

class NASA(Dataset):
    def __init__(self, root="data", dataset="msl", filename="P-1_", mode="train", window_size=10):
        super().__init__()

        datasets_possibles = ["smap", "msl"]
        assert dataset in datasets_possibles, f"dataset sould be in {datasets_possibles}"

        # SMAP features dim=25
        # MSL features dim=55

        self.mode = mode
        self.window_size = window_size

        ano_file = os.path.join(root, "labeled_anomalies.csv")
        values = pd.read_csv(ano_file)
        values = values[values["spacecraft"]==dataset.upper()]

        filenames_possible = values['chan_id'].values.tolist()
        assert filename in filenames_possible, f"filename must be in {filenames_possible}"

        indices = values[values['chan_id']==filename]['anomaly_sequences'].values[0]
        indices = indices.replace("]", "").replace("[", "").split(', ')
        indices = [int(i) for i in indices]
        
        normal = np.load(os.path.join(root, "train", f"{filename}.npy"))
        scaler = StandardScaler()
        x_normal_scaled = scaler.fit_transform(normal)

        normal = pd.DataFrame(x_normal_scaled)


        attack = np.load(os.path.join(root, "test", f"{filename}.npy"))
        x_attack_scaled = scaler.transform(attack)
        attack = pd.DataFrame(x_attack_scaled)

        labels = np.zeros(len(attack))
        for i in range(0, len(indices), 2):
            labels[indices[i]:indices[i+1]] = 1
        
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
    

def get_loaders(root_dir="data/nasa", dataset="msl", filename="M-7", window_size=10, batch_size=32):

    trainset = NASA(mode="train", dataset=dataset, filename=filename, window_size=window_size, root=root_dir)
    testset = NASA(mode="test", dataset=dataset, filename=filename,  window_size=window_size, root=root_dir)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=21)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=21)

    return trainloader, testloader


smapfiles = ['P-1', 'S-1', 'E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 
             'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'A-1', 'D-1', 'P-2', 'P-3', 'D-2', 
             'D-3', 'D-4', 'A-2', 'A-3', 'A-4', 'G-1', 'G-2', 'D-5', 'D-6', 'D-7', 
             'F-1', 'P-4', 'G-3', 'T-1', 'T-2', 'D-8', 'D-9', 'F-2', 'G-4', 'T-3', 'D-11',
             'D-12', 'B-1', 'G-6', 'G-7', 'P-7', 'R-1', 'A-5', 'A-6', 'A-7', 'D-13', 
             'P-2', 'A-8', 'A-9', 'F-3']

mslfiles = ['M-6', 'M-1', 'M-2', 'S-2', 'P-10', 'T-4', 'T-5', 'F-7', 'M-3', 'M-4', 
             'M-5', 'P-15', 'C-1', 'C-2', 'T-12', 'T-13', 'F-4', 'F-5', 'D-14', 
             'T-9', 'P-14', 'T-8', 'P-11', 'D-15', 'D-16', 'M-7', 'F-8']