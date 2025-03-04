import os
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader  

class NABdata(Dataset):
    def __init__(
        self, 
        root_dir="data/nab", 
        dataset="nyc_taxi", 
        window_size=10,
        mode="train",
    ):
        self.root_dir = root_dir
        self.dataset = dataset
        self.window_size = window_size
        self.mode = mode

        with open(os.path.join(self.root_dir, "config.json"), "r") as f:
            self.config = json.load(f)[dataset]

        self.df = pd.read_csv(
            os.path.join(self.root_dir, f"{self.dataset}.csv"), 
            index_col="timestamp", 
            parse_dates=True
        )

        test_date = pd.to_datetime(self.config["test_date"])
        self.test_date = test_date

        test_idx = self.df.index.searchsorted(test_date, side="left")

        train = self.df.iloc[:test_idx].copy()
        test = self.df.iloc[test_idx:].copy()

        scaler = StandardScaler()
        train["value_normed"] = scaler.fit_transform(train[["value"]])
        test["value_normed"] = scaler.transform(test[["value"]])

        self.df = pd.concat([train["value_normed"], test["value_normed"]])

        if self.mode == "train":
            self.data = self.df.iloc[:test_idx]
        elif self.mode == "test":
            self.data = self.df.iloc[test_idx - self.window_size:]

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, index):

        start = index
        end = index + self.window_size

        anomaly = 0
        timestamp = self.data.index[end]
        if timestamp in pd.to_datetime(self.config["anomaly_dates"]):
            anomaly = 1

        features = self.data.iloc[start:end+1]
        return torch.tensor(features.values, dtype=torch.float32).unsqueeze(1), anomaly


    
def get_loaders(window_size=10, root_dir="data/nab", dataset="nyc_taxi", batch_size=128):
    train = NABdata(root_dir=root_dir, dataset=dataset, mode="train", window_size=window_size)
    test = NABdata(root_dir=root_dir, dataset=dataset, mode="test", window_size=window_size)

    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=21)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=21)
    return trainloader, testloader