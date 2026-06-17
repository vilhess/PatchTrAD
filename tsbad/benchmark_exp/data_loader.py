import os
import pandas as pd
import numpy as np
from TSB_AD.utils.slidingWindows import find_length_rank

def load_file(file_path):
    df = pd.read_csv(file_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
    return data, label, slidingWindow

def train_split(data, filename):
    train_index = filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]
    return data_train