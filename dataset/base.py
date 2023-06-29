import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os


class RANSDataset:
    def __init__(
        self,
        data_dir,
        train_ratio=0.6,
        valid_ratio=0.2,
        test_ratio=0.2,
        batch_size=128,
        load_subset=False,
        ):
        self.batch_size = batch_size
        X_df = pd.read_csv(os.path.join(data_dir, 'data_train_input.csv'))
        y_df = pd.read_csv(os.path.join(data_dir, 'data_train_output.csv'))
        X_df.fillna(0, inplace=True)
        if load_subset:
            X_df = X_df[:10000]
            y_df = y_df[:10000]
        X = X_df.values
        y = y_df.values
        train_X, valid_test_X, train_y, valid_test_y = train_test_split(X, y, train_size=train_ratio)
        valid_X, test_X, valid_y, test_y = train_test_split(valid_test_X, valid_test_y, train_size=valid_ratio/(valid_ratio+test_ratio))
        
        self.train_data = RANSBase(train_X, train_y)
        self.valid_data = RANSBase(valid_X, valid_y)
        self.test_data = RANSBase(test_X, test_y)
    
    @property
    def train_loader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    
    @property
    def valid_loader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size)
    
    @property
    def test_loader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


class RANSBase(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
