from torch.utils.data import Dataset
import torch 
import numpy as np 

class Data(Dataset):
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train=torch.from_numpy(X_train.to_numpy())
        self.Y_train=torch.from_numpy(Y_train)
        self.X_test = torch.from_numpy(X_test.to_numpy())
        self.Y_test = torch.from_numpy(np.asarray(Y_test))
        self.len=self.X_train.shape[0]
    def __getitem__(self,index):      
        return self.X_train[index], self.Y_train[index]
    def __len__(self):
        return self.len 