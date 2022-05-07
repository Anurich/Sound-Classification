import config 
import pickle
import torch
from torch.utils.data import  Dataset

class dataset(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.data = self._loadpickle()
        labels = self.data["Y"]
        Xdata  = self.data["X"]
        self.Xdata = torch.tensor(Xdata)
        self.labels = torch.tensor(labels)

    def _loadpickle(self):
        return pickle.load(open(self.filename,"rb"))    
    
    def __len__(self):
        return len(self.Xdata)

    def __getitem__(self, index):
        return self.Xdata[index], self.labels[index]
