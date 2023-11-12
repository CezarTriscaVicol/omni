import torch
import numpy as np
from torch.utils.data import Dataset
class numpy_dataset(Dataset):  
    def __init__(self, data, target, label_mapping):
        self.data = torch.from_numpy(data).unsqueeze(1).to(torch.float32)  
        self.target = torch.from_numpy(np.vectorize(label_mapping.get)(target)).to(torch.long)  # Change to torch.long

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)