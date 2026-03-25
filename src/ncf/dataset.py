\"\"\"
PyTorch Dataset definition for NCF.
\"\"\"
import torch
from torch.utils.data import Dataset

class NCFDataset(Dataset):
    def __init__(self, user_inputs, item_inputs, labels):
        self.user_inputs = user_inputs
        self.item_inputs = item_inputs
        self.labels = labels

    def __len__(self):
        return len(self.user_inputs)

    def __getitem__(self, idx):
        return self.user_inputs[idx], self.item_inputs[idx], self.labels[idx]
