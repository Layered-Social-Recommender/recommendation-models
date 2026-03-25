\"\"\"
PyTorch Model definition for NCF.
\"\"\"
import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=8, hidden_layers=[64, 32, 16, 8]):
        super().__init__()
        pass

    def forward(self, user_idx, item_idx):
        pass
