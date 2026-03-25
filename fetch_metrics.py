import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

from ncf.evaluate import evaluate_model
from ncf.dataset import get_dataloaders
from ncf.model import NCF
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(project_root, 'outputs', 'models', 'ncf_model.pth')
processed_dir = os.path.join(project_root, 'data', 'processed')

checkpoint = torch.load(model_path, weights_only=False, map_location=device)
model = NCF(
    num_users=checkpoint['num_users'],
    num_items=checkpoint['num_items'],
    embed_dim=checkpoint['embed_dim'],
    hidden_layers=checkpoint['hidden_layers']
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

_, test_loader, _, _ = get_dataloaders(processed_dir, batch_size=256)

model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for users, items, labels in test_loader:
        users = users.to(device)
        items = items.to(device)
        logits = model(users, items)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

from sklearn.metrics import accuracy_score, precision_score, recall_score
a = accuracy_score(all_labels, all_preds)
p = precision_score(all_labels, all_preds, zero_division=0)
r = recall_score(all_labels, all_preds, zero_division=0)
print(f"MY_METRICS:|{a:.4f}|{p:.4f}|{r:.4f}|")
