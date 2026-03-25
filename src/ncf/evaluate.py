"""
Phase 9 — NCF Evaluation

Evaluates the trained NCF model on the test dataset.
Computes standard binary classification metrics:
  - Accuracy
  - Precision
  - Recall

Note: Recommendation-specific metrics like HR@K and NDCG@K
can be added here in future phases.
"""
import torch
import sys
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ncf.model import NCF
from ncf.dataset import get_dataloaders

def evaluate_model(model, test_loader, device):
    """
    Evaluates the model on the test dataloader.
    Returns basic metrics (accuracy, precision, recall).
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    
    print("\nEvaluating model on test set...")
    with torch.no_grad():
        for users, items, labels in test_loader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)
            
            # Forward pass -> raw logits
            logits = model(users, items)
            
            # Convert logits to probabilities, then to binary predictions
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    print("-" * 40)
    print("  Basic Evaluation Metrics")
    print("-" * 40)
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print("-" * 40)
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Negative (0)', 'Positive (1)']))
    
    return accuracy, precision, recall

def run_evaluation(model_path, processed_dir):
    """Loads the best model and runs evaluation."""
    print("=" * 60)
    print("  Phase 9 — NCF Evaluation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    
    # Initialize the model with the same architecture
    model = NCF(
        num_users=checkpoint['num_users'],
        num_items=checkpoint['num_items'],
        embed_dim=checkpoint['embed_dim'],
        hidden_layers=checkpoint['hidden_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded (trained for {checkpoint['epoch']} epochs, best val_loss: {checkpoint['val_loss']:.4f})")
    
    # Load dataloaders
    print("Loading test data...")
    _, test_loader, _, _ = get_dataloaders(processed_dir, batch_size=256)
    
    # Run evaluation
    evaluate_model(model, test_loader, device)
    
    print("=" * 60)
    print("  Evaluation complete!")
    print("=" * 60)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    model_path = os.path.join(project_root, 'outputs', 'models', 'ncf_model.pth')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    
    run_evaluation(model_path, processed_dir)
