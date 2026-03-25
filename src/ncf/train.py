"""
Phase 8 — Training Pipeline for NCF

Trains the NCF model on MovieLens implicit interaction data.

Config:
  Loss      : BCEWithLogitsLoss
  Optimizer : Adam (lr=0.001)
  Batch size: 256
  Epochs    : 5
  Save to   : outputs/models/ncf_model.pth
"""
import torch
import torch.nn as nn
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ncf.model import NCF
from ncf.dataset import get_dataloaders


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for users, items, labels in train_loader:
        users = users.to(device)
        items = items.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(users, items)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, test_loader, criterion, device):
    """Evaluate the model on the test set and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for users, items, labels in test_loader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)

            logits = model(users, items)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def train(epochs=5, batch_size=256, lr=0.001, embed_dim=32, hidden_layers=[64, 32, 16]):
    """Full training pipeline."""
    print("=" * 60)
    print("  Phase 8 — NCF Training Pipeline")
    print("=" * 60)

    # ── Device ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── Paths ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    model_save_dir = os.path.join(project_root, 'outputs', 'models')
    os.makedirs(model_save_dir, exist_ok=True)

    # ── Data ──
    print("\n[1/4] Loading data...")
    train_loader, test_loader, num_users, num_items = get_dataloaders(
        processed_dir, batch_size=batch_size
    )

    # ── Model ──
    print("\n[2/4] Initializing model...")
    model = NCF(num_users, num_items, embed_dim=embed_dim, hidden_layers=hidden_layers)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── Loss & Optimizer ──
    print("\n[3/4] Setting up loss and optimizer...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"  Loss     : BCEWithLogitsLoss")
    print(f"  Optimizer: Adam (lr={lr})")

    # ── Training Loop ──
    print(f"\n[4/4] Training for {epochs} epochs...")
    print("-" * 60)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Time':>8}")
    print("-" * 60)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate(model, test_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {elapsed:>7.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(model_save_dir, 'ncf_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'num_users': num_users,
                'num_items': num_items,
                'embed_dim': embed_dim,
                'hidden_layers': hidden_layers,
            }, save_path)

    print("-" * 60)
    print(f"\nTraining complete!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Model saved to: {save_path}")
    print("=" * 60)

    return model


if __name__ == '__main__':
    train(epochs=5, batch_size=256, lr=0.001)
