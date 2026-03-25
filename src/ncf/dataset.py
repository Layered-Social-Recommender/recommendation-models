"""
Phase 6 — PyTorch Dataset & DataLoader for NCF

Custom Dataset class that loads ncf_training_data.csv and returns
(user_idx, item_idx, label) tensors. Includes a helper to create
train/test DataLoaders with batching.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


class NCFDataset(Dataset):
    """
    PyTorch Dataset for NCF.
    Each sample returns: (user_idx, item_idx, label)
    """
    def __init__(self, users, items, labels):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


def load_and_split(processed_dir, test_size=0.2, seed=42):
    """
    Load ncf_training_data.csv and split into train/test sets.

    Returns:
        train_dataset, test_dataset, num_users, num_items
    """
    data_path = os.path.join(processed_dir, 'ncf_training_data.csv')
    print(f"Loading training data from {data_path}...")
    df = pd.read_csv(data_path)

    num_users = df['user_idx'].max() + 1
    num_items = df['item_idx'].max() + 1

    # Split into train and test
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df['label']
    )

    print(f"Train size: {len(train_df)}  |  Test size: {len(test_df)}")
    print(f"Users: {num_users}  |  Items: {num_items}")

    train_dataset = NCFDataset(
        train_df['user_idx'].values,
        train_df['item_idx'].values,
        train_df['label'].values
    )

    test_dataset = NCFDataset(
        test_df['user_idx'].values,
        test_df['item_idx'].values,
        test_df['label'].values
    )

    return train_dataset, test_dataset, num_users, num_items


def get_dataloaders(processed_dir, batch_size=256, test_size=0.2, seed=42):
    """
    Creates train and test DataLoaders from the processed data.

    Returns:
        train_loader, test_loader, num_users, num_items
    """
    train_dataset, test_dataset, num_users, num_items = load_and_split(
        processed_dir, test_size, seed
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}")
    print(f"Batch size: {batch_size}")

    return train_loader, test_loader, num_users, num_items


if __name__ == '__main__':
    # Quick test to verify DataLoader works
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    processed_dir = os.path.join(project_root, 'data', 'processed')

    print("=" * 60)
    print("  Phase 6 — DataLoader Test")
    print("=" * 60)

    train_loader, test_loader, num_users, num_items = get_dataloaders(
        processed_dir, batch_size=256
    )

    # Grab one batch and verify
    users, items, labels = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Users  shape: {users.shape}  dtype: {users.dtype}")
    print(f"  Items  shape: {items.shape}  dtype: {items.dtype}")
    print(f"  Labels shape: {labels.shape}  dtype: {labels.dtype}")
    print(f"\n  First 5 users : {users[:5].tolist()}")
    print(f"  First 5 items : {items[:5].tolist()}")
    print(f"  First 5 labels: {labels[:5].tolist()}")

    print("=" * 60)
    print("  DataLoader is working correctly!")
    print("=" * 60)
