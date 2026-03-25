"""
Phase 4 — Negative Sampling
Generates negative samples for NCF training.

For each user, randomly samples movies they have NOT interacted with
and assigns label = 0. Combined with positive interactions (label = 1)
to create the final NCF training dataset.

Ratio: 1 positive : 3 negatives (configurable)
"""
import pandas as pd
import numpy as np
import os


def generate_negative_samples(positive_df, num_items, num_negatives=3, seed=42):
    """
    For each positive interaction, sample `num_negatives` items that the
    user has NOT interacted with and assign label = 0.

    Args:
        positive_df: DataFrame with columns [user_idx, item_idx, label]
        num_items: Total number of unique items
        num_negatives: Number of negatives per positive (default: 3)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with all positive + negative samples
    """
    np.random.seed(seed)

    print(f"[Phase 4] Generating negative samples (ratio 1:{num_negatives})...")

    # Build a set of interacted items per user for fast lookup
    user_positive_items = positive_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

    all_items = set(range(num_items))

    negative_rows = []
    total_users = len(user_positive_items)

    for idx, (user, pos_items) in enumerate(user_positive_items.items()):
        # Items this user has NOT interacted with
        negative_pool = list(all_items - pos_items)

        # Number of negatives = num_negatives * number of positive interactions
        num_neg_samples = len(pos_items) * num_negatives

        # If not enough negatives available, sample with replacement
        if num_neg_samples > len(negative_pool):
            sampled = np.random.choice(negative_pool, size=num_neg_samples, replace=True)
        else:
            sampled = np.random.choice(negative_pool, size=num_neg_samples, replace=False)

        for item in sampled:
            negative_rows.append({
                'user_idx': user,
                'item_idx': item,
                'label': 0
            })

        # Progress update every 100 users
        if (idx + 1) % 100 == 0:
            print(f"         Processed {idx + 1}/{total_users} users...")

    negative_df = pd.DataFrame(negative_rows)

    # Combine positive and negative
    combined_df = pd.concat([positive_df[['user_idx', 'item_idx', 'label']], negative_df], ignore_index=True)

    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"         Positive samples : {len(positive_df)}")
    print(f"         Negative samples : {len(negative_df)}")
    print(f"         Total samples    : {len(combined_df)}")

    return combined_df


def save_training_data(df, processed_dir):
    """Save the final NCF training data CSV."""
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, 'ncf_training_data.csv')
    df.to_csv(output_path, index=False)
    print(f"[Phase 4] Saved training data to {output_path}")
    return output_path


def run_negative_sampling(processed_dir, num_negatives=3):
    """Run the full negative sampling pipeline."""
    print("=" * 60)
    print("  Phase 4 — Negative Sampling Pipeline")
    print("=" * 60)

    # Load positive interactions from Phase 3
    pos_path = os.path.join(processed_dir, 'positive_interactions.csv')
    print(f"Loading positive interactions from {pos_path}...")
    positive_df = pd.read_csv(pos_path)

    num_items = positive_df['item_idx'].max() + 1
    print(f"Number of items: {num_items}")

    # Generate negatives
    training_df = generate_negative_samples(positive_df, num_items, num_negatives)

    # Save
    output_path = save_training_data(training_df, processed_dir)

    print("=" * 60)
    print("  Negative sampling complete!")
    print(f"  Output : {output_path}")
    print(f"  Ratio  : 1 positive : {num_negatives} negatives")
    print(f"  Total  : {len(training_df)} samples")
    print("=" * 60)

    return training_df


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    processed_dir = os.path.join(project_root, 'data', 'processed')

    run_negative_sampling(processed_dir, num_negatives=3)
