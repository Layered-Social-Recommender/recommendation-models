"""
Phase 3 — Data Preprocessing
Converts raw MovieLens ratings into NCF-ready interaction data.

Steps:
  1. Load ratings.csv
  2. Keep userId, movieId, rating
  3. Convert to implicit interaction format (label = 1 for all ratings)
  4. Encode raw IDs into contiguous integer indices starting from 0
  5. Save processed positive interactions to data/processed/positive_interactions.csv
"""
import pandas as pd
import os


def load_ratings(raw_data_dir):
    """Step 1: Load the raw ratings file."""
    ratings_path = os.path.join(raw_data_dir, 'ratings.csv')
    print(f"[Step 1] Loading ratings from {ratings_path}...")
    df = pd.read_csv(ratings_path)
    print(f"         Loaded {len(df)} raw ratings.")
    return df


def keep_needed_columns(df):
    """Step 2: Keep only userId, movieId, rating."""
    print("[Step 2] Keeping columns: userId, movieId, rating")
    df = df[['userId', 'movieId', 'rating']].copy()
    return df


def convert_to_implicit(df):
    """Step 3: Convert explicit ratings to implicit interactions (label=1)."""
    print("[Step 3] Converting to implicit interaction format (label=1 for all)...")
    df['label'] = 1
    df = df[['userId', 'movieId', 'label']].copy()
    print(f"         Total positive interactions: {len(df)}")
    return df


def encode_ids(df):
    """
    Step 4: Encode raw userId and movieId into contiguous integer indices
    starting from 0. Stores the mappings for later use.
    """
    print("[Step 4] Encoding user and item IDs to contiguous indices...")

    # Create mappings: raw ID -> index (0-based)
    user_ids = df['userId'].unique()
    item_ids = df['movieId'].unique()

    user2idx = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
    item2idx = {iid: idx for idx, iid in enumerate(sorted(item_ids))}

    df['user_idx'] = df['userId'].map(user2idx)
    df['item_idx'] = df['movieId'].map(item2idx)

    num_users = len(user2idx)
    num_items = len(item2idx)

    print(f"         Unique users : {num_users}")
    print(f"         Unique items : {num_items}")

    return df, user2idx, item2idx, num_users, num_items


def save_processed(df, processed_dir):
    """Step 5: Save the processed positive interactions CSV."""
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, 'positive_interactions.csv')
    df_out = df[['user_idx', 'item_idx', 'label']].copy()
    df_out.to_csv(output_path, index=False)
    print(f"[Step 5] Saved {len(df_out)} positive interactions to {output_path}")
    return output_path


def run_preprocessing(raw_data_dir, processed_dir):
    """Run the full preprocessing pipeline."""
    print("=" * 60)
    print("  Phase 3 — Data Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Load
    df = load_ratings(raw_data_dir)

    # Step 2: Keep needed columns
    df = keep_needed_columns(df)

    # Step 3: Convert to implicit
    df = convert_to_implicit(df)

    # Step 4: Encode IDs
    df, user2idx, item2idx, num_users, num_items = encode_ids(df)

    # Step 5: Save
    output_path = save_processed(df, processed_dir)

    print("=" * 60)
    print("  Preprocessing complete!")
    print(f"  Output: {output_path}")
    print(f"  Users: {num_users}  |  Items: {num_items}  |  Interactions: {len(df)}")
    print("=" * 60)

    return df, user2idx, item2idx, num_users, num_items


if __name__ == '__main__':
    # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    raw_dir = os.path.join(project_root, 'data', 'raw')
    processed_dir = os.path.join(project_root, 'data', 'processed')

    run_preprocessing(raw_dir, processed_dir)
