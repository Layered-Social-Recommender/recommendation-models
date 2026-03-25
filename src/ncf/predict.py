"""
Phase 10 — Generate Top-K Recommendations

For selected users, scores all candidate movies (those not already interacted with),
sorts by predicted score, and outputs top-K recommendations.

Output: outputs/recommendations/top_k_recommendations.csv
"""
import torch
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ncf.model import NCF


def load_trained_model(model_path, device):
    """Load the trained NCF model from checkpoint."""
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model = NCF(
        num_users=checkpoint['num_users'],
        num_items=checkpoint['num_items'],
        embed_dim=checkpoint['embed_dim'],
        hidden_layers=checkpoint['hidden_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint['num_users'], checkpoint['num_items']


def get_user_history(processed_dir):
    """Load positive interactions to know which movies each user already interacted with."""
    pos_path = os.path.join(processed_dir, 'positive_interactions.csv')
    df = pd.read_csv(pos_path)
    user_history = df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    return user_history


def recommend_for_user(model, user_idx, num_items, user_history, top_k=10, device='cpu'):
    """
    Generate top-K recommendations for a single user.

    Scores all items the user has NOT interacted with,
    then returns the top-K highest scored items.
    """
    # Items already interacted with
    seen_items = user_history.get(user_idx, set())

    # Candidate items = all items minus already seen
    candidate_items = [i for i in range(num_items) if i not in seen_items]

    # Create tensors
    user_tensor = torch.LongTensor([user_idx] * len(candidate_items)).to(device)
    item_tensor = torch.LongTensor(candidate_items).to(device)

    # Score all candidates
    with torch.no_grad():
        logits = model(user_tensor, item_tensor)
        scores = torch.sigmoid(logits).cpu().numpy()

    # Build results and sort by score descending
    results = pd.DataFrame({
        'user_idx': user_idx,
        'movie_idx': candidate_items,
        'predicted_score': scores
    })
    results = results.sort_values('predicted_score', ascending=False).head(top_k)
    results = results.reset_index(drop=True)

    return results


def generate_recommendations(user_list, top_k=10):
    """Generate and save top-K recommendations for a list of users."""
    print("=" * 60)
    print("  Phase 10 — Generate Top-K Recommendations")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    model_path = os.path.join(project_root, 'outputs', 'models', 'ncf_model.pth')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    rec_dir = os.path.join(project_root, 'outputs', 'recommendations')
    os.makedirs(rec_dir, exist_ok=True)

    # Load model
    print("Loading trained model...")
    model, num_users, num_items = load_trained_model(model_path, device)
    print(f"Model loaded. Users: {num_users}, Items: {num_items}")

    # Load user history
    print("Loading user interaction history...")
    user_history = get_user_history(processed_dir)

    # Load movie titles for readable output
    movies_path = os.path.join(project_root, 'data', 'raw', 'movies.csv')
    movies_df = pd.read_csv(movies_path)
    # Build item_idx -> movie title mapping
    pos_df = pd.read_csv(os.path.join(processed_dir, 'positive_interactions.csv'))
    raw_ratings = pd.read_csv(os.path.join(project_root, 'data', 'raw', 'ratings.csv'))
    # Map: movieId -> item_idx
    movie_ids = sorted(raw_ratings['movieId'].unique())
    movieid2idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    idx2movieid = {idx: mid for mid, idx in movieid2idx.items()}
    # Map: movieId -> title
    movieid2title = dict(zip(movies_df['movieId'], movies_df['title']))

    # Generate recommendations
    all_recs = []
    for user_idx in user_list:
        if user_idx >= num_users:
            print(f"\n  [!] User {user_idx} out of range (max: {num_users - 1}), skipping.")
            continue

        recs = recommend_for_user(model, user_idx, num_items, user_history, top_k, device)

        # Add movie titles
        recs['movieId'] = recs['movie_idx'].map(idx2movieid)
        recs['title'] = recs['movieId'].map(movieid2title)

        num_seen = len(user_history.get(user_idx, set()))
        print(f"\n  User {user_idx} (watched {num_seen} movies) — Top {top_k} Recommendations:")
        print("  " + "-" * 56)
        for rank, (_, row) in enumerate(recs.iterrows(), 1):
            print(f"  {rank:>3}. {row['title']:<40s}  (score: {row['predicted_score']:.4f})")

        all_recs.append(recs)

    # Combine and save
    combined = pd.concat(all_recs, ignore_index=True)
    output_path = os.path.join(rec_dir, 'top_k_recommendations.csv')
    combined[['user_idx', 'movie_idx', 'predicted_score', 'title']].to_csv(output_path, index=False)
    print(f"\n  Saved {len(combined)} recommendations to {output_path}")

    print("=" * 60)
    print("  Recommendations generated successfully!")
    print("=" * 60)

    return combined


if __name__ == '__main__':
    # Generate top-10 recommendations for 5 sample users
    sample_users = [0, 50, 100, 200, 400]
    generate_recommendations(user_list=sample_users, top_k=10)
