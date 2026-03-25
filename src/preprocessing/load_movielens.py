"""
Loads raw MovieLens dataset files.
"""
import pandas as pd
import os

def load_data(raw_data_dir):
    """
    Loads ratings from raw data directory and converts to implicit feedback.
    Treats any given rating as interaction = 1.
    """
    ratings_path = os.path.join(raw_data_dir, 'ratings.csv')
    
    print(f"Loading ratings from {ratings_path}...")
    df = pd.read_csv(ratings_path)
    
    # Keep relevant columns
    df = df[['userId', 'movieId', 'rating', 'timestamp']]
    
    # Convert explicit ratings to implicit interactions (interaction = 1 if rated)
    df['interaction'] = 1
    
    print(f"Loaded {len(df)} interactions.")
    return df[['userId', 'movieId', 'interaction', 'timestamp']]
