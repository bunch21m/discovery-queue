from src.ingest.initialize_game_embeddings import initialize_game_embeddings_database
from src.db.tools.game_functions import load_all_games_from_database

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# ID embeddings removed - they were fixed hashes that added noise without learning
# Now features are: genre (~30), price (4), popularity (1), rating (1), rating_buckets (5)


def compute_game_features(df, mlb=None):
    """
    Computes game embedding features from a DataFrame.
    Returns:
        np.ndarray: Matrix of features (N_games, Dim).
    """
    # 1. Genre Processing (L1-Normalized Multi-Hot)
    # Handle missing genres
    genres = df['genres'].apply(lambda x: x if isinstance(x, list) else [])
    
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=False)
        genre_matrix_multi_hot = mlb.fit_transform(genres)
    else:
        genre_matrix_multi_hot = mlb.transform(genres)

    # L1-normalize genre vectors so they sum to 1.0
    # This matches the user embedding normalization and ensures compatible scales
    row_sums = genre_matrix_multi_hot.sum(axis=1, keepdims=True) + 1e-9
    genre_features = genre_matrix_multi_hot / row_sums

    # 4. Price Bucketing (Vectorized)
    # Map: 0 -> 0, <10 -> 1, <30 -> 2, >=30 -> 3
    prices = df['price'].fillna(0).values
    price_buckets = np.digitize(prices, bins=[0.01, 10.00, 30.00]) 
    # One-hot encode the buckets
    price_features = np.eye(4)[price_buckets]

    # 5. Popularity Feature (Log1p Normalized)
    # Norm constant ~ log(5,000,000) ~ 15.4
    # Ensure 'positive' exists, defaulting to 0
    if 'positive' not in df.columns:
        positive_counts = np.zeros(len(df), dtype=np.float32)
    else:
        positive_counts = df['positive'].fillna(0).values.astype(np.float32)
    
    if 'negative' not in df.columns:
        negative_counts = np.zeros(len(df), dtype=np.float32)
    else:
        negative_counts = df['negative'].fillna(0).values.astype(np.float32)
    
    pop_features = np.log1p(positive_counts) / 15.0
    pop_features = pop_features.reshape(-1, 1)
    
    # 5b. Popularity Bucket (One-hot) - matches user pop_bucket_prefs
    # Buckets: niche (<1k), moderate (1k-10k), popular (10k-100k), mega-popular (100k+)
    pop_buckets = np.zeros(len(df), dtype=np.int32)
    for i in range(len(df)):
        p = positive_counts[i]
        if p < 1000:
            pop_buckets[i] = 0  # niche
        elif p < 10000:
            pop_buckets[i] = 1  # moderate
        elif p < 100000:
            pop_buckets[i] = 2  # popular
        else:
            pop_buckets[i] = 3  # mega-popular
    pop_bucket_features = np.eye(4)[pop_buckets].astype(np.float32)
    
    # 6. Rating Ratio Feature (Quality Signal)
    # Range: 0.0 to 1.0
    total_reviews = positive_counts + negative_counts + 1.0  # +1 to avoid div by zero
    rating_ratio = positive_counts / total_reviews
    rating_ratio = rating_ratio.reshape(-1, 1).astype(np.float32)
    
    # 7. Rating Category Buckets (Steam-style one-hot)
    # Categories: Overwhelmingly Positive (>95%), Very Positive (>80%), Positive (>70%), Mixed (>40%), Negative (<=40%)
    # Only meaningful if there are enough reviews
    min_reviews_for_rating = 10
    has_enough_reviews = (positive_counts + negative_counts) >= min_reviews_for_rating
    
    rating_buckets = np.zeros(len(df), dtype=np.int32)
    for i in range(len(df)):
        if not has_enough_reviews[i]:
            rating_buckets[i] = 2  # Default to "Positive" if not enough data
        elif rating_ratio[i, 0] > 0.95:
            rating_buckets[i] = 0  # Overwhelmingly Positive
        elif rating_ratio[i, 0] > 0.80:
            rating_buckets[i] = 1  # Very Positive
        elif rating_ratio[i, 0] > 0.70:
            rating_buckets[i] = 2  # Positive
        elif rating_ratio[i, 0] > 0.40:
            rating_buckets[i] = 3  # Mixed
        else:
            rating_buckets[i] = 4  # Negative
    
    rating_bucket_features = np.eye(5)[rating_buckets].astype(np.float32)

    # Combine All Features
    # Shape: (N_Games, GENRE_DIM + PRICE_DIM + POP_BUCKET + 1 + 1 + 5) = ~45 dims
    # genre_features = L1-normalized multi-hot genres (~30 dims)
    # price_features = one-hot price buckets (4 dims)
    # pop_bucket_features = one-hot popularity bucket (4 dims)
    # pop_features = log(positive_count) for popularity signal (1 dim)
    # rating_ratio = continuous quality signal 0.0 to 1.0 (1 dim)
    # rating_bucket_features = Steam-style quality categories (5 dims)
    final_item_vectors = np.concatenate([
        genre_features, 
        price_features, 
        pop_bucket_features,
        pop_features,
        rating_ratio,
        rating_bucket_features
    ], axis=1).astype(np.float32)

    return final_item_vectors

def main():
    
    #ensure gameEmbeddings table exists
    initialize_game_embeddings_database()
    
    # get all games in database
    raw_data = load_all_games_from_database()
    df = pd.DataFrame.from_dict(raw_data, orient='index')
    df['app_id'] = df.index

    final_item_vectors = compute_game_features(df)

    print(f"Processed {len(df)} games. Feature Vector Shape: {final_item_vectors.shape}")
    # print(f"Sample Feature Vector for AppID {df['app_id'].iloc[0]}: {final_item_vectors[0]}")
    


if __name__ == "__main__":
    main()
        

