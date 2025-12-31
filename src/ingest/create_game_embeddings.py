from src.ingest.initialize_game_embeddings import initialize_game_embeddings_database
from src.db.game_functions import load_all_games_from_database

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer
import hashlib

# -------- Configurable hyperparameters --------
ID_EMB_DIM = 16      # deterministic ID embedding size  
GENRE_PROJ_DIM = 64  # projected dimension for genres
# ---------------------------------------------


def embed_app_id_deterministic(app_id_series):
    """Vectorized hashing of AppIDs to deterministic floats."""
    def hash_id(aid):
        hash_bytes = hashlib.md5(str(aid).encode('utf-8')).digest()
        # Convert first N bytes to floats and normalize
        ints = np.frombuffer(hash_bytes, dtype=np.uint8)[:ID_EMB_DIM]
        vec = ints.astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)
    
    return np.stack(app_id_series.apply(hash_id).values)





def compute_game_features(df):
    """
    Computes game embedding features from a DataFrame.
    Returns:
        np.ndarray: Matrix of features (N_games, Dim).
    """
    # 2. ID Embeddings (Static Hashing)
    id_features = embed_app_id_deterministic(df['app_id'])

    # 3. Genre Processing (Multi-Hot + SVD)
    mlb = MultiLabelBinarizer(sparse_output=False)
    # Handle missing genres
    genres = df['genres'].apply(lambda x: x if isinstance(x, list) else [])
    
    genre_matrix_multi_hot = mlb.fit_transform(genres)

    # Apply dimensionality reduction globally
    actual_dim = min(GENRE_PROJ_DIM, genre_matrix_multi_hot.shape[1] - 1)
    if actual_dim > 0:
        svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        genre_features = svd.fit_transform(genre_matrix_multi_hot)
    else:
        genre_features = genre_matrix_multi_hot

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

    # 8. Combine All Features
    # Shape: (N_Games, ID_DIM + GENRE_DIM + PRICE_DIM + 1 + 5)
    # ID features = deterministic hash embeddings
    # genre_features = SVD projected multi-hot genres
    # price_features = one-hot price buckets
    # pop_features = log(positive_count) for popularity signal
    # rating_bucket_features = quality signal (Steam-style categories)
    final_item_vectors = np.concatenate([
        id_features, 
        genre_features, 
        price_features, 
        pop_features, 
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
        

