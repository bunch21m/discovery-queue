from src.ingest.initialize_game_embeddings import initialize_game_embeddings_database
from src.models.common_model_utils import load_all_games_from_database

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




def main():
    
    #ensure gameEmbeddings table exists
    initialize_game_embeddings_database()
    
    # get all games in database
    raw_data = load_all_games_from_database()
    df = pd.DataFrame.from_dict(raw_data, orient='index')
    df['appID'] = df.index

    # 2. ID Embeddings (Static Hashing)
    id_features = embed_app_id_deterministic(df['appID'])

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

    # 5. Combine All Features
    # Shape: (N_Games, ID_DIM + GENRE_DIM + PRICE_DIM)
    final_item_vectors = np.concatenate([id_features, genre_features, price_features], axis=1).astype(np.float32)

    print(f"Processed {len(df)} games. Feature Vector Shape: {final_item_vectors.shape}")
    print(f"Sample Feature Vector for AppID {df['appID'].iloc[0]}: {final_item_vectors[0]}")
    
    # TODO: Train for embedding finalItemVectors in a two-tower NN approach and store in pgvector database.
    # TODO: Add check to avoid duplicate work if embeddings already exist.


if __name__ == "__main__":
    main()
        

