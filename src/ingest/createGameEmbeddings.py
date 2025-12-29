from src.ingest.initializeGameEmbeddings import initializeGameEmbeddingsDatabase
from src.models.CommonModelUtils import loadAllGamesFromDatabase

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer
import hashlib

# -------- Configurable hyperparameters --------
ID_EMB_DIM = 16      # deterministic ID embedding size  
GENRE_PROJ_DIM = 64  # projected dimension for genres
# ---------------------------------------------


def embedAppIdDeterministic(appIdSeries):
    """Vectorized hashing of AppIDs to deterministic floats."""
    def hashId(aid):
        hashBytes = hashlib.md5(str(aid).encode('utf-8')).digest()
        # Convert first N bytes to floats and normalize
        ints = np.frombuffer(hashBytes, dtype=np.uint8)[:ID_EMB_DIM]
        vec = ints.astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)
    
    return np.stack(appIdSeries.apply(hashId).values)




def main():
    
    #ensure gameEmbeddings table exists
    initializeGameEmbeddingsDatabase()
    
    # get all games in database
    rawData = loadAllGamesFromDatabase() 
    df = pd.DataFrame.from_dict(rawData, orient='index')
    df['appID'] = df.index

    # 2. ID Embeddings (Static Hashing)
    idFeatures = embedAppIdDeterministic(df['appID'])

    # 3. Genre Processing (Multi-Hot + SVD)
    mlb = MultiLabelBinarizer(sparse_output=False)
    # Handle missing genres
    genres = df['genres'].apply(lambda x: x if isinstance(x, list) else [])
    
    genreMatrixMultiHot = mlb.fit_transform(genres)

    # Apply dimensionality reduction globally
    actualDim = min(GENRE_PROJ_DIM, genreMatrixMultiHot.shape[1] - 1)
    if actualDim > 0:
        svd = TruncatedSVD(n_components=actualDim, random_state=42)
        genreFeatures = svd.fit_transform(genreMatrixMultiHot)
    else:
        genreFeatures = genreMatrixMultiHot

    # 4. Price Bucketing (Vectorized)
    # Map: 0 -> 0, <10 -> 1, <30 -> 2, >=30 -> 3
    prices = df['price'].fillna(0).values
    priceBuckets = np.digitize(prices, bins=[0.01, 10.00, 30.00]) 
    # One-hot encode the buckets
    priceFeatures = np.eye(4)[priceBuckets]

    # 5. Combine All Features
    # Shape: (N_Games, ID_DIM + GENRE_DIM + PRICE_DIM)
    finalItemVectors = np.concatenate([idFeatures, genreFeatures, priceFeatures], axis=1).astype(np.float32)

    print(f"Processed {len(df)} games. Feature Vector Shape: {finalItemVectors.shape}")
    print(f"Sample Feature Vector for AppID {df['appID'].iloc[0]}: {finalItemVectors[0]}")
    
    # TODO: Train for embedding finalItemVectors in a two-tower NN approach and store in pgvector database.
    # TODO: Add check to avoid duplicate work if embeddings already exist.


if __name__ == "__main__":
    main()
        

