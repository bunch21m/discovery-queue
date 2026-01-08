from src.db.user_functions import get_user_by_username
from src.db.interaction_functions import get_users_interactions_from_database
from src.db.tools.game_functions import get_game_from_database
from src.db.tools.game_functions import load_all_games_from_database

import numpy as np
import pandas as pd
import hashlib

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer

# -------- Configurable hyperparameters --------
ID_EMB_DIM = 16      # deterministic ID embedding size  
GENRE_PROJ_DIM = 64  # projected dimension for genres
# ---------------------------------------------
# Global cache to avoid reloading massive game table every request
_GAMES_CACHE = None
_GENRE_PROCESSORS_CACHE = None

def get_cached_game_data():
    """Helper to lazily load and cache the entire games table."""
    global _GAMES_CACHE, _GENRE_PROCESSORS_CACHE
    if _GAMES_CACHE is None:
        print("Loading games table into cache for embedding computation...")
        games = load_all_games_from_database()
        _GAMES_CACHE = pd.DataFrame.from_dict(games, orient='index')
        _GAMES_CACHE['app_id'] = _GAMES_CACHE.index
        
        print("Pre-fitting genre processors for cache...")
        _GENRE_PROCESSORS_CACHE = prepare_genre_processors(_GAMES_CACHE)
        
    return _GAMES_CACHE, _GENRE_PROCESSORS_CACHE



def embed_user_id_deterministic(user_id_series):
    """Vectorized hashing of AppIDs to deterministic floats."""
    def hash_id(aid):
        hash_bytes = hashlib.md5(str(aid).encode('utf-8')).digest()
        # Convert first N bytes to floats and normalize
        ints = np.frombuffer(hash_bytes, dtype=np.uint8)[:ID_EMB_DIM]
        vec = ints.astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)
    
    return np.stack(user_id_series.apply(hash_id).values)

def compute_user_embedding(user_id, interactions_df, games_df, svd=None, mlb=None, genre_matrix_multi_hot=None):
    """
    Core logic to compute user embedding vectors.
    Args:
        user_id (str/int): User ID.
        interactions_df (pd.DataFrame): User interactions.
        games_df (pd.DataFrame): All games interactions with genres and price. 
                                 Index should be appID.
        svd (TruncatedSVD): Pre-fitted SVD model.
        mlb (MultiLabelBinarizer): Pre-fitted MLB.
    """
    
    # ID Embedding
    user_id_emb = embed_user_id_deterministic(pd.Series([user_id]))
    
    # Genre Processing
    # Genre Processing
    if genre_matrix_multi_hot is None:
        if mlb is None:
            mlb = MultiLabelBinarizer(sparse_output=False)
            genres = games_df['genres'].apply(lambda x: x if isinstance(x, list) else [])
            genre_matrix_multi_hot = mlb.fit_transform(genres)
            
            # Auto-fit SVD if not provided and needed
            actual_dim = min(GENRE_PROJ_DIM, genre_matrix_multi_hot.shape[1] - 1)
            if actual_dim > 0 and svd is None:
                svd = TruncatedSVD(n_components=actual_dim, random_state=42)
                svd.fit(genre_matrix_multi_hot)
        else:
            genres = games_df['genres'].apply(lambda x: x if isinstance(x, list) else [])
            genre_matrix_multi_hot = mlb.transform(genres)
            # SVD must be provided if passed outside or fitted here previously.
    
    # We need actual_dim for user embedding
    actual_dim = min(GENRE_PROJ_DIM, genre_matrix_multi_hot.shape[1] - 1)

    
    user_genres = np.zeros(genre_matrix_multi_hot.shape[1], dtype=np.float32)
    wishlisted_prices = []
    
    # Process interactions
    if not interactions_df.empty:
        # Standardize column name
        if 'appid' in interactions_df.columns:
            interactions_df = interactions_df.rename(columns={'appid': 'app_id'})

        # Filter for valid games
        valid_interactions = interactions_df[interactions_df['app_id'].isin(games_df.index)]
        
        for _, row in valid_interactions.iterrows():
            app_id = row['app_id']
            interaction_type = row['interactiontype']
            
            if interaction_type == 'wishlist':
                # Get index in games_df
                # Assuming games_df index is unique appIDs
                try:
                    idx = games_df.index.get_loc(app_id)
                    # If duplicate index, get_loc returns slice or array, handle simple case
                    if isinstance(idx, int):
                        user_genres += genre_matrix_multi_hot[idx]
                        price = games_df.iloc[idx]['price']
                        wishlisted_prices.append(price)
                    elif isinstance(idx, slice) or isinstance(idx, np.ndarray):
                        # Take first match
                        if isinstance(idx, slice):
                            start = idx.start
                            user_genres += genre_matrix_multi_hot[start]
                            price = games_df.iloc[start]['price']
                        else:
                            # bool array or int array
                            first_idx = np.where(idx)[0][0] if idx.dtype == bool else idx[0]
                            user_genres += genre_matrix_multi_hot[first_idx]
                            price = games_df.iloc[first_idx]['price']
                        wishlisted_prices.append(price)

                except KeyError:
                    continue

    # Apply same dimensionality transformation to user genre preferences
    if user_genres.sum() > 0 and actual_dim > 0:
        # Normalize the vector so 100 action games isn't 100x stronger than 1 action game
        # We want the direction of the preference, not just the magnitude
        normalized_user_genres = user_genres / (user_genres.sum() + 1e-9)
        
        if svd:
             user_genre_embedding = svd.transform(normalized_user_genres.reshape(1, -1)).flatten()
        else:
             user_genre_embedding = normalized_user_genres
    else:
        user_genre_embedding = np.zeros(actual_dim if actual_dim > 0 else genre_matrix_multi_hot.shape[1])

    # Wishlist stats
    wishlisted_prices = np.array(wishlisted_prices)
    if interactions_df.empty:
        wishlist_rate = 0.0
    else:
        wishlist_rate = sum(interactions_df['interactiontype'] == 'wishlist') / len(interactions_df)

    free_wishlisted = (wishlisted_prices == 0).sum()
    low_wishlisted = ((wishlisted_prices > 0) & (wishlisted_prices < 10)).sum()
    mid_wishlisted = ((wishlisted_prices >= 10) & (wishlisted_prices < 30)).sum()
    high_wishlisted = (wishlisted_prices >= 30).sum()

    # recent skip rate
    if not interactions_df.empty and 'timestamp' in interactions_df.columns:
        recent_interactions = interactions_df.sort_values(by='timestamp', ascending=False).head(20)
        recent_skip_rate = sum(recent_interactions['interactiontype'] == 'skip') / len(recent_interactions)
    elif not interactions_df.empty:
        # no timestamp, just take head
        recent_interactions = interactions_df.head(20)
        recent_skip_rate = sum(recent_interactions['interactiontype'] == 'skip') / len(recent_interactions)
    else:
        recent_skip_rate = 0.0

    return np.concatenate(
        [
            user_id_emb[0].astype(np.float32),
            user_genre_embedding.astype(np.float32),
            np.array(
                [
                    wishlist_rate,
                    free_wishlisted,
                    low_wishlisted,
                    mid_wishlisted,
                    high_wishlisted,
                    recent_skip_rate,
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    ).astype(np.float32)

def concat_user_features(username):
    """
    Concatenates user features for the two-tower embedding model.

    Args:
            username (str): The username of the user to process.
    """
    user = get_user_by_username(username)
    if not user:
        raise ValueError(f"User {username} not found in database")

    interactions = get_users_interactions_from_database(user['userid'])
    interactions_df = pd.DataFrame.from_dict(interactions, orient='index')
    
    # Use cached data to avoid loading 300MB+ of games on every request
    games_df, (mlb, svd, genre_matrix_multi_hot) = get_cached_game_data()
    
    return compute_user_embedding(
        user['userid'], 
        interactions_df, 
        games_df, 
        mlb=mlb, 
        svd=svd, 
        genre_matrix_multi_hot=genre_matrix_multi_hot
    )

def prepare_genre_processors(games_df):
    """
    Helper to pre-fit MLB and SVD and pre-compute genre matrix for batch processing.
    Returns:
        tuple: (mlb, svd, genre_matrix_multi_hot)
    """
    mlb = MultiLabelBinarizer(sparse_output=False)
    genres = games_df['genres'].apply(lambda x: x if isinstance(x, list) else [])
    genre_matrix_multi_hot = mlb.fit_transform(genres)
    
    svd = None
    actual_dim = min(GENRE_PROJ_DIM, genre_matrix_multi_hot.shape[1] - 1)
    if actual_dim > 0:
        svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        svd.fit(genre_matrix_multi_hot)
        
    return mlb, svd, genre_matrix_multi_hot