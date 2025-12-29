from src.db.user_functions import get_user_by_username
from src.db.interaction_functions import get_users_interactions_from_database
from src.db.game_functions import get_game_from_database
from src.models.common_model_utils import load_all_games_from_database

import numpy as np
import pandas as pd
import hashlib

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer

# -------- Configurable hyperparameters --------
ID_EMB_DIM = 16      # deterministic ID embedding size  
GENRE_PROJ_DIM = 64  # projected dimension for genres
# ---------------------------------------------




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

def concat_user_features(username):
    """
    Concatenates user features for the two-tower embedding model.

    Args:
            username (str): The username of the user to process.
    """
    
    
    user = get_user_by_username(username)
    if not user:
        raise ValueError(f"User {username} not found in database")

    
    feature_list = []
    
    # ID Embedding
    user_id_emb = embed_user_id_deterministic(pd.Series([user['userid']]))
    
    interactions = get_users_interactions_from_database(user['userid'])
    interactions_df = pd.DataFrame.from_dict(interactions, orient='index')
    
    # get all games to create multihot genre feature
    games = load_all_games_from_database() 
    games_df = pd.DataFrame.from_dict(games, orient='index')
    games_df['appID'] = games_df.index

    # Genre Processing
    # First create multi-hot encoding for all games
    # Use svd to reduce dimensionality
    # Apply same transformation to user genre preferences
    
    
    
    mlb = MultiLabelBinarizer(sparse_output=False)
    # Handle missing genres
    genres = games_df['genres'].apply(lambda x: x if isinstance(x, list) else [])
    
    genre_matrix_multi_hot = mlb.fit_transform(genres)

    # Apply dimensionality reduction globally
    actual_dim = min(GENRE_PROJ_DIM, genre_matrix_multi_hot.shape[1] - 1)
    if actual_dim > 0:
        svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        genre_features = svd.fit_transform(genre_matrix_multi_hot)
    else:
        genre_features = genre_matrix_multi_hot

    
    wishlisted_games = []
    user_genres = np.zeros(genre_matrix_multi_hot.shape[1], dtype=np.float32)

    # Count genres from wishlisted games and fill wishlistedGames
    for _, row in interactions_df.iterrows():
        app_id = row['appid']
        interaction_type = row['interactiontype']

        if app_id not in games_df.index:
            continue

        if interaction_type == 'wishlist':
            game = get_game_from_database(app_id)
            if game:
                wishlisted_games.append(game)
                user_genres += genre_matrix_multi_hot[games_df.index.get_loc(app_id)]

    # Apply same dimensionality transformation to user genre preferences
    if user_genres.sum() > 0 and actual_dim > 0:
        user_genre_embedding = svd.transform(user_genres.reshape(1, -1)).flatten()
    else:
        user_genre_embedding = np.zeros(actual_dim if actual_dim > 0 else genre_matrix_multi_hot.shape[1])

    wishlisted_games_df = pd.DataFrame(wishlisted_games)
    
    # Wishlist Rate
    wishlist_rate = sum(interactions_df['interactiontype'] == 'wishlist') / len(interactions_df) if len(interactions_df) > 0 else 0.0
    
    # Bucket wishlisted games by price ranges (Free, <10, <30, >=30)
    if not wishlisted_games_df.empty and 'price' in wishlisted_games_df.columns:
        free_wishlisted = (wishlisted_games_df['price'] == 0).sum()
        low_wishlisted = ((wishlisted_games_df['price'] > 0) & (wishlisted_games_df['price'] < 10)).sum()
        mid_wishlisted = ((wishlisted_games_df['price'] >= 10) & (wishlisted_games_df['price'] < 30)).sum()
        high_wishlisted = (wishlisted_games_df['price'] >= 30).sum()
    else:
        free_wishlisted = low_wishlisted = mid_wishlisted = high_wishlisted = 0

    # recent skip rate(last 20 interactions)
    recent_interactions = interactions_df.sort_values(by='timestamp', ascending=False).head(20)
    recent_skip_rate = sum(recent_interactions['interactiontype'] == 'skip') / len(recent_interactions) if len(recent_interactions) > 0 else 0.0
    
    
    
    # Concatenate all features
    feature_list.append(user_id_emb[0])
    print(f"userIdEmb: {user_id_emb[0]}")
    feature_list.append(user_genre_embedding)
    print(f"userGenreEmbedding: {user_genre_embedding}")
    feature_list.append(wishlist_rate)
    print(f"wishlistRate: {wishlist_rate}")
    feature_list.append(free_wishlisted)
    print(f"freeWishlisted: {free_wishlisted}")
    feature_list.append(low_wishlisted)
    print(f"lowWishlisted: {low_wishlisted}")
    feature_list.append(mid_wishlisted)
    print(f"midWishlisted: {mid_wishlisted}")
    feature_list.append(high_wishlisted)
    print(f"highWishlisted: {high_wishlisted}")
    feature_list.append(recent_skip_rate)
    print(f"recentSkipRate: {recent_skip_rate}")  
    

    

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
    
    