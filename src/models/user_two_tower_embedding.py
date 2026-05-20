from src.db.user_functions import get_user_by_username
from src.db.interaction_functions import get_users_interactions_from_database
from src.db.tools.game_functions import get_game_from_database
from src.db.tools.game_functions import load_all_games_from_database

import pickle
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

# ID embeddings removed - they were fixed hashes that added noise without learning
# User features now: genre preferences (~30), behavioral signals (8)

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
        
        # Try loading from disk first to ensure consistency with trained model
        _GENRE_PROCESSORS_CACHE = load_genre_processors()
        if _GENRE_PROCESSORS_CACHE is None:
            print("No saved genre processors found. Fitting new ones...")
            _GENRE_PROCESSORS_CACHE = prepare_genre_processors(_GAMES_CACHE)
            save_genre_processors(_GENRE_PROCESSORS_CACHE[0])
        else:
            print("Successfully loaded genre processors from disk.")
            # Re-generate the multi-hot matrix with loaded MLB
            mlb = _GENRE_PROCESSORS_CACHE
            genres = _GAMES_CACHE['genres'].apply(lambda x: x if isinstance(x, list) else [])
            genre_matrix = mlb.transform(genres)
            _GENRE_PROCESSORS_CACHE = (mlb, genre_matrix)
        
    return _GAMES_CACHE, _GENRE_PROCESSORS_CACHE

def save_genre_processors(mlb):
    """Saves MLB to disk."""
    print("Saving genre processors to disk...")
    os.makedirs('data', exist_ok=True)
    with open('data/genre_mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)

def load_genre_processors():
    """Loads MLB from disk."""
    if os.path.exists('data/genre_mlb.pkl'):
        try:
            with open('data/genre_mlb.pkl', 'rb') as f:
                mlb = pickle.load(f)
            return mlb
        except Exception as e:
            print(f"Error loading genre processors: {e}")
    return None






def compute_user_embedding(user_id, interactions_df, games_df, mlb=None, genre_matrix_multi_hot=None):
    """
    Core logic to compute user embedding vectors.
    Args:
        user_id (str/int): User ID.
        interactions_df (pd.DataFrame): User interactions.
        games_df (pd.DataFrame): All games interactions with genres and price. 
                                 Index should be appID.
        mlb (MultiLabelBinarizer): Pre-fitted MLB.
    """
    # Genre Processing (L1-Normalized Multi-Hot)
    if genre_matrix_multi_hot is None:
        if mlb is None:
            mlb = MultiLabelBinarizer(sparse_output=False)
            genres = games_df['genres'].apply(lambda x: x if isinstance(x, list) else [])
            genre_matrix_multi_hot = mlb.fit_transform(genres)
        else:
            genres = games_df['genres'].apply(lambda x: x if isinstance(x, list) else [])
            genre_matrix_multi_hot = mlb.transform(genres)
    
    user_genres = np.zeros(genre_matrix_multi_hot.shape[1], dtype=np.float32)
    wishlisted_prices = []
    wishlisted_pops = []
    wishlisted_ratings = []
    
    # Process interactions
    if not interactions_df.empty:
        # Standardize column name and TYPE
        if 'appid' in interactions_df.columns:
            interactions_df = interactions_df.rename(columns={'appid': 'app_id'})
        
        # CRITICAL: Ensure app_id is string to match games_df.index
        interactions_df['app_id'] = interactions_df['app_id'].astype(str)

        # Filter for valid games
        valid_interactions = interactions_df[interactions_df['app_id'].isin(games_df.index)]
        

        
        wishlist_count = 0
        for _, row in valid_interactions.iterrows():
            app_id = row['app_id']
            interaction_type = row['interactiontype']
            
            if interaction_type == 'wishlist':
                try:
                    idx = games_df.index.get_loc(app_id)
                    if isinstance(idx, int):
                        user_genres += genre_matrix_multi_hot[idx]
                        game_row = games_df.iloc[idx]
                        wishlisted_prices.append(game_row['price'])
                        wishlisted_pops.append(np.log1p(float(game_row.get('positive', 0))) / 15.0)
                        wishlisted_ratings.append(float(game_row.get('positive', 0)) / (float(game_row.get('positive', 0)) + float(game_row.get('negative', 0)) + 1.0))
                        wishlist_count += 1
                    elif isinstance(idx, (slice, np.ndarray)):
                        # Take first match if duplicates exist
                        first_idx = idx.start if isinstance(idx, slice) else (np.where(idx)[0][0] if idx.dtype == bool else idx[0])
                        user_genres += genre_matrix_multi_hot[first_idx]
                        game_row = games_df.iloc[first_idx]
                        wishlisted_prices.append(game_row['price'])
                        wishlisted_pops.append(np.log1p(float(game_row.get('positive', 0))) / 15.0)
                        wishlisted_ratings.append(float(game_row.get('positive', 0)) / (float(game_row.get('positive', 0)) + float(game_row.get('negative', 0)) + 1.0))
                        wishlist_count += 1

                except KeyError:
                    continue
        


    # Scale by max value instead of L1 normalization
    # This preserves the relative magnitude of preferences (action-heavy user stays action-heavy)
    # L1 norm was collapsing all users to similar vectors
    if user_genres.max() > 0:
        user_genre_embedding = user_genres / (user_genres.max() + 1e-9)
    else:
        user_genre_embedding = np.zeros(genre_matrix_multi_hot.shape[1], dtype=np.float32)
    
    # Add dominant genre one-hot feature
    # This makes the PRIMARY preference explicit and harder to lose in averaging
    dominant_genre_idx = np.argmax(user_genres) if user_genres.max() > 0 else 0
    dominant_genre_onehot = np.zeros(genre_matrix_multi_hot.shape[1], dtype=np.float32)
    if user_genres.max() > 0:
        dominant_genre_onehot[dominant_genre_idx] = 1.0

    # Wishlist stats
    wishlisted_prices = np.array(wishlisted_prices)
    n_interactions = len(interactions_df) if not interactions_df.empty else 1
    
    if interactions_df.empty:
        wishlist_rate = 0.0
    else:
        wishlist_rate = sum(interactions_df['interactiontype'] == 'wishlist') / n_interactions

    # Normalize counts into proportions (matching training distribution)
    # Total scale of 0.0 to 1.0 (proportion of wishlist that falls in this bucket)
    n_wishes = len(wishlisted_prices) if len(wishlisted_prices) > 0 else 1
    free_wishlisted = (wishlisted_prices == 0).sum() / n_wishes
    low_wishlisted = ((wishlisted_prices > 0) & (wishlisted_prices < 10)).sum() / n_wishes
    mid_wishlisted = ((wishlisted_prices >= 10) & (wishlisted_prices < 30)).sum() / n_wishes
    high_wishlisted = (wishlisted_prices >= 30).sum() / n_wishes

    # Average Popularity/Rating Preferences
    avg_wishlist_pop = np.mean(wishlisted_pops) if wishlisted_pops else 0.5
    avg_wishlist_rating = np.mean(wishlisted_ratings) if wishlisted_ratings else 0.5
    
    # Popularity bucket preferences (like price buckets)
    # Buckets: niche (<1k), moderate (1k-10k), popular (10k-100k), mega-popular (100k+)
    # Use raw positive counts from wishlisted games
    pop_bucket_prefs = np.zeros(4, dtype=np.float32)
    if wishlisted_pops:  # wishlisted_pops are already log1p normalized / 15.0
        # Convert back to raw counts: exp(pop * 15) - 1
        raw_pops = [np.expm1(p * 15.0) for p in wishlisted_pops]
        n_pop_wishes = len(raw_pops)
        pop_bucket_prefs[0] = sum(1 for p in raw_pops if p < 1000) / n_pop_wishes      # niche
        pop_bucket_prefs[1] = sum(1 for p in raw_pops if 1000 <= p < 10000) / n_pop_wishes  # moderate
        pop_bucket_prefs[2] = sum(1 for p in raw_pops if 10000 <= p < 100000) / n_pop_wishes  # popular
        pop_bucket_prefs[3] = sum(1 for p in raw_pops if p >= 100000) / n_pop_wishes  # mega-popular
    else:
        pop_bucket_prefs[2] = 1.0  # Default to "popular" preference

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

    # Compute average rating bucket preference (one-hot style)
    # This mirrors the game's rating_bucket_features structure
    avg_rating_bucket = np.zeros(5, dtype=np.float32)
    if wishlisted_ratings:
        for rating in wishlisted_ratings:
            if rating > 0.95:
                avg_rating_bucket[0] += 1  # Overwhelmingly Positive
            elif rating > 0.80:
                avg_rating_bucket[1] += 1  # Very Positive
            elif rating > 0.70:
                avg_rating_bucket[2] += 1  # Positive
            elif rating > 0.40:
                avg_rating_bucket[3] += 1  # Mixed
            else:
                avg_rating_bucket[4] += 1  # Negative
        # Normalize to sum to 1
        avg_rating_bucket = avg_rating_bucket / (avg_rating_bucket.sum() + 1e-9)
    else:
        avg_rating_bucket[2] = 1.0  # Default to "Positive" preference

    # Combine All User Features - ENHANCED WITH DOMINANT GENRE + POPULARITY BUCKETS
    # User: [genre(~30), dominant_genre(~30), price_prefs(4), pop_bucket_prefs(4), avg_pop(1), avg_rating(1), rating_bucket_prefs(5)]
    return np.concatenate(
        [
            user_genre_embedding.astype(np.float32),    # ~30 dims (scaled genre preferences)
            dominant_genre_onehot.astype(np.float32),   # ~30 dims (explicit dominant genre)
            np.array([
                free_wishlisted,     # Price bucket 0 preference
                low_wishlisted,      # Price bucket 1 preference
                mid_wishlisted,      # Price bucket 2 preference
                high_wishlisted,     # Price bucket 3 preference
            ], dtype=np.float32),
            pop_bucket_prefs,    # 4 dims (popularity bucket preferences)
            np.array([avg_wishlist_pop], dtype=np.float32),    # 1 dim
            np.array([avg_wishlist_rating], dtype=np.float32), # 1 dim
            avg_rating_bucket,   # 5 dims (rating bucket preferences)
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
    games_df, (mlb, genre_matrix_multi_hot) = get_cached_game_data()
    
    return compute_user_embedding(
        user['userid'], 
        interactions_df, 
        games_df, 
        mlb=mlb, 
        genre_matrix_multi_hot=genre_matrix_multi_hot
    )

def prepare_genre_processors(games_df):
    """
    Helper to pre-fit MLB and pre-compute genre matrix for batch processing.
    Returns:
        tuple: (mlb, genre_matrix_multi_hot)
    """
    mlb = MultiLabelBinarizer(sparse_output=False)
    genres = games_df['genres'].apply(lambda x: x if isinstance(x, list) else [])
    genre_matrix_multi_hot = mlb.fit_transform(genres)
        
    return mlb, genre_matrix_multi_hot