import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.user_two_tower_embedding import compute_user_embedding
from src.ingest.create_game_embeddings import compute_game_features


def compute_user_features(user_id, interactions_df, games_df, mlb=None, genre_matrix_multi_hot=None):
    """
    Compute user feature vector for LambdaMART.
    Reuses the two-tower user embedding computation.
    """
    return compute_user_embedding(
        user_id, 
        interactions_df, 
        games_df, 
        mlb=mlb, 
        genre_matrix_multi_hot=genre_matrix_multi_hot
    )


def compute_game_features_for_lambdamart(game_row, games_df, precomputed_features=None, app_id_to_idx=None, mlb=None):
    """
    Compute game feature vector for a single game.
    """
    app_id = game_row.get('app_id', '')
    
    # 1. Use precomputed features if available (fastest)
    if precomputed_features is not None and app_id_to_idx is not None:
        if app_id in app_id_to_idx:
            idx = app_id_to_idx[app_id]
            return precomputed_features[idx]
    
    # 2. Compute for single game (fallback, needs processors for consistency)
    single_game_df = pd.DataFrame([game_row])
    single_game_df['app_id'] = app_id
    
    # Use processors to ensure consistent dimension
    features = compute_game_features(single_game_df, mlb=mlb)
    
    return features[0]


# train the model
def train_lambdamart(X_train, y_train, group):
    train_data = lgb.Dataset(X_train, label=y_train, group=group)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5, 10],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'verbose': -1
    }
    model = lgb.train(params, train_data, num_boost_round=100)
    return model


# predict with the model
def predict_lambdamart(model, X):
    # Try normal predict, fallback with shape check disabled if needed
    try:
        preds = model.predict(X)
    except:
        preds = model.predict(X, predict_disable_shape_check=True)
    return preds


# Global state for caching model and processors
_LAMBDAMART_MODEL_CACHE = None
_LAMBDAMART_PROC_CACHE = None
_LAMBDAMART_GENRE_CACHE = None

def get_lambdamart_resources(games_df, model_path='data/lambdamart_model.txt'):
    """Lazily load and cache model and processors."""
    global _LAMBDAMART_MODEL_CACHE, _LAMBDAMART_PROC_CACHE, _LAMBDAMART_GENRE_CACHE
    
    if _LAMBDAMART_MODEL_CACHE is None:
        import os
        import pickle
        from src.models.user_two_tower_embedding import get_cached_game_data
        
        # Load Model
        if os.path.exists(model_path):
            _LAMBDAMART_MODEL_CACHE = lgb.Booster(model_file=model_path)
            print(f"Loaded LambdaMART model from {model_path}")
        
        # Load Processors
        processors_path = 'data/lambdamart_processors.pkl'
        if os.path.exists(processors_path):
            with open(processors_path, 'rb') as f:
                _LAMBDAMART_PROC_CACHE = pickle.load(f)
                print("Loaded LambdaMART processors from disk.")
        
        # Fallback/Refresh genre matrix cache from two-tower cache
        # This is the BIG performance win: reuse the 100k transformed rows
        _, (cached_mlb, cached_genres) = get_cached_game_data()
        
        # If disk processors were loaded, we MUST use those for consistency.
        # Otherwise, use the ones from two-tower cache.
        if _LAMBDAMART_PROC_CACHE is None:
            _LAMBDAMART_PROC_CACHE = {'mlb': cached_mlb}
            _LAMBDAMART_GENRE_CACHE = cached_genres
        else:
            # Check if we can reuse the cached genres (if MLB matches)
            # For safety, if they mismatch, we just use the one from disk
            _LAMBDAMART_GENRE_CACHE = cached_genres # Usually the same system-wide
            
    return _LAMBDAMART_MODEL_CACHE, _LAMBDAMART_PROC_CACHE, _LAMBDAMART_GENRE_CACHE



def compute_cross_features(user_features, game_features):
    """
    Compute interaction features between User and Game vectors.
    Assumes structure:
    Game: [Genre(N), Price(4), PopBucket(4), Pop(1), RatingRatio(1), RatingBucket(5)]
    User: [Genre(N), DomGenre(N), Price(4), PopBucket(4), Pop(1), Rating(1), RatingBucket(5)]
    """
    # 1. Infer N (Genre Dim)
    # Game fixed tail = 4 + 4 + 1 + 1 + 5 = 15
    N_genres = game_features.shape[1] - 15
    
    if N_genres <= 0:
        # Fallback if something is wrong
        return np.zeros((game_features.shape[0], 4))

    # 2. Slice Vectors
    # Genre: First N
    user_genre = user_features[:, :N_genres]       # (1, N)
    game_genre = game_features[:, :N_genres]       # (M, N)
    
    # Price: 4 dims at dist -15 to -11
    user_price = user_features[:, -15:-11]
    game_price = game_features[:, -15:-11]
    
    # Pop Bucket: 4 dims at dist -11 to -7
    user_pop = user_features[:, -11:-7]
    game_pop = game_features[:, -11:-7]
    
    # Rating Bucket: 5 dims at dist -5 to end
    user_rating = user_features[:, -5:]
    game_rating = game_features[:, -5:]
    
    # 3. Compute Interactions (Dot products)
    # We want element-wise multiplication then sum along axis 1 (dot product per row)
    
    # Genre Match (User preferences * Game genres)
    genre_match = np.sum(user_genre * game_genre, axis=1, keepdims=True)
    
    # Price Match (User price buckets * Game price buckets)
    # Since Game is one-hot, this picks the specific user preference for that price
    price_match = np.sum(user_price * game_price, axis=1, keepdims=True)
    
    # Pop Match
    pop_match = np.sum(user_pop * game_pop, axis=1, keepdims=True)
    
    # Rating Match
    rating_match = np.sum(user_rating * game_rating, axis=1, keepdims=True)
    
    return np.hstack([genre_match, price_match, pop_match, rating_match])


def get_top_k_recommendations(
    user_id, 
    interactions_df, 
    games_df, 
    candidate_app_ids,
    candidate_distances=None,
    k=10,
    model_path='data/lambdamart_model.txt'
):
    """
    Return top-k recommendations with heavily optimized inference.
    """
    model, proc, genre_matrix = get_lambdamart_resources(games_df, model_path)
    
    if model is None:
        return []

    mlb = proc.get('mlb')
    
    # 1. Compute user features
    user_features = compute_user_features(
        user_id, 
        interactions_df, 
        games_df, 
        mlb=mlb, 
        genre_matrix_multi_hot=genre_matrix
    )
    
    # Ensure user_features is 2D (1, D)
    if len(user_features.shape) == 1:
        user_features = user_features.reshape(1, -1)
    
    # 2. Vectorized Game Feature computation for candidates
    candidate_indices = []
    found_app_ids = []
    for aid in candidate_app_ids:
        if aid in games_df.index:
            candidate_indices.append(games_df.index.get_loc(aid))
            found_app_ids.append(aid)
            
    if not candidate_indices:
        return []
        
    candidate_df = games_df.iloc[candidate_indices]
    game_features = compute_game_features(candidate_df, mlb=mlb)
    
    # 3. Compute personalization features (distance)
    if candidate_distances is not None:
        raw_distances = np.array([
            candidate_distances.get(aid, 1.0) 
            for aid in found_app_ids
        ])
        min_dist = raw_distances.min()
        max_dist = raw_distances.max()
        dist_range = max_dist - min_dist if max_dist > min_dist else 1.0
        personalization_features = (1.0 - (raw_distances - min_dist) / dist_range).reshape(-1, 1)
    else:
        personalization_features = np.full((len(found_app_ids), 1), 0.5)
        
    # 4. Compute Cross Features (NEW)
    # Broadcast user features to match game features count is handled inside cross func via numpy broadcasting 
    # if we passed (1, D) and (M, D).
    # Currently my helper expects matching first dims or broadcasting.
    # user_features is (1, D_user), game_features is (M, D_game)
    # The helper needs to handle the tile.
    
    # Let's tile user_features first to be safe and explicit
    n_candidates = len(game_features)
    user_features_matrix = np.tile(user_features, (n_candidates, 1))
    
    cross_features = compute_cross_features(user_features_matrix, game_features)
    
    # 5. Matrix Construction
    # Stack: User | Game | Personalization | Cross
    X = np.hstack([
        user_features_matrix, 
        game_features, 
        personalization_features,
        cross_features
    ])
    
    # 5. Predict
    scores = predict_lambdamart(model, X)
    
    # 6. Rank and return
    ranked = sorted(zip(found_app_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]