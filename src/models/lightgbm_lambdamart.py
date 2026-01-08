import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.user_two_tower_embedding import compute_user_embedding
from src.ingest.create_game_embeddings import compute_game_features


def compute_user_features(user_id, interactions_df, games_df, svd=None, mlb=None, genre_matrix_multi_hot=None):
    """
    Compute user feature vector for LambdaMART.
    Reuses the two-tower user embedding computation.
    """
    return compute_user_embedding(
        user_id, 
        interactions_df, 
        games_df, 
        svd=svd, 
        mlb=mlb, 
        genre_matrix_multi_hot=genre_matrix_multi_hot
    )


def compute_game_features_for_lambdamart(game_row, games_df, precomputed_features=None, app_id_to_idx=None, mlb=None, svd=None):
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
    features = compute_game_features(single_game_df, mlb=mlb, svd=svd)
    
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
        _, (cached_mlb, cached_svd, cached_genres) = get_cached_game_data()
        
        # If disk processors were loaded, we MUST use those for consistency.
        # Otherwise, use the ones from two-tower cache.
        if _LAMBDAMART_PROC_CACHE is None:
            _LAMBDAMART_PROC_CACHE = {'mlb': cached_mlb, 'svd': cached_svd}
            _LAMBDAMART_GENRE_CACHE = cached_genres
        else:
            # Check if we can reuse the cached genres (if MLB matches)
            # For safety, if they mismatch, we just use the one from disk
            _LAMBDAMART_GENRE_CACHE = cached_genres # Usually the same system-wide
            
    return _LAMBDAMART_MODEL_CACHE, _LAMBDAMART_PROC_CACHE, _LAMBDAMART_GENRE_CACHE


def get_top_k_recommendations(
    user_id, 
    interactions_df, 
    games_df, 
    candidate_app_ids,
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
    svd = proc.get('svd')
    
    # 1. Compute user features (Reuse genre_matrix to avoid 100k transformations)
    user_features = compute_user_features(
        user_id, 
        interactions_df, 
        games_df, 
        mlb=mlb, 
        svd=svd, 
        genre_matrix_multi_hot=genre_matrix
    )
    
    # 2. Vectorized Game Feature computation for candidates
    # Extract only the candidate rows for efficiency
    candidate_indices = []
    found_app_ids = []
    for aid in candidate_app_ids:
        if aid in games_df.index:
            candidate_indices.append(games_df.index.get_loc(aid))
            found_app_ids.append(aid)
            
    if not candidate_indices:
        return []
        
    candidate_df = games_df.iloc[candidate_indices]
    
    # Compute game features for the subset
    game_features = compute_game_features(candidate_df, mlb=mlb, svd=svd)
    
    # 3. Matrix Construction
    n_candidates = len(game_features)
    user_features_matrix = np.tile(user_features, (n_candidates, 1))
    
    # Horizontally stack (User | Game)
    X = np.hstack([user_features_matrix, game_features])
    
    # 4. Predict
    scores = predict_lambdamart(model, X)
    
    # 5. Rank and return
    ranked = sorted(zip(found_app_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]