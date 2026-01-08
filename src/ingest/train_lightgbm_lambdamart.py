"""
LightGBM LambdaMART Training Data Generator

Generates training data for the LambdaMART learning-to-rank model.
Uses the same persona-based approach as generate_training_pairs.py,
but structures data as query groups with graded relevance labels.

Output format:
    - X_train: (N_samples, feature_dim) combined user+game features
    - y_train: (N_samples,) graded relevance labels (0-3)
    - group: list of group sizes (candidates per query)
"""

import json
import numpy as np
import pandas as pd
import random
import uuid
import os
import pickle

from src.models.lightgbm_lambdamart import compute_user_features, compute_game_features_for_lambdamart
from src.ingest.create_game_embeddings import compute_game_features


def get_weights(df):
    """
    Calculates sampling weights based on Rating Quality and Popularity Volume.
    Formula: (Positive_Ratio) * (Log(Positive_Count)^1.5)
    """
    pos = df['positive'].fillna(0)
    neg = df['negative'].fillna(0)
    
    ratio = pos / (pos + neg + 1.0)
    volume_boost = np.log1p(pos) ** 1.5
    
    return (ratio * volume_boost) + 0.1


def compute_relevance_score(game_row, persona_name, persona_data, games_df):
    """
    Compute graded relevance score (0-3) for a game given a persona.
    
    Scoring varies by persona type:
    - Mainstream personas: Higher relevance for popular games in liked genres
    - IndieLover: Higher relevance for cheap, indie games (ignores popularity)
    - NicheHunter: Higher relevance for obscure games (inverts popularity preference)
    - TrendFollower: Only cares about popularity
    
    Returns:
        int: Relevance score 0-3
    """
    genres = game_row.get('genres', [])
    if not isinstance(genres, list):
        genres = []
    
    positive = game_row.get('positive', 0)
    price = game_row.get('price', 0)
    
    liked_genres = persona_data.get('likes', [])
    disliked_genres = persona_data.get('dislikes', [])
    min_positive = persona_data.get('min_positive', 0)
    
    # Check genre match
    has_liked_genre = any(g in genres for g in liked_genres) if liked_genres else False
    has_disliked_genre = any(g in genres for g in disliked_genres) if disliked_genres else False
    
    # Persona-specific scoring
    if persona_name == "TrendFollower":
        # Only cares about popularity
        if positive >= 10000:
            return 3
        elif positive >= 1000:
            return 2
        elif positive >= 100:
            return 1
        else:
            return 0
    
    elif persona_name == "IndieLover":
        # Prefers cheap indie games, doesn't care about popularity
        is_indie = "Indie" in genres or "Casual" in genres
        is_cheap = price <= persona_data.get('price_max', 20)
        
        if has_disliked_genre:
            return 0
        elif is_indie and is_cheap:
            return 3
        elif is_indie or is_cheap:
            return 2
        elif not has_disliked_genre:
            return 1
        else:
            return 0
    
    elif persona_name == "NicheHunter":
        # Prefers obscure games - INVERTS popularity preference
        has_preferred_genre = any(g in genres for g in liked_genres) if liked_genres else True
        
        if has_preferred_genre:
            if positive < 100:  # Very obscure = high relevance
                return 3
            elif positive < 500:
                return 2
            elif positive < 2000:
                return 1
            else:
                return 0  # Too popular for niche hunter
        else:
            return 0
    
    else:
        # Mainstream personas (ActionFan, RPGPlayer, StrategySimFan, EclecticExplorer)
        if has_disliked_genre:
            return 0
        
        if has_liked_genre:
            # Genre match - now check popularity
            if positive >= min_positive * 10:  # Very popular
                return 3
            elif positive >= min_positive:
                return 2
            elif positive >= min_positive / 2:
                return 1
            else:
                return 0  # Below popularity threshold
        else:
            # No genre match - check if at least popular
            if positive >= min_positive * 5:
                return 1
            else:
                return 0


def generate_lambdamart_training_data():
    """
    Generate training data for LightGBM LambdaMART model.
    
    Creates query groups where each synthetic user has a pool of
    candidate games with graded relevance labels.
    """
    output_path = 'data/lambdamart_training_data.pkl'
    
    if os.path.exists(output_path):
        print(f"Training data already exists at {output_path}, skipping generation.")
        return
    
    print("Loading games...")
    with open('data/games.json', 'r', encoding='utf-8') as f:
        games_dict = json.load(f)
    
    # Convert to DataFrame
    processed_games = []
    for app_id, data in games_dict.items():
        price = data.get('price', 0)
        if isinstance(price, str):
            try:
                price = float(price.replace('$', ''))
            except:
                price = 0.0
        
        genres = data.get('genres', [])
        if not isinstance(genres, list):
            genres = []
        
        processed_games.append({
            'app_id': str(app_id),
            'name': data.get('name', ''),
            'genres': genres,
            'price': float(price),
            'positive': int(data.get('positive', 0)),
            'negative': int(data.get('negative', 0))
        })
    
    games_df = pd.DataFrame(processed_games)
    games_df.set_index('app_id', inplace=True)
    games_df['app_id'] = games_df.index
    
    print(f"Loaded {len(games_df)} games.")
    
    # Pre-compute all game features once for consistent SVD encoding
    print("Pre-computing game features...")
    # Use prepare_genre_processors to get consistent MLB/SVD
    from src.models.user_two_tower_embedding import prepare_genre_processors
    mlb, svd, _ = prepare_genre_processors(games_df)
    
    # Save processors for inference consistency
    os.makedirs('data', exist_ok=True)
    with open('data/lambdamart_processors.pkl', 'wb') as f:
        pickle.dump({'mlb': mlb, 'svd': svd}, f)
    print("Saved processors to data/lambdamart_processors.pkl")
    
    precomputed_game_features = compute_game_features(games_df, mlb=mlb, svd=svd)
    app_id_to_idx = {aid: i for i, aid in enumerate(games_df.index)}
    
    # Personas - same as two-tower training
    MAINSTREAM_MIN_POSITIVE = 100
    
    personas = {
        # Mainstream Personas (filtered to popular games)
        "ActionFan": {
            "likes": ["Action", "FPS", "Shooter"],
            "dislikes": ["Strategy", "Puzzle"],
            "min_positive": MAINSTREAM_MIN_POSITIVE
        },
        "RPGPlayer": {
            "likes": ["RPG", "Adventure", "JRPG"],
            "dislikes": ["Sports", "Racing"],
            "min_positive": MAINSTREAM_MIN_POSITIVE
        },
        "StrategySimFan": {
            "likes": ["Strategy", "Simulation", "City Builder"],
            "dislikes": ["FPS", "Action"],
            "min_positive": MAINSTREAM_MIN_POSITIVE
        },
        "EclecticExplorer": {
            "likes": ["Puzzle", "Strategy", "RPG", "Indie"],
            "dislikes": [],
            "min_positive": MAINSTREAM_MIN_POSITIVE
        },
        "TrendFollower": {
            "likes": [],
            "dislikes": [],
            "min_positive": 1000
        },
        
        # Niche Personas
        "IndieLover": {
            "likes": ["Indie", "Casual"],
            "dislikes": ["AAA", "Sports"],
            "price_max": 20,
            "min_positive": 0
        },
        "NicheHunter": {
            "likes": ["Indie", "Simulation", "Visual Novel"],
            "dislikes": [],
            "min_positive": 0,
            "prefers_obscure": True
        },
    }
    
    # Training data accumulators
    X_train_list = []
    y_train_list = []
    group_list = []
    
    n_users_per_persona = 100
    n_candidates_per_query = 50
    
    print("Generating training queries...")
    
    for persona_name, persona_data in personas.items():
        print(f"  Generating {n_users_per_persona} users for persona: {persona_name}")
        
        for _ in range(n_users_per_persona):
            user_id = str(uuid.uuid4())
            
            # Sample candidate pool
            # Mix of: liked genres, disliked genres, random games
            candidates = []
            
            liked_genres = persona_data.get('likes', [])
            disliked_genres = persona_data.get('dislikes', [])
            
            # ~40% from liked genres
            if liked_genres:
                liked_mask = games_df['genres'].apply(
                    lambda g: any(l in g for l in liked_genres) if isinstance(g, list) else False
                )
                liked_games = games_df[liked_mask]
                if len(liked_games) > 0:
                    n_liked = min(len(liked_games), int(n_candidates_per_query * 0.4))
                    sampled_liked = liked_games.sample(n=n_liked, weights=get_weights(liked_games))
                    candidates.extend(sampled_liked.index.tolist())
            
            # ~20% from disliked genres
            if disliked_genres:
                disliked_mask = games_df['genres'].apply(
                    lambda g: any(d in g for d in disliked_genres) if isinstance(g, list) else False
                )
                disliked_games = games_df[disliked_mask]
                if len(disliked_games) > 0:
                    n_disliked = min(len(disliked_games), int(n_candidates_per_query * 0.2))
                    sampled_disliked = disliked_games.sample(n=n_disliked, weights=get_weights(disliked_games))
                    candidates.extend(sampled_disliked.index.tolist())
            
            # Fill rest with random games
            remaining = n_candidates_per_query - len(candidates)
            if remaining > 0:
                available = games_df[~games_df.index.isin(candidates)]
                if len(available) > 0:
                    n_random = min(len(available), remaining)
                    sampled_random = available.sample(n=n_random, weights=get_weights(available))
                    candidates.extend(sampled_random.index.tolist())
            
            # Deduplicate
            candidates = list(dict.fromkeys(candidates))[:n_candidates_per_query]
            
            if len(candidates) < 10:
                # Skip if not enough candidates
                continue
            
            # Generate fake interactions for user embedding
            # Sample some wishlisted games from liked genres
            wishlisted_ids = []
            if liked_genres:
                liked_mask = games_df['genres'].apply(
                    lambda g: any(l in g for l in liked_genres) if isinstance(g, list) else False
                )
                liked_pool = games_df[liked_mask]
                if len(liked_pool) > 0:
                    n_wishlist = random.randint(3, 8)
                    wishlisted = liked_pool.sample(n=min(len(liked_pool), n_wishlist))
                    wishlisted_ids = wishlisted.index.tolist()
            
            interactions = []
            for wid in wishlisted_ids:
                interactions.append({
                    'app_id': wid,
                    'interactiontype': 'wishlist',
                    'timestamp': random.randint(0, 1000)
                })
            
            interactions_df = pd.DataFrame(interactions) if interactions else pd.DataFrame()
            
            # Compute user features (passing mlb/svd for consistency)
            user_features = compute_user_features(user_id, interactions_df, games_df, mlb=mlb, svd=svd)
            
            # Score each candidate
            query_X = []
            query_y = []
            
            for app_id in candidates:
                game_row = games_df.loc[app_id].to_dict()
                game_row['app_id'] = app_id  # Add app_id since it's the index
                
                # Compute relevance
                relevance = compute_relevance_score(
                    game_row, persona_name, persona_data, games_df
                )
                
                # Compute game features (use precomputed for consistency)
                game_features = compute_game_features_for_lambdamart(
                    game_row, games_df, precomputed_game_features, app_id_to_idx, mlb=mlb, svd=svd
                )
                
                # Concatenate user + game features
                combined_features = np.concatenate([user_features, game_features])
                
                query_X.append(combined_features)
                query_y.append(relevance)
            
            # Add to training data
            if query_X:
                X_train_list.extend(query_X)
                y_train_list.extend(query_y)
                group_list.append(len(query_X))
    
    # Convert to arrays
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)
    
    print(f"\nGenerated training data:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  Number of query groups: {len(group_list)}")
    print(f"  Total samples: {sum(group_list)}")
    print(f"  Relevance distribution: {np.bincount(y_train.astype(int))}")
    
    # Save
    training_data = {
        'X_train': X_train,
        'y_train': y_train,
        'group': group_list
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"\nSaved training data to {output_path}")

def train_lambdamart_model():
    """
    Train the LightGBM LambdaMART model using generated training data.
    """
    from src.models.lightgbm_lambdamart import train_lambdamart
    
    data_path = 'data/lambdamart_training_data.pkl'
    model_path = 'data/lambdamart_model.txt'
    
    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}. Please generate it first.")
        return
    
    if os.path.exists(model_path):
        print(f"LambdaMART Model already exists at {model_path}, skipping training.")
        return
    
    print("Loading training data...")
    with open(data_path, 'rb') as f:
        training_data = pickle.load(f)
    
    X_train = training_data['X_train']
    y_train = training_data['y_train']
    group = training_data['group']
    
    print("Training LambdaMART model...")
    model = train_lambdamart(X_train, y_train, group)
    
    print(f"Saving model to {model_path}...")
    model.save_model(model_path)
    
    print("Model training complete.")


if __name__ == "__main__":
    generate_lambdamart_training_data()
    train_lambdamart_model()
    
    
