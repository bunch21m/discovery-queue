
import json
import numpy as np
import pandas as pd
import random
import uuid
import sys
import os
import pickle


from src.models.user_two_tower_embedding import compute_user_embedding, prepare_genre_processors
from src.ingest.create_game_embeddings import compute_game_features

def get_weights(df):
    """
    Calculates sampling weights based on Rating Quality and Popularity Volume.
    Formula: (Positive_Ratio) * (Log(Positive_Count)^1.5)
    
    Balanced weighting - still favors popular games but not as extremely.
    - A 100% positive game with 10 reviews: 1.0 * (2.4^1.5) ~= 3.7
    - A 95% positive game with 1M reviews: 0.95 * (13.8^1.5) ~= 49
    Result: Popular games are ~13x more likely (vs 200x before).
    """
    pos = df['positive'].fillna(0)
    neg = df['negative'].fillna(0)
    
    # Balanced score: Ratio * Moderate Volume Boost
    ratio = pos / (pos + neg + 1.0)
    volume_boost = np.log1p(pos) ** 1.5  # Reduced from ^3 to ^1.5
    
    return (ratio * volume_boost) + 0.1

def generate_pairs():
    if os.path.exists('data/two_tower_training_pairs.pkl'):
        print("Training pairs already exist, skipping generation.")
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
                price = float(price.replace('$',''))
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

    # Compute Game Embeddings 
    # This uses the logic in src/ingest/create_game_embeddings.py
    print("Computing game embeddings...")
    game_vectors_matrix = compute_game_features(games_df)
    game_vectors = {aid: vec for aid, vec in zip(games_df.index, game_vectors_matrix)}

    # PRE-COMPUTE GENRE PROCESSORS FOR SPEED
    print("Pre-computing genre matrices...")
    mlb, svd, genre_matrix_multi_hot = prepare_genre_processors(games_df)

    # Personas
    # IMPORTANT: min_positive is a HARD FILTER.
    # Only IndieLover and NicheHunter can interact with games < 100 reviews.
    # This teaches the model that tiny games are generally undesirable.
    MAINSTREAM_MIN_POSITIVE = 100  # Hard floor for most users
    
    personas = {
        # Mainstream Personas (filtered to popular games)
        "ActionFan": {"likes": ["Action", "FPS", "Shooter"], "dislikes": ["Strategy", "Puzzle"], "min_positive": MAINSTREAM_MIN_POSITIVE},
        "RPGPlayer": {"likes": ["RPG", "Adventure", "JRPG"], "dislikes": ["Sports", "Racing"], "min_positive": MAINSTREAM_MIN_POSITIVE},
        "StrategySimFan": {"likes": ["Strategy", "Simulation", "City Builder"], "dislikes": ["FPS", "Action"], "min_positive": MAINSTREAM_MIN_POSITIVE},
        "EclecticExplorer": {"likes": ["Puzzle", "Strategy", "RPG", "Indie"], "dislikes": [], "min_positive": MAINSTREAM_MIN_POSITIVE},
        "TrendFollower": {"likes": [], "dislikes": [], "min_positive": 1000},  # Strongly popular only
        
        # Niche Personas (can interact with tiny games - teaching model these ARE valid for some users)
        "IndieLover": {"likes": ["Indie", "Casual"], "dislikes": ["AAA", "Sports"], "price_max": 20, "min_positive": 0},
        "NicheHunter": {"likes": ["Indie", "Simulation", "Visual Novel"], "dislikes": [], "min_positive": 0, "prefers_obscure": True},
    }

    pairs = []
    
    n_users_per_persona = 200
    n_interactions = 15
    current_time = 1000
    
    print("Generating consistent users...")

    for pname, pdata in personas.items():
        for _ in range(n_users_per_persona):
            user_id = str(uuid.uuid4())
            wishes = []
            skips = []
            
            # Get persona's minimum popularity threshold
            min_pos = pdata.get('min_positive', 0)
            
            # Select potential positive games
            if pname == "TrendFollower":
                candidates = games_df[games_df['positive'] > pdata.get('min_positive', 1000)]
            else:
                mask = games_df['genres'].apply(lambda g: any(l in g for l in pdata['likes']))
                if 'price_max' in pdata:
                    mask = mask & (games_df['price'] <= pdata['price_max'])
                candidates = games_df[mask]
                
                # APPLY POPULARITY FILTER (unless niche persona)
                if min_pos > 0:
                    candidates = candidates[candidates['positive'] >= min_pos]

            if candidates.empty:
                candidates = games_df.sample(n=min(len(games_df), 100))

            # GENRE STAPLES: Sometimes include top game from each liked genre
            # 50% of users get 1 staple per genre (toned down from 80% / 2 staples)
            staple_wishes = []
            if pname != "TrendFollower" and pname != "NicheHunter" and random.random() < 0.50:
                liked_genres = pdata.get('likes', [])
                for genre in liked_genres:
                    genre_mask = games_df['genres'].apply(lambda g: genre in g if isinstance(g, list) else False)
                    genre_games = games_df[genre_mask].sort_values('positive', ascending=False)
                    if not genre_games.empty:
                        # Get top 1 staple per genre (reduced from 2)
                        top_staple = genre_games.head(1)
                        for _, g in top_staple.iterrows():
                            if g.name not in [w['app_id'] for w in staple_wishes]:
                                ts = current_time - random.randint(0, 500)
                                staple_wishes.append({'app_id': g.name, 'interactiontype': 'wishlist', 'timestamp': ts})

            # Wishlist (Positive) - Start with staples, then add random weighted samples
            wishes = staple_wishes.copy()
            n_pos = np.random.randint(3, n_interactions)
            remaining_pos = max(0, n_pos - len(wishes))
            
            if len(candidates) > 0 and remaining_pos > 0:
                # Exclude already wishlisted staples
                already_wishlisted = [w['app_id'] for w in wishes]
                remaining_candidates = candidates[~candidates.index.isin(already_wishlisted)]
                
                if not remaining_candidates.empty:
                    pos_games = remaining_candidates.sample(
                        n=min(len(remaining_candidates), remaining_pos),
                        weights=get_weights(remaining_candidates)
                    )
                    for _, g in pos_games.iterrows():
                        ts = current_time - random.randint(0, 500)
                        wishes.append({'app_id': g.name, 'interactiontype': 'wishlist', 'timestamp': ts})

            # Skips (Negative) - From disliked genres
            n_neg = n_interactions - len(wishes)
            dislikes = pdata.get('dislikes', [])
            if dislikes:
                neg_mask = games_df['genres'].apply(lambda g: any(d in g for d in dislikes))
                neg_candidates = games_df[neg_mask]
            else:
                neg_candidates = games_df

            if neg_candidates.empty:
                neg_candidates = games_df

            if len(neg_candidates) > 0:
                 neg_games = neg_candidates.sample(
                     n=min(len(neg_candidates), n_neg),
                     weights=get_weights(neg_candidates)
                 )
                 for _, g in neg_games.iterrows():
                     ts = current_time - random.randint(0, 500)
                     skips.append({'app_id': g.name, 'interactiontype': 'skip', 'timestamp': ts})
            
            # For mainstream users, add explicit SKIP training for tiny games
            # This teaches: "Even if genre matches, unpopular = skip"
            if min_pos > 0:  # Only for mainstream personas
                tiny_games = games_df[games_df['positive'] < min_pos]
                if not tiny_games.empty and len(tiny_games) > 0:
                    # Sample 3-5 tiny games to skip
                    n_tiny_skip = min(5, len(tiny_games))
                    tiny_sample = tiny_games.sample(n=n_tiny_skip)
                    for _, g in tiny_sample.iterrows():
                        ts = current_time - random.randint(0, 500)
                        skips.append({'app_id': g.name, 'interactiontype': 'skip', 'timestamp': ts})
            
            interactions = wishes + skips
            random.shuffle(interactions)
            
            # Compute User Vector
            # Optimized call using pre-computed processors
            user_vec = compute_user_embedding(
                user_id, 
                pd.DataFrame(interactions), 
                games_df, 
                mlb=mlb, 
                svd=svd, 
                genre_matrix_multi_hot=genre_matrix_multi_hot
            )
            
            # Create training pairs
            for x in interactions:
                aid = x['app_id']
                if aid in game_vectors:
                    label = 1 if x['interactiontype'] == 'wishlist' else 0
                    pairs.append({
                        "user_vector": user_vec.tolist(),
                        "game_vector": game_vectors[aid].tolist(),
                        "label": label
                    })

    # Drifting Users (Fickle)
    print("Generating drifting users...")
    # Define drift patterns: (Old Preferences, New Preferences) 
    # Tuple of (likes, dislikes)
    drifts = [
        # Action -> Strategy (Burnout on action)
        ((["Action", "FPS"], ["Strategy"]), (["Strategy", "Simulation"], ["Action", "FPS"])),
        # RPG -> Sports (Complete switch)
        ((["RPG", "JRPG"], ["Sports"]), (["Sports", "Racing"], ["RPG"])),
        # Indie -> AAA
        ((["Indie"], ["AAA"]), (["AAA"], ["Indie"]))
    ]

    for old_prefs, new_prefs in drifts:
        old_likes, old_dislikes = old_prefs
        new_likes, new_dislikes = new_prefs
        
        for _ in range(50): # 50 users per drift pattern
            user_id = str(uuid.uuid4())
            wishes = []
            skips = []
            
            # PHASE 1: OLD (Time 0-500)
            # Wishlist old likes
            mask_old = games_df['genres'].apply(lambda g: any(l in g for l in old_likes))
            cands_old = games_df[mask_old]
            if not cands_old.empty:
                # 5 old wishlists
                for _, g in cands_old.sample(n=min(len(cands_old), 5), weights=get_weights(cands_old)).iterrows():
                    wishes.append({'app_id': g.name, 'interactiontype': 'wishlist', 'timestamp': random.randint(0, 500)})

            # PHASE 2: NEW (Time 501-1000)
            # Wishlist new likes
            mask_new = games_df['genres'].apply(lambda g: any(l in g for l in new_likes))
            cands_new = games_df[mask_new]
            if not cands_new.empty:
                # 5 new wishlists
                for _, g in cands_new.sample(n=min(len(cands_new), 5), weights=get_weights(cands_new)).iterrows():
                    wishes.append({'app_id': g.name, 'interactiontype': 'wishlist', 'timestamp': random.randint(501, 1000)})
            
            # Skip new dislikes (which are old likes!)
            mask_new_dis = games_df['genres'].apply(lambda g: any(l in g for l in new_dislikes))
            cands_new_dis = games_df[mask_new_dis]
            if not cands_new_dis.empty:
                # 5 new skips
                for _, g in cands_new_dis.sample(n=min(len(cands_new_dis), 5), weights=get_weights(cands_new_dis)).iterrows():
                    skips.append({'app_id': g.name, 'interactiontype': 'skip', 'timestamp': random.randint(501, 1000)})

            interactions = wishes + skips
            random.shuffle(interactions)
            
            # This user vector will now reflect the mix + the recent skip rate on old genres
            user_vec = compute_user_embedding(
                user_id, 
                pd.DataFrame(interactions), 
                games_df, 
                mlb=mlb, 
                svd=svd, 
                genre_matrix_multi_hot=genre_matrix_multi_hot
            )
            
            for x in interactions:
                aid = x['app_id']
                if aid in game_vectors:
                    label = 1 if x['interactiontype'] == 'wishlist' else 0
                    pairs.append({
                        "user_vector": user_vec.tolist(),
                        "game_vector": game_vectors[aid].tolist(),
                        "label": label
                    })

    print(f"Generated {len(pairs)} pairs.")
    
    with open('data/two_tower_training_pairs.pkl', 'wb') as f:
        pickle.dump(pairs, f)
    print("Saved to data/two_tower_training_pairs.pkl")

if __name__ == "__main__":
    generate_pairs()
