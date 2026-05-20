
import json
import numpy as np
import pandas as pd
import random
import uuid
import sys
import os
import pickle
import gc
from scipy.special import expit  # Sigmoid function

from src.models.user_two_tower_embedding import compute_user_embedding, prepare_genre_processors, save_genre_processors
from src.ingest.create_game_embeddings import compute_game_features

# --- CONFIGURATION ---
NUM_USERS = 15000  # Increased to 15k to support DNN learning & Niche archetypes
INTERACTIONS_PER_USER_MEAN = 60
INTERACTIONS_PER_USER_STD = 20
GLOBAL_TIME_START = 0
GLOBAL_TIME_END = 1000

# Feature weights for the "Ground Truth" probability model
# These determine what users actually care about when deciding to wishlist
W_AFFINITY = 4.0    # Genre match (reduced to balance with quality)
W_QUALITY = 4.0     # Rating ratio - higher = better games preferred
W_PRICE = -3.0      # Price is a negative factor
W_POPULARITY = 3.0  # Popularity signal - popular games are more likely to be good
BIAS = -3.0         # Base probability is low (most games are ignored)

# Minimum review threshold - games below this are excluded from training
MIN_REVIEWS_FOR_TRAINING = 10

def load_and_preprocess_games():
    """Load games and compute normalized features for the simulator."""
    print("Loading games from JSON...")
    with open('data/games.json', 'r', encoding='utf-8') as f:
        games_dict = json.load(f)
    
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
            
        # Skip games with insufficient reviews
        positive_reviews = int(data.get('positive', 0))
        if positive_reviews < MIN_REVIEWS_FOR_TRAINING:
            continue
        
        # Skip games with no genres (no useful signal)
        if not genres:
            continue
            
        processed_games.append({
            'app_id': str(app_id),
            'name': data.get('name', ''),
            'genres': genres,
            'price': float(price),
            'positive': int(data.get('positive', 0)),
            'negative': int(data.get('negative', 0))
        })
    
    df = pd.DataFrame(processed_games)
    df.set_index('app_id', inplace=True)
    df['app_id'] = df.index  # Keep as column too
    
    # --- Feature Engineering for Simulation (MATCHING MODEL INPUTS) ---
    
    # 1. Quality & Popularity (Linear, not Product)
    # The model sees these as separate features, so we must simulate them as additive utility
    pos = df['positive'].fillna(0)
    total = pos + df['negative'].fillna(0) + 1e-9
    
    df['sim_rating_ratio'] = (pos / total).fillna(0)
    df['sim_log_popularity'] = np.log1p(total) / 15.0  # Normalized to ~0-1 range
    
    # 2. Price Buckets (Discrete)
    # Model sees: Free, <10, <30, >30. 
    # We map these to discrete penalty levels to match model visibility
    # 0 = Free, 1 = Cheap, 2 = Mid, 3 = AAA
    prices = df['price'].fillna(0).values
    # Buckets: 0-0.01, 10, 30
    price_buckets = np.digitize(prices, bins=[0.01, 10.00, 30.00])
    
    # Map buckets to "Perceived Cost" for simulation (0.0 to 1.0)
    # Free=0.0, Cheap=0.2, Mid=0.5, AAA=1.0
    bucket_costs = np.array([0.0, 0.2, 0.6, 1.0])
    df['sim_price_proxy'] = bucket_costs[price_buckets]
    
    return df

def generate_user_latent_factors(all_genres):
    """Generate a random user profile with latent preferences."""
    profile = {}
    
    # 1. Genre Preferences (Sparse vector)
    # Pick 1-2 "Core" genres they love (clearer signals)
    num_cores = np.random.choice([1, 2], p=[0.5, 0.5])
    core_genres = np.random.choice(all_genres, size=num_cores, replace=False)
    
    # Affinity map: Genre -> Weight [0, 1]
    # Initialize non-core genres to ZERO (cleaner signal, no noise)
    profile['genre_affinity'] = {g: 0.0 for g in all_genres}
    
    # Boost core genres significantly (stronger signal)
    for g in core_genres:
        profile['genre_affinity'][g] = np.random.uniform(0.9, 1.0)
        
    profile['price_sensitivity'] = np.random.beta(2, 5) * 1.5 
    profile['quality_threshold'] = np.random.beta(5, 2)
    
    return profile

def generate_archetype_profile(archetype_name, all_genres):
    """Generate a specific user archetype profile with balanced genre coverage."""
    profile = {
        'genre_affinity': {g: np.random.uniform(0.0, 0.1) for g in all_genres}, 
        'price_sensitivity': np.random.uniform(0.1, 0.9),
        'quality_threshold': 0.7,  # Default
        'min_reviews': 500  # Default for Normal
    }
    
    # Archetype definitions: (core_genres, quality_threshold, min_reviews, max_reviews)
    # max_reviews = None means no upper limit
    ARCHETYPES = {
        # Normal variants (500+ reviews)
        'ActionNormal': (['Action', 'Shooter'], 0.70, 500, None),
        'StrategyNormal': (['Strategy', 'Simulation'], 0.70, 500, None),
        'RPGNormal': (['RPG', 'Adventure'], 0.70, 500, None),
        'IndieNormal': (['Indie', 'Adventure'], 0.65, 500, None),
        'CasualNormal': (['Casual', 'Simulation'], 0.60, 500, None),
        'MMONormal': (['Massively Multiplayer'], 0.65, 500, None),
        'RacingSportsNormal': (['Racing', 'Sports'], 0.70, 500, None),
        'SimBuilderNormal': (['Simulation', 'Strategy'], 0.70, 500, None),
        
        # Popular variants (10k+ reviews)
        'ActionPopular': (['Action', 'Shooter'], 0.80, 10000, None),
        'StrategyPopular': (['Strategy', 'Simulation'], 0.80, 10000, None),
        'RPGPopular': (['RPG', 'Adventure'], 0.80, 10000, None),
        'IndiePopular': (['Indie', 'Adventure'], 0.75, 10000, None),
        'CasualPopular': (['Casual', 'Simulation'], 0.70, 10000, None),
        'MMOPopular': (['Massively Multiplayer'], 0.75, 10000, None),
        'RacingSportsPopular': (['Racing', 'Sports'], 0.80, 10000, None),
        'SimBuilderPopular': (['Simulation', 'Strategy'], 0.80, 10000, None),
        
        # Niche variants (prefer <5k reviews) - NEW
        'ActionNiche': (['Action', 'Shooter'], 0.70, 100, 5000),
        'StrategyNiche': (['Strategy', 'Simulation'], 0.70, 100, 5000),
        'RPGNiche': (['RPG', 'Adventure'], 0.70, 100, 5000),
        'IndieNiche': (['Indie', 'Adventure'], 0.65, 100, 5000),
        'CasualNiche': (['Casual', 'Simulation'], 0.60, 100, 5000),
        'MMONiche': (['Massively Multiplayer'], 0.65, 100, 5000),
        'RacingSportsNiche': (['Racing', 'Sports'], 0.70, 100, 5000),
        'SimBuilderNiche': (['Simulation', 'Strategy'], 0.70, 100, 5000),
        
        # Hybrid variants (cross-genre fans)
        'ActionStrategyFan': (['Action', 'Strategy'], 0.70, 100, None),
        'RPGSimFan': (['RPG', 'Simulation'], 0.70, 100, None),
        'IndieActionFan': (['Indie', 'Action'], 0.70, 100, None),
        'CasualRPGFan': (['Casual', 'RPG'], 0.65, 100, None),
        
        # Software/Creator variants (target non-game apps)
        'SoftwareUser': (['Utilities', 'Web Publishing', 'Software Training', 'Accounting'], 0.60, 10, None),
        'Creator': (['Audio Production', 'Video Production', 'Design & Illustration', 'Photo Editing', 'Animation & Modeling'], 0.70, 10, None),
        
        # Expanded Genre Coverage
        'EarlyAdopter': (['Early Access', 'Indie'], 0.60, 10, None),
        'F2PWhale': (['Free to Play', 'Free To Play', 'Massively Multiplayer'], 0.60, 10, None),
        'MatureGamer': (['Violent', 'Gore', 'Sexual Content', 'Nudity'], 0.60, 10, None),
        'Learner': (['Education', 'Tutorial', 'Game Development'], 0.60, 10, None),
        'Viewer': (['Movie', 'Documentary', '360 Video', 'Short', 'Episodic'], 0.50, 5, None),
    }
    
    if archetype_name in ARCHETYPES:
        cores, quality, min_reviews, max_reviews = ARCHETYPES[archetype_name]
        profile['quality_threshold'] = quality
        profile['min_reviews'] = min_reviews
        profile['max_reviews'] = max_reviews  # None means no limit
        
        for c in cores:
            matching = [g for g in all_genres if c.lower() in g.lower()]
            for m in matching:
                profile['genre_affinity'][m] = np.random.uniform(0.90, 1.0)
    
    return profile

def simulate_browsing_session(user_profile, games_df, num_impressions=200):
    """
    Simulate a user browsing a set of games.
    Returns list of interaction dicts.
    """
    # Get review requirements from profile
    min_reviews = user_profile.get('min_reviews', 0)
    max_reviews = user_profile.get('max_reviews', None)  # None = no limit
    
    # Filter games by min_reviews and max_reviews
    filtered_df = games_df
    if min_reviews > 0:
        filtered_df = filtered_df[filtered_df['positive'] >= min_reviews]
    if max_reviews is not None:
        filtered_df = filtered_df[filtered_df['positive'] <= max_reviews]
    
    # Fall back to all games if filter is too restrictive
    if len(filtered_df) < num_impressions:
        filtered_df = games_df
    
    # 1. Select Candidates (The "Impression" set)
    # Calculate probabilities for this filtered set
    weights = filtered_df['sim_log_popularity'] + 0.05
    p = (weights / weights.sum()).values
    indices = filtered_df.index.values
    
    # Clamp num_impressions to available games
    actual_impressions = min(num_impressions, len(filtered_df))

    # Sample indices
    candidate_indices = np.random.choice(
        indices, 
        size=actual_impressions, 
        p=p, 
        replace=False
    )
    
    candidates = games_df.loc[candidate_indices]
    num_impressions = len(candidates)  # Update for downstream logic
    
    # 2. Vectorized Interaction Logic
    
    # A. Genre Scores
    # Hard to fully vectorize dictionary lookup against list column without padding.
    # List comprehension is fastest for this specific "simulated user" structure.
    
    def calc_genre_score(g_list):
        if not g_list: return 0.0
        # Model Logic: Sum(UserAffinity[g] * (1/NumGenres))
        # optimize: g_list is small (1-5 items)
        s = sum(user_profile['genre_affinity'].get(g, 0) for g in g_list)
        return s / len(g_list)

    # Use map/list comprehension for genre scores (bottleneck, but better than iterrows)
    genre_scores = np.array([calc_genre_score(g) for g in candidates['genres']])
    
    # B. Vectorized Utility
    rating_vals = candidates['sim_rating_ratio'].values
    pop_vals = candidates['sim_log_popularity'].values
    price_scores = candidates['sim_price_proxy'].values
    
    # Calculate Logits (Vectorized)
    logits = BIAS + \
             (genre_scores * W_AFFINITY) + \
             (rating_vals * W_QUALITY) + \
             (pop_vals * W_POPULARITY) + \
             (price_scores * user_profile['price_sensitivity'] * W_PRICE) + \
             np.random.normal(0, 0.1, size=num_impressions)  # Reduced noise from 0.2 to 0.1
             
    probs_wishlist = expit(logits)
    
    # C. Decide Actions (Vectorized)
    rolls = np.random.random(size=num_impressions)
    timestamps = np.random.randint(GLOBAL_TIME_START, GLOBAL_TIME_END, size=num_impressions)
    
    interactions = []
    
    # Wishlists
    wishlist_mask = rolls < probs_wishlist
    if np.any(wishlist_mask):
        w_ids = candidates.index[wishlist_mask]
        w_probs = probs_wishlist[wishlist_mask]
        w_times = timestamps[wishlist_mask]
        
        for aid, prob, ts in zip(w_ids, w_probs, w_times):
            interactions.append({
                'app_id': aid,
                'interactiontype': 'wishlist',
                'timestamp': int(ts), # int for serialization
                'prob': float(prob)
            })
            
    # Skips (Hard Negatives)
    # roll < prob * 4.0 BUT NOT wishlist
    # equivalent to: probs_wishlist <= roll < probs_wishlist * 4.0
    skip_threshold = probs_wishlist * 4.0
    skip_mask = (rolls >= probs_wishlist) & (rolls < skip_threshold)
    
    if np.any(skip_mask):
        s_ids = candidates.index[skip_mask]
        s_probs = probs_wishlist[skip_mask]
        s_times = timestamps[skip_mask]
        
        for aid, prob, ts in zip(s_ids, s_probs, s_times):
            interactions.append({
                'app_id': aid,
                'interactiontype': 'skip',
                'timestamp': int(ts),
                'prob': float(prob)
            })
            
    return interactions

def sample_genre_hard_negatives_batch(user_profile, games_df, interacted_ids, genre_to_indices, num_negatives=4):
    """
    OPTIMIZED: Batch sample hard negatives using pre-computed genre index.
    Returns list of game indices that are hard negatives.
    """
    # 1. Identify User's Top Genres
    preferred_genres = [g for g, w in user_profile['genre_affinity'].items() if w > 0.7]
    if not preferred_genres:
        return []
    
    # Get user's personal thresholds
    quality_threshold = user_profile.get('quality_threshold', 0.7)
    price_sensitivity = user_profile.get('price_sensitivity', 0.5)
    price_threshold = 0.8 - (price_sensitivity * 0.4)
    
    # 2. Get all games in preferred genres (using pre-computed index)
    candidate_indices = set()
    for genre in preferred_genres:
        if genre in genre_to_indices:
            candidate_indices.update(genre_to_indices[genre])
    
    # Remove already interacted games
    candidate_indices = candidate_indices - interacted_ids
    
    if not candidate_indices:
        return []
    
    # 3. Convert to list for random sampling
    candidates = list(candidate_indices)
    random.shuffle(candidates)
    
    # 4. Find hard negatives (low quality OR high price)
    hard_negatives = []
    for idx in candidates[:100]:  # Check at most 100 candidates
        if len(hard_negatives) >= num_negatives:
            break
        try:
            game = games_df.loc[idx]
            # Low quality OR high price
            if game['sim_rating_ratio'] < quality_threshold or game['sim_price_proxy'] > price_threshold:
                hard_negatives.append(idx)
        except:
            continue
    
    return hard_negatives

def sample_wrong_genre_hard_negatives(user_profile, games_df, interacted_ids, genre_to_indices, num_negatives=2):
    """
    Sample HIGH QUALITY games from WRONG genres.
    Forces the model to learn genre discrimination, not just quality.
    """
    # 1. Identify User's Top Genres
    preferred_genres = [g for g, w in user_profile['genre_affinity'].items() if w > 0.7]
    if not preferred_genres:
        return []
    
    # 2. Get all games NOT in preferred genres
    preferred_indices = set()
    for genre in preferred_genres:
        if genre in genre_to_indices:
            preferred_indices.update(genre_to_indices[genre])
    
    # All game indices minus preferred genres
    all_indices = set(games_df.index)
    wrong_genre_indices = all_indices - preferred_indices - interacted_ids
    
    if not wrong_genre_indices:
        return []
    
    # 3. Sample from wrong-genre pool, preferring HIGH QUALITY games
    candidates = list(wrong_genre_indices)
    random.shuffle(candidates)
    
    wrong_genre_negatives = []
    for idx in candidates[:150]:  # Check more candidates
        if len(wrong_genre_negatives) >= num_negatives:
            break
        try:
            game = games_df.loc[idx]
            # HIGH quality (this is the key difference!)
            # We want games that would normally be recommended but are WRONG genre
            if game['sim_rating_ratio'] >= 0.75 and game['sim_log_popularity'] >= 0.3:
                wrong_genre_negatives.append(idx)
        except:
            continue
    
    return wrong_genre_negatives

def generate_pairs():
    print("Starting Advanced Probabilistic Data Generation...")
    
    if os.path.exists('data/two_tower_training_data.npz'):
        print("Training pairs already exist, skipping generation.")
        return
    
    # 1. Load Data
    games_df = load_and_preprocess_games()
    print(f"Loaded {len(games_df)} games.")
    
    # Extract all unique genres
    all_genres = set()
    for g_list in games_df['genres']:
        all_genres.update(g_list)
    all_genres = list(all_genres)
    print(f"Found {len(all_genres)} unique genres.")
    
    # PRE-COMPUTE: Genre -> Game Indices mapping (for fast hard negative sampling)
    print("Building genre index for fast hard negative sampling...")
    genre_to_indices = {}
    for idx, row in games_df.iterrows():
        for genre in row['genres']:
            if genre not in genre_to_indices:
                genre_to_indices[genre] = set()
            genre_to_indices[genre].add(idx)
    print(f"Built index for {len(genre_to_indices)} genres.")
    
    # 2. Compute Game Features & Embeddings (Standard Procedure)
    print("Computing game embeddings...")
    game_vectors_matrix = compute_game_features(games_df)
    
    # Filter out "Ghost Games" (Zero Vectors)
    game_vectors = {}
    ghost_count = 0
    for aid, vec in zip(games_df.index, game_vectors_matrix):
        if np.sum(vec) == 0:
            ghost_count += 1
            continue
        game_vectors[aid] = vec
        
    print(f"Computed embeddings. Dropped {ghost_count} Ghost Games (Zero Vectors). Active Pool: {len(game_vectors)}")
    
    print("Pre-computing genre processors...")
    mlb, genre_matrix_multi_hot = prepare_genre_processors(games_df)
    save_genre_processors(mlb)
    
    # 3. Simulate Users in Parallel (Conceptually)
    # OPTIMIZATION: Process in batches to prevent OOM
    BATCH_SIZE = 500
    
    # Storage for accumulated numpy arrays (efficient)
    user_vec_batches = []
    pos_vec_batches = []
    neg_vec_batches = []
    
    current_batch_pairs = []
    
    print(f"Simulating {NUM_USERS} probabilistic users...")
    
    # Progress tracking
    milestone = NUM_USERS // 10
    
    for i in range(NUM_USERS):
        if i % milestone == 0:
            print(f"Generated {i}/{NUM_USERS} users...")
            
        user_id = str(uuid.uuid4())
        
        # A. Create Profile (90% Archetype, 10% Random)
        # 28 archetypes with balanced distribution
        archetypes = [
            # Normal variants (100+ reviews)
            'ActionNormal', 'StrategyNormal', 'RPGNormal', 'IndieNormal',
            'CasualNormal', 'MMONormal', 'RacingSportsNormal', 'SimBuilderNormal',
            # Popular variants (10k+ reviews)  
            'ActionPopular', 'StrategyPopular', 'RPGPopular', 'IndiePopular',
            'CasualPopular', 'MMOPopular', 'RacingSportsPopular', 'SimBuilderPopular',
            # Niche variants (100-5000 reviews) - NEW
            'ActionNiche', 'StrategyNiche', 'RPGNiche', 'IndieNiche',
            'CasualNiche', 'MMONiche', 'RacingSportsNiche', 'SimBuilderNiche',
            # Hybrid variants (cross-genre fans)
            'ActionStrategyFan', 'RPGSimFan', 'IndieActionFan', 'CasualRPGFan',
            # Creator/Software variants
            'SoftwareUser', 'Creator',
            # Expanded Coverage
            'EarlyAdopter', 'F2PWhale', 'MatureGamer', 'Learner', 'Viewer',
        ]
        
        # 90% archetype, 10% random explorers
        if np.random.rand() < 0.9:
            arch = np.random.choice(archetypes)
            profile = generate_archetype_profile(arch, all_genres)
        else:
            # Random noise for robustness
            profile = generate_user_latent_factors(all_genres)
        
        # B. Simulate Session
        # B. Simulate Session
        # REALISM UPDATE: Use Log-Normal distribution.
        # Real users follow a "long tail": Most interactions are low, some are massive.
        # Normal(60, 20) was too consistent.
        # LogNormal(mean=4.0, sigma=0.8) -> Median ~55, but tail goes to 500+
        num_interactions_target = int(np.random.lognormal(mean=4.0, sigma=0.9))
        
        # Interactions are roughly 20% of impressions (Conversion rate)
        # So we need 5x impressions to get the target interactions
        num_impressions = num_interactions_target * 5
        
        # Clamp to reasonable bounds
        num_impressions = max(20, min(num_impressions, 5000))
        
        interactions = simulate_browsing_session(profile, games_df, num_impressions=num_impressions)
        
        # D. Form Triplets (Advanced: Split Context vs Target)
        # To avoid Leakage (where the User Vector literally contains the Positive Item),
        # we split the interactions:
        # Context (80%) -> Builds the User Vector
        # Target (20%) -> Serves as the Positive Items for training
        
        positives = [x for x in interactions if x['interactiontype'] == 'wishlist']
        negatives = [x for x in interactions if x['interactiontype'] == 'skip']
        
        # Need enough positives to split
        if len(positives) < 5:
            continue
            
        # TEMPORAL SPLIT (Professional Standard)
        # Sort by timestamp to predict FUTURE items from PAST history
        positives.sort(key=lambda x: x['timestamp'])
        
        # PROGRESSIVE SNAPSHOTS: Create training pairs at multiple history stages
        # This teaches the model to generalize from partial user profiles
        # Snapshots at 25%, 50%, 75%, 80% of history
        snapshot_pcts = [0.25, 0.50, 0.75, 0.80]
        
        for pct in snapshot_pcts:
            split_idx = max(2, int(len(positives) * pct))  # At least 2 context items
            if split_idx >= len(positives):
                continue
                
            context_pos = positives[:split_idx]  # PAST
            # Target: next item(s) after this snapshot
            next_idx = min(split_idx + 2, len(positives))  # Take 1-2 items as targets
            target_pos = positives[split_idx:next_idx]   # FUTURE
            
            if not target_pos:
                continue
            
            # Build Context Interactions (All negatives + context positives)
            context_interactions = context_pos + negatives
            context_interactions.sort(key=lambda x: x['timestamp'])
            
            context_df = pd.DataFrame(context_interactions)
            
            try:
                user_vec = compute_user_embedding(
                    user_id,
                    context_df,
                    games_df,
                    mlb=mlb,
                    genre_matrix_multi_hot=genre_matrix_multi_hot
                )
            except Exception as e:
                continue
            
            # Compute interacted_ids for THIS snapshot only (prevents leakage)
            # Only includes context items + target items (not future items beyond target)
            interacted_ids = set([x['app_id'] for x in context_interactions])
            interacted_ids.update([x['app_id'] for x in target_pos])
            
            # Generate Triplets for TARGET items at this snapshot
            for pos in target_pos:
                pos_id = pos['app_id']
                if pos_id not in game_vectors: continue
                pos_vec = game_vectors[pos_id].tolist()
            
                # 1. Pair with Random Negatives
                attempts = 0
                count_random = 0
                while count_random < 2 and attempts < 30:
                    rand_id = random.choice(games_df.index)
                    if rand_id not in interacted_ids and rand_id in game_vectors:
                        current_batch_pairs.append({
                            "user_vector": user_vec.tolist(),
                            "positive_game": pos_vec,
                            "negative_game": game_vectors[rand_id].tolist()
                        })
                        count_random += 1
                    attempts += 1
                    
                # 2. Pair with "Same-Genre Hard Negatives" (low quality)
                hard_neg_ids = sample_genre_hard_negatives_batch(profile, games_df, interacted_ids, genre_to_indices, num_negatives=2)
                for hard_neg_id in hard_neg_ids:
                    if hard_neg_id in game_vectors:
                        current_batch_pairs.append({
                            "user_vector": user_vec.tolist(),
                            "positive_game": pos_vec,
                            "negative_game": game_vectors[hard_neg_id].tolist()
                        })
                
                # 3. Pair with "Wrong-Genre Hard Negatives" (high quality, wrong genre)
                # This forces the model to learn genre discrimination, not just quality
                wrong_neg_ids = sample_wrong_genre_hard_negatives(profile, games_df, interacted_ids, genre_to_indices, num_negatives=2)
                for wrong_neg_id in wrong_neg_ids:
                    if wrong_neg_id in game_vectors:
                        current_batch_pairs.append({
                            "user_vector": user_vec.tolist(),
                            "positive_game": pos_vec,
                            "negative_game": game_vectors[wrong_neg_id].tolist()
                        })
        
        # Check if batch is full
        if len(current_batch_pairs) >= 50000 or i == NUM_USERS - 1:
             # Process batch
             if current_batch_pairs:
                 print(f"Processing batch of {len(current_batch_pairs)} pairs (User {i})...")
                 
                 # Do NOT shuffle within batch - we want to keep User blocks together
                 # random.shuffle(current_batch_pairs)
                 
                 # Convert to numpy
                 b_user = np.array([p['user_vector'] for p in current_batch_pairs], dtype=np.float32)
                 b_pos = np.array([p['positive_game'] for p in current_batch_pairs], dtype=np.float32)
                 b_neg = np.array([p['negative_game'] for p in current_batch_pairs], dtype=np.float32)
                 
                 user_vec_batches.append(b_user)
                 pos_vec_batches.append(b_pos)
                 neg_vec_batches.append(b_neg)
                 
                 # Clear memory
                 current_batch_pairs = []
                 gc.collect()

    print("Concatenating all batches...")
    
    if not user_vec_batches:
        print("No pairs generated!")
        return

    # Extract arrays
    user_vectors = np.concatenate(user_vec_batches)
    pos_vectors = np.concatenate(pos_vec_batches)
    neg_vectors = np.concatenate(neg_vec_batches)

    print(f"Total triplets generated: {len(user_vectors)}")

    print("Saving to data/two_tower_training_data.npz...")
    np.savez_compressed(
        'data/two_tower_training_data.npz',
        user_vectors=user_vectors,
        pos_vectors=pos_vectors,
        neg_vectors=neg_vectors
    )
    print("Done. Saved compressed numpy file.")

if __name__ == "__main__":
    generate_pairs()
