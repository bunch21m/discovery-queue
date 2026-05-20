
import math
import random
import time
import numpy as np
import pandas as pd
import torch
import psycopg2

from src.models.two_tower_candidate_pooler import TwoTowerRecommender
from src.db.user_functions import get_user_by_username, add_user_to_database
from src.db.interaction_functions import (
    get_users_interactions_from_database, 
    add_user_interaction_to_database,
    delete_user_interaction_from_database
)
from src.db.tools.auto_wishlister import set_user_config, score_game, action_for_score
from src.db.tools.game_functions import load_all_games_from_database
from src.models.user_two_tower_embedding import concat_user_features
from src.ingest.initialize_game_embeddings import build_database_url
from src.models.train_two_tower_model import TwoTowerModel


def get_test_item_positions(user_embedding, test_app_ids):
    """Query DB to find where test items rank in cosine distance from user embedding."""
    db_url = build_database_url()
    conn = psycopg2.connect(db_url)
    positions = {}
    
    try:
        with conn.cursor() as cur:
            user_vec_str = str(user_embedding.tolist())
            
            for test_id in test_app_ids:
                # Get ranking of this specific game
                query = """
                SELECT COUNT(*) + 1 as position
                FROM gameEmbeddings
                WHERE embedding <=> %s::vector < (
                    SELECT embedding <=> %s::vector 
                    FROM gameEmbeddings 
                    WHERE appid = %s
                );
                """
                cur.execute(query, (user_vec_str, user_vec_str, str(test_id)))
                row = cur.fetchone()
                if row:
                    positions[test_id] = row[0]
                else:
                    positions[test_id] = None
    except Exception as e:
        print(f"Error getting positions: {e}")
    finally:
        conn.close()
    
    return positions

def calculate_ndcg(rank):
    """Calculates NDCG for a single item (Leave-One-Out)."""
    if rank <= 0 or rank > 10:
        return 0.0
    return 1.0 / math.log2(rank + 1)

def run_evaluation(username='ndcg_action_eval', num_samples=10):
    print(f"--- Starting NDCG@10 Evaluation for user: {username} ---")
    
    # 1. Ensure user exists
    user = get_user_by_username(username)
    if not user:
        print(f"Creating evaluation user '{username}'...")
        add_user_to_database(username)
        user = get_user_by_username(username)
    
    user_id = user['userid']
    
    # 2. Configure Auto-Wishlister for "StrategyGamer" Profile
    # STRICT: Only likes Strategy/Simulation, strong penalties otherwise
    config = {
        "positive": [
            # STRONG popularity requirement - penalize low-review games
            {"comparison": "lt", "value": 500, "score": -200},   # Hard block <500
            {"comparison": "lt", "value": 2000, "score": -100},  # Penalty <2000
            {"comparison": "lt", "value": 5000, "score": -50},   # Mild penalty <5000
            {"comparison": "gt", "value": 5000, "score": 30},    # Bonus 5k+
            {"comparison": "gt", "value": 20000, "score": 20},   # Extra bonus 20k+
        ],
        "genres": [
            # STRICT: Only Strategy and Simulation get high scores
            {"value": "strategy", "score": 150},      
            {"value": "simulation", "score": 150},    
            
            
            # Strong penalties - must have Strategy/Sim to overcome
            {"value": "action", "score": -100},       
            {"value": "shooter", "score": -150},      
            {"value": "fps", "score": -150},
            {"value": "casual", "score": -80},
            {"value": "sports", "score": -150},
            {"value": "racing", "score": -150},
            {"value": "adventure", "score": -60},
            {"value": "rpg", "score": -40},
        ],
        "price": [{"comparison": "lt", "value": 40, "score": 10}]
    }
    set_user_config(username, config)
    
    # 3. Populate Interactions - require minimum WISHLISTS
    MIN_WISHLISTS_REQUIRED = 10  # Reduced from 20
    existing_interactions = get_users_interactions_from_database(user_id)
    existing_wishlists = [
        row for row in existing_interactions.values() 
        if row['interactiontype'] == 'wishlist'
    ]
    
    if len(existing_wishlists) < MIN_WISHLISTS_REQUIRED:
        print(f"Populating interactions (need {MIN_WISHLISTS_REQUIRED} wishlists, have {len(existing_wishlists)})...")
        all_games = load_all_games_from_database()
        # Sort by popularity first, then shuffle within tiers to add variety
        game_list = sorted(all_games.values(), key=lambda g: g.get('positive', 0), reverse=True)
        # Only consider top 20,000 by popularity (matches retrieval pool size)
        game_list = game_list[:20000]
        random.shuffle(game_list)
        
        wishlist_count = len(existing_wishlists)
        interaction_count = 0
        max_interactions = 500  # Cap total to avoid infinite loop
        
        for game in game_list:
            if wishlist_count >= MIN_WISHLISTS_REQUIRED or interaction_count >= max_interactions:
                break
            score = score_game(game, config)
            action = action_for_score(score)
            add_user_interaction_to_database(game['app_id'], user_id, action)
            if action == 'wishlist':
                wishlist_count += 1
            interaction_count += 1
        
        print(f"Generated {interaction_count} interactions, {wishlist_count} wishlists")
        existing_interactions = get_users_interactions_from_database(user_id)
    
    wishlisted_app_ids = [
        str(row['appid']) for row in existing_interactions.values() 
        if row['interactiontype'] == 'wishlist'
    ]
    
    if len(wishlisted_app_ids) < 10:
        print(f"Not enough wishlisted items ({len(wishlisted_app_ids)}) to run a meaningful evaluation.")
        return

    # --- DIAGNOSTIC: Print wishlisted game details ---
    print(f"\n--- Wishlisted Games ({len(wishlisted_app_ids)} total) ---")
    all_games = load_all_games_from_database()
    for app_id in wishlisted_app_ids[:15]:  # Show first 15
        game = all_games.get(app_id, {})
        name = game.get('name', 'Unknown')[:40]
        genres = game.get('genres', [])[:4]  # First 4 genres
        positive = game.get('positive', 0)
        print(f"  {app_id}: {name} | Genres: {genres} | Reviews: {positive}")
    if len(wishlisted_app_ids) > 15:
        print(f"  ... and {len(wishlisted_app_ids) - 15} more")
    print("--- End Wishlisted Games ---\n")

    # 4. Temporal Split Evaluation
    
    # Needs robust interactions with timestamps
    all_interactions = list(existing_interactions.values())
    
    # Sort by timestamp (simulate if missing)
    # If timestamps are all 0 or missing, we can't do temporal.
    # But our generator adds random ints.
    all_interactions.sort(key=lambda x: x.get('timestamp', 0))
    
    # If timestamps are identical/bogus, fallback to random shuffle to simulate "sequence"
    if all_interactions[-1].get('timestamp', 0) == all_interactions[0].get('timestamp', 0):
       random.shuffle(all_interactions)
       
    # Identify "Wishlists" in the sequence
    wishlist_sequence = [x for x in all_interactions if x['interactiontype'] == 'wishlist']
    
    if len(wishlist_sequence) < 10:
        print("Not enough wishlists for temporal split (need >10).")
        return

    # Split: Hold out the LAST 5 items (The "Future")
    # Train: Everything before that
    TEST_SIZE = 5
    test_set = wishlist_sequence[-TEST_SIZE:]
    train_set = [x for x in all_interactions if x not in test_set] # Keep negatives in train if they happened before
    
    # Currently `concat_user_features` reads directly from DB. 
    # To do a strict Temporal Split, we must simulate "Ignoring" the test items during embedding generation.
    # The `TwoTowerRecommender` doesn't easily support "masking" interactions without deleting them.
    # SO: We will DELETE the test items from DB temporarily, compute embedding, then RESTORE them.
    
    print(f"\nRunning Temporal Split Evaluation (Last {TEST_SIZE} items)...")
    print(f"Test Set (The 'Future'): {[x['appid'] for x in test_set]}")
    
    # 1. Hide Test Set from DB
    hidden_interactions = []
    for interaction in test_set:
        app_id = str(interaction['appid'])
        # Hack: Delete and store for restore
        # Ideally we'd pass 'excluded_ids' to the embedding function, but that requires deeper refactor.
        delete_user_interaction_from_database(app_id, user_id)
        hidden_interactions.append(interaction)
    
    recommender = TwoTowerRecommender()
    
    stats = {
        'ndcg': [],
        'hits': 0,
        'recalls': 0
    }
    
    try:
        # 2. Generate Recommendations (based on 'Past' only)
        # We start ONE recommendation session.
        # Can we retreive ALL test items in a single top-K list?
        # Standard Recommender asks for k=10. 
        # Metric: Recall@10 (How many of the 5 test items are in the top 10?)
        # Metric: NDCG@10 (Ranking quality of those items)
        
        print("Generating recommendations based on 'Past' history...")
        # Get top 250 for Hit Rate@250 evaluation
        recommendations = recommender.get_recommendations(username, k=250)
        
        rec_app_ids = [str(r['app_id']) for r in recommendations]
        print(f"Top 10 Recommendations: {rec_app_ids[:10]}")
        
        # DEBUG: Query DB directly for test item positions in Two-Tower ranking
        # Compute user embedding to match what Two-Tower uses
        test_app_ids_list = [str(x['appid']) for x in test_set]
        try:
            user_features = concat_user_features(username)
            user_tensor = torch.tensor(user_features, dtype=torch.float32).unsqueeze(0)
            
            # Load model and compute embedding
            model = TwoTowerModel(user_input_dim=73, game_input_dim=44)
            checkpoint = torch.load('data/two_tower_model.pth', map_location='cpu')
            # Handle both formats: direct state_dict or wrapped in 'model' key
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            
            with torch.no_grad():
                user_embedding = model.user_tower(user_tensor)
                user_embedding = torch.nn.functional.normalize(user_embedding, p=2, dim=1)
            
            user_emb_np = user_embedding.numpy().flatten()
            positions = get_test_item_positions(user_emb_np, test_app_ids_list)
            
            print(f"\n--- Test Item Positions in Two-Tower Pool (out of 42k) ---")
            for test_id in test_app_ids_list:
                pos = positions.get(test_id)
                if pos:
                    print(f"  {test_id}: Position {pos}")
                else:
                    print(f"  {test_id}: NOT FOUND in embeddings")
        except Exception as e:
            print(f"Error computing positions: {e}")
        
        # 3. Evaluate
        # For 'Session' evaluation, we treat the Test Set as the "Ground Truth" relevant items.
        # We calculate NDCG considering these 5 items as Relevance=1, others=0.
        
        dcg = 0.0
        idcg = 0.0
        hits_10 = 0
        hits_100 = 0
        hits_250 = 0
        
        test_app_ids = set([str(x['appid']) for x in test_set])
        
        # Calculate DCG and Recall at various K
        for i, rec_id in enumerate(rec_app_ids):
            if rec_id in test_app_ids:
                if i < 10:
                    hits_10 += 1
                    dcg += 1.0 / math.log2((i + 1) + 1)
                if i < 100:
                    hits_100 += 1
                if i < 250:
                    hits_250 += 1
                
        # Calculate Ideal DCG (Best possible ranking)
        for i in range(min(len(test_set), 10)):
            idcg += 1.0 / math.log2((i + 1) + 1)
            
        ndcg = dcg / idcg if idcg > 0 else 0.0
        recall_10 = hits_10 / len(test_set)
        recall_100 = hits_100 / len(test_set)
        recall_250 = hits_250 / len(test_set)
        

        # Genre Precision - NARROW target to Strategy/Simulation only (not Indie/RPG which are too broad)
        target_genres = {'strategy', 'simulation', 'rts', '4x', 'turn-based', 'city builder', 'management'}
        genre_matches = 0
        
        # DEBUG: Check what genres are actually in recommendations
        print("\nDEBUG: Top 10 recommendation genres:")
        for i, rec in enumerate(recommendations[:10]):
            rec_genres = rec.get('genres', [])
            rec_genres_lower = set(g.lower() for g in rec_genres) if rec_genres else set()
            match = '✓' if rec_genres_lower & target_genres else '✗'
            print(f"  {i+1}. {rec.get('app_id')}: {rec_genres} {match}")
            if rec_genres_lower & target_genres:
                genre_matches += 1
        
        genre_precision = genre_matches / 10
        
        # MORE METRIC: Compare to Random Baseline
        # Calculate what % of ALL games are in the target genres (random baseline)
        random_genre_matches = 0
        total_games = len(all_games)
        for game in all_games.values():
            game_genres = set(g.lower() for g in game.get('genres', []))
            if game_genres & target_genres:
                random_genre_matches += 1
        random_baseline = random_genre_matches / total_games if total_games > 0 else 0
        
        # Lift = How much better is our model than random?
        lift = genre_precision / random_baseline if random_baseline > 0 else 0
        
        print(f"\n--- Temporal Evaluation Results ---")
        print(f"Goal: Retrieve {len(test_set)} specific future items from 42k+ candidates.")
        print(f"Hits@10: {hits_10} | Hits@100: {hits_100} | Hits@250: {hits_250}")
        print(f"NDCG@10: {ndcg:.4f}")
        print(f"Recall@10: {recall_10:.4f} | Recall@100: {recall_100:.4f} | Recall@250: {recall_250:.4f}")
        
        # Random baseline for Hit Rate@250: P(hit) = 250/42000 * 5 test items = ~3%
        random_hit_baseline = (250 / 42000) * len(test_set)
        hit_rate_lift = hits_250 / random_hit_baseline if random_hit_baseline > 0 else 0
        
        print(f"\n--- MORE METRICS ---")
        print(f"Genre Precision@10: {genre_precision:.1%}")
        print(f"Random Baseline: {random_baseline:.1%}")
        print(f"Lift over Random: {lift:.1f}x improvement")
        print(f"\nHit Rate@250 Random Baseline: {random_hit_baseline:.2f} expected hits")
        print(f"Hit Rate@250 Lift: {hit_rate_lift:.1f}x vs random")
        
    finally:
        # 4. Restore DB State
        print("Restoring test interactions...")
        for interaction in hidden_interactions:
            add_user_interaction_to_database(interaction['appid'], user_id, interaction['interactiontype'])

if __name__ == "__main__":
    run_evaluation()
