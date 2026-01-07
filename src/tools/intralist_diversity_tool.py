
import torch
import time

from src.models.two_tower_candidate_pooler import TwoTowerRecommender
from src.models.user_two_tower_embedding import concat_user_features
from src.models.reranker import mmr_rerank, get_intralist_diversity
from src.db.user_functions import get_user_by_username
from src.db.interaction_functions import get_users_interactions_from_database, add_user_interaction_to_database
from src.db.tools.auto_wishlister import score_game_for_user, action_for_score, set_user_config

def run_auto_swipe(username='test', total_recommendations=1000, batch_size=10):
    print(f"Starting auto-swiper for user: {username}")
    
    # 0. Setup Auto-Wishlister Config
    config = {
        "positive": [
            {"comparison": "gt", "value": 1000, "score": 5},
            {"comparison": "lt", "value": 100, "score": -15}  # Penalize games with very few reviews
        ],
        "price": [
            {"comparison": "lt", "value": 1000, "score": 5},
            {"comparison": "gt", "value": 3000, "score": -10}  # Penalize expensive games
        ],
        "negative": [
            {"comparison": "gt", "value": 500, "score": -10}  # Penalize games with many negative reviews
        ],
        "positive_ratio": [
            {"comparison": "lt", "value": 0.7, "score": -15}  # Penalize games with less than 70% positive
        ],
        "genres": [
            {"value": "rpg", "score": 5},
            {"value": "adventure", "score": 5},
            {"value": "casual", "score": -5},
            {"value": "sports", "score": -5}
        ]
    }
    set_user_config(username, config)
    print("User configuration set for auto-wishlister.")
    recommender = TwoTowerRecommender()
    
    # Check model loaded
    if not recommender.model:
        print("Model failed to load.")
        return

    # Loop
    num_batches = total_recommendations // batch_size
    
    diversity_total_before = 0.0
    diversity_total_after = 0.0
    
    for i in range(num_batches):
        print(f"\n--- Batch {i+1}/{num_batches} ---")
        
        # 1. Get user and interactions
        user_data = get_user_by_username(username)
        if not user_data:
            print(f"User {username} not found")
            return
        user_id = user_data['userid']
        interactions = get_users_interactions_from_database(user_id)
        seen_app_ids = [str(row['appid']) for row in interactions.values()]
        
        # 2. Compute embedding
        try:
            user_features_np = concat_user_features(username)
            user_features = torch.tensor(user_features_np, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                user_embedding = recommender.model.user_tower(user_features)
                user_embedding = torch.nn.functional.normalize(user_embedding, p=2, dim=1)
            user_vec = user_embedding.numpy().flatten().tolist()
        except Exception as e:
            print(f"Error computing user embedding: {e}")
            continue
        
        # 3. Retrieve neighbors
        # We can use the private method _retrieve_nearest_neighbors
        # excluding already seen games
        pool_size = 10000 
        candidates = recommender._retrieve_nearest_neighbors(user_vec, pool_size, exclude_app_ids=seen_app_ids)
        
        if not candidates:
            print("No candidates found.")
            break
            
        # 4. Diversity Before
        original_top10 = list(candidates.values())[:batch_size]
        div_before = get_intralist_diversity(original_top10)
        print(f"Diversity Before Rerank: {div_before}")
        diversity_total_before += div_before
        
        # 5. Rerank
        mmr_pool = list(candidates.values())[:100]
        mmr_reranked = mmr_rerank(mmr_pool, lambda_param=0.3, num_recommendations=batch_size)
        
        # 6. Diversity After
        div_after = get_intralist_diversity(mmr_reranked)
        print(f"Diversity After Rerank: {div_after}")
        diversity_total_after += div_after
        
        # 7. Swipe
        for game in mmr_reranked:
            app_id = game['app_id']
            try:
                score = score_game_for_user(game, username)
                action = action_for_score(score)
                #print(f"Game: {game.get('name')} (ID: {app_id}), Score: {score:.2f}, Action: {action}")
                
                add_user_interaction_to_database(app_id, user_id, action)
            except Exception as e:
                print(f"Error processing game {app_id}: {e}")
    
    print("\n--- Auto-Swiper Summary ---")
    print(f"Average Diversity Before Rerank: {diversity_total_before / num_batches}")
    print(f"Average Diversity After Rerank: {diversity_total_after / num_batches}")
    print(f"Diversity Improvement Percentage: {(diversity_total_after - diversity_total_before) / diversity_total_before * 100:.2f}%")
if __name__ == "__main__":
    run_auto_swipe()
