import random
import os
import sys
import time
import torch
import psycopg2
import pandas as pd
from psycopg2.extras import RealDictCursor


from src.models.train_two_tower_model import TwoTowerModel
from src.models.user_two_tower_embedding import compute_user_embedding, concat_user_features
from src.ingest.initialize_game_embeddings import build_database_url
from src.db.tools.game_functions import get_game_from_database, get_games_from_database, load_all_games_from_database
from src.db.user_functions import get_user_by_username
from src.db.interaction_functions import get_users_interactions_from_database
from src.models.reranker import mmr_rerank, get_intralist_diversity
from src.models.lightgbm_lambdamart import get_top_k_recommendations

class TwoTowerRecommender:
    def __init__(self, model_path='data/two_tower_model.pth'):
        self.model = None
        self.model_path = model_path
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Model file {self.model_path} not found.")
            return

        try:
            state_dict = torch.load(self.model_path)
            
            # Infer dimensions
            fw_weight = state_dict['game_tower.0.weight']
            saved_game_dim = fw_weight.shape[1]
            
            uw_weight = state_dict['user_tower.0.weight']
            saved_user_dim = uw_weight.shape[1]
            
            self.model = TwoTowerModel(saved_user_dim, saved_game_dim)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("TwoTowerModel loaded successfully.")
        except Exception as e:
            print(f"Failed to load user model: {e}")

    def get_recommendations(self, username, k=10):
        """
        Generates recommendations for a user.
        Args:
             username (str): Username to recommend for.
             k (int): Number of recommendations.
        Returns:
             tuple: (recommended_game_names, recommended_app_ids, recommended_movie_urls)
        """
        if not self.model:
            print("Model not loaded, falling back or returning empty.")
            return [], [], []

        # Fetch seen games to exclude
        seen_app_ids = []
        has_interactions = False
        try:
             user_data = get_user_by_username(username)
             if user_data:
                 user_id = user_data['userid']
                 interactions = get_users_interactions_from_database(user_id)
                 seen_app_ids = [str(row['appid']) for row in interactions.values()]
                 has_interactions = len(seen_app_ids) > 0
        except Exception as e:
             print(f"Error fetching interactions for filtering: {e}")

        # COLD START
        if not has_interactions:
            print("Inserting preset games for cold start")
            cold_start_app_ids = [
                "730",      # Placeholder 1 (CS:GO)
                "570",      # Placeholder 2 (Dota 2)
                "646570",      # Placeholder 3 (Slay the Spire)
                "271590",   # Placeholder 4 (GTA V)
                "2124490",   # Placeholder 5 (Silent Hill 2)
                "250320",   # Placeholder 6 (The Wolf Among Us)
                "1017900",  # Placeholder 7 (Age of Empires: Definitive Edition)
                "292030",   # Placeholder 8 (Witcher 3)
                "1245620",  # Placeholder 9 (Elden Ring)
                "413150",   # Placeholder 10 (Stardew Valley)
            ]
            cold_start_games = self._fetch_cold_start_games(cold_start_app_ids)

            final_games = []
            for app_id in cold_start_app_ids:
                if app_id in cold_start_games:
                    final_games.append(cold_start_games[app_id])

            return final_games

        # 1. Compute User Embedding
        try:
            # We use the wrapper function which handles fetching and computing
            user_features_np = concat_user_features(username)
            
            
            # Convert to tensor and batch dim
            user_features = torch.tensor(user_features_np, dtype=torch.float32).unsqueeze(0)
            
            # Pass through User Tower
            with torch.no_grad():
                user_embedding = self.model.user_tower(user_features)
                user_embedding = torch.nn.functional.normalize(user_embedding, p=2, dim=1)
            

            
            user_vec = user_embedding.numpy().flatten().tolist()
            
            # 2. ANN Retrieval from DB
            # Pool size to determine how many candidates to fetch
            
            print("Retrieving nearest neighbors...")
            # Reduced pool size from 20,000 to 10,000 to improve latency while keeping safe recall
            pool_size = 10000 
            start_time = time.perf_counter()
            candidates = self._retrieve_nearest_neighbors(user_vec, pool_size, exclude_app_ids=seen_app_ids)
            print(f"Time it took to retrieve {len(candidates)} candidates: {time.perf_counter() - start_time:.2f} seconds")
            
            if not candidates:
                return {}
            
            print(f"Top 10 candidates from two-tower: {[list(candidates.keys())[i] for i in range(min(10, len(candidates)))]}")
            
            # LambdaMART scoring
            # Score top candidates using LambdaMART
            lambdamart_model_path = 'data/lambdamart_model.txt'
            if os.path.exists(lambdamart_model_path):
                print("Applying LambdaMART scoring...")
                start_time = time.perf_counter()
                
                # Use cached data to avoid loading 300MB+ of games on every request
                from src.models.user_two_tower_embedding import get_cached_game_data
                games_df, _ = get_cached_game_data()
                
                # Get user interactions for feature computation
                interactions = get_users_interactions_from_database(user_id)
                interactions_df = pd.DataFrame.from_dict(interactions, orient='index')
                if 'appid' in interactions_df.columns:
                    interactions_df = interactions_df.rename(columns={'appid': 'app_id'})
                
                # Get top K from two-tower as candidates for LambdaMART
                candidate_app_ids = list(candidates.keys())[:5000]
                
                # Extract distances for personalization feature
                candidate_distances = {
                    aid: candidates[aid].get('distance', 1.0) 
                    for aid in candidate_app_ids
                }
                
                # Score with LambdaMART - get top 500 for MMR   
                lambdamart_results = get_top_k_recommendations(
                    user_id=user_id,
                    interactions_df=interactions_df,
                    games_df=games_df,
                    candidate_app_ids=candidate_app_ids,
                    candidate_distances=candidate_distances,  # NEW: pass distances
                    k=500,
                    model_path=lambdamart_model_path
                )
                
                print(f"LambdaMART scoring took {time.perf_counter() - start_time:.2f} seconds")
                
                # Reorder candidates dict based on LambdaMART scores
                if lambdamart_results:
                    reordered_candidates = {}
                    for app_id, score in lambdamart_results:
                        if app_id in candidates:
                            candidates[app_id]['lambdamart_score'] = score
                            reordered_candidates[app_id] = candidates[app_id]
                    
                    # Add remaining candidates that weren't in LambdaMART results
                    for app_id in candidates:
                        if app_id not in reordered_candidates:
                            reordered_candidates[app_id] = candidates[app_id]
                    
                    candidates = reordered_candidates
                    print(f"Top 10 after LambdaMART: {list(candidates.keys())[:10]}")
            else:
                print(f"LambdaMART model not found at {lambdamart_model_path}, skipping LambdaMART scoring")
            
            # === MMR Diversity Reranking ===
            original_top10 = list(candidates.values())[:10]
            div_before = get_intralist_diversity(original_top10)
            print(f"Intralist diversity before MMR: {div_before}")
            
            # Rerank using MMR - rerank top 500 to match LambdaMART output
            candidate_list = list(candidates.values())
            mmr_pool = candidate_list[:500]
            mmr_reranked = mmr_rerank(mmr_pool, lambda_param=0.5, num_recommendations=k)
            
            div_after = get_intralist_diversity(mmr_reranked[:10])
            print(f"Top 10 candidates after MMR: {[mmr_reranked[i]['app_id'] for i in range(min(10, len(mmr_reranked)))]}")
            print(f"Intralist diversity after MMR: {div_after}")
            
            # Save metrics for analysis
            self.latest_metrics = {
                'diversity_before': div_before,
                'diversity_after': div_after,
                'diversity_improvement': div_after - div_before
            }
                
            # Hybrid Strategy: Top 5 Nearest + 5 Random from the rest
            # We assume k=10 usually. We'll split k roughly in half.
            
            # n_nearest = min(5, k)
            # n_random = k - n_nearest
            
            # # If total candidates are less than k, just return all
            # if len(candidates) <= k:
            #     return candidates
                
            # # 1. Top Nearest (Indices 0 to n_nearest-1)
            # final_indices = list(range(n_nearest))
            
            # # 2. Random from the rest (Indices n_nearest to end)
            # remaining_indices = list(range(n_nearest, len(candidates)))
            
            # if n_random > 0 and remaining_indices:
            #      # Sample random indices from the remainder
            #      # Ensure we don't sample more than available
            #      n_to_sample = min(n_random, len(remaining_indices))
            #      random_indices = random.sample(remaining_indices, n_to_sample)
            #      final_indices.extend(random_indices)
            
            return mmr_reranked
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return {}   
        
    def _retrieve_nearest_neighbors(self, vector, k, exclude_app_ids=None):
        db_url = build_database_url()
        conn = psycopg2.connect(db_url)
        results = []
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # HNSW ef_search MUST be >= k to retrieve k results
                # pgvector caps ef_search at 1000
                ef_search = min(max(k, 400), 1000)
                cur.execute(f"SET hnsw.ef_search = {ef_search};")
                
                query_params = [str(vector)]
                filter_clause = ""
                
                if exclude_app_ids:
                    placeholders = ','.join(['%s'] * len(exclude_app_ids))
                    filter_clause = f"WHERE appid NOT IN ({placeholders})"
                    query_params.extend(exclude_app_ids)
                
                query_params.append(k)
                
                query = f"""
                SELECT appid, embedding <=> %s::vector AS distance
                FROM gameEmbeddings
                {filter_clause}
                ORDER BY distance ASC
                LIMIT %s;
                """
                
                cur.execute(query, tuple(query_params))
                rows = cur.fetchall()
                results = [(row['appid'], row['distance']) for row in rows]
                
        except Exception as e:
            print(f"DB Retrieval error: {e}")
        finally:
            conn.close()
            
        # 3. Fetch Game Details (Name, Movie)
        # We can use common utility or just query `games` table again.
        # Since we just have IDs, let's load all games or query individually.
        # Loading all might be cached/fast enough since we did it elsewhere.
        # Or faster: query `games` table for these IDs.
        
        games_data = {}
        
        # Let's batch query the games table
        if results:
            games_data = self._fetch_game_details(results)
        
        return games_data

    def _fetch_game_details(self, app_ids_with_distances):
        """
        Fetches game details from IN-MEMORY CACHE (Lazy Hydration).
        Avoids hitting the DB for 10k items.
        """
        from src.models.user_two_tower_embedding import get_cached_game_data
        
        # Ensure cache is loaded
        games_df, _ = get_cached_game_data()
        
        games_data = {}
        
        # Batch lookup from DataFrame (Hash Map)
        # much faster than SQL IN (...)
        for item in app_ids_with_distances:
            if isinstance(item, tuple):
                aid, dist = item
            else:
                aid, dist = item, None
            
            # Use string ID for lookup
            aid_str = str(aid)
            
            if aid_str in games_df.index:
                # Convert row to dict
                # Note: This gives us 'genres', 'prices', etc.
                # It suffices for LambdaMART and MMR.
                game_dict = games_df.loc[aid_str].to_dict()
                game_dict['app_id'] = aid_str # Ensure explicitly set
                game_dict['distance'] = dist
                games_data[aid_str] = game_dict
                
        return games_data

    def _fetch_cold_start_games(self, app_ids):
        """
        Fetches game details for cold start app IDs.
        Returns tuple: (game_names, app_ids, movie_urls)
        """
        games_data = self._fetch_game_details(app_ids)
        
        return games_data
