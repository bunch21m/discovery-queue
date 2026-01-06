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
from src.db.tools.game_functions import get_game_from_database, get_games_from_database
from src.db.user_functions import get_user_by_username
from src.db.interaction_functions import get_users_interactions_from_database
from src.models.reranker import mmr_rerank

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
            pool_size = 10000
            start_time = time.perf_counter()
            candidates = self._retrieve_nearest_neighbors(user_vec, pool_size, exclude_app_ids=seen_app_ids)
            print(f"Time it took to retrieve {pool_size} candidates: {time.perf_counter() - start_time:.2f} seconds")
            
            if not candidates:
                return {}
            
            
            
            print(f"Top 10 candidates before rerank: {[list(candidates.keys())[i] for i in range(min(10, len(candidates)))]}")
            # Rerank using MMR - only rerank top 200 to keep it fast
            candidate_list = list(candidates.values())
            mmr_pool = candidate_list[:200]
            mmr_reranked = mmr_rerank(mmr_pool, lambda_param=0.5, num_recommendations=k)
            print(f"Top 10 candidates after rerank: {[mmr_reranked[i]['app_id'] for i in range(min(10, len(mmr_reranked)))]}")
                
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
            
            # Construct final result
            final_games =  []
            for idx in range(len(mmr_reranked)):
                app_id = list(candidates.keys())[idx]
                game_data = candidates[app_id]
                final_games.append(game_data)
            
            return final_games
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return {}   
        
    def _retrieve_nearest_neighbors(self, vector, k, exclude_app_ids=None):
        db_url = build_database_url()
        conn = psycopg2.connect(db_url)
        results = []
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # pgvector cosine distance operator is <=>
                
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
        Fetches game details in batch.
        """
        app_ids = []
        distances = {}
        for item in app_ids_with_distances:
            if isinstance(item, tuple):
                aid, dist = item
            else:
                aid, dist = item, None
            app_ids.append(aid)
            distances[aid] = dist
            
        games_data = get_games_from_database(app_ids)
        
        # Add the distance back to the objects
        for aid, game in games_data.items():
            game['distance'] = distances.get(aid)
                
        return games_data

    def _fetch_cold_start_games(self, app_ids):
        """
        Fetches game details for cold start app IDs.
        Returns tuple: (game_names, app_ids, movie_urls)
        """
        games_data = self._fetch_game_details(app_ids)
        
        return games_data
