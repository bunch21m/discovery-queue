
import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import psycopg2
from psycopg2.extras import execute_values


from src.models.train_two_tower_model import TwoTowerModel
from src.ingest.create_game_embeddings import compute_game_features
from src.ingest.initialize_game_embeddings import build_database_url

def load_data_and_model():
    # Load Games
    print("Loading games...")
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
            
        processed_games.append({
            'app_id': str(app_id),
            'name': data.get('name', ''),
            'genres': genres,
            'price': float(price),
            'positive': int(data.get('positive', 0))
        })
    
    games_df = pd.DataFrame(processed_games)
    games_df.set_index('app_id', inplace=True)
    games_df['app_id'] = games_df.index # Ensure column existence for compute function using app_id

    # Compute raw features
    print("Computing game features...")
    game_features = compute_game_features(games_df)
    game_dim = game_features.shape[1]
    
    # Load Model
    print("Loading model...")

    
    state_dict = torch.load('data/two_tower_model.pth')
    
    # Infer dimensions from state dict
    # game_tower.0.weight shape is (64, game_input_dim)
    # user_tower.0.weight shape is (64, user_input_dim)
    
    fw_weight = state_dict['game_tower.0.weight']
    saved_game_dim = fw_weight.shape[1]
    
    uw_weight = state_dict['user_tower.0.weight']
    saved_user_dim = uw_weight.shape[1]
    
    print(f"Inferred dimensions from model file - User: {saved_user_dim}, Game: {saved_game_dim}")
    
    if saved_game_dim != game_dim:
         print(f"WARNING: Computed game feature dim {game_dim} != Model expected {saved_game_dim}")
         # This might happen if SVD behaved differently.
    
    model = TwoTowerModel(saved_user_dim, saved_game_dim)
    model.load_state_dict(state_dict)
    model.eval()
    
    return games_df, game_features, model

def store_embeddings():
    # Check if data exists first to avoid expensive computation
    db_url = build_database_url()
    try:
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM gameEmbeddings;")
            count = cur.fetchone()[0]
        conn.close()
        
        if count > 0:
            print(f"Database already contains {count} embeddings. Skipping storage.")
            return
    except Exception as e:
        print(f"Warning: Failed to check existing embeddings: {e}")

    games_df, game_features, model = load_data_and_model()
    
    # Generate Embeddings
    print("Generating embeddings...")
    game_tensor = torch.tensor(game_features, dtype=torch.float32)
    with torch.no_grad():
        embeddings = model.game_tower(game_tensor)
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    embeddings_np = embeddings.numpy()
    


    
    db_url = build_database_url()
    conn = psycopg2.connect(db_url)
    
    try:
        with conn.cursor() as cur:
            # Clear existing data before inserting new embeddings
            cur.execute("TRUNCATE TABLE gameEmbeddings;")
            
            print("Inserting embeddings into DB...")
            
            values = []
            for i, (app_id, row) in enumerate(games_df.iterrows()):
                vec = embeddings_np[i].tolist()
                # app_id is index
                values.append((str(app_id), vec))
                
            execute_values(cur, 
                           "INSERT INTO gameEmbeddings (appid, embedding) VALUES %s", 
                           values)
            
            conn.commit()
            print(f"Stored {len(values)} embeddings.")
            
    finally:
        conn.close()

if __name__ == "__main__":
    store_embeddings()
