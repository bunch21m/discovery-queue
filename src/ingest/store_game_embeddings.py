
import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import psycopg2
from psycopg2.extras import execute_values


from src.models.train_two_tower_model import TwoTowerModel
from src.models.user_two_tower_embedding import load_genre_processors
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
    mlb = load_genre_processors()
    if mlb:
        print("Loaded genre processors from disk for game embedding.")
        game_features = compute_game_features(games_df, mlb=mlb)
    else:
        print("WARNING: No genre processors found on disk. Fitting NEW ones (Consistency risk!).")
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
    
    print(f"Dimensions from model file - User: {saved_user_dim}, Game: {saved_game_dim}")
    
    if saved_game_dim != game_dim:
         print(f"WARNING: Computed game feature dim {game_dim} != Model expected {saved_game_dim}")
         # This might happen if SVD behaved differently.
    
    model = TwoTowerModel(saved_user_dim, saved_game_dim)
    model.load_state_dict(state_dict)
    model.eval()
    
    return games_df, game_features, model

def generate_and_store_all_game_embeddings(model, games_df, game_features):
    """
    Generates embeddings for all games and stores them in the database.
    """
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
                values.append((str(app_id), vec))
                
            execute_values(cur, 
                           "INSERT INTO gameEmbeddings (appid, embedding) VALUES %s", 
                           values)
            
            conn.commit()
            print(f"Stored {len(values)} embeddings.")
            
    finally:
        conn.close()

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
    
    # DEBUG: Print game feature dimensions and sample values
    print(f"DEBUG: Game features shape: {game_features.shape}")
    print(f"DEBUG: Game features first sample (first 10): {game_features[0, :10]}")
    print(f"DEBUG: Game features first sample (last 10): {game_features[0, -10:]}")
    
    game_tensor = torch.tensor(game_features, dtype=torch.float32)
    with torch.no_grad():
        embeddings = model.game_tower(game_tensor)
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    print(f"DEBUG: Game embedding (first sample, first 5): {embeddings[0, :5].tolist()}")
    
    embeddings_np = embeddings.numpy()
    


    
    db_url = build_database_url()
    conn = psycopg2.connect(db_url)
    
    embedding_dim = embeddings_np.shape[1]
    print(f"Embedding dimension: {embedding_dim}")
    
    try:
        with conn.cursor() as cur:
            # Always drop and recreate to ensure correct dimensions
            print(f"Dropping and recreating gameEmbeddings table with {embedding_dim} dimensions...")
            cur.execute("DROP TABLE IF EXISTS gameEmbeddings;")
            cur.execute(f"""
                CREATE TABLE gameEmbeddings (
                    id bigserial PRIMARY KEY,
                    appid VARCHAR(255),
                    embedding vector({embedding_dim})
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS game_embeddings_appid_idx ON gameEmbeddings (appid);")
            
            # Create HNSW index for vector search
            print("Creating HNSW index...")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS game_embeddings_embedding_idx 
                ON gameEmbeddings USING hnsw (embedding vector_cosine_ops);
            """)
            conn.commit()
            
            print("Inserting embeddings into DB...")
            
            values = []
            skipped_count = 0
            low_review_count = 0
            MIN_REVIEWS = 10  # Minimum positive reviews required to be in candidate pool
            
            for i, (app_id, row) in enumerate(games_df.iterrows()):
                # CHECK 1: Minimum Review Threshold
                # Games with too few reviews are low-quality and should not be recommended
                positive_reviews = row.get('positive', 0)
                if positive_reviews < MIN_REVIEWS:
                    low_review_count += 1
                    continue
                
                # CHECK 2: Zero Feature Vector (Ghost Games)
                if np.sum(game_features[i]) == 0:
                    skipped_count += 1
                    continue
                
                # CHECK 3: Empty Genres (no signal for genre-based recommendation)
                genres = row.get('genres', [])
                if not genres or len(genres) == 0:
                    skipped_count += 1
                    continue

                vec = embeddings_np[i].tolist()
                # app_id is index
                values.append((str(app_id), vec))
            
            print(f"Skipped {low_review_count} games with < {MIN_REVIEWS} reviews.")
            print(f"Skipped {skipped_count} games with ZERO features (Ghost Games).")
                
            execute_values(cur, 
                           "INSERT INTO gameEmbeddings (appid, embedding) VALUES %s", 
                           values)
            
            conn.commit()
            print(f"Stored {len(values)} embeddings.")
            
    finally:
        conn.close()

if __name__ == "__main__":
    store_embeddings()
