import os
import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values

from src.models.train_two_tower_model import TwoTowerModel, TripletLoss
from src.models.user_two_tower_embedding import compute_user_embedding, prepare_genre_processors
from src.ingest.create_game_embeddings import compute_game_features
from src.ingest.initialize_game_embeddings import build_database_url
from src.db.interaction_functions import (
    get_all_interactions_after_timestamp_from_database, 
    get_users_interactions_from_database,
    get_interactions_for_users_from_database
)
from src.db.tools.game_functions import load_all_games_from_database
from src.ingest.store_game_embeddings import generate_and_store_all_game_embeddings

# Match initial training parameters
TRIPLET_MARGIN = 0.33


def get_last_training_timestamp():
    """Get the timestamp of the last training run."""
    timestamp_file = 'data/last_training_timestamp.txt'
    if os.path.exists(timestamp_file):
        with open(timestamp_file, 'r') as f:
            return f.read().strip()
    # If no timestamp file exists, use a default (e.g., 30 days ago)
    default_timestamp = (datetime.now() - timedelta(days=30)).isoformat()
    return default_timestamp

def save_training_timestamp():
    """Save the current timestamp as the last training time."""
    timestamp_file = 'data/last_training_timestamp.txt'
    current_timestamp = datetime.now().isoformat()
    with open(timestamp_file, 'w') as f:
        f.write(current_timestamp)

def load_existing_model():
    """Load the existing model and determine dimensions."""
    if not os.path.exists('data/two_tower_model.pth'):
        print("No existing model found. Please run initial training first.")
        return None, None, None

    # Load model state dict to infer dimensions
    state_dict = torch.load('data/two_tower_model.pth')

    fw_weight = state_dict['game_tower.0.weight']
    saved_game_dim = fw_weight.shape[1]

    uw_weight = state_dict['user_tower.0.weight']
    saved_user_dim = uw_weight.shape[1]

    model = TwoTowerModel(saved_user_dim, saved_game_dim)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded existing model with dimensions - User: {saved_user_dim}, Game: {saved_game_dim}")
    return model, saved_user_dim, saved_game_dim

def generate_triplets_from_new_interactions(last_timestamp):
    """
    Generate training TRIPLETS from new interactions since last training.
    Uses same triplet structure as initial training for consistency.
    """

    print(f"Fetching interactions after {last_timestamp}...")

    # Get new interactions
    new_interactions = get_all_interactions_after_timestamp_from_database(last_timestamp)

    if not new_interactions:
        print("No new interactions found since last training.")
        return []

    print(f"Found {len(new_interactions)} new interactions.")

    # Group interactions by user
    user_interactions = {}
    for interaction in new_interactions.values():
        user_id = interaction['userid']
        if user_id not in user_interactions:
            user_interactions[user_id] = []
        user_interactions[user_id].append(interaction)

    # Load games data from database
    print("Loading games data...")
    raw_data = load_all_games_from_database()
    games_df = pd.DataFrame.from_dict(raw_data, orient='index')
    games_df['app_id'] = games_df.index

    # Compute game features
    print("Computing game features...")
    game_vectors_matrix = compute_game_features(games_df)
    game_vectors = {aid: vec for aid, vec in zip(games_df.index, game_vectors_matrix)}
    
    # Get list of valid game IDs for negative sampling
    valid_game_ids = list(game_vectors.keys())

    # PRE-COMPUTE GENRE PROCESSORS FOR SPEED
    print("Pre-computing genre matrices...")
    mlb, genre_matrix_multi_hot = prepare_genre_processors(games_df)
    
    triplets = []

    # Fetch ALL interactions for ALL relevant users in one go
    user_id_list = list(user_interactions.keys())
    all_users_all_interactions = get_interactions_for_users_from_database(user_id_list)

    # Generate triplets for each user with new interactions
    for user_id, new_user_interactions in user_interactions.items():
        # Get ALL interactions for this user
        all_interactions = all_users_all_interactions.get(user_id, {})
        all_interactions_df = pd.DataFrame.from_dict(all_interactions, orient='index')

        # Compute user embedding using ALL of this user's interactions
        try:
            user_vec = compute_user_embedding(
                user_id,
                all_interactions_df,
                games_df,
                mlb=mlb,
                genre_matrix_multi_hot=genre_matrix_multi_hot
            )

            # Separate positive (wishlist) interactions
            positives = [i for i in new_user_interactions if i['interactiontype'] == 'wishlist']
            
            if not positives:
                continue  # Skip users with no wishlists in new interactions
                
            # Get set of all interacted games for this user
            interacted_ids = set(str(i['appid']) for i in all_interactions.values()) if all_interactions else set()

            # Create triplets: (user, positive_game, negative_game)
            for pos_interaction in positives:
                pos_id = str(pos_interaction['appid'])
                if pos_id not in game_vectors:
                    continue
                    
                pos_vec = game_vectors[pos_id]
                
                # Sample random negatives (matching initial training strategy)
                num_negatives = 5  # Match initial training
                attempts = 0
                neg_count = 0
                
                while neg_count < num_negatives and attempts < 60:
                    rand_id = random.choice(valid_game_ids)
                    if rand_id not in interacted_ids and rand_id in game_vectors:
                        triplets.append({
                            "user_vector": user_vec.tolist(),
                            "pos_vector": pos_vec.tolist(),
                            "neg_vector": game_vectors[rand_id].tolist()
                        })
                        neg_count += 1
                    attempts += 1

        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            continue

    print(f"Generated {len(triplets)} training triplets from new interactions.")
    return triplets



def train_on_new_interactions():
    """Main function to train the model on new interactions using TripletLoss."""

    print("Starting incremental training on new interactions...")

    # Get last training timestamp
    last_timestamp = get_last_training_timestamp()
    print(f"Last training timestamp: {last_timestamp}")

    # Load existing model
    model, user_dim, game_dim = load_existing_model()
    if model is None:
        return

    # Generate triplets from new interactions
    triplets = generate_triplets_from_new_interactions(last_timestamp)

    if not triplets:
        print("No new training data available.")
        return

   
    # Split Train/Val
    random.shuffle(triplets)
    split_idx = int(0.8 * len(triplets))
    
    # Vectorize data
    print("Vectorizing data...")
    user_vectors = np.array([t['user_vector'] for t in triplets], dtype=np.float32)
    pos_vectors = np.array([t['pos_vector'] for t in triplets], dtype=np.float32)
    neg_vectors = np.array([t['neg_vector'] for t in triplets], dtype=np.float32)

    # Convert to Tensors
    all_user_tensor = torch.tensor(user_vectors)
    all_pos_tensor = torch.tensor(pos_vectors)
    all_neg_tensor = torch.tensor(neg_vectors)

    train_indices = list(range(split_idx))
    val_indices = list(range(split_idx, len(triplets)))

    train_dataset = torch.utils.data.TensorDataset(
        all_user_tensor[train_indices], 
        all_pos_tensor[train_indices], 
        all_neg_tensor[train_indices]
    )
    val_dataset = torch.utils.data.TensorDataset(
        all_user_tensor[val_indices], 
        all_pos_tensor[val_indices], 
        all_neg_tensor[val_indices]
    )

    # Check for empty datasets
    if len(train_dataset) == 0:
        print("Not enough data to train.")
        return

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Match initial training batch size
    BATCH_SIZE = 512
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    if len(train_loader) == 0:
        print("Not enough data for a full batch. Skipping training.")
        return

    model.to(device)

    # Use TripletLoss matching initial training
    criterion = TripletLoss(margin=TRIPLET_MARGIN)
    # Lower learning rate for incremental training to avoid catastrophic forgetting
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    epochs = 20  # Fewer epochs for incremental training
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5

    print(f"Starting triplet training (margin={TRIPLET_MARGIN})...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for user_feat, pos_game, neg_game in train_loader:
            user_feat = user_feat.to(device)
            pos_game = pos_game.to(device)
            neg_game = neg_game.to(device)

            optimizer.zero_grad()
            
            # Get embeddings
            user_emb, pos_emb = model(user_feat, pos_game)
            _, neg_emb = model(user_feat, neg_game)
            
            loss = criterion(user_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for user_feat, pos_game, neg_game in val_loader:
                user_feat = user_feat.to(device)
                pos_game = pos_game.to(device)
                neg_game = neg_game.to(device)

                user_emb, pos_emb = model(user_feat, pos_game)
                _, neg_emb = model(user_feat, neg_game)
                
                loss = criterion(user_emb, pos_emb, neg_emb)
                val_loss += loss.item()
                
                # Accuracy: positive similarity > negative similarity
                pos_sim = (user_emb * pos_emb).sum(dim=1)
                neg_sim = (user_emb * neg_emb).sum(dim=1)
                correct += (pos_sim > neg_sim).sum().item()
                total += user_feat.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Step the learning rate scheduler
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'data/two_tower_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training complete. Model saved to data/two_tower_model.pth")
    
    # Load all games and compute features
    raw_data = load_all_games_from_database()
    games_df = pd.DataFrame.from_dict(raw_data, orient='index')
    games_df['app_id'] = games_df.index
    game_features = compute_game_features(games_df)

    # Repopulate database with fresh embeddings
    generate_and_store_all_game_embeddings(model, games_df, game_features)
    
    
    # Save new timestamp
    save_training_timestamp()
    print("Training timestamp updated.")

    print("Incremental training complete!")

if __name__ == "__main__":
    train_on_new_interactions()
