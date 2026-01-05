import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values

from src.models.train_two_tower_model import TwoTowerModel, TwoTowerDataset
from src.models.user_two_tower_embedding import compute_user_embedding, prepare_genre_processors
from src.ingest.create_game_embeddings import compute_game_features
from src.ingest.initialize_game_embeddings import build_database_url
from src.db.interaction_functions import get_all_interactions_after_timestamp_from_database, get_users_interactions_from_database
from src.db.tools.game_functions import load_all_games_from_database
from src.ingest.store_game_embeddings import generate_and_store_all_game_embeddings

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

def generate_pairs_from_new_interactions(last_timestamp):
    """Generate training pairs from new interactions since last training."""

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

    # Load games data from database (since all games are already there)
    print("Loading games data...")
    raw_data = load_all_games_from_database()
    games_df = pd.DataFrame.from_dict(raw_data, orient='index')
    games_df['app_id'] = games_df.index

    # Compute game features
    print("Computing game features...")
    game_vectors_matrix = compute_game_features(games_df)
    game_vectors = {aid: vec for aid, vec in zip(games_df.index, game_vectors_matrix)}

    # PRE-COMPUTE GENRE PROCESSORS FOR SPEED
    print("Pre-computing genre matrices...")
    mlb, svd, genre_matrix_multi_hot = prepare_genre_processors(games_df)
    
    pairs = []

    # Generate pairs for each user with new interactions
    for user_id, new_interactions in user_interactions.items():
        # Get ALL interactions for this user to compute proper user embedding with multihot genres
        all_interactions = get_users_interactions_from_database(user_id)
        all_interactions_df = pd.DataFrame.from_dict(all_interactions, orient='index')

        # Compute user embedding using ALL of this user's interactions
        # This ensures the multihot genre encoding reflects all wishlisted genres across user's history
        try:
            user_vec = compute_user_embedding(
                user_id,
                all_interactions_df,
                games_df,
                mlb=mlb,
                svd=svd,
                genre_matrix_multi_hot=genre_matrix_multi_hot
            )

            # Create positive and negative pairs from new interactions
            for interaction in new_interactions:
                app_id = str(interaction['appid'])
                if app_id in game_vectors:
                    label = 1 if interaction['interactiontype'] == 'wishlist' else 0
                    pairs.append({
                        "user_vector": user_vec.tolist(),
                        "game_vector": game_vectors[app_id].tolist(),
                        "label": label
                    })

        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            continue

    print(f"Generated {len(pairs)} training pairs from new interactions.")
    return pairs



def train_on_new_interactions():
    """Main function to train the model on new interactions."""

    print("Starting incremental training on new interactions...")

    # Get last training timestamp
    last_timestamp = get_last_training_timestamp()
    print(f"Last training timestamp: {last_timestamp}")

    # Load existing model
    model, user_dim, game_dim = load_existing_model()
    if model is None:
        return

    # Generate pairs from new interactions
    pairs = generate_pairs_from_new_interactions(last_timestamp)

    if not pairs:
        print("No new training data available.")
        return

   
   # Split Train/Val
    np.random.shuffle(pairs)
    split_idx = int(0.8 * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    train_dataset = TwoTowerDataset(train_pairs)
    val_dataset = TwoTowerDataset(val_pairs)

    # Check for empty datasets
    if len(train_pairs) == 0:
        print("Not enough data to train.")
        return

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    # Lower learning rate for incremental training to avoid catastrophic forgetting
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 

    epochs = 50
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for user_feat, game_feat, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(user_feat, game_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for user_feat, game_feat, labels in val_loader:
                outputs = model(user_feat, game_feat)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'data/two_tower_model.pth')

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
