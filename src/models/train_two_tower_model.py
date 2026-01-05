
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import psycopg2
from datetime import datetime

from src.ingest.initialize_game_embeddings import build_database_url


# Removing hardcoded USER_INPUT_DIM and GAME_INPUT_DIM
EMBEDDING_DIM = 32

class TwoTowerModel(nn.Module):
    def __init__(self, user_input_dim, game_input_dim):
        super(TwoTowerModel, self).__init__()
        
        self.user_tower = nn.Sequential(
            nn.Linear(user_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, EMBEDDING_DIM)
        )
        
        self.game_tower = nn.Sequential(
            nn.Linear(game_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, EMBEDDING_DIM)
        )
        
    def forward(self, user_features, game_features):
        user_emb = self.user_tower(user_features)
        game_emb = self.game_tower(game_features)
        
        # Normalize embeddings for cosine similarity focus (optional but good for retrieval)
        user_emb = torch.nn.functional.normalize(user_emb, p=2, dim=1)
        game_emb = torch.nn.functional.normalize(game_emb, p=2, dim=1)
        
        # Dot product
        # Element-wise multiply and sum
        dot_product = (user_emb * game_emb).sum(dim=1)
        return dot_product

class TwoTowerDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        return (
            torch.tensor(item['user_vector'], dtype=torch.float32),
            torch.tensor(item['game_vector'], dtype=torch.float32),
            torch.tensor(item['label'], dtype=torch.float32)
        )

def save_training_timestamp():
    """Save the current timestamp as the last training time."""
    timestamp_file = 'data/last_training_timestamp.txt'
    current_timestamp = datetime.now().isoformat()
    with open(timestamp_file, 'w') as f:
        f.write(current_timestamp)

def train_model():
    if os.path.exists('data/two_tower_model.pth'):
        print("Model already exists, skipping training.")
        return

    data_path = 'data/two_tower_training_pairs.pkl'
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Run generation script first.")
        return

    print("Loading training data...")
    with open(data_path, 'rb') as f:
        pairs = pickle.load(f)
    print(f"Loaded {len(pairs)} pairs.")

    if len(pairs) == 0:
        print("No pairs found.")
        return

    # Determine dimensions dynamically
    sample_user = pairs[0]['user_vector']
    sample_game = pairs[0]['game_vector']
    
    user_dim = len(sample_user)
    game_dim = len(sample_game)
    
    print(f"Detected Dimensions - User: {user_dim}, Game: {game_dim}")

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

    model = TwoTowerModel(user_dim, game_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    
    # Save training timestamp for incremental training
    save_training_timestamp()
    print("Training timestamp saved.")
    
    # Clear game embeddings table to ensure fresh embeddings are inserted
    print("Truncating gameEmbeddings table for fresh embeddings...")
    try:
        db_url = build_database_url()
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE gameEmbeddings;")
        conn.commit()
        conn.close()
        print("gameEmbeddings table truncated successfully.")
    except Exception as e:
        print(f"Warning: Could not truncate gameEmbeddings: {e}")

if __name__ == "__main__":
    train_model()
