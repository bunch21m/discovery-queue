
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import sys
import psycopg2
from datetime import datetime

from src.ingest.initialize_game_embeddings import build_database_url


# Removing hardcoded USER_INPUT_DIM and GAME_INPUT_DIM
EMBEDDING_DIM = 64  # Reduced to 64 to match model bottleneck (Linear(..., 64))
TEMPERATURE = 0.07  # Temperature for InfoNCE loss (lower = sharper distinctions)


class TripletLoss(nn.Module):
    """
    Triplet Margin Loss for (user, positive_game, negative_game) triplets.
    Encourages: distance(user, positive) + margin < distance(user, negative)
    """
    def __init__(self, margin=0.5):  # Reduced from 1.0 to 0.5
        super().__init__()
        self.margin = margin
    
    def forward(self, user_emb, pos_emb, neg_emb):
        """
        Args:
            user_emb: (batch_size, embedding_dim) - L2 normalized user embeddings
            pos_emb: (batch_size, embedding_dim) - L2 normalized positive game embeddings
            neg_emb: (batch_size, embedding_dim) - L2 normalized negative game embeddings
        """
        # Compute cosine similarities (dot product of normalized vectors)
        pos_similarity = (user_emb * pos_emb).sum(dim=1)  # Higher = more similar
        neg_similarity = (user_emb * neg_emb).sum(dim=1)  # Should be lower
        
        # Triplet margin loss: we want pos_similarity > neg_similarity + margin
        # Loss = max(0, margin - (pos_similarity - neg_similarity))
        loss = F.relu(self.margin - (pos_similarity - neg_similarity))
        
        return loss.mean()


class TwoTowerModel(nn.Module):
    def __init__(self, user_input_dim, game_input_dim):
        super(TwoTowerModel, self).__init__()
        
        # UPGRADED: Deep Neural Network (DNN)
        # 2-Layer MLP allows learning non-linear feature interactions (e.g. XOR problems)
        self.user_tower = nn.Sequential(
            nn.Linear(user_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM)
        )
        
        self.game_tower = nn.Sequential(
            nn.Linear(game_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM)
        )
        
    def forward(self, user_features, game_features):
        user_emb = self.user_tower(user_features)
        game_emb = self.game_tower(game_features)
        
        # Normalize embeddings for cosine similarity focus
        user_emb = F.normalize(user_emb, p=2, dim=1)
        game_emb = F.normalize(game_emb, p=2, dim=1)
        
        return user_emb, game_emb
    
    def compute_similarity(self, user_emb, game_emb):
        """Compute dot product similarity (for inference)."""
        return (user_emb * game_emb).sum(dim=1) * 10.0  # Scale for BCE compatibility

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

    data_path = 'data/two_tower_training_data.npz'
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Run generation script first.")
        return

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading training data from Numpy...")
    with np.load(data_path) as data:
        user_vectors = data['user_vectors']
        pos_vectors = data['pos_vectors']
        neg_vectors = data['neg_vectors']
    
    print(f"Loaded {len(user_vectors)} triplets.")

    if len(user_vectors) == 0:
        print("No pairs found.")
        return

    # Determine dimensions dynamically
    user_dim = user_vectors.shape[1]
    game_dim = pos_vectors.shape[1]
    
    print(f"Detected Dimensions - User: {user_dim}, Game: {game_dim}")

    # Convert to Tensors
    print("Converting to Tensors...")
    train_user_tensor = torch.tensor(user_vectors)
    train_pos_tensor = torch.tensor(pos_vectors)
    train_neg_tensor = torch.tensor(neg_vectors)

    # Split Train/Val
    # SHUFFLE before split to ensure similar distributions in train and val
    # Previous sequential split caused validation to have different user archetypes
    dataset_size = len(user_vectors)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)  # CRITICAL: Shuffle to mix all user types
    
    split_idx = int(0.8 * dataset_size)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = torch.utils.data.TensorDataset(
        train_user_tensor[train_indices], 
        train_pos_tensor[train_indices], 
        train_neg_tensor[train_indices]
    )
    val_dataset = torch.utils.data.TensorDataset(
        train_user_tensor[val_indices], 
        train_pos_tensor[val_indices], 
        train_neg_tensor[val_indices]
    )

    # Check for empty datasets
    if len(train_dataset) == 0:
        print("Not enough data to train.")
        return

    # Reduced batch size for more gradient noise (regularization effect)
    BATCH_SIZE = 512
    print(f"Using batch size: {BATCH_SIZE}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    model = TwoTowerModel(user_dim, game_dim).to(device)
    # Increased margin to 0.5 for harder training task
    criterion = TripletLoss(margin=0.5)
    # Simple model can handle higher LR without memorizing
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    epochs = 30 
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5 

    print(f"Starting triplet training (margin=0.5, embedding_dim={EMBEDDING_DIM})...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for user_feat, pos_game, neg_game in train_loader:
            # Move batch to device
            user_feat = user_feat.to(device)
            pos_game = pos_game.to(device)
            neg_game = neg_game.to(device)
            
            optimizer.zero_grad()
            
            # Get embeddings
            user_emb, pos_emb = model(user_feat, pos_game)
            _, neg_emb = model(user_feat, neg_game)  # Reuse user features
            
            loss = criterion(user_emb, pos_emb, neg_emb)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
                
                # Accuracy
                pos_sim = (user_emb * pos_emb).sum(dim=1)
                neg_sim = (user_emb * neg_emb).sum(dim=1)
                correct += (pos_sim > neg_sim).sum().item()
                total += user_feat.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
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
                print(f"Early stopping at epoch {epoch+1} (no improvement for {max_patience} epochs)")
                break

    print("Training complete. Model saved to data/two_tower_model.pth")
    
    # Save training timestamp
    save_training_timestamp()
    print("Training timestamp saved.")
    
    # Clear game embeddings table
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
