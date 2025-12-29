from src.db.userFunctions import getUserByUsername
from src.db.interactionFunctions import getUsersInteractionsFromDatabase
from src.db.gameFunctions import getGameFromDatabase
from src.models.CommonModelUtils import loadAllGamesFromDatabase

import numpy as np
import pandas as pd
import hashlib

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer

# -------- Configurable hyperparameters --------
ID_EMB_DIM = 16      # deterministic ID embedding size  
GENRE_PROJ_DIM = 64  # projected dimension for genres
# ---------------------------------------------




def embedUserIdDeterministic(userIdSeries):
    """Vectorized hashing of AppIDs to deterministic floats."""
    def hashId(aid):
        hashBytes = hashlib.md5(str(aid).encode('utf-8')).digest()
        # Convert first N bytes to floats and normalize
        ints = np.frombuffer(hashBytes, dtype=np.uint8)[:ID_EMB_DIM]
        vec = ints.astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)
    
    return np.stack(userIdSeries.apply(hashId).values)

def concatUserFeatures(username):
    """
    Concatenates user features for the two-tower embedding model.

    Args:
            username (str): The username of the user to process.
    """
    
    
    user = getUserByUsername(username)
    if not user:
        raise ValueError(f"User {username} not found in database")

    
    feature_list = []
    
    # ID Embedding
    userIdEmb = embedUserIdDeterministic(pd.Series([user['userid']]))
    
    interactions = getUsersInteractionsFromDatabase(user['userid'])
    interactionsdf = pd.DataFrame.from_dict(interactions, orient='index')
    
    # get all games to create multihot genre feature
    games = loadAllGamesFromDatabase() 
    gamesdf = pd.DataFrame.from_dict(games, orient='index')
    gamesdf['appID'] = gamesdf.index

    # Genre Processing
    # First create multi-hot encoding for all games
    # Use svd to reduce dimensionality
    # Apply same transformation to user genre preferences
    
    
    
    mlb = MultiLabelBinarizer(sparse_output=False)
    # Handle missing genres
    genres = gamesdf['genres'].apply(lambda x: x if isinstance(x, list) else [])
    
    genreMatrixMultiHot = mlb.fit_transform(genres)

    # Apply dimensionality reduction globally
    actualDim = min(GENRE_PROJ_DIM, genreMatrixMultiHot.shape[1] - 1)
    if actualDim > 0:
        svd = TruncatedSVD(n_components=actualDim, random_state=42)
        genreFeatures = svd.fit_transform(genreMatrixMultiHot)
    else:
        genreFeatures = genreMatrixMultiHot

    
    wishlistedGames = []
    userGenres = np.zeros(genreMatrixMultiHot.shape[1], dtype=np.float32)

    # Count genres from wishlisted games and fill wishlistedGames
    for _, row in interactionsdf.iterrows():
        appid = row['appid']
        interactionType = row['interactiontype']

        if appid not in gamesdf.index:
            continue

        if interactionType == 'wishlist':
            game = getGameFromDatabase(appid)
            if game:
                wishlistedGames.append(game)
                userGenres += genreMatrixMultiHot[gamesdf.index.get_loc(appid)]

    # Apply same dimensionality transformation to user genre preferences
    if userGenres.sum() > 0 and actualDim > 0:
        userGenreEmbedding = svd.transform(userGenres.reshape(1, -1)).flatten()
    else:
        userGenreEmbedding = np.zeros(actualDim if actualDim > 0 else genreMatrixMultiHot.shape[1])

    wishlistedGamesDf = pd.DataFrame(wishlistedGames)
    
    # Wishlist Rate
    wishlistRate = sum(interactionsdf['interactiontype'] == 'wishlist') / len(interactionsdf) if len(interactionsdf) > 0 else 0.0
    
    # Bucket wishlisted games by price ranges (Free, <10, <30, >=30)
    if not wishlistedGamesDf.empty and 'price' in wishlistedGamesDf.columns:
        freeWishlisted = (wishlistedGamesDf['price'] == 0).sum()
        lowWishlisted = ((wishlistedGamesDf['price'] > 0) & (wishlistedGamesDf['price'] < 10)).sum()
        midWishlisted = ((wishlistedGamesDf['price'] >= 10) & (wishlistedGamesDf['price'] < 30)).sum()
        highWishlisted = (wishlistedGamesDf['price'] >= 30).sum()
    else:
        freeWishlisted = lowWishlisted = midWishlisted = highWishlisted = 0

    # recent skip rate(last 20 interactions)
    recentInteractions = interactionsdf.sort_values(by='timestamp', ascending=False).head(20)
    recentSkipRate = sum(recentInteractions['interactiontype'] == 'skip') / len(recentInteractions) if len(recentInteractions) > 0 else 0.0
    
    
    
    # Concatenate all features
    feature_list.append(userIdEmb[0])
    print(f"userIdEmb: {userIdEmb[0]}")
    feature_list.append(userGenreEmbedding)
    print(f"userGenreEmbedding: {userGenreEmbedding}")
    feature_list.append(wishlistRate)
    print(f"wishlistRate: {wishlistRate}")
    feature_list.append(freeWishlisted)
    print(f"freeWishlisted: {freeWishlisted}")
    feature_list.append(lowWishlisted)
    print(f"lowWishlisted: {lowWishlisted}")
    feature_list.append(midWishlisted)
    print(f"midWishlisted: {midWishlisted}")
    feature_list.append(highWishlisted)
    print(f"highWishlisted: {highWishlisted}")
    feature_list.append(recentSkipRate)
    print(f"recentSkipRate: {recentSkipRate}")  
    

    

    return np.concatenate(
    [
        userIdEmb[0].astype(np.float32),
        userGenreEmbedding.astype(np.float32),
        np.array(
            [
                wishlistRate,
                freeWishlisted,
                lowWishlisted,
                midWishlisted,
                highWishlisted,
                recentSkipRate,
            ],
            dtype=np.float32,
        ),
    ],
    axis=0,
).astype(np.float32)
    
    