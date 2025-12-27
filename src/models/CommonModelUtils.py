# Holds common utility functions and constants for models

import os
import json


def readSecret(path):
    """
    Reads a secret from a file.

    Args:
            path (str): The path to the secret file.
    Returns:
            str: The secret read from the file.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return None

def loadAllGamesFromJSON(pathToDataset: str):
    """
    Loads all games from a JSON dataset file.

    Args:
            pathToDataset (str): The path to the JSON dataset file.
    Returns:
            dict: A dictionary of all games loaded from the dataset holding many features.
    """
    dataset = {}
    if os.path.exists(pathToDataset):
        with open(pathToDataset, 'r', encoding='utf-8') as fin:
            text = fin.read()
            if len(text) > 0:
                dataset = json.loads(text)

    for appID in dataset:
        dataset[appID]['appID'] = appID
    
    return dataset


def loadAllGamesFromDatabase():
    """
    Loads all games from the PostgreSQL database.

    Returns:
            dict: A dictionary of all games loaded from the database.
    """
    dataset = {}
    
    user = readSecret('/run/secrets/postgres_user')
    password = readSecret('/run/secrets/postgres_password')
    dbName = readSecret('/run/secrets/postgres_db')
    
    if not (user and password and dbName):
        raise RuntimeError("Database credentials not found")

    dbUrl = f"postgresql://{user}:{password}@db:5432/{dbName}"

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(dbUrl)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT appid, name, positive, tags, data FROM games")
            rows = cur.fetchall()
            for row in rows:
                appid = row.get('appid')
                game = {
                    'name': row.get('name'),
                    'positive': int(row.get('positive') or 0),
                    'tags': row.get('tags') or []
                }
                if row.get('data'):
                    game.update(row.get('data'))
                
                game['appID'] = str(appid)
                dataset[str(appid)] = game
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return dataset
