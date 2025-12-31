# Holds various utility functions for interacting with the games database

def read_secret(path):
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

def load_all_games_from_database():
    """
    Loads all games from the PostgreSQL database.

    Returns:
            dict: A dictionary of all games loaded from the database.
    """
    dataset = {}
    
    user = read_secret('/run/secrets/postgres_user')
    password = read_secret('/run/secrets/postgres_password')
    db_name = read_secret('/run/secrets/postgres_db')
    
    if not (user and password and db_name):
        raise RuntimeError("Database credentials not found")

    db_url = f"postgresql://{user}:{password}@db:5432/{db_name}"

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(db_url)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT appid, name, positive, negative, tags, genres, price, data FROM games")
            rows = cur.fetchall()
            for row in rows:
                app_id = row.get('appid')
                game = {
                    'name': row.get('name'),
                    'positive': int(row.get('positive') or 0),
                    'negative': int(row.get('negative') or 0),
                    'tags': row.get('tags') or [],
                    'genres': row.get('genres') or [],
                    'price': float(row.get('price') or 0)
                }
                if row.get('data'):
                    game.update(row.get('data'))
                
                game['app_id'] = str(app_id)
                dataset[str(app_id)] = game
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return dataset

def get_game_from_database(app_id):
    """
    Loads a game by app_id from the PostgreSQL database.

    Returns:
            dict: A dictionary of the game's data loaded from the database.
    """
    game_data = None
    
    user = read_secret('/run/secrets/postgres_user')
    password = read_secret('/run/secrets/postgres_password')
    db_name = read_secret('/run/secrets/postgres_db')
    
    if not (user and password and db_name):
        raise RuntimeError("Database credentials not found")

    db_url = f"postgresql://{user}:{password}@db:5432/{db_name}"

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(db_url)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM games WHERE appid = '{app_id}';")
            # game_data = cur.fetchone()
            row = cur.fetchone()
            if row:
                game_data = {
                    'name': row.get('name'),
                    'positive': int(row.get('positive') or 0),
                    'negative': int(row.get('negative') or 0),
                    'tags': row.get('tags') or [],
                    'genres': row.get('genres') or [],
                    'price': float(row.get('price') or 0)
                }
                if row.get('data'):
                    game_data.update(row.get('data'))
                
                game_data['app_id'] = str(row.get('appid'))
            
            
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return game_data
