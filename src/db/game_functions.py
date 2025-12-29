from src.models.common_model_utils import read_secret

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
            game_data = cur.fetchone()
            
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return game_data