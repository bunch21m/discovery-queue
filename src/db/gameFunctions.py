from src.models.CommonModelUtils import readSecret

def getGameFromDatabase(appid):
    """
    Loads a game by appid from the PostgreSQL database.

    Returns:
            dict: A dictionary of the game's data loaded from the database.
    """
    gameData = None
    
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
            cur.execute(f"SELECT * FROM games WHERE appid = '{appid}';")
            gameData = cur.fetchone()
            
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return gameData