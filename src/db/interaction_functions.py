from src.db.game_functions import read_secret

def get_users_interactions_from_database(user_id):
    """
    Loads all interactions for a specific user from the PostgreSQL database.

    Returns:
            dict: A dictionary of all interactions for a specific user loaded from the database.
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
            cur.execute(f"SELECT * FROM interactions WHERE userid = {user_id};")
            rows = cur.fetchall()
            for row in rows:
                interaction_id = row['interactionid']
                dataset[interaction_id] = row
            
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return dataset

def add_user_interaction_to_database(app_id, user_id, interaction_type):
    """
    Adds a user interaction to the PostgreSQL database.

    Returns:
            bool: True if the interaction was added successfully, False otherwise.
    """
    
    user = read_secret('/run/secrets/postgres_user')
    password = read_secret('/run/secrets/postgres_password')
    db_name = read_secret('/run/secrets/postgres_db')
    
    if not (user and password and db_name):
        raise RuntimeError("Database credentials not found")

    db_url = f"postgresql://{user}:{password}@db:5432/{db_name}"

    try:
        import psycopg2

        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO interactions (appid, userid, interactiontype, timestamp)
                VALUES (%s, %s, %s, now());
                """,
                (app_id, user_id, interaction_type)
            )
            conn.commit()
        conn.close()
        return True
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return False