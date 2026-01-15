from src.db.tools.game_functions import read_secret


def get_latest_user_interaction_from_database(user_id):
    """
    Loads the latest interaction for a specific user from the PostgreSQL database.

    Returns:
            dict: A dictionary of the latest interaction for a specific user loaded from the database.
    """
    interaction = None
    
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
            cur.execute("SELECT * FROM interactions WHERE userid = %s ORDER BY timestamp DESC LIMIT 1;", (user_id,))
            row = cur.fetchone()
            if row:
                interaction = row
            
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return interaction

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
            cur.execute("SELECT * FROM interactions WHERE userid = %s;", (user_id,))
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


def delete_user_interaction_from_database(app_id, user_id):
    """
    Deletes a specific user interaction from the PostgreSQL database.
    Used for evaluation purposes (e.g., Leave-One-Out validation).

    Returns:
            bool: True if the interaction was deleted successfully, False otherwise.
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
                DELETE FROM interactions 
                WHERE appid = %s AND userid = %s;
                """,
                (app_id, user_id)
            )
            conn.commit()
        conn.close()
        return True
    except Exception as e:
        raise RuntimeError(f"Database error during deletion: {e}")

    return False

def get_all_interactions_from_database():
    """
    Loads all interactions from the PostgreSQL database.

    Returns:
            dict: A dictionary of all interactions loaded from the database.
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
            cur.execute("SELECT * FROM interactions;")
            rows = cur.fetchall()
            for row in rows:
                interaction_id = row['interactionid']
                dataset[interaction_id] = row
            
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return dataset


def get_all_interactions_after_timestamp_from_database(timestamp):
    """
    Loads all interactions after a specific timestamp from the PostgreSQL database.

    Returns:
            dict: A dictionary of all interactions after the specified timestamp loaded from the database.
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
            cur.execute("SELECT * FROM interactions WHERE timestamp > %s;", (timestamp,))
            rows = cur.fetchall()
            for row in rows:
                interaction_id = row['interactionid']
                dataset[interaction_id] = row
            
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return dataset


def get_interactions_for_users_from_database(user_ids):
    """
    Loads all interactions for a list of users in a single query.
    """
    if not user_ids:
        return {}
        
    user = read_secret('/run/secrets/postgres_user')
    password = read_secret('/run/secrets/postgres_password')
    db_name = read_secret('/run/secrets/postgres_db')
    
    if not (user and password and db_name):
        raise RuntimeError("Database credentials not found")

    db_url = f"postgresql://{user}:{password}@db:5432/{db_name}"
    dataset = {}

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(db_url)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM interactions WHERE userid IN %s;", (tuple(user_ids),))
            rows = cur.fetchall()
            for row in rows:
                user_id = row['userid']
                if user_id not in dataset:
                    dataset[user_id] = {}
                interaction_id = row['interactionid']
                dataset[user_id][interaction_id] = row
            
        conn.close()
    except Exception as e:
        print(f"Database error in batch interaction fetch: {e}")

    return dataset