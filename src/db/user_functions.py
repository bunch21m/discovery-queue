from src.models.common_model_utils import read_secret

def get_all_users_from_database():
    """
    Loads all users from the PostgreSQL database.

    Returns:
            dict: A dictionary of all users loaded from the database.
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
            cur.execute("SELECT * FROM users;")
            rows = cur.fetchall()
            for row in rows:
                user_id = row['userid']
                dataset[user_id] = row
            
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return dataset

def get_user_by_username(username):
    """
    Loads a user by username from the PostgreSQL database.

    Returns:
            dict: A dictionary of the user data loaded from the database.
    """
    user_data = {}
    
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
            cur.execute(f"SELECT * FROM users WHERE username = '{username}';")
            rows = cur.fetchall()
            for row in rows:
                user_data = row

        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return user_data

def add_user_to_database(username, data={}):
    """
    Adds a new user to the PostgreSQL database.

    Args:
        username (str): The username of the new user.
        data (dict): Additional data to store for the user.

    Returns:
        int: The user_id of the newly created user.
    """
    
    user = read_secret('/run/secrets/postgres_user')
    password = read_secret('/run/secrets/postgres_password')
    db_name = read_secret('/run/secrets/postgres_db')
    
    if not (user and password and db_name):
        raise RuntimeError("Database credentials not found")

    db_url = f"postgresql://{user}:{password}@db:5432/{db_name}"

    try:
        import psycopg2
        import json

        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            
            cur.execute(
                "INSERT INTO users (username, data) VALUES (%s, %s) ON CONFLICT (username) DO NOTHING RETURNING userid;",
                (username, json.dumps(data))
            )
            result = cur.fetchone()
            if result:
                user_id = result[0]
            else:
                # User already exists
                user_id = None
            conn.commit()
        conn.close()
        return user_id
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")