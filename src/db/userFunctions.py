from src.models.CommonModelUtils import readSecret

def getAllUsersFromDatabase():
    """
    Loads all users from the PostgreSQL database.

    Returns:
            dict: A dictionary of all users loaded from the database.
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
            cur.execute("SELECT * FROM users;")
            rows = cur.fetchall()
            for row in rows:
                userid = row['userid']
                dataset[userid] = row
            
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return dataset

def getUserByUsername(username):
    """
    Loads a user by username from the PostgreSQL database.

    Returns:
            dict: A dictionary of all users loaded from the database.
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
            cur.execute(f"SELECT * FROM users WHERE username = '{username}';")
            rows = cur.fetchall()
            for row in rows:
                dataset = row

        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return dataset

def addUserToDatabase(username, data={}):
    """
    Adds a new user to the PostgreSQL database.

    Args:
        username (str): The username of the new user.
        data (dict): Additional data to store for the user.

    Returns:
        int: The userid of the newly created user.
    """
    
    user = readSecret('/run/secrets/postgres_user')
    password = readSecret('/run/secrets/postgres_password')
    dbName = readSecret('/run/secrets/postgres_db')
    
    if not (user and password and dbName):
        raise RuntimeError("Database credentials not found")

    dbUrl = f"postgresql://{user}:{password}@db:5432/{dbName}"

    try:
        import psycopg2
        import json

        conn = psycopg2.connect(dbUrl)
        with conn.cursor() as cur:
            
            cur.execute(
                "INSERT INTO users (username, data) VALUES (%s, %s) ON CONFLICT (username) DO NOTHING RETURNING userid;",
                (username, json.dumps(data))
            )
            result = cur.fetchone()
            if result:
                userid = result[0]
            else:
                # User already exists
                userid = None
            conn.commit()
        conn.close()
        return userid
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")