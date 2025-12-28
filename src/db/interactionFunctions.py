from src.models.CommonModelUtils import readSecret

def getUsersInteractionsFromDatabase(userid):
    """
    Loads all interactions for a specific user from the PostgreSQL database.

    Returns:
            dict: A dictionary of all interactions for a specific user loaded from the database.
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
            cur.execute(f"SELECT * FROM interactions WHERE userid = {userid};")
            rows = cur.fetchall()
            for row in rows:
                interactionid = row['interactionid']
                dataset[interactionid] = row
            
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return dataset

def addUserInteractionToDatabase(appid, userid, interactionType):
    """
    Adds a user interaction to the PostgreSQL database.

    Returns:
            bool: True if the interaction was added successfully, False otherwise.
    """
    
    user = readSecret('/run/secrets/postgres_user')
    password = readSecret('/run/secrets/postgres_password')
    dbName = readSecret('/run/secrets/postgres_db')
    
    if not (user and password and dbName):
        raise RuntimeError("Database credentials not found")

    dbUrl = f"postgresql://{user}:{password}@db:5432/{dbName}"

    try:
        import psycopg2

        conn = psycopg2.connect(dbUrl)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO interactions (appid, userid, interaction_type, timestamp)
                VALUES (%s, %s, %s, now());
                """,
                (appid, userid, interactionType)
            )
            conn.commit()
        conn.close()
        return True
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    return False