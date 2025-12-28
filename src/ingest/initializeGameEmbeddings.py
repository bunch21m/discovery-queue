import os
import time
import psycopg2


def _readSecret(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return None


def buildDatabaseUrl():

    user = _readSecret('/run/secrets/postgres_user') 
    password = _readSecret('/run/secrets/postgres_password') 
    dbName = _readSecret('/run/secrets/postgres_db') 

    if user and password and dbName:
        return f"postgresql://{user}:{password}@gameEmbeddings:5432/{dbName}"
    return None


def waitForDb(dbUrl, retries=20, delay=1.0):
    if not dbUrl:
        print("No DATABASE_URL available to wait on")
        return False
    for i in range(retries):
        try:
            conn = psycopg2.connect(dbUrl)
            conn.close()
            print("Database is available")
            return True
        except Exception as e:
            print(f"Waiting for database ({i+1}/{retries})... {e}")
            time.sleep(delay)
    return False


def initializeGameEmbeddings(dbUrl, dim=30):
    """Create the pgvector extension and gameEmbeddings table if they do not exist."""
    if not dbUrl:
        raise RuntimeError("DATABASE_URL not provided")

    conn = psycopg2.connect(dbUrl)
    try:
        with conn.cursor() as cur:
            # Ensure pgvector extension exists
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # Create gameEmbeddings table with vector column of given dimension

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS gameEmbeddings (
                    id bigserial PRIMARY KEY,
                    embedding vector({dim})
                );
                """
            )
            
        conn.commit()
        print(f"GameEmbeddings table is available (dim={dim})")
    finally:
        conn.close()

def initializeGameEmbeddingsDatabase():
    dimEnv = os.environ.get('EMBEDDING_DIM')
    try:
        dim = int(dimEnv) if dimEnv else 30
    except Exception:
        dim = 30

    dbUrl = buildDatabaseUrl()
    if not waitForDb(dbUrl):
        raise RuntimeError("Database did not become available")

    initializeGameEmbeddings(dbUrl, dim=dim)
    

def main():
    dimEnv = os.environ.get('EMBEDDING_DIM')
    try:
        dim = int(dimEnv) if dimEnv else 30
    except Exception:
        dim = 30

    dbUrl = buildDatabaseUrl()
    if not waitForDb(dbUrl):
        raise RuntimeError("Database did not become available")

    initializeGameEmbeddings(dbUrl, dim=dim)


if __name__ == '__main__':
    main()
