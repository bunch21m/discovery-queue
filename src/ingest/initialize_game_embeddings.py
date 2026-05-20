import os
import time
import psycopg2


def _read_secret(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return None


def build_database_url():

    user = _read_secret('/run/secrets/postgres_user') 
    password = _read_secret('/run/secrets/postgres_password') 
    db_name = _read_secret('/run/secrets/postgres_db') 

    if user and password and db_name:
        return f"postgresql://{user}:{password}@gameEmbeddings:5432/{db_name}"
    return None


def wait_for_db(db_url, retries=20, delay=1.0):
    if not db_url:
        print("No DATABASE_URL available to wait on")
        return False
    for i in range(retries):
        try:
            conn = psycopg2.connect(db_url)
            conn.close()
            print("Database is available")
            return True
        except Exception as e:
            print(f"Waiting for database ({i+1}/{retries})... {e}")
            time.sleep(delay)
    return False


def initialize_game_embeddings(db_url, dim=128):
    """Create the pgvector extension and gameEmbeddings table if they do not exist.
    If table exists with different dimension, drop and recreate it."""
    if not db_url:
        raise RuntimeError("DATABASE_URL not provided")

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            # Ensure pgvector extension exists
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Check if table exists and has correct dimension
            cur.execute("""
                SELECT COUNT(*) FROM information_schema.columns 
                WHERE table_name = 'gameembeddings' AND column_name = 'embedding';
            """)
            table_exists = cur.fetchone()[0] > 0
            
            if table_exists:
                # Check current dimension by looking at a sample or the type
                cur.execute("""
                    SELECT atttypmod FROM pg_attribute 
                    WHERE attrelid = 'gameembeddings'::regclass 
                    AND attname = 'embedding';
                """)
                result = cur.fetchone()
                current_dim = result[0] if result and result[0] > 0 else 0
                
                if current_dim != dim and current_dim > 0:
                    print(f"Dimension mismatch: table has {current_dim}, need {dim}. Dropping table...")
                    cur.execute("DROP TABLE IF EXISTS gameEmbeddings;")
                    table_exists = False
            
            if not table_exists:
                # Create gameEmbeddings table with vector column of given dimension
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS gameEmbeddings (
                        id bigserial PRIMARY KEY,
                        appid VARCHAR(255),
                        embedding vector({dim})
                    );
                    """
                )
                # Create HNSW index for fast approximate nearest neighbor search
                # vector_cosine_ops is required for <=> operator
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS game_embeddings_embedding_idx 
                    ON gameEmbeddings USING hnsw (embedding vector_cosine_ops);
                """)
                # Standard btree index on appid for lookups
                cur.execute("CREATE INDEX IF NOT EXISTS game_embeddings_appid_idx ON gameEmbeddings (appid);")
            
        conn.commit()
        print(f"GameEmbeddings table is available (dim={dim})")
    finally:
        conn.close()

def initialize_game_embeddings_database():
    dim_env = os.environ.get('EMBEDDING_DIM')
    try:
        dim = int(dim_env) if dim_env else 128
    except Exception:
        dim = 128

    db_url = build_database_url()
    if not wait_for_db(db_url):
        raise RuntimeError("Database did not become available")

    initialize_game_embeddings(db_url, dim=dim)
    

def main():
    dim_env = os.environ.get('EMBEDDING_DIM')
    try:
        dim = int(dim_env) if dim_env else 128
    except Exception:
        dim = 128

    db_url = build_database_url()
    if not wait_for_db(db_url):
        raise RuntimeError("Database did not become available")

    initialize_game_embeddings(db_url, dim=dim)


if __name__ == '__main__':
    main()
