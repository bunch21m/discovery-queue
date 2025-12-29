import os
import time
import json
import psycopg2
from psycopg2.extras import Json


def _read_secret(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return None


user = _read_secret('/run/secrets/postgres_user')
password = _read_secret('/run/secrets/postgres_password')
db_name = _read_secret('/run/secrets/postgres_db')
db_url = None
if user and password and db_name:
    db_url = f"postgresql://{user}:{password}@db:5432/{db_name}"

def wait_for_db(retries=20, delay=1.0):
    if not db_url:
        print("DATABASE_URL not set, skipping DB init.")
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
    raise RuntimeError("Database not available")

def init_schema(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS games (
                appid TEXT PRIMARY KEY,
                name TEXT,
                positive INTEGER,
                tags TEXT[],
                genres TEXT[],
                price FLOAT,
                data JSONB
            );
            """
        )
        
        
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                userid INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                username TEXT UNIQUE,
                data JSONB
            );
            """
        )
        
        
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                interactionid INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                appid TEXT REFERENCES games(appid),
                userid INT REFERENCES users(userid),
                interactiontype TEXT,
                timestamp TIMESTAMP,
                data JSONB
            );
            """
        )
        
        cur.execute(
            """
            INSERT INTO users (username, data) VALUES ('test', '{}')
            ON CONFLICT (username) DO NOTHING;
            """
        )
                     
        
        
        
        
    conn.commit()

def load_data(conn, json_path='data/games.json'):
    if not os.path.exists(json_path):
        print(f"Data file {json_path} not found, skipping seed.")
        return

    with open(json_path, 'r', encoding='utf-8') as fin:
        text = fin.read()
        if not text:
            print("Data file empty, skipping seed.")
            return
        dataset = json.loads(text)

    with conn.cursor() as cur:
        for app_id, game in dataset.items():
            name = game.get('name')
            positive = int(game.get('positive') or 0)
            tags = game.get('tags') or []
            if isinstance(tags, dict):
                tags = list(tags.keys())
            genres = game.get('genres') or []
            if isinstance(genres, dict):
                genres = list(genres.keys())
            price = float(game.get('price') or 0)

            cur.execute(
                """
                INSERT INTO games (appid, name, positive, tags, genres, price, data)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (appid) DO UPDATE SET
                    name = EXCLUDED.name,
                    positive = EXCLUDED.positive,
                    tags = EXCLUDED.tags,
                    genres = EXCLUDED.genres,
                    price = EXCLUDED.price,
                    data = EXCLUDED.data;
                """,
                (str(app_id), name, positive, tags, genres, price, Json(game))
            )
    conn.commit()

def main():
    if not db_url:
        print("DATABASE_URL not set, skipping DB init.")
        return

    wait_for_db()

    conn = psycopg2.connect(db_url)
    try:
        init_schema(conn)
        load_data(conn)
        print("DB init/seed complete")
    finally:
        conn.close()

if __name__ == '__main__':
    main()
