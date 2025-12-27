import os
import time
import json
import psycopg2
from psycopg2.extras import Json


def _readSecret(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return None


user = _readSecret('/run/secrets/postgres_user')
password = _readSecret('/run/secrets/postgres_password')
dbName = _readSecret('/run/secrets/postgres_db')
dbUrl = None
if user and password and dbName:
    dbUrl = f"postgresql://{user}:{password}@db:5432/{dbName}"

def waitForDb(retries=20, delay=1.0):
    if not dbUrl:
        print("DATABASE_URL not set, skipping DB init.")
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
    raise RuntimeError("Database not available")

def initSchema(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS games (
                appid TEXT PRIMARY KEY,
                name TEXT,
                positive INTEGER,
                tags TEXT[],
                data JSONB
            );
            """
        )
    conn.commit()

def loadData(conn, jsonPath='data/games.json'):
    if not os.path.exists(jsonPath):
        print(f"Data file {jsonPath} not found, skipping seed.")
        return

    with open(jsonPath, 'r', encoding='utf-8') as fin:
        text = fin.read()
        if not text:
            print("Data file empty, skipping seed.")
            return
        dataset = json.loads(text)

    with conn.cursor() as cur:
        for appid, game in dataset.items():
            name = game.get('name')
            positive = int(game.get('positive') or 0)
            tags = game.get('tags') or []
            if isinstance(tags, dict):
                tags = list(tags.keys())
                
            cur.execute(
                """
                INSERT INTO games (appid, name, positive, tags, data)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (appid) DO UPDATE SET
                    name = EXCLUDED.name,
                    positive = EXCLUDED.positive,
                    tags = EXCLUDED.tags,
                    data = EXCLUDED.data;
                """,
                (str(appid), name, positive, tags, Json(game))
            )
    conn.commit()

def main():
    if not dbUrl:
        print("DATABASE_URL not set, skipping DB init.")
        return

    waitForDb()

    conn = psycopg2.connect(dbUrl)
    try:
        initSchema(conn)
        loadData(conn)
        print("DB init/seed complete")
    finally:
        conn.close()

if __name__ == '__main__':
    main()
