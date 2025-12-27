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
dbname = _read_secret('/run/secrets/postgres_db')
if user and password and dbname:

    DB_URL = f"postgresql://{user}:{password}@db:5432/{dbname}"

def wait_for_db(retries=20, delay=1.0):
    if not DB_URL:
        print("DATABASE_URL not set, skipping DB init.")
        return False
    for i in range(retries):
        try:
            conn = psycopg2.connect(DB_URL)
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
                data JSONB
            );
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
        for appid, game in dataset.items():
            name = game.get('name')
            positive = int(game.get('positive') or 0)
            cur.execute(
                """
                INSERT INTO games (appid, name, positive, data)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (appid) DO UPDATE SET
                    name = EXCLUDED.name,
                    positive = EXCLUDED.positive,
                    data = EXCLUDED.data;
                """,
                (str(appid), name, positive, Json(game))
            )
    conn.commit()

def main():
    if not DB_URL:
        print("DATABASE_URL not set, skipping DB init.")
        return

    wait_for_db()

    conn = psycopg2.connect(DB_URL)
    try:
        init_schema(conn)
        load_data(conn)
        print("DB init/seed complete")
    finally:
        conn.close()

if __name__ == '__main__':
    main()
