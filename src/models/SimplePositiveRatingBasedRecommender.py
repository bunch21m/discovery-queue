import os
import json


def getRecommendations(*args, **kwargs):
    """
    Recommends a game simply by returning the top
    numRecommendationsToMake highest rated games.
    """
    for key, value in kwargs.items():
        if key == "numRecommendationsToMake":
            numRecommendationsToMake = value

    dataset = {}


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
        db_url = f"postgresql://{user}:{password}@db:5432/{dbname}"

    if db_url:
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            conn = psycopg2.connect(db_url)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT appid, name, positive, data FROM games")
                rows = cur.fetchall()
                for row in rows:
                    appid = row.get('appid')
                    game = {
                        'name': row.get('name'),
                        'positive': int(row.get('positive') or 0)
                    }
                    if row.get('data'):
                        game.update(row.get('data'))
                    dataset[str(appid)] = game
            conn.close()
        except Exception:
            raise RuntimeError("Database not available")
    else:
        raise RuntimeError("Database not available")

    # Sort and return top-N names
    sorted_games = sorted(dataset.values(), key=lambda x: int(x.get('positive', 0)), reverse=True)
    sorted_games = sorted_games[:numRecommendationsToMake]
    sorted_games = [game.get('name') for game in sorted_games]

    return sorted_games