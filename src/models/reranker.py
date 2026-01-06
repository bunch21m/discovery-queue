
import psycopg2
from src.ingest.initialize_game_embeddings import build_database_url

# helper for cosine similarity
def get_cosine_similarity(cur, app_id_1, app_id_2):
    """
    Computes cosine similarity between two game embeddings using an existing cursor.
    """
    similarity = 0.0
    query = """
    SELECT 
        1 - (g1.embedding <=> g2.embedding) AS cosine_similarity
    FROM 
        gameEmbeddings g1, gameEmbeddings g2
    WHERE 
        g1.appid = %s AND g2.appid = %s;
    """
    cur.execute(query, (str(app_id_1), str(app_id_2)))
    row = cur.fetchone()
    if row:
        similarity = row[0]
    return similarity


# Implements a simple MMR (Maximal Marginal Relevance) reranking algorithm
def mmr_rerank(recommendations, lambda_param=0.5, num_recommendations=10):
    """
    Reranks the given recommendations using the MMR algorithm.
    """
    if not recommendations:
        return []
    
    selected = [recommendations[0]]  # Start with the top recommendation
    remaining = recommendations[1:]  # Remaining candidates
    
    db_url = build_database_url()
    conn = psycopg2.connect(db_url)
    
    try:
        with conn.cursor() as cur:
            # add next best recommendation based on MMR
            while len(selected) < num_recommendations and remaining:
                mmr_scores = {}
                
                for candidate in remaining:
                    # Relevance: higher for lower distance (more similar to user)
                    relevance = -candidate['distance']
                    
                    # Max similarity to already selected items
                    # We reuse the cursor 'cur' to avoid re-connecting
                    max_sim = max([get_cosine_similarity(cur, candidate['app_id'], s['app_id']) for s in selected]) if selected else 0
                    
                    # MMR score: balance relevance vs diversity
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                    mmr_scores[candidate['app_id']] = (mmr_score, candidate)
                    
                # Select the candidate with highest MMR score
                next_best_id = max(mmr_scores, key=lambda x: mmr_scores[x][0])
                next_best = mmr_scores[next_best_id][1]
                selected.append(next_best)
                remaining.remove(next_best)
    finally:
        conn.close()
    
    return selected

    