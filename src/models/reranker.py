
import psycopg2
from src.ingest.initialize_game_embeddings import build_database_url

#helper for cosine similarity
def cosine_similarity(app_id_1, app_id_2):
    """
    Computes cosine similarity between two game embeddings stored in the database.
    
    :param app_id_1: App ID of the first game.
    :param app_id_2: App ID of the second game.
    :return: Cosine similarity score.
    """
    db_url = build_database_url()
    conn = psycopg2.connect(db_url)
    similarity = 0.0
    try:
        with conn.cursor() as cur:
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
    except Exception as e:
        print(f"Error computing cosine similarity: {e}")
    finally:
        conn.close()
    
    return similarity


# Implements a simple MMR (Maximal Marginal Relevance) reranking algorithm
def mmr_rerank(recommendations, lambda_param=0.5, num_recommendations=10):
    """
    Reranks the given recommendations using the MMR algorithm.
    
    :param recommendations: List of games to rerank.
    :param lambda_param: The lambda parameter for MMR, balancing relevance and diversity
    :param num_recommendations: Number of recommendations to return after reranking
    """
    
    if not recommendations:
        return []
    
    selected = [recommendations[0]]  # Start with the top recommendation
    remaining = recommendations[1:]  # Remaining candidates
    
    # add next best recommendation based on MMR
    while len(selected) < num_recommendations and remaining:
        mmr_scores = {}
        
        for candidate in remaining:
            # Relevance: higher for lower distance (more similar to user)
            relevance = -candidate['distance']  # Negative because lower distance = higher relevance
            
            # Max similarity to already selected items
            max_sim = max([cosine_similarity(candidate['app_id'], s['app_id']) for s in selected]) if selected else 0
            
            # MMR score: balance relevance vs diversity
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores[candidate['app_id']] = (mmr_score, candidate)
            
        # Select the candidate with highest MMR score
        next_best_id = max(mmr_scores, key=lambda x: mmr_scores[x][0])
        next_best = mmr_scores[next_best_id][1]
        selected.append(next_best)
        remaining.remove(next_best)
    
    return selected
    