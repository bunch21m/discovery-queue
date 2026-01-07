
import psycopg2
from src.ingest.initialize_game_embeddings import build_database_url


def get_intralist_diversity(recommendations):
    """
    Calculates the Intralist Diversity (ILD) of a list of recommendations.
    ILD is defined as the average pairwise distance between all items in the list.
    """
    if not recommendations or len(recommendations) < 2:
        return 0.0
        
    db_url = build_database_url()
    conn = psycopg2.connect(db_url)
    
    total_distance = 0.0
    pair_count = 0
    
    try:
        with conn.cursor() as cur:
            for i in range(len(recommendations)):
                for j in range(i + 1, len(recommendations)):
                    
                    id_i = recommendations[i]['app_id']
                    id_j = recommendations[j]['app_id']
                    
                    dist = get_cosine_distance(cur, id_i, id_j)
                    total_distance += dist
                    pair_count += 1
    finally:
        conn.close()
        
    if pair_count == 0:
        return 0.0
        
    return total_distance / pair_count

# helper for cosine distance
def get_cosine_distance(cur, app_id_1, app_id_2):
    """
    Computes cosine distance between two game embeddings using an existing cursor.
    """
    distance = 1.0 # Max distance
    query = """
    SELECT 
        (g1.embedding <=> g2.embedding) AS cosine_distance
    FROM 
        gameEmbeddings g1, gameEmbeddings g2
    WHERE 
        g1.appid = %s AND g2.appid = %s;
    """
    cur.execute(query, (str(app_id_1), str(app_id_2)))
    row = cur.fetchone()
    if row and row[0] is not None:
        distance = row[0]
    return distance


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
                    # We convert distance to similarity: 1 - distance (<=> is cosine distance)
                    relevance = 1 - candidate['distance']
                    
                    # Max similarity to already selected items
                    # We reuse the cursor 'cur' to avoid re-connecting
                    # Convert distance to similarity for MMR: 1 - distance
                    max_sim = max([1 - get_cosine_distance(cur, candidate['app_id'], s['app_id']) for s in selected]) if selected else 0
                    
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

    