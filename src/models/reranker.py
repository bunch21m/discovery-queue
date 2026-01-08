
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


def calculate_genre_similarity(genres1, genres2):
    """
    Computes Jaccard similarity between two lists of genres.
    """
    if not genres1 or not genres2:
        return 0.0
    set1 = set(genres1)
    set2 = set(genres2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0


# Implements a simple MMR (Maximal Marginal Relevance) reranking algorithm
def mmr_rerank(recommendations, lambda_param=0.3, num_recommendations=10):
    """
    Reranks the given recommendations using the MMR algorithm.
    """
    if not recommendations:
        return []
    
    # 1. Extract raw relevance scores
    raw_relevances = []
    for r in recommendations:
        if 'lambdamart_score' in r:
            raw_relevances.append(r['lambdamart_score'])
        else:
            # Fallback to (1 - distance) if no LambdaMART score
            raw_relevances.append(1 - r.get('distance', 1.0))
    
    # 2. Normalize relevance scores to [0, 1] for balanced MMR
    min_rel = min(raw_relevances)
    max_rel = max(raw_relevances)
    rel_range = max_rel - min_rel if max_rel > min_rel else 1.0
    
    for i, r in enumerate(recommendations):
        r['_mmr_relevance'] = (raw_relevances[i] - min_rel) / rel_range

    selected = [recommendations[0]]  # Start with the most relevant
    remaining = recommendations[1:]
    
    db_url = build_database_url()
    conn = psycopg2.connect(db_url)
    
    try:
        with conn.cursor() as cur:
            while len(selected) < num_recommendations and remaining:
                mmr_scores = {}
                
                for candidate in remaining:
                    relevance = candidate['_mmr_relevance']
                    
                    # Compute max similarity to selected items
                    # COMBINE: 1. Embedding Similarity + 2. Explicit Genre Overlap
                    similarities = []
                    for s in selected:
                        # a. Embedding-based similarity
                        emb_sim = 1 - get_cosine_distance(cur, candidate['app_id'], s['app_id'])
                        
                        # b. Explicit Genre Overlap (Jaccard)
                        genre_sim = calculate_genre_similarity(candidate.get('genres', []), s.get('genres', []))
                        
                        # Weight both - genre similarity is much more "interpretable" for the user
                        combined_sim = 0.5 * emb_sim + 0.5 * genre_sim
                        similarities.append(combined_sim)
                    
                    max_sim = max(similarities)
                    
                    # MMR score
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

    