
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
def mmr_rerank(recommendations, lambda_param=0.7, num_recommendations=10):
    """
    Reranks the given recommendations using the MMR algorithm.
    OPTIMIZED: Batch fetch embeddings, compute similarities in NumPy.
    """
    if not recommendations:
        return []
    
    # 1. Extract raw relevance scores
    raw_relevances = []
    for r in recommendations:
        if 'lambdamart_score' in r:
            raw_relevances.append(r['lambdamart_score'])
        else:
            raw_relevances.append(1 - r.get('distance', 1.0))
    
    # 2. Normalize relevance scores to [0, 1]
    min_rel = min(raw_relevances)
    max_rel = max(raw_relevances)
    rel_range = max_rel - min_rel if max_rel > min_rel else 1.0
    
    for i, r in enumerate(recommendations):
        r['_mmr_relevance'] = (raw_relevances[i] - min_rel) / rel_range

    # 3. OPTIMIZATION: Batch fetch all embeddings upfront
    app_ids = [r['app_id'] for r in recommendations]
    embeddings = _batch_fetch_embeddings(app_ids)
    
    # Build embedding lookup and genre lookup
    emb_lookup = {aid: emb for aid, emb in embeddings}
    genre_lookup = {r['app_id']: r.get('genres', []) for r in recommendations}
    
    # 4. MMR Selection (all in Python, no more DB calls)
    import numpy as np
    
    selected = [recommendations[0]]
    remaining = list(recommendations[1:])
    
    while len(selected) < num_recommendations and remaining:
        best_score = float('-inf')
        best_candidate = None
        
        for candidate in remaining:
            relevance = candidate['_mmr_relevance']
            
            # Compute max similarity to selected items (in memory)
            max_sim = 0.0
            cand_emb = emb_lookup.get(candidate['app_id'])
            cand_genres = genre_lookup.get(candidate['app_id'], [])
            
            for s in selected:
                sel_emb = emb_lookup.get(s['app_id'])
                sel_genres = genre_lookup.get(s['app_id'], [])
                
                # Embedding similarity (cosine)
                if cand_emb is not None and sel_emb is not None:
                    # Cosine similarity = 1 - cosine distance
                    dot = np.dot(cand_emb, sel_emb)
                    norm = np.linalg.norm(cand_emb) * np.linalg.norm(sel_emb)
                    emb_sim = dot / norm if norm > 0 else 0.0
                else:
                    emb_sim = 0.0
                
                # Genre similarity (Jaccard)
                genre_sim = calculate_genre_similarity(cand_genres, sel_genres)
                
                combined_sim = 0.5 * emb_sim + 0.5 * genre_sim
                max_sim = max(max_sim, combined_sim)
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = candidate
        
        if best_candidate:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
    
    return selected


def _batch_fetch_embeddings(app_ids):
    """Batch fetch embeddings for multiple app IDs in a single query."""
    if not app_ids:
        return []
    
    db_url = build_database_url()
    conn = psycopg2.connect(db_url)
    embeddings = []
    
    try:
        with conn.cursor() as cur:
            placeholders = ','.join(['%s'] * len(app_ids))
            query = f"""
            SELECT appid, embedding
            FROM gameEmbeddings
            WHERE appid IN ({placeholders});
            """
            cur.execute(query, tuple(str(aid) for aid in app_ids))
            rows = cur.fetchall()
            
            import numpy as np
            for row in rows:
                app_id = row[0]
                # Parse the vector string to numpy array
                emb_str = row[1]
                if isinstance(emb_str, str):
                    # Format: [0.1,0.2,...]
                    emb_str = emb_str.strip('[]')
                    emb = np.array([float(x) for x in emb_str.split(',')])
                else:
                    emb = np.array(emb_str)
                embeddings.append((app_id, emb))
    finally:
        conn.close()
    
    return embeddings
