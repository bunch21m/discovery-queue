
import math
import random
import time
import numpy as np
import pandas as pd
import torch
import psycopg2

from src.models.two_tower_candidate_pooler import TwoTowerRecommender
from src.db.user_functions import get_user_by_username, add_user_to_database
from src.db.interaction_functions import (
    get_users_interactions_from_database, 
    add_user_interaction_to_database,
    delete_user_interaction_from_database
)
from src.db.tools.auto_wishlister import set_user_config, score_game, action_for_score
from src.db.tools.game_functions import load_all_games_from_database
from src.models.user_two_tower_embedding import concat_user_features
from src.ingest.initialize_game_embeddings import build_database_url
from src.models.train_two_tower_model import TwoTowerModel
from src.models.reranker import get_intralist_diversity

# --- CONFIGURATION ---
# --- CONFIGURATION ---
# Valid Genres from genre_check_output.txt:
# Action, Adventure, Casual, Indie, Massively Multiplayer, RPG, Racing, Simulation, Sports, Strategy,
# 360 Video, Accounting, Animation & Modeling, Audio Production, Design & Illustration, Documentary,
# Early Access, Education, Episodic, Free To Play, Game Development, Gore, Movie, Nudity,
# Photo Editing, Sexual Content, Short, Software Training, Tutorial, Utilities, Video Production,
# Violent, Web Publishing

PERSONAS = {
    'StrategyPro': {
        "positive": [
             # STRICT POPULARITY (Mainstream)
             {"comparison": "lt", "value": 500, "score": -500},   # Hard block <500
             {"comparison": "lt", "value": 10000, "score": -150},  # Penalty <10k
             {"comparison": "gt", "value": 20000, "score": 50},   # Big Bonus >20k
             {"comparison": "gt", "value": 50000, "score": 100}   # Massive Bonus >50k
        ],
        "genres": [
            {"value": "strategy", "score": 150}, {"value": "simulation", "score": 150}, 
            # STRICT PENALTIES
            {"value": "action", "score": -100}, {"value": "shooter", "score": -150},
            {"value": "sports", "score": -150}, {"value": "racing", "score": -150},
            {"value": "rpg", "score": -40}
        ],
        "target_genres": {'strategy', 'simulation'} 
    },
    'CozyExplorer': {
        "positive": [
            # STRICT NICHE / INDIE (Avoid Blockbusters)
            {"comparison": "lt", "value": 1000, "score": -500},     # Avoid very low popularity
            {"comparison": "lt", "value": 20000, "score": 100},   # Loves Niche Titles
            {"comparison": "gt", "value": 50000, "score": -50}   # Mild preference
        ],
        "genres": [
            {"value": "casual", "score": 150}, {"value": "indie", "score": 100},
            # STRICT PENALTIES
            {"value": "violent", "score": -500}, {"value": "gore", "score": -500}, 
            {"value": "action", "score": -100}, {"value": "shooter", "score": -200},
            {"value": "horror", "score": -200}, {"value": "sports", "score": -150}
        ],
        "target_genres": {'casual', 'indie'}
    },
    'ActionFanatic': {
        "positive": [
             # ULTRA STRICT POPULARITY (AAA Gamer)
             {"comparison": "lt", "value": 2000, "score": -500},  # Hard block <2000
             {"comparison": "lt", "value": 30000, "score": -100},  # Penalty <30k
             {"comparison": "gt", "value": 50000, "score": 50},   # Loves AAA
             {"comparison": "gt", "value": 100000, "score": 100}  # Obsessed with Mega-Hits
        ], 
        "genres": [
            {"value": "action", "score": 150}, {"value": "massively multiplayer", "score": 100},
            # STRICT PENALTIES
            {"value": "strategy", "score": -100}, {"value": "puzzle", "score": -150},
            {"value": "simulation", "score": -100}, {"value": "casual", "score": -100},
            {"value": "sports", "score": -100}
        ],
        "target_genres": {'action', 'massively multiplayer'}
    },
    'RPGFanatic': {
        "positive": [
             # STRICT POPULARITY
             {"comparison": "lt", "value": 500, "score": -500},   # Hard block <500
             {"comparison": "lt", "value": 10000, "score": -150},  # Penalty <10k
             {"comparison": "gt", "value": 10000, "score": 30},   # Bonus >10k
             {"comparison": "gt", "value": 50000, "score": 80}    # Big Bonus >50k
        ],
        "genres": [
            {"value": "rpg", "score": 150}, {"value": "adventure", "score": 100},
            # STRICT PENALTIES
            {"value": "sports", "score": -200}, {"value": "racing", "score": -150},
            {"value": "strategy", "score": -50}, {"value": "simulation", "score": -50}
        ],
        "target_genres": {'rpg', 'adventure'}
    }
}

def get_test_item_positions(user_embedding, test_app_ids):
    """Query DB to find where test items rank in cosine distance from user embedding."""
    db_url = build_database_url()
    conn = psycopg2.connect(db_url)
    positions = {}
    try:
        with conn.cursor() as cur:
            user_vec_str = str(user_embedding.tolist())
            for test_id in test_app_ids:
                query = """
                SELECT COUNT(*) + 1 
                FROM gameEmbeddings 
                WHERE embedding <=> %s::vector < (
                    SELECT embedding <=> %s::vector FROM gameEmbeddings WHERE appid = %s
                );
                """
                cur.execute(query, (user_vec_str, user_vec_str, str(test_id)))
                row = cur.fetchone()
                positions[test_id] = row[0] if row else None
    except Exception as e:
        print(f"Error getting positions: {e}")
    finally:
        conn.close()
    return positions

def run_single_persona_evaluation(username, config, all_games):
    """Runs temporal split evaluation for a single persona."""
    print(f"\nExample user: {username} ({list(config['target_genres'])[:2]}...)")
    
    # 1. Setup User
    user = get_user_by_username(username)
    if not user:
        add_user_to_database(username)
        user = get_user_by_username(username)
    user_id = user['userid']
    set_user_config(username, config)
    
    # 2. Populate Interactions (if needed)
    MIN_WISHLISTS = 15
    interactions = get_users_interactions_from_database(user_id)
    wishlists = [r for r in interactions.values() if r['interactiontype'] == 'wishlist']
    
    if len(wishlists) < MIN_WISHLISTS:
        # Generate interactions
        game_list = sorted(list(all_games.values()), key=lambda g: g.get('positive', 0), reverse=True)[:30000]
        random.shuffle(game_list)
        count = 0
        for game in game_list:
            if len(wishlists) >= MIN_WISHLISTS: break
            score = score_game(game, config)
            action = action_for_score(score)
            add_user_interaction_to_database(game['app_id'], user_id, action)
            if action == 'wishlist': 
                wishlists.append({'appid': game['app_id']}) # Mock
            count += 1
        interactions = get_users_interactions_from_database(user_id) # Refresh
        
    # 3. Temporal Split (Last 5 items)
    all_ints = list(interactions.values())
    all_ints.sort(key=lambda x: x.get('timestamp', 0))
    # Fallback sort if timestamps missing
    if len(all_ints) > 1 and all_ints[0].get('timestamp', 0) == all_ints[-1].get('timestamp', 0):
        random.shuffle(all_ints)
        
    wishlist_seq = [x for x in all_ints if x['interactiontype'] == 'wishlist']
    if len(wishlist_seq) < 10: return None # Fail
    
    TEST_SIZE = 5
    test_set = wishlist_seq[-TEST_SIZE:]
    
    # Hide Future Items
    hidden = []
    for x in test_set:
        delete_user_interaction_from_database(str(x['appid']), user_id)
        hidden.append(x)
        
    metrics = {}
    try:
        # 4. Generate Recommendations
        recommender = TwoTowerRecommender()
        recommendations = recommender.get_recommendations(username, k=250) # Top 250 for Hit Rate
        
        # 5. Compute Metrics
        rec_ids = [str(r['app_id']) for r in recommendations]
        test_ids = set([str(x['appid']) for x in test_set])
        
        # Retrieval Metrics
        hits_10 = 0
        hits_250 = 0
        rec_ranks = []
        
        for i, x in enumerate(rec_ids):
            if x in test_ids:
                rank = i + 1
                rec_ranks.append(rank)
                if i < 10: hits_10 += 1
                if i < 250: hits_250 += 1
                
        metrics['hit_rate_250'] = hits_250 / len(test_set)
        metrics['recall_10'] = hits_10 / len(test_set)
        metrics['best_rec_rank'] = min(rec_ranks) if rec_ranks else None
        metrics['binary_hit_250'] = 1.0 if hits_250 > 0 else 0.0
        
        # Ranking Quality (NDCG@10)
        dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(rec_ids[:10]) if x in test_ids)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(10, len(test_set))))
        metrics['ndcg_10'] = dcg / idcg if idcg > 0 else 0.0
        
        # Genre Precision (Relevance) - Top 10
        matches = 0
        target = config['target_genres']
        for r in recommendations[:10]:
            r_genres = set(g.lower() for g in r.get('genres', []))
            if r_genres & target: matches += 1
        metrics['genre_precision'] = matches / 10.0
        
        # Diversity (ILD) - Top 10
        metrics['diversity'] = get_intralist_diversity(recommendations[:10])
        
        # Capture MMR Improvement
        if hasattr(recommender, 'latest_metrics') and recommender.latest_metrics:
            metrics['diversity_before'] = recommender.latest_metrics.get('diversity_before', 0)
            metrics['diversity_improvement_pct'] = (metrics['diversity'] - metrics['diversity_before']) / metrics['diversity_before'] if metrics['diversity_before'] > 0 else 0
        else:
            metrics['diversity_improvement_pct'] = 0.0

        # Position Check (Diagnostic)
        # Compute user embedding
        user_feats = concat_user_features(username)
        user_tensor = torch.tensor(user_feats, dtype=torch.float32).unsqueeze(0)
        model = TwoTowerModel(73, 44)
        ckpt = torch.load('data/two_tower_model.pth', map_location='cpu') # Load model 
        if isinstance(ckpt, dict) and 'model' in ckpt: model.load_state_dict(ckpt['model'])
        else: model.load_state_dict(ckpt)
        model.eval()
        with torch.no_grad():
            u_emb = model.user_tower(user_tensor)
            u_emb = torch.nn.functional.normalize(u_emb, p=2, dim=1).numpy().flatten()
            
        pos_dict = get_test_item_positions(u_emb, [str(x['appid']) for x in test_set])
        ranks = [p for p in pos_dict.values() if p]
        metrics['avg_emb_rank'] = sum(ranks)/len(ranks) if ranks else 42000
        metrics['best_emb_rank'] = min(ranks) if ranks else 42000
        
    finally:
        # Restore
        for x in hidden:
            add_user_interaction_to_database(x['appid'], user_id, x['interactiontype'])
            
    return metrics

def run_full_evaluation():
    print("--- Starting Comprehensive Evaluation Suite ---")
    all_games = load_all_games_from_database()
    
    # Configuration
    NUM_TRIALS = 3  # Run 3 independent users per persona to smooth out noise
    
    results = []
    
    for name, config in PERSONAS.items():
        print(f"\n--- Evaluating Persona: {name} ({NUM_TRIALS} Trials) ---")
        persona_metrics = []
        
        for i in range(NUM_TRIALS):
            # Unique user for each trial to ensure independent history
            trial_user = f"eval_{name.lower()}_{i}" 
            print(f"  Trial {i+1}/{NUM_TRIALS} (User: {trial_user})...")
            
            m = run_single_persona_evaluation(trial_user, config, all_games)
            if m:
                # Handle None values for rank
                best_rec = m.get('best_rec_rank')
                best_rec_str = str(best_rec) if best_rec else "N/A"
                print(f"    > HitRate@250: {m['hit_rate_250']:.1%} | Precision: {m['genre_precision']:.1%} | Best Rec Rank: {best_rec_str}")
                persona_metrics.append(m)
            else:
                print("    > Failed (not enough data)")
        
        if persona_metrics:
            # Average metrics for this persona
            df_p = pd.DataFrame(persona_metrics)
            
            # Numeric columns only
            numeric_cols = ['hit_rate_250', 'recall_10', 'genre_precision', 'diversity', 
                            'avg_emb_rank', 'best_emb_rank', 'ndcg_10', 'binary_hit_250']
                            
            avg_m = df_p[numeric_cols].mean().to_dict()
            std_m = df_p[numeric_cols].std().fillna(0).to_dict() # Capture deviation
            
            avg_m['persona'] = name
            avg_m['std'] = std_m # Store STD for reporting
            
            # Store improvement separately to handles cases where it might be 0
            avg_m['diversity_improvement_pct'] = df_p['diversity_improvement_pct'].mean()
            
            # Capture PEAK performance (Min Rank) to explain Recall hits
            avg_m['peak_rec_rank'] = df_p['best_rec_rank'].min()
            avg_m['avg_rec_rank'] = df_p['best_rec_rank'].mean()
            
            results.append(avg_m)
            print(f"  >>> AVG for {name}: HitRate: {avg_m['hit_rate_250']:.1%}")

    if not results:
        print("No results generated.")
        return

    # Aggregate
    df = pd.DataFrame(results)
    
    lines = []
    lines.append("="*80)
    lines.append(f"FINAL SYSTEM REPORT (Metrics - {NUM_TRIALS} Trials Avg)")
    lines.append("="*80)
    
    lines.append(f"\nNOTE: 'Hit Rate@250' is Recall (Items Found / 5). 'Binary Hit' is Session Success Rate (At least 1 item found).")
    
    lines.append("\n--- PER-PERSONA BREAKDOWN (Mean ± Std Dev) ---")
    for r in results:
        std = r['std']
        lines.append(f"\nPERSONA: {r['persona']}")
        lines.append(f"  Retrieval (Recall@250): {r['hit_rate_250']:.1%} ±{std['hit_rate_250']:.1%} | Binary Hit Rate: {r['binary_hit_250']:.1%}")
        lines.append(f"  Ranking (NDCG@10):      {r['ndcg_10']:.4f} ±{std['ndcg_10']:.4f}       | Recall@10:       {r['recall_10']:.1%}")
        lines.append(f"  Precision@10:           {r['genre_precision']:.1%} ±{std['genre_precision']:.1%}   | Diversity:       {r['diversity']:.3f} ±{std['diversity']:.3f}")
        
        rec_rank = f"{r['avg_rec_rank']:.1f}" if pd.notnull(r['avg_rec_rank']) else "N/A"
        peak_rank = f"{r['peak_rec_rank']:.0f}" if pd.notnull(r['peak_rec_rank']) else "N/A"
        
        lines.append(f"  Avg Rec Rank: {rec_rank} | Peak Rec Rank: {peak_rank}")
        lines.append(f"  MMR Gain:     {r.get('diversity_improvement_pct', 0):+.1%}")

    lines.append("\n" + "-"*80)
    lines.append("AGGREGATE SYSTEM AVERAGES")
    lines.append("-"*80)
    
    lines.append(f"\n1. TWO-TOWER NEURAL NETWORK (Candidate Generation)")
    lines.append(f"   Avg Rank (Catalog):    {df['avg_emb_rank'].mean():.0f} (Top {df['avg_emb_rank'].mean()/42800:.1%} of 42k items)")
    lines.append(f"   Avg Peak Rank (Mean):  {df['best_emb_rank'].mean():.0f} (Avg of best retrieved items)")
    lines.append(f"   Best Peak Rank (Abs):  {df['best_emb_rank'].min()} (Best performance observed)")
    
    lines.append(f"\n2. FINAL SYSTEM (MMR Reranking + Policy)")
    lines.append(f"   Avg Recall@250:        {df['hit_rate_250'].mean():.1%} (Items retained)")
    lines.append(f"   Avg Binary Hit Rate:   {df['binary_hit_250'].mean():.1%} (Sessions with success)")
    lines.append(f"   Avg NDCG@10:           {df['ndcg_10'].mean():.4f}")
    lines.append(f"   Genre Precision@10:    {df['genre_precision'].mean():.1%}")
    lines.append(f"   Intra-List Diversity:  {df['diversity'].mean():.3f}")
    lines.append(f"   MMR Diversity Gain:    +{df['diversity_improvement_pct'].mean():.1%}")
    
    lines.append("\n" + "="*80)
    
    report_text = "\n".join(lines)
    print(report_text)
    
    with open('metrics_output.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("\nSaved output to metrics_output.txt")

if __name__ == "__main__":
    run_full_evaluation()
