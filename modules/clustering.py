import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# Initialize SBERT (Global to avoid reloading)
print("Loading SBERT model (this may take a moment)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def group_stories(df):
    """
    Groups stories into events using Event Threading with Time Decay.
    Reference: Nallapati et al. (2004)
    """
    if df.empty: return df
    
    print(f"Clustering {len(df)} stories using Time Decay...")
    
    # 1. Embeddings
    titles = df['title'].fillna('').tolist()
    embeddings = model.encode(titles, show_progress_bar=True)
    
    # 2. Time Processing
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    # Convert to days for distance calculation
    dates = (df['publish_date'] - df['publish_date'].min()).dt.days.values.reshape(-1, 1)
    
    # 3. Compute Similarity Matrices
    # Cosine distance (0 to 2) -> Similarity (1 to -1)
    # We map distance 0->1, 1->0, 2->-1. 
    # Actually, standard cosine similarity is dot product of normalized vectors.
    # sklearn cosine_distances = 1 - cosine_similarity.
    # So Similarity = 1 - Distance.
    content_dist = cosine_distances(embeddings)
    content_sim = 1 - content_dist
    
    # Ensure range [0, 1] for stability (clip negative values if any)
    content_sim = np.clip(content_sim, 0, 1)
    
    # Time distance in days
    time_dist = euclidean_distances(dates)
    
    # 4. Apply Time Decay
    # Formula: Sim_combined = Sim_content * exp(-alpha * delta_t)
    ALPHA = 0.15 # Decay rate
    
    decay_factor = np.exp(-ALPHA * time_dist)
    combined_sim = content_sim * decay_factor
    
    # 5. Convert back to Distance for Clustering
    # Distance = 1 - Similarity
    final_dist = 1 - combined_sim
    
    # Ensure no negative distances due to floating point errors
    final_dist = np.clip(final_dist, 0, 1)
    
    # 6. Clustering
    # Threshold: If Sim < 0.5, then Dist > 0.5.
    # With decay, even identical stories (Sim=1) will drop below 0.5 after:
    # exp(-0.15 * t) = 0.5 => -0.15 * t = ln(0.5) => t ~= 4.6 days.
    # So identical stories > 5 days apart will likely be split.
    # Very similar stories (0.8) will be split sooner.
    clustering_model = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=0.5, 
        metric='precomputed', 
        linkage='average' 
    )
    
    df['cluster_id'] = clustering_model.fit_predict(final_dist)
    print(f"Identified {len(set(df['cluster_id']))} clusters.")
    return df
