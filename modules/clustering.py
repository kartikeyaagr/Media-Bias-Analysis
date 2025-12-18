import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import gc

# Initialize SBERT (Global to avoid reloading)
print("Loading SBERT model (this may take a moment)...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def group_stories(df):
    """
    Groups stories into events using Event Threading with Time Decay.
    Reference: Nallapati et al. (2004)
    Non-Blocking Optimization: Uses float32 and in-place operations to minimize RAM usage.
    """
    if df.empty:
        return df

    print(f"Clustering {len(df)} stories using Time Decay...")

    # 1. Embeddings
    titles = df["title"].fillna("").tolist()
    # Ensure embeddings are float32
    print("Generating embeddings...")
    embeddings = model.encode(titles, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    # 2. Time Processing
    df["publish_date"] = pd.to_datetime(df["publish_date"])
    # Convert to days for distance calculation
    dates = (df["publish_date"] - df["publish_date"].min()).dt.days.values.reshape(
        -1, 1
    )
    dates = dates.astype(np.float32)

    # 3. Compute Distance/Similarity Matrices (In-Place Optimization)

    # A. Content Similarity
    # Calculate cosine distance (float32)
    # Note: sklearn returns float64 by default, so we cast immediately.
    print("Computing content distance...")
    # We use a single large matrix 'sim_matrix' to save memory.
    # Initially effectively content_dist
    sim_matrix = cosine_distances(embeddings).astype(np.float32)

    # Free embeddings memory
    del embeddings
    gc.collect()

    # Convert distance to similarity: Sim = 1 - Dist
    # Operation: sim_matrix = 1 - sim_matrix
    print("Converting to content similarity...")
    sim_matrix *= -1
    sim_matrix += 1

    # Clip to [0, 1] in-place
    np.clip(sim_matrix, 0, 1, out=sim_matrix)

    # B. Time Decay
    print("Computing time decay...")
    time_dist = euclidean_distances(dates).astype(np.float32)

    # Decay factor = exp(-alpha * time_dist)
    ALPHA = 0.15

    # In-place: time_dist = -ALPHA * time_dist
    time_dist *= -ALPHA

    # In-place: time_dist = exp(time_dist) -> This becomes our decay_factor matrix
    np.exp(time_dist, out=time_dist)

    # C. Combined Similarity
    # Sim_combined = Sim_content * Decay_factor
    print("Applying time decay to similarity...")
    sim_matrix *= time_dist

    # Free time matrix
    del time_dist
    gc.collect()

    # 4. Convert back to Distance for Clustering
    # Final_Dist = 1 - Combined_Sim
    print("Converting to final distance matrix...")
    sim_matrix *= -1
    sim_matrix += 1

    # Clip safe
    np.clip(sim_matrix, 0, 1, out=sim_matrix)

    # 5. Clustering
    print("Running Agglomerative Clustering...")
    clustering_model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.5, metric="precomputed", linkage="average"
    )

    df["cluster_id"] = clustering_model.fit_predict(sim_matrix)
    print(f"Identified {len(set(df['cluster_id']))} clusters.")

    # Cleanup final matrix
    del sim_matrix
    gc.collect()

    return df
