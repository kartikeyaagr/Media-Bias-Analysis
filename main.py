import os

# Disable joblib's resource tracker component to prevent "leaked semaphore" warnings
# This must be set before any other imports that might use joblib
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
import config
from modules import data_loader, clustering, analysis, visualization
import pandas as pd


import shutil


def run_analysis_pipeline():
    print("Starting Media Bias Analysis Pipeline...")
    print(f"Output Directory: {config.OUTPUT_DIR}")

    # Clear previous outputs
    if os.path.exists(config.OUTPUT_DIR):
        print("Clearing previous outputs...")
        shutil.rmtree(config.OUTPUT_DIR)

    # Ensure output directories exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    for folder_path in config.DIRS.values():
        os.makedirs(folder_path, exist_ok=True)

    # Iterate through all topics defined in config
    for topic_name in config.QUERIES.keys():
        print(f"\n{'='*50}")
        print(f"Processing Topic: {topic_name}")
        print(f"{'='*50}")

        # 1. Load Data
        try:
            df = data_loader.get_data(topic_name)
            if df.empty:
                print(f"No stories found for {topic_name}. Skipping.")
                continue
        except Exception as e:
            print(f"Error loading data for {topic_name}: {e}")
            continue

        # 2. Clustering
        df = clustering.group_stories(df)
        num_clusters = df["cluster_id"].nunique()
        print(
            f"DEBUG: Topic '{topic_name}' - Formed {num_clusters} clusters from {len(df)} stories."
        )

        # 3. Analysis
        df = analysis.analyze_sentiment(df)

        # Keywords
        print("Extracting keywords...")
        keywords = analysis.extract_keywords(df)

        # Network
        G = analysis.build_network(df)

        # 4. Visualization
        print("Generating visualizations...")
        visualization.plot_coverage_over_time(df, topic_name)
        visualization.plot_sentiment_distribution(df, topic_name)
        visualization.plot_event_framing(df, topic_name)
        visualization.plot_source_network(G, topic_name)
        visualization.plot_top_keywords(df, topic_name)

        # 5. Generate Text Report
        report_path = os.path.join(config.OUTPUT_DIR, f"{topic_name}_report.txt")
        print(f"Generating report: {report_path}")
        with open(report_path, "w") as f:
            f.write(f"Analysis Report for {topic_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Stories: {len(df)}\n")
            f.write(f"Total Clusters: {len(set(df['cluster_id']))}\n\n")

            f.write("Top Keywords:\n")
            for k, v in keywords.items():
                f.write(f"  - {k}: {v:.2f}\n")
            f.write("\n")

            f.write("Top Event Clusters:\n")
            top_clusters = df["cluster_id"].value_counts().head(10)
            for cid, count in top_clusters.items():
                cluster_df = df[df["cluster_id"] == cid]
                sample_title = cluster_df["title"].iloc[0]
                f.write(f"  - Cluster {cid} ({count} stories): {sample_title}\n")

        # 6. Generate JSON Export
        json_path = os.path.join(config.OUTPUT_DIR, f"{topic_name}_clusters.json")
        print(f"Generating JSON export: {json_path}")

        # Group by cluster_id
        clusters_data = []
        for cid, cluster_df in df.groupby("cluster_id"):
            stories = []
            for _, row in cluster_df.iterrows():
                stories.append(
                    {
                        "title": row.get("title", ""),
                        "url": row.get("url", ""),
                        "publish_date": str(row.get("publish_date", "")),
                        "media_name": row.get("media_name", ""),
                        "sentiment": row.get("sentiment_label", ""),
                    }
                )

            clusters_data.append(
                {"cluster_id": int(cid), "size": len(stories), "stories": stories}
            )

        # Sort by size descending
        clusters_data.sort(key=lambda x: x["size"], reverse=True)

        import json

        with open(json_path, "w") as f:
            json.dump(
                {
                    "topic": topic_name,
                    "total_stories": len(df),
                    "total_clusters": len(clusters_data),
                    "clusters": clusters_data,
                },
                f,
                indent=2,
            )

        # Clean up memory
        print(f"Cleaning up memory for {topic_name}...")
        del df
        if "G" in locals():
            del G
        if "keywords" in locals():
            del keywords
        import gc

        gc.collect()

    print("\nPipeline Completed Successfully.")


if __name__ == "__main__":
    run_analysis_pipeline()
