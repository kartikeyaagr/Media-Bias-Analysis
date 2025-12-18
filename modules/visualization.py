import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import os
import community.community_louvain as community_louvain
import matplotlib.dates as mdates
import config
from modules import analysis

# Set Style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def save_plot(fig, folder, filename):
    """Helper to save plots to the correct directory."""
    path = os.path.join(config.DIRS[folder], filename)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved plot: {path}")


def plot_coverage_over_time(df, topic_name):
    """Plots the coverage volume of top event clusters over time."""
    if df.empty or "cluster_id" not in df.columns:
        return

    df["publish_date"] = pd.to_datetime(df["publish_date"])
    top_clusters = df["cluster_id"].value_counts().head(5).index
    df_top = df[df["cluster_id"].isin(top_clusters)]

    # Create descriptive labels
    cluster_labels = {}
    for cid in top_clusters:
        cluster_df = df[df["cluster_id"] == cid]
        # Use the most frequent title as the representative label
        title_counts = cluster_df["title"].value_counts()
        if not title_counts.empty:
            best_title = title_counts.index[0]
            # Truncate to 50 chars
            if len(best_title) > 60:
                best_title = best_title[:57] + "..."
        else:
            best_title = "Unknown Event"

        cluster_labels[cid] = f"{best_title} (ID:{cid})"

    monthly_counts = (
        df_top.groupby([pd.Grouper(key="publish_date", freq="ME"), "cluster_id"])
        .size()
        .unstack(fill_value=0)
    )

    # Rename columns
    monthly_counts.rename(columns=cluster_labels, inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_counts.plot(kind="area", stacked=True, alpha=0.7, ax=ax, cmap="tab10")

    ax.set_title(f"{topic_name}: Top 5 Events Coverage Over Time", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Stories")

    # Format x-axis
    start_date = pd.to_datetime(config.START_DATE)
    end_date = pd.to_datetime(config.END_DATE)
    ax.set_xlim(start_date, end_date)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    ax.legend(title="Event Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")

    save_plot(fig, "coverage", f"{topic_name}_coverage.png")


def plot_sentiment_distribution(df, topic_name):
    """Plots overall sentiment distribution and subjectivity."""
    if df.empty or "sentiment_label" not in df.columns:
        return

    # 1. Sentiment Count Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    # Explicit colors: Negative=Red, Neutral=Grey, Positive=Green
    palette = {"Negative": "#ff9999", "Neutral": "#d3d3d3", "Positive": "#99ff99"}

    sns.countplot(
        x="sentiment_label",
        data=df,
        order=["Negative", "Neutral", "Positive"],
        hue="sentiment_label",
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_title(f"{topic_name}: Overall Sentiment Distribution", fontsize=14)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")

    save_plot(fig, "sentiment", f"{topic_name}_sentiment_counts.png")

    # 2. Subjectivity Histogram
    if "subjectivity_score" in df.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.histplot(
            df["subjectivity_score"], bins=20, kde=True, color="purple", ax=ax2
        )
        ax2.set_title(
            f"{topic_name}: Subjectivity Distribution (0=Fact, 1=Opinion)", fontsize=14
        )
        ax2.set_xlabel("Subjectivity Score")

        save_plot(fig2, "sentiment", f"{topic_name}_subjectivity.png")


def plot_event_framing(df, topic_name):
    """Plots how different sources frame the top events."""
    if df.empty or "cluster_id" not in df.columns:
        return

    top_clusters = df["cluster_id"].value_counts().head(3).index.tolist()

    for cluster_id in top_clusters:
        cluster_df = df[df["cluster_id"] == cluster_id]
        if len(cluster_df) < 5:
            continue

        top_sources = cluster_df["media_name"].value_counts().head(5).index.tolist()
        source_df = cluster_df[cluster_df["media_name"].isin(top_sources)].copy()

        if source_df.empty:
            continue

        # Calculate Net Sentiment (Pos - Neg) for sorting
        source_stats = []
        for source in top_sources:
            s_df = source_df[source_df["media_name"] == source]
            n = len(s_df)
            if n == 0:
                continue

            pos_pct = (s_df["sentiment_label"] == "Positive").mean() * 100
            neg_pct = (s_df["sentiment_label"] == "Negative").mean() * 100
            net_sentiment = pos_pct - neg_pct
            source_stats.append(
                {"source": source, "net_sentiment": net_sentiment, "n": n}
            )

        # Sort by Net Sentiment
        source_stats.sort(key=lambda x: x["net_sentiment"])
        sorted_sources = [x["source"] for x in source_stats]

        # Rename sources with n=X
        source_labels = {
            x["source"]: f"{x['source']} (n={x['n']})" for x in source_stats
        }
        source_df["display_name"] = source_df["media_name"].map(source_labels)

        sentiment_counts = (
            source_df.groupby(["display_name", "sentiment_label"])
            .size()
            .unstack(fill_value=0)
        )

        # Reorder based on sorted sources
        sorted_labels = [source_labels[s] for s in sorted_sources if s in source_labels]
        sentiment_counts = sentiment_counts.reindex(sorted_labels)

        sentiment_pcts = (
            sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
        )

        # Ensure columns exist
        for col in ["Negative", "Neutral", "Positive"]:
            if col not in sentiment_pcts.columns:
                sentiment_pcts[col] = 0
        sentiment_pcts = sentiment_pcts[["Negative", "Neutral", "Positive"]]

        fig, ax = plt.subplots(figsize=(10, 5))
        sentiment_pcts.plot(
            kind="bar", stacked=True, color=["#ff9999", "#d3d3d3", "#99ff99"], ax=ax
        )

        sample_title = cluster_df["title"].iloc[0][:50] + "..."
        ax.set_title(f'Framing of Event {cluster_id}: "{sample_title}"', fontsize=12)
        ax.set_ylabel("Percentage")
        ax.set_xlabel("Source")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        save_plot(fig, "framing", f"{topic_name}_framing_{cluster_id}.png")


def plot_source_network(G, topic_name):
    """Plots the source similarity network."""
    if G is None:
        return

    # Community detection
    try:
        partition = community_louvain.best_partition(G)
    except Exception as e:
        print(f"Community detection failed: {e}")
        partition = {n: 0 for n in G.nodes()}

    fig, ax = plt.subplots(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5, seed=42)

    # Node sizing based on degree
    degrees = dict(G.degree())
    # scikit-network-style sizing (scaled)
    # INCREASED SIZE: Base 1000, Multiplier 300
    node_sizes = [degrees[n] * 300 + 1000 for n in G.nodes()]

    # Node coloring
    node_colors = [partition.get(n) for n in G.nodes()]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.get_cmap("tab10"),
        alpha=0.8,
        ax=ax,
    )

    # Edge styling
    # INCREASED THICKNESS: Multiplier 5 (was 2)
    weights = [G[u][v].get("weight", 0) * 5 for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="black", width=weights, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black", ax=ax)

    ax.set_title(
        f"{topic_name}: Source Similarity Network (Echo Chambers)", fontsize=16
    )
    ax.axis("off")

    # Create legend for communities (optional but good)
    # We can skip explicit legend for now as requested just "color the nodes"

    save_plot(fig, "networks", f"{topic_name}_network.png")


def plot_top_keywords(df, topic_name):
    """Plots comparison of top keywords in Positive vs Negative headlines."""
    if df.empty or "sentiment_label" not in df.columns:
        return

    # Split by sentiment
    pos_df = df[df["sentiment_label"] == "Positive"]
    neg_df = df[df["sentiment_label"] == "Negative"]

    pos_kw = analysis.extract_keywords(pos_df, top_n=10)
    neg_kw = analysis.extract_keywords(neg_df, top_n=10)

    if not pos_kw and not neg_kw:
        return

    # Prepare data for plotting
    data = []
    for k, v in pos_kw.items():
        data.append({"Keyword": k, "Score": v, "Sentiment": "Positive"})
    for k, v in neg_kw.items():
        data.append({"Keyword": k, "Score": v, "Sentiment": "Negative"})

    df_kw = pd.DataFrame(data)

    if df_kw.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot with hue for comparison
    sns.barplot(
        x="Score",
        y="Keyword",
        hue="Sentiment",
        data=df_kw,
        palette={"Positive": "#99ff99", "Negative": "#ff9999"},
        ax=ax,
    )

    ax.set_title(f"{topic_name}: Top Keywords by Sentiment", fontsize=14)
    ax.set_xlabel("TF-IDF Score")

    save_plot(fig, "keywords", f"{topic_name}_keywords_comparison.png")
