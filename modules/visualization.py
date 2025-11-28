import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import os
import config

# Set Style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def save_plot(fig, folder, filename):
    """Helper to save plots to the correct directory."""
    path = os.path.join(config.DIRS[folder], filename)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved plot: {path}")

def plot_coverage_over_time(df, topic_name):
    """Plots the coverage volume of top event clusters over time."""
    if df.empty or 'cluster_id' not in df.columns: return
    
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    top_clusters = df['cluster_id'].value_counts().head(5).index
    df_top = df[df['cluster_id'].isin(top_clusters)]
    
    monthly_counts = df_top.groupby([pd.Grouper(key='publish_date', freq='M'), 'cluster_id']).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_counts.plot(kind='area', stacked=True, alpha=0.7, ax=ax, cmap='tab10')
    
    ax.set_title(f'{topic_name}: Top 5 Events Coverage Over Time', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Stories')
    ax.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_plot(fig, 'coverage', f'{topic_name}_coverage.png')

def plot_sentiment_distribution(df, topic_name):
    """Plots overall sentiment distribution and subjectivity."""
    if df.empty or 'sentiment_label' not in df.columns: return
    
    # 1. Sentiment Count Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    # Explicit colors: Negative=Red, Neutral=Grey, Positive=Green
    palette = {'Negative': '#ff9999', 'Neutral': '#d3d3d3', 'Positive': '#99ff99'}
    
    sns.countplot(x='sentiment_label', data=df, order=['Negative', 'Neutral', 'Positive'], palette=palette, ax=ax)
    ax.set_title(f'{topic_name}: Overall Sentiment Distribution', fontsize=14)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    
    save_plot(fig, 'sentiment', f'{topic_name}_sentiment_counts.png')
    
    # 2. Subjectivity Histogram
    if 'subjectivity_score' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.histplot(df['subjectivity_score'], bins=20, kde=True, color='purple', ax=ax2)
        ax2.set_title(f'{topic_name}: Subjectivity Distribution (0=Fact, 1=Opinion)', fontsize=14)
        ax2.set_xlabel('Subjectivity Score')
        
        save_plot(fig2, 'sentiment', f'{topic_name}_subjectivity.png')

def plot_event_framing(df, topic_name):
    """Plots how different sources frame the top events."""
    if df.empty or 'cluster_id' not in df.columns: return
    
    top_clusters = df['cluster_id'].value_counts().head(3).index.tolist()
    
    for cluster_id in top_clusters:
        cluster_df = df[df['cluster_id'] == cluster_id]
        if len(cluster_df) < 5: continue
        
        top_sources = cluster_df['media_name'].value_counts().head(5).index.tolist()
        source_df = cluster_df[cluster_df['media_name'].isin(top_sources)]
        
        if source_df.empty: continue
        
        sentiment_counts = source_df.groupby(['media_name', 'sentiment_label']).size().unstack(fill_value=0)
        sentiment_pcts = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
        
        # Ensure columns exist
        for col in ['Negative', 'Neutral', 'Positive']:
            if col not in sentiment_pcts.columns: sentiment_pcts[col] = 0
        sentiment_pcts = sentiment_pcts[['Negative', 'Neutral', 'Positive']]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sentiment_pcts.plot(kind='bar', stacked=True, color=['#ff9999', '#d3d3d3', '#99ff99'], ax=ax)
        
        sample_title = cluster_df['title'].iloc[0][:50] + "..."
        ax.set_title(f'Framing of Event {cluster_id}: "{sample_title}"', fontsize=12)
        ax.set_ylabel('Percentage')
        ax.set_xlabel('Source')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        save_plot(fig, 'framing', f'{topic_name}_framing_{cluster_id}.png')

def plot_source_network(G, topic_name):
    """Plots the source similarity network."""
    if G is None: return
    
    fig, ax = plt.subplots(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='#1f77b4', alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', ax=ax)
    
    ax.set_title(f'{topic_name}: Source Similarity Network (Echo Chambers)', fontsize=16)
    ax.axis('off')
    
    save_plot(fig, 'networks', f'{topic_name}_network.png')

def plot_top_keywords(keywords, topic_name, suffix=""):
    """Plots bar chart of top keywords."""
    if not keywords: return
    
    df_kw = pd.DataFrame(list(keywords.items()), columns=['Keyword', 'Score'])
    df_kw = df_kw.sort_values('Score', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Score', y='Keyword', data=df_kw, palette='viridis', ax=ax)
    
    ax.set_title(f'{topic_name}: Top Keywords {suffix}', fontsize=14)
    ax.set_xlabel('TF-IDF Score')
    
    save_plot(fig, 'keywords', f'{topic_name}_keywords{suffix}.png')
