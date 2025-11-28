import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

def analyze_sentiment(df):
    """
    Applies VADER sentiment and TextBlob subjectivity analysis.
    """
    if df.empty: return df
    
    print("Running Sentiment & Subjectivity Analysis...")
    analyzer = SentimentIntensityAnalyzer()
    
    def get_vader_scores(text):
        return analyzer.polarity_scores(str(text))['compound']
    
    def get_subjectivity(text):
        return TextBlob(str(text)).sentiment.subjectivity

    df['sentiment_score'] = df['title'].apply(get_vader_scores)
    df['subjectivity_score'] = df['title'].apply(get_subjectivity)
    
    # Categorize
    def get_label(score):
        if score >= 0.05: return 'Positive'
        if score <= -0.05: return 'Negative'
        return 'Neutral'
        
    df['sentiment_label'] = df['sentiment_score'].apply(get_label)
    
    return df

def extract_keywords(df, top_n=10):
    """
    Extracts top keywords using TF-IDF.
    Returns a dictionary of {keyword: score}.
    """
    if df.empty: return {}
    
    try:
        # Stop words + common news terms
        stop_words = 'english' 
        # We could add custom stop words here if needed
        
        vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=top_n)
        tfidf_matrix = vectorizer.fit_transform(df['title'].fillna(''))
        
        feature_names = vectorizer.get_feature_names_out()
        # Sum tfidf scores for each term across all documents
        dense = tfidf_matrix.todense()
        denselist = dense.tolist()
        df_tfidf = pd.DataFrame(denselist, columns=feature_names)
        
        keywords = df_tfidf.sum().sort_values(ascending=False).head(top_n).to_dict()
        return keywords
    except ValueError:
        # Handle empty vocabulary or other issues
        return {}

def build_network(df):
    """
    Builds a source similarity network based on shared event clusters.
    """
    if df.empty or 'cluster_id' not in df.columns: return None
    
    print("Building source network...")
    # Filter for top sources to keep graph manageable
    top_sources = df['media_name'].value_counts().head(20).index.tolist()
    df_top = df[df['media_name'].isin(top_sources)]
    
    source_clusters = df_top.groupby('media_name')['cluster_id'].apply(set).to_dict()
    
    G = nx.Graph()
    for s in top_sources: G.add_node(s)
    
    sources = list(source_clusters.keys())
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            s1, s2 = sources[i], sources[j]
            set1, set2 = source_clusters[s1], source_clusters[s2]
            
            union_len = len(set1.union(set2))
            if union_len > 0:
                jaccard = len(set1.intersection(set2)) / union_len
                if jaccard > 0.1: # Threshold for edge
                    G.add_edge(s1, s2, weight=jaccard)
    return G
