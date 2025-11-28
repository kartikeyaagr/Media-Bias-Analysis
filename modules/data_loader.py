import pandas as pd
import mediacloud.api
import datetime
from tqdm import tqdm
import config


def get_sources(col_id):
    MC_API_KEY = config.MC_API_KEY
    if not MC_API_KEY:
        raise ValueError("MC_API_KEY not found in .env file")

    mc_directory = mediacloud.api.DirectoryApi(MC_API_KEY)
    sources = []
    offset = 0
    while True:
        response = mc_directory.source_list(
            collection_id=col_id, limit=100, offset=offset
        )
        sources += response["results"]
        if response["next"] is None:
            break
        offset += len(response["results"])
    return sources


def load_sources():
    """Loads national and local sources from MediaCloud API."""
    try:
        print("Fetching national sources...")
        national = get_sources(config.INDIA_NATIONAL_COL)
        print(f"Fetched {len(national)} national sources.")
        
        print("Fetching local sources...")
        local = get_sources(config.INDIA_STATE_LOCAL_COL)
        print(f"Fetched {len(local)} local sources.")
        
        # Combine and deduplicate
        all_sources = national + local
        # Deduplicate by ID
        seen_ids = set()
        unique_sources = []
        for s in all_sources:
            if s['id'] not in seen_ids:
                unique_sources.append(s)
                seen_ids.add(s['id'])
                
        return pd.DataFrame(unique_sources)
    except Exception as e:
        print(f"Error loading sources: {e}")
        return pd.DataFrame()


import time


def fetch_stories(topic_name, query):
    """Fetches stories from MediaCloud using logic from analysis.ipynb."""
    print(f"Fetching stories for {topic_name}...")

    # Logic from notebook's get_stories
    api_key = config.MC_API_KEY
    search_api = mediacloud.api.SearchApi(api_key)

    # Load sources
    sources_df = load_sources()
    source_ids = []
    if not sources_df.empty:
        source_ids = sources_df["id"].tolist()
        source_ids = [int(sid) for sid in source_ids if pd.notna(sid)]
        # Deduplicate
        source_ids = list(set(source_ids))

    start_date = datetime.datetime.strptime(config.START_DATE, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(config.END_DATE, "%Y-%m-%d").date()
    limit = config.STORY_LIMIT

    all_stories = []
    more_stories = True
    pagination_token = None

    # Using print instead of tqdm to avoid conflict with main pipeline output
    print(f"Fetching max {limit} stories...")

    while more_stories and len(all_stories) < limit:
        try:
            page, pagination_token = search_api.story_list(
                query,
                start_date,
                end_date,
                source_ids=source_ids,
                pagination_token=pagination_token,
            )
            all_stories += page
            print(f"Fetched {len(page)} stories. Total: {len(all_stories)}")
            more_stories = pagination_token is not None
        except Exception as e:
            print(f"Error fetching stories: {e}")
            time.sleep(5)
            continue

    stories_df = pd.DataFrame(all_stories)
    print(f"Fetched {len(stories_df)} stories for {topic_name}.")
    return stories_df


def get_data(topic_name):
    """Main entry point to get data for a topic."""
    if topic_name not in config.QUERIES:
        raise ValueError(f"Topic {topic_name} not found in config.")

    query = config.QUERIES[topic_name]
    return fetch_stories(topic_name, query)
