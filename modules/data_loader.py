import pandas as pd
import os
import config

# Map topic names from config.QUERIES to filenames in data/
TOPIC_FILE_MAP = {
    "Elections": "elections.csv",
    "Budget": "budget.csv",
    "Supreme Court": "judgements.csv",
    "Military": "military.csv",
    "Foreign Policy": "foreign.csv",
}


def get_data(topic_name):
    """
    Loads data for a given topic from the local 'data' directory.
    Replaces previous MediaCloud API fetching logic.
    """
    if topic_name not in TOPIC_FILE_MAP:
        print(f"Warning: No local file mapping found for topic '{topic_name}'.")
        return pd.DataFrame()

    filename = TOPIC_FILE_MAP[topic_name]
    # Construct absolute path to data directory
    # Assuming data/ is at the project root, same level as main.py
    # We can use a relative path or construct it relative to this file
    # But since main.py is entry point, 'data/' works if cwd is project root.
    # To be safer, let's look for it relative to config.py or use a known path.
    # The user provided path: /Users/karti/Desktop/Ashoka/Monsoon25/Capstone Project/data

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", filename)

    print(f"Loading data for {topic_name} from {data_path}...")

    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return pd.DataFrame()

    try:
        # Try UTF-8 first
        try:
            df = pd.read_csv(data_path, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            print(
                f"Warning: UTF-8 decoding failed for {filename}. Retrying with 'latin-1'..."
            )
            df = pd.read_csv(data_path, encoding="latin-1", on_bad_lines="skip")

        # Ensure 'publish_date' is standard format if needed, though main.py just passes it through mostly.
        # But 'main.py' logic might expect it to be a string or date object.
        # Let's check main.py usage. It writes str(row.get("publish_date")) to json.
        # analysis.py might parse it.

        print(f"Loaded {len(df)} stories for {topic_name}.")
        return df

    except Exception as e:
        print(f"Error reading {data_path}: {e}")
        return pd.DataFrame()
