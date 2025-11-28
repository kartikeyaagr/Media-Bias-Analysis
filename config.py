import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
MC_API_KEY = os.getenv("MC_API_KEY")
if not MC_API_KEY:
    raise ValueError("MC_API_KEY not found in .env file")

# Analysis Constants
START_DATE = "2020-01-01"
END_DATE = "2025-06-30"
STORY_LIMIT = 20000

# Collection IDs
INDIA_NATIONAL_COL = 34412118
INDIA_STATE_LOCAL_COL = 38379954

# Output Directories
OUTPUT_DIR = "output"
DIRS = {
    "coverage": os.path.join(OUTPUT_DIR, "coverage"),
    "networks": os.path.join(OUTPUT_DIR, "networks"),
    "sentiment": os.path.join(OUTPUT_DIR, "sentiment"),
    "framing": os.path.join(OUTPUT_DIR, "framing"),
    "keywords": os.path.join(OUTPUT_DIR, "keywords"),
}

# Queries
QUERIES = {
    "Elections": """("national election*" OR "general election*" OR "lok sabha" OR "parliamentary election*" OR title:("modi" OR "rahul" OR "bjp" OR "congress" OR "nda" OR "india bloc")) AND (poll* OR bypoll* OR turnout OR "phase ?" OR "counting day" OR "election commission"~3 OR ECI OR "model code"~3 OR MCC) AND (vote* OR ballot* OR elector* OR EVM* OR VVPAT* OR franchise OR constituency OR manifesto OR candidate*) AND NOT (cricket OR bollywood OR IPL OR "box office" OR tennis OR hockey)""",
    "Budget": """(title:("union budget" OR "budget 20*" OR "interim budget" OR "railway budget") OR "finance bill" OR "economic survey" OR "annual financial statement") AND (tax* OR fiscal OR deficit OR capex OR "capital expenditure" OR GDP OR inflation OR "customs duty" OR GST OR slab* OR cess OR surcharge OR "finance minister"~3 OR "nirmala sitharaman") AND NOT ("film budget" OR "movie budget" OR "box office" OR "big budget" OR cricket OR bollywood)""",
    "Supreme Court": """(title:("supreme court" OR "apex court" OR "top court" OR "constitution bench") OR "supreme court of india" OR "SC bench"~3 OR "CJI"~3 OR "chief justice"~3) AND (verdict OR judgment OR judgement OR "suo motu" OR plea OR PIL OR "public interest litigation" OR "curative petition" OR "review petition" OR stay OR bail OR "contempt of court" OR affidavit OR "status quo" OR collegium OR "bar council") AND NOT ("scheduled caste" OR "SC/ST" OR tennis OR badminton OR basketball)""",
    "Military": """("indian army" OR "indian navy" OR "indian air force" OR IAF OR "defense ministry" OR MoD OR "ministry of defence" OR "border security force" OR BSF OR CRPF OR "coast guard" OR "chief of defence staff" OR CDS OR DRDO) AND (operation* OR mission OR encounter OR terror* OR infiltration OR "counter insurgency" OR "line of control" OR LOC OR "line of actual control" OR LAC OR galwan OR tawang OR ladakh OR doklam OR "surgical strike" OR "joint exercise" OR wargame* OR "test fire*" OR missile OR drone) AND NOT ("bts army" OR k-pop OR cricket OR football OR "defense mechanism")""",
    "Foreign Policy": """(title:("prime minister" OR "PM modi" OR "external affairs" OR jaishankar OR MEA) OR "foreign ministry" OR "diplomatic corps" OR ambassador OR envoy OR "high commission*") AND (G20 OR G7 OR BRICS OR SCO OR QUAD OR ASEAN OR "global south" OR "united nations" OR UNSC OR "security council" OR "world economic forum" OR davos OR "non-aligned movement" OR NAM) AND (summit OR bilateral OR trilateral OR multilateral OR "joint statement" OR MoU OR "memorandum of understanding" OR pact OR treaty OR "strategic partnership" OR "state visit" OR extradition OR diaspora) AND NOT ("investor summit" OR "business summit" OR "tech summit" OR "foreign direct investment" OR FDI)""",
}
