import os
from dotenv import load_dotenv

load_dotenv()

FAISS_DIR = "./faiss_store"  # must match build_index.py
EMBED_MODEL = "all-mpnet-base-v2"

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

TOP_K_VECTOR = 6
TOP_K_KEYWORD = 8
TOP_K_RERANKED = 8

TEMPERATURE = 0.1
MIN_EVIDENCE = 2
DEBUG_MODE = True

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many use cases to keep per query (configurable)
MAX_USE_CASES = 2

QUERIES = [
    "Generate positive, negative, and boundary test cases for Booking.com flight search based on the available PRD and screenshots.",
    "Generate use cases and test cases for the dashboard feature focusing on chart management, chart limit, and real-time updates.",
]


def validate_config() -> None:
    errors = []
    if not AZURE_ENDPOINT:
        errors.append("AZURE_OPENAI_ENDPOINT missing")
    if not AZURE_API_KEY:
        errors.append("AZURE_OPENAI_API_KEY missing")
    if not os.path.exists(FAISS_DIR):
        errors.append("FAISS directory './faiss_store' not found (run build_index.py)")
    if errors:
        print("❌ Config errors:")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)
