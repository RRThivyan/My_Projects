"""
Configuration file for BFSI Call Center AI Assistant
Update paths according to your Google Drive structure
"""

import os

# ============================================================
# GOOGLE DRIVE PATHS
# ============================================================

# Base directory in Google Drive
DRIVE_BASE = "/content/drive/MyDrive/Vertical_AI"

# Dataset path
DATASET_PATH = os.path.join(DRIVE_BASE, "bfsi_cleaned_dataset.json")

# Model paths
MODEL_BASE_PATH = "microsoft/Phi-3.5-mini-instruct"
LORA_ADAPTER_PATH = os.path.join(DRIVE_BASE, "phi35-bfsi-lora")
MERGED_MODEL_PATH = os.path.join(DRIVE_BASE, "phi35-bfsi-merged")

# Documents and index paths
DOCS_PATH = os.path.join(DRIVE_BASE, "rag_docs")
INDEX_PATH = os.path.join(DRIVE_BASE, "rag_index")

# ============================================================
# MODEL SETTINGS
# ============================================================

# Tier 1: Dataset Similarity
TIER1_MODEL = "all-MiniLM-L6-v2"
TIER1_THRESHOLD = 0.72
TIER1_TOP_K = 3

# Tier 2: Fine-tuned SLM
TIER2_MAX_TOKENS = 200
TIER2_TEMPERATURE = 0.7
TIER2_TOP_P = 0.9

# Tier 3: RAG
TIER3_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TIER3_RERANKER_MODEL = "BAAI/bge-reranker-base"
TIER3_CHUNK_SIZE = 1000
TIER3_CHUNK_OVERLAP = 200
TIER3_TOP_K = 5
TIER3_CONFIDENCE_THRESHOLD = 0.3

# ============================================================
# SYSTEM SETTINGS
# ============================================================

# Use merged model (True) or LoRA adapter (False)
USE_MERGED_MODEL = False  # Set to True if you've merged the model

# ChromaDB collection name
CHROMA_COLLECTION_NAME = "bfsi_docs_final"

# Device settings
DEVICE = "auto"  # "auto", "cuda", "cpu"
TORCH_DTYPE = "float16"  # "float16" or "float32"

# ============================================================
# OUT-OF-DOMAIN KEYWORDS
# ============================================================

OUT_OF_DOMAIN_KEYWORDS = [
    "weather", "sports", "movie", "cricket", "politics",
    "recipe", "song", "game", "celebrity", "news"
]

# ============================================================
# GRADIO SETTINGS
# ============================================================

GRADIO_SHARE = True  # Create public link
GRADIO_DEBUG = True  # Show debug info
GRADIO_PORT = 7860

# ============================================================
# VALIDATION
# ============================================================

def validate_config():
    """Validate that all required paths exist"""
    errors = []
    
    # Check dataset
    if not os.path.exists(DATASET_PATH):
        errors.append(f"Dataset not found: {DATASET_PATH}")
    
    # Check model paths
    if USE_MERGED_MODEL:
        if not os.path.exists(MERGED_MODEL_PATH):
            errors.append(f"Merged model not found: {MERGED_MODEL_PATH}")
    else:
        if not os.path.exists(LORA_ADAPTER_PATH):
            errors.append(f"LoRA adapter not found: {LORA_ADAPTER_PATH}")
    
    # Check docs path
    if not os.path.exists(DOCS_PATH):
        errors.append(f"Documents folder not found: {DOCS_PATH}")
    
    # Create index directory if it doesn't exist
    os.makedirs(INDEX_PATH, exist_ok=True)
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("✅ Configuration validated successfully")
    return True

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_model_path():
    """Get the appropriate model path based on settings"""
    return MERGED_MODEL_PATH if USE_MERGED_MODEL else LORA_ADAPTER_PATH

def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Documents: {DOCS_PATH}")
    print(f"Index: {INDEX_PATH}")
    print(f"Model: {'Merged' if USE_MERGED_MODEL else 'LoRA Adapter'}")
    print(f"Model Path: {get_model_path()}")
    print("=" * 60)
