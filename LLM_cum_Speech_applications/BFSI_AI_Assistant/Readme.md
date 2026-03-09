# 🏦 BFSI Call Center AI Assistant - Demo

A lightweight, production-ready demo system for handling Banking, Financial Services, and Insurance (BFSI) call center queries with a 3-tier architecture. Built for local deployment with strict compliance and safety guidelines.

## 📋 Table of Contents
- [Demo Overview](#demo-overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Dataset](#dataset)
- [Model](#model)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Module Documentation](#module-documentation)
- [Configuration](#configuration)
- [Demo Performance](#demo-performance)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 🎯 Demo Overview

This demo system provides intelligent, compliant responses to BFSI queries using a three-tier architecture designed for local deployment:

| Tier | Type | Coverage | Response Time | Use Case |
|------|------|----------|---------------|----------|
| **Tier 1** | Dataset Similarity | ~70% | ~15ms | Common queries (interest rates, eligibility) |
| **Tier 2** | Fine-tuned SLM | ~20% | ~500ms | Variations of common queries |
| **Tier 3** | RAG with Documents | ~10% | ~2s | Complex policy questions |

### Key Demo Features
- ✅ **Local Execution**: Runs entirely on-premises with no API calls
- ✅ **Compliant**: Strict guardrails for financial regulations
- ✅ **Fast**: 70% of queries answered in milliseconds
- ✅ **Accurate**: Hybrid search with reranking for precision
- ✅ **Modular**: Easy to maintain and extend
- ✅ **Demo Ready**: Pre-configured with sample data and models

---

## 🏗️ System Architecture
User Query
↓
Orchestrator (Priority Router)
↓
┌─────────────────────────────────────┐
│ Tier 1: Dataset Similarity Matcher │ (70% coverage)
│ • 524 curated BFSI Q&A pairs │
│ • all-MiniLM-L6-v2 embeddings │
│ • 0.72 similarity threshold │
└─────────────────────────────────────┘
↓ (if no match >0.72)
┌─────────────────────────────────────┐
│ Tier 2: Fine-tuned SLM │ (20% coverage)
│ • Phi-3.5-mini (LoRA fine-tuned) │
│ • 189 training steps on BFSI data │
│ • 4-bit quantized for efficiency │
└─────────────────────────────────────┘
↓ (if complex query)
┌─────────────────────────────────────┐
│ Tier 3: RAG System │ (10% coverage)
│ • 11 PDFs with 19 tables │
│ • Hybrid search (BM25 + semantic) │
│ • Cross-encoder reranking │
│ • 62 chunks with vector embeddings │
└─────────────────────────────────────┘



---

## 📊 Dataset

The demo uses a curated dataset of **524 BFSI conversation samples** in Alpaca format:

```json
{
  "instruction": "What is the interest rate for personal loans?",
  "input": "",
  "output": "The interest rate for personal loans ranges from 10.50% to 11.50% per annum..."
}
Dataset Statistics:

Total Samples: 524 (exceeds demo requirement of 150+)

Format: Alpaca (Instruction, Input, Output)

Categories: Loans, Cards, Accounts, Transfers, Disputes

Safety: Includes out-of-domain and rejection responses

🤖 Model
Fine-tuned Model (Tier 2)
The demo uses a LoRA fine-tuned Phi-3.5-mini model specifically trained on BFSI data.

Model Details:

Base Model: microsoft/Phi-3.5-mini-instruct

Fine-tuning: LoRA (Low-Rank Adaptation)

Training Steps: 189

Format: 4-bit quantized for efficient inference

Download Demo Model:
The fine-tuned model weights are available on Google Drive:


📁 https://drive.google.com/drive/folders/1792OpDdYbAbRHXGMyAXXOWZX-LR8s2tr?usp=sharing
After downloading, place the phi35-bfsi-lora folder in:


Vertical_AI/phi35-bfsi-lora/

Document Knowledge Base (Tier 3)
11 PDF Documents covering:

✅ Personal Loans MITC

✅ Home Loans Master Circular

✅ Credit Cards MITC

✅ Vehicle Loans

✅ Savings & Current Accounts

✅ Insurance Products Guide

✅ Fixed & Recurring Deposits

✅ Digital Banking Services

✅ SBI Scholar Loan

✅ Unsecured Personal Loans

Knowledge Base Stats:

Total Documents: 11

Extracted Tables: 19

Total Chunks: 62

Embedding Model: BAAI/bge-base-en-v1.5

Reranker: BAAI/bge-reranker-base

🚀 Quick Start
Prerequisites
Python 3.8+

Google Colab (recommended for demo)

12GB+ RAM

15GB free storage

Step 1: Clone/Download Files
bash
# Create project directory
mkdir -p Vertical_AI/bfsi_assistant
cd Vertical_AI/bfsi_assistant

# Upload all Python files from this repository
Step 2: Download Demo Model
Download the fine-tuned model from:


https://drive.google.com/drive/folders/1792OpDdYbAbRHXGMyAXXOWZX-LR8s2tr?usp=sharing
Place it at: /content/drive/MyDrive/Vertical_AI/phi35-bfsi-lora/

Step 3: Install Dependencies
python
# Install all required packages directly
!pip install -r requirements.txt
Step 4: Configure Paths
Edit config.py and update the base path:

python
DRIVE_BASE = "/content/drive/MyDrive/Vertical_AI"  # Update to your path
Step 5: Create Vector Database (One-time Setup)
python
!python main.py --mode setup-db
This will process the 11 PDFs and create 62 chunks with embeddings.

💻 Usage
Option 1: Gradio Web Interface (Best for Demo)
python
!python main.py --mode ui
This launches a professional web interface with a public shareable link - perfect for demonstrations!

Option 2: Command Line Interface
python
!python main.py --mode cli
Interactive CLI mode for testing and debugging.

Option 3: Quick Test
python
!python main.py --mode test
Runs predefined test queries to verify all three tiers are working.

Option 4: Python API
python
from orchestrator import BFSIOrchestrator

# Initialize (takes ~30 seconds)
orchestrator = BFSIOrchestrator()
orchestrator.initialize()

# Query examples
queries = [
    "What is the interest rate for personal loans?",  # Tier 1
    "Tell me about loan eligibility",                  # Tier 1/2
    "What is LTV ratio for home loans?"               # Tier 3
]

for query in queries:
    result = orchestrator.query(query)
    print(f"Q: {query}")
    print(f"A: {result['answer'][:100]}...")
    print(f"Tier: {result['tier']}")
    print(f"Confidence: {result['confidence']}\n")
📁 Demo Project Structure

bfsi_assistant/
├── config.py                  # Configuration & paths
├── requirements.txt           # Dependencies (pip install directly)
├── main.py                    # Entry point
├── orchestrator.py            # Query router (3-tier logic)
├── tier1_dataset_matcher.py   # Dataset similarity (Tier 1)
├── tier2_slm.py              # Fine-tuned SLM (Tier 2)
├── tier3_rag.py              # RAG system (Tier 3)
├── vector_db_creator.py      # Vector DB creation
├── gradio_ui.py              # Web interface
├── COLAB_QUICKSTART.py       # One-click Colab setup
├── README.md                 # This file
└── USAGE_GUIDE.md            # Detailed usage instructions
⚙️ Demo Configuration
Key settings in config.py for the demo:

python
# Paths (update these for your setup)
DRIVE_BASE = "./Vertical_AI"
DATASET_PATH = os.path.join(DRIVE_BASE, "bfsi_cleaned_dataset.json")
LORA_ADAPTER_PATH = os.path.join(DRIVE_BASE, "phi35-bfsi-lora")
DOCS_PATH = os.path.join(DRIVE_BASE, "rag_docs")
INDEX_PATH = os.path.join(DRIVE_BASE, "rag_index")

# Tier 1 Settings
TIER1_THRESHOLD = 0.72  # Similarity threshold for dataset matches
TIER1_MODEL = "all-MiniLM-L6-v2"

# Tier 2 Settings
TIER2_MAX_TOKENS = 200
TIER2_TEMPERATURE = 0.7

# Tier 3 Settings
TIER3_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TIER3_RERANKER_MODEL = "BAAI/bge-reranker-base"
CHROMA_COLLECTION_NAME = "bfsi_docs_final"
📊 Demo Performance Metrics
Metric	Target	Demo Actual
Tier 1 Coverage	70%	~75%
Tier 2 Coverage	20%	~18%
Tier 3 Coverage	10%	~7%
Average Response Time	<1s	~650ms
Dataset Size	150+	524 ✓
Documents	N/A	11 PDFs
Tables Extracted	N/A	19
Vector Chunks	N/A	62
🔧 Troubleshooting
Common Demo Issues
Issue: "Collection not found"

python
# The database exists but path might be wrong
import chromadb
import os
import config

# Check the actual database location
db_path = os.path.join(config.INDEX_PATH, 'chroma')
print(f"Looking for DB at: {db_path}")
print(f"Exists: {os.path.exists(db_path)}")

# List available collections
client = chromadb.PersistentClient(path=db_path)
print("Collections:", [c.name for c in client.list_collections()])
Issue: Model not loading

python
# Verify model path
import os
import config
print(f"Model path: {config.LORA_ADAPTER_PATH}")
print(f"Exists: {os.path.exists(config.LORA_ADAPTER_PATH)}")
Issue: Vector DB empty or not created

python
# Recreate the database
!python main.py --mode setup-db
Issue: Out of memory in Colab

python
# Restart runtime and use these settings in config.py
USE_MERGED_MODEL = False  # Use LoRA adapter (more memory efficient)
TORCH_DTYPE = "float16"   # Use half precision
Issue: Gradio not launching

python
# Upgrade gradio
!pip install --upgrade gradio
🎯 Demo Examples
Try these sample queries to test the system:

python
test_queries = [
    # Tier 1 (Dataset Match)
    "What is the interest rate for personal loans?",
    "How do I check my loan status?",
    "I forgot my password",
    
    # Tier 2 (SLM Generation)
    "Tell me about personal loan features",
    "What documents do I need for a home loan?",
    
    # Tier 3 (RAG from PDFs)
    "What is the LTV ratio for home loans above 75 lakhs?",
    "Explain the foreclosure policy for loans"
]

for query in test_queries:
    result = orchestrator.query(query)
    print(f"{result['tier'].upper()}: {query} → {result['confidence']}")
📝 License
This demo is provided for evaluation and demonstration purposes only. All financial documents and models are for illustrative use.

🙏 Acknowledgments
Microsoft for Phi-3.5-mini model

Hugging Face for transformers library

ChromaDB for vector storage

Sentence-Transformers for embeddings

Gradio for the web interface

📧 Support
For demo-related issues or questions:

Check the USAGE_GUIDE.md

Run !python main.py --mode test for diagnostics

Review module-specific documentation

Check the troubleshooting section above

Built for BFSI Call Center Operations Demo 🏦
Production-ready code • Compliant by design • Easy to demo