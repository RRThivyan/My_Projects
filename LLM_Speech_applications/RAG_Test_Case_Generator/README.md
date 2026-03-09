## RAG‑Powered Test Case Generator

A Retrieval‑Augmented Generation (RAG) system that automatically generates **use cases and test cases** from product documentation (PDFs, DOCX, images). It combines hybrid retrieval (vector + keyword search) with cross‑encoder reranking and Azure OpenAI to produce high‑quality, structured JSON test scenarios.

This repository also includes a **vision‑language pipeline** using Qwen3‑VL to read UI screenshots (e.g., Booking.com) and a **Streamlit UI** for interactive test‑case generation.

---

## Features

- **Multi‑format document ingestion**
  - PDFs (`.pdf`)
  - Word documents (`.docx`)
  - UI screenshots (`.png`, `.jpg`, `.jpeg`)
- **Vision‑language extraction** using `Qwen/Qwen3‑VL‑2B‑Instruct` for text and structured info from screenshots
- **Hybrid retrieval**
  - FAISS vector search over sentence‑embedding chunks
  - BM25 keyword search over the same corpus
- **Intelligent reranking** via a cross‑encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Structured JSON output** for test cases (positive, negative, boundary scenarios)
- **Azure OpenAI integration** for LLM‑based test‑case generation
- **CLI and Streamlit UI** for running the pipeline

---

## Project Structure

```text
devassure/
├── app.py                     # Streamlit UI – query over existing FAISS index
├── newapp.py                  # Streamlit UI – query + upload new docs to extend index
├── build_index.py             # Offline ingestion and FAISS index builder from sample_data/
├── generate_testcases_singlecode.py
│                               # Standalone script version of the hybrid RAG pipeline
├── main.py                    # CLI entrypoint wrapper (calls rag_testcases.cli:main)
├── rag_testcases/
│   ├── cli.py                 # CLI: run predefined queries and write JSON files to outputs/
│   ├── config.py              # Configuration (paths, model names, queries, thresholds)
│   ├── ingestion.py           # Ingestion utilities + incremental FAISS update
│   ├── models.py              # Vector store and reranker loaders
│   ├── retrieval.py           # Hybrid retrieval (FAISS + BM25) and reranking
│   ├── prompting.py           # Prompt template and context formatting
│   ├── llm.py                 # Azure OpenAI LLM wrapper + JSON parsing
│   ├── pipeline.py            # End‑to‑end pipeline orchestration and validation hook
│   └── evaluation.py          # Basic JSON schema validation for outputs
├── extractor/
│   └── qwen_vl_extractor.py   # Qwen3‑VL vision‑language extractor and chunking helpers
├── sample_data/               # Demo PDFs, DOCX files, and Booking.com screenshots
├── faiss_store/               # Persisted FAISS index (index.faiss, index.pkl)
├── outputs/                   # JSON test‑case files produced by CLI runs
├── requirements.txt           # Python dependencies
├── installation_cmd.txt       # Example pip install command (with extra PyTorch index)
├── .env                       # Local environment variables (not for production use)
└── Readme.txt                 # Original high‑level notes (superseded by this README)
```

---

## Installation

### Prerequisites

- Python **3.9+**
- (Optional) CUDA‑compatible GPU for faster Qwen‑VL and embedding inference
- Azure OpenAI account and deployment (GPT‑based model)

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
```

### 2. Install dependencies

From the `devassure/` directory:

```bash
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

> The extra index is required to pull GPU‑enabled PyTorch wheels. If you are on CPU‑only, you can usually omit the `--extra-index-url` flag and install standard CPU PyTorch builds instead.

### 3. Configure environment variables

Create a `.env` file in `devassure/` (or update the existing one) with your Azure OpenAI details and FAISS settings. **Do not commit real keys to source control.**

```text
# =========================
# FAISS CONFIGURATION
# =========================
FAISS_DIR=faiss_store
EMBED_MODEL=all-mpnet-base-v2

# =========================
# AZURE OPENAI CONFIGURATION
# =========================
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_DEPLOYMENT=your_model_deployment_name   # e.g. gpt-5.1
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# Optional: other RAG tuning flags (if you want to mirror config.py)
TOP_K=5
TEMPERATURE=0.2
SHOW_SOURCES=False
DEBUG_MODE=False
```

The `rag_testcases/config.py` module also defines constants such as:

- `FAISS_DIR`, `EMBED_MODEL`
- `TOP_K_VECTOR`, `TOP_K_KEYWORD`, `TOP_K_RERANKED`
- `TEMPERATURE`, `MIN_EVIDENCE`, `DEBUG_MODE`
- `RERANK_MODEL_NAME`
- `MAX_USE_CASES`
- `QUERIES` – default demo queries for the CLI/UI

Update those if you need different defaults.

---

## Preparing Data & Building the Vector Index

### 1. Place your documents

Put your source documents under `sample_data/`:

- Product requirements documents (PRDs) in **PDF** or **DOCX** format
- UI screenshots for flows/features you want test cases for

You can use the included Booking.com samples as a reference.

### 2. Build the FAISS index (offline ingestion)

```bash
cd devassure
python build_index.py
```

This will:

1. Walk `sample_data/` and inspect each file
2. For PDFs and DOCX files:
   - Extract text using PyMuPDF / python‑docx
   - Clean and chunk text into overlapping segments
3. For images:
   - Call Qwen3‑VL (`qwen_vl_extractor.py`) to extract text + structured info
   - Normalize and chunk the output
4. Embed all chunks with `all-mpnet-base-v2`
5. Build a FAISS index and save it under `faiss_store/`

If no valid documents are found, `build_index.py` will raise an error.

---

## Running the RAG Test Case Generator

You can run the pipeline either via **CLI** or via a **Streamlit UI**.

### Option 1 – CLI (batch JSON outputs)

The CLI uses the configuration and pipeline in the `rag_testcases/` package. It runs a set of predefined queries (from `rag_testcases/config.py`) and writes one JSON file per query to `outputs/`.

From inside `devassure/`:

```bash
python main.py
```

This is equivalent to running `python -m rag_testcases.cli`. For each query in `QUERIES`, the CLI will:

1. Load the FAISS vector store from `faiss_store/`
2. Build a BM25 corpus over all chunks
3. Load the cross‑encoder reranker
4. Run **hybrid retrieval → rerank → LLM → JSON validation**
5. Save the JSON output to `outputs/testcases_q<i>_<timestamp>.json`

See the existing samples under `outputs/` for the expected JSON structure.

### Option 2 – Streamlit UI (interactive)

There are two Streamlit apps:

#### `app.py` – query over existing index

```bash
cd devassure
streamlit run app.py
```

Features:

- Choose between **predefined** queries (`QUERIES`) or a **custom** query
- Run the same hybrid RAG pipeline as the CLI
- Inspect the generated JSON in the browser and download it as a file

#### `newapp.py` – upload docs + enhanced UI

```bash
cd devassure
streamlit run newapp.py
```

Additional capabilities:

- Upload PDFs, DOCX, and image files directly from the sidebar
- Files are converted into `langchain_core.documents.Document` objects:
  - PDFs → text pages, falling back to Qwen‑VL when pages are image‑only
  - DOCX → concatenated paragraphs
  - Images → Qwen‑VL extraction
- The uploaded content is embedded and **added on top of** the existing FAISS index using `rag_testcases.ingestion.build_or_update_faiss`
- BM25 corpus is rebuilt to reflect the extended index
- Rich UI showing upload progress, embedding status, and basic metrics
- Visual pipeline steps (Retrieve → Rerank → Generate → Output)
- Summary tab that counts positive/negative/boundary test cases (if present in the JSON)

Both apps use the same underlying RAG pipeline defined in `rag_testcases/`.

---

## RAG Pipeline Overview

Key components (as implemented in `rag_testcases/`):

1. **Ingestion (`ingestion.py`)**
   - `pdf_to_docs`, `docx_to_docs`, `image_to_docs`
   - `build_or_update_faiss` to create or extend the FAISS index

2. **Retrieval (`retrieval.py`)**
   - `build_bm25_corpus` constructs a BM25 index over all stored documents
   - `hybrid_retrieve` combines FAISS similarity search and BM25 keyword search
   - `rerank_docs` reorders merged results via a cross‑encoder and keeps top‑K

3. **Prompting (`prompting.py`)**
   - Builds a structured QA/test‑case prompt with constraints
   - Limits number of `use_cases` via `MAX_USE_CASES`
   - Formats retrieved documents into a readable context block

4. **LLM (`llm.py`)**
   - Wraps `AzureChatOpenAI` with your deployment
   - Enforces JSON‑only output via `JsonOutputParser`

5. **Pipeline (`pipeline.py`)**
   - `run_single_query`: hybrid retrieve → rerank → build context → LLM → clamp `use_cases`
   - `run_and_validate`: adds basic JSON schema validation

6. **Evaluation (`evaluation.py`)**
   - Checks top‑level keys (`query`, `use_cases`, `assumptions`, `missing_info`)
   - Validates that required fields exist in each use case

---

## Output Format

All generators (CLI and Streamlit) ultimately produce a JSON object with this shape (simplified):

```json
{
  "query": "Original query string",
  "use_cases": [
    {
      "use_case_title": "string",
      "goal": "string",
      "preconditions": ["string"],
      "test_data": {"key": "value"},
      "steps": ["string"],
      "expected_results": ["string"],
      "negative_cases": ["string"],
      "boundary_cases": ["string"]
    }
  ],
  "assumptions": ["string"],
  "missing_info": ["string"]
}
```

See the files in `outputs/` for realistic, fully populated examples.

---

## Customization

### Change default queries

Edit `QUERIES` in `rag_testcases/config.py`:

```python
QUERIES = [
    "Generate positive, negative, and boundary test cases for ...",
    "Generate use cases and test cases for ...",
]
``;

Streamlit apps (`app.py` / `newapp.py`) and the CLI will automatically pick up these values.

### Tune retrieval and generation

- Retrieval knobs (in `rag_testcases/config.py`):
  - `TOP_K_VECTOR`, `TOP_K_KEYWORD`, `TOP_K_RERANKED`
  - `MIN_EVIDENCE` (minimum chunks after rerank required to use real context)
- Generation knobs:
  - `TEMPERATURE` for the Azure OpenAI model
  - `MAX_USE_CASES` to limit the number of use cases per query

### Swap embedding or reranker models

Update these in `rag_testcases/config.py` (and `build_index.py` when changing embeddings):

- `EMBED_MODEL` (e.g., another sentence‑transformers model)
- `RERANK_MODEL_NAME` (e.g., a different cross‑encoder)

Make sure the chosen models are compatible with your hardware and available via Hugging Face.

---

## Notes & Limitations

- Vision extraction is only as good as the Qwen‑VL model and screenshot quality
- Complex PDF layouts (tables, multi‑column text) may require extra preprocessing
- Very large document sets may need chunking and indexing tweaks for memory/performance
- Azure OpenAI usage is subject to your subscription limits and costs

---

## Security & Secrets

- Never commit real API keys or sensitive endpoints to version control
- The `.env` file in this repository is for **local experimentation only** – replace any placeholder or demo values with your own secrets and keep them private
- Consider using a secret manager (Key Vault, environment‑level secrets, etc.) in production

---

## License & Usage

This project is intended for educational and internal testing tooling use cases. Ensure you comply with the licenses and terms of:

- Qwen3‑VL and associated models
- Azure OpenAI service
- FAISS and other open‑source libraries listed in `requirements.txt`

